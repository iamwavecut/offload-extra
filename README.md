# offload-extra

Partial weight offload for multi-component Diffusers and Transformers models on one GPU.

Model offload in Diffusers moves whole components: before each stage it evicts every other model and uploads its own
in full, so every pipeline call copies every weight to the GPU again. `offload-extra` splits each model into
blocks, keeps blocks on the GPU across stages and calls, and uploads only the blocks that are missing. When a block
needs room, it evicts the resident block whose next use lies furthest ahead, as recorded in the trace of the previous
call (Belady's rule). Eviction swaps the block back to its host copy and costs no device-to-host copy.
The policy is chosen with `policy=`: `"belady"` (default) and `"lru"` exist today, and the package is meant to
collect more offload algorithms.

On an RTX 4060 Ti 16 GB, a 2048×2048 MiniMax-H3-Image (24.8 GiB of weights) takes 14.8 s instead of 17.4 s with
whole-model offload, and Sana 1600M on a simulated 8 GiB card takes 5.7 s instead of 9.0 s with Diffusers model or
group offload, with identical pixels. See [Results](#results).

Status: 0.1.0, experimental. Not affiliated with Hugging Face.

## Install

```bash
pip install git+https://github.com/iamwavecut/offload-extra
```

With Diffusers and Transformers:

```bash
pip install "offload-extra[hf] @ git+https://github.com/iamwavecut/offload-extra"
```

Requirements: Python 3.10 or later, PyTorch 2.4 or later with CUDA, `accelerate` 0.30 or later, `safetensors` 0.4 or
later. These are the declared minimums; the package was tested only with the versions listed in
[Results](#results). It works with any Diffusers pipeline and any PyTorch model, including Transformers models and
models with custom kernels whose weights are plain tensors.

## When it helps

- A pipeline whose components do not fit in VRAM together and that you call more than once. Each call saves the
  upload of every block that stays resident. The saving grows when the host copies come from disk instead of RAM.
- It does not help when the whole pipeline fits in VRAM (keep it resident), for a single call, or when compute
  dominates: MiniMax-H3-Image at 4096×4096 runs 93.5 s against 95.0 s with whole-model offload.
- The first call at a new image size behaves like whole-model offload, and the second reloads what the first one
  evicted. From the third call on, uploads drop to the steady level. Reserves are learned and kept per size, so a
  size seen before does not repeat this warm-up.

## Quick start

A Diffusers pipeline. Load it on the CPU and do not call `pipe.to("cuda")`:

```python
import torch
from diffusers import DiffusionPipeline
import offload_extra

pipe = DiffusionPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
                                         variant="bf16", torch_dtype=torch.bfloat16)
offload = offload_extra.enable(pipe, "cuda")
image = pipe("a red fox in fresh snow").images[0]
print(offload.last_call)   # H2D GiB, loads, hits, evictions and resident GiB per component
```

Any set of models that run one after another:

```python
offload = offload_extra.PartialOffload("cuda")
offload.add_model("text_encoder", text_encoder)
offload.add_model("denoiser", denoiser)
offload.add_model("decoder", decoder)
with offload.call(signature=(height, width)):   # one pipeline call
    ...
```

A Transformers model with `generate`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import offload_extra

repo = "..."   # any causal language model
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)   # stays on the CPU
offload = offload_extra.PartialOffload("cuda")
offload.add_model("llm", model)
ids = tokenizer("The capital of France is", return_tensors="pt").input_ids   # generate moves it to model.device
print(tokenizer.decode(model.generate(ids, max_new_tokens=64)[0]))
```

Without `call(...)`, a new call starts when the first block of the current one runs again: each token step of
`generate` is one call.

Memory-mapped host copies (`host="mmap"`) write each model's state once to a safetensors file in `cache_dir` and
make its tensors views of that file. The weights are then reclaimable page cache instead of process memory, and
models that a loader builds on the GPU can be registered one by one, each freeing its GPU memory before the next
loads. See [`examples/minimax_h3_image.py`](examples/minimax_h3_image.py).

## How it works

**Units.** Each registered model ("component") is split into eviction units. Every module whose class is listed in
`_no_split_modules` or `_repeated_blocks` (Diffusers and Transformers declare their block classes there) is a unit;
any other subtree of at least `min_unit_bytes` without such blocks is a unit when it is at most `max_unit_bytes` or
has no children with tensors (a large embedding). Everything else, and tensors shared between units (tied
embeddings), is the component's remainder. A unit loads right before its forward and stays resident after it.

**Stages.** A component's stage starts when any of its modules runs, or when Diffusers calls its accelerate hook
(`vae.decode`, the modular text-encoder step). The stage's remainder stays pinned until another stage starts.

**Calls and the scheduler.** The order of unit uses in a call is recorded. In the next call, a missing unit evicts
the unpinned resident unit with the furthest next use in that trace; a unit not used again in this call is ranked by
its first use in the next call. The first call falls back to LRU. Plain LRU is available as `policy="lru"` for
comparison: on a pipeline whose call cycle does not fit in VRAM it evicts every block before its reuse.

**Budget.** The budget is total device memory minus memory outside PyTorch's allocator, minus tensors that are alive
at the start of a call and are not ours (latents, a KV cache, other models), minus `margin` (1 GiB, for allocator
fragmentation). The resident weights may use the budget minus the running stage's activation reserve. The reserve is
learned per call signature (the image size) from `torch.cuda.max_memory_allocated` between hook events. For an unseen
signature a stage reserves everything except its own weights, as whole-model offload would. So the first call at a
new image size evicts like whole-model offload, the second call reloads what the first one evicted, and from the
third call on each call uploads only the units that do not stay resident.

**Device.** `model.device` of every Diffusers and Transformers model in a component reports the execution device
(`generate` places its inputs there), and each component carries an accelerate hook with `execution_device`, which
Diffusers pipelines read.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `policy` | `"belady"` | `"belady"` or `"lru"` |
| `max_memory` | all free memory | bytes for weights and activations |
| `margin` | 1 GiB | kept free for allocator fragmentation |
| `min_reserve` | 1 GiB | smallest stage reserve before the stage's peak is learned |
| `host` | `"keep"` | `"keep"`: the CPU tensors the models hold; `"mmap"`: safetensors files in `cache_dir` |
| `min_unit_bytes`, `max_unit_bytes` | 16 MiB, 2 GiB | unit discovery limits |
| `host_embeddings` | `False` | gather token rows from the host table and copy only those rows |
| `learn_reserve` | `True` | learn stage reserves (resets PyTorch's peak memory statistics) |
| `release_runtimes` | `True` | release OrbitQuant fused runtimes when their stage ends |
| `timing` | `False` | synchronize around uploads and report upload seconds |
| `expandable_segments` | `True` | turn on expandable segments in PyTorch's CUDA allocator, unless `PYTORCH_CUDA_ALLOC_CONF` sets them |

Expandable segments matter. Evicted units leave freed blocks between resident ones. With the default allocator, a
large activation (a VAE decoder's 512 MiB buffer) can fail to find a contiguous block although more than enough
memory is free: the first Sana run at a simulated 8 GiB failed this way with 1.24 GiB reserved but unallocated.
The option changes a process-wide allocator setting through a private PyTorch function; set
`PYTORCH_CUDA_ALLOC_CONF` yourself to control it.

`offload.last_call` reports the latest call; `offload.describe()` lists the units; `offload.offload_all()` evicts
everything; `offload.detach()` (or `offload_extra.disable(pipe)`) restores the models.

## Troubleshooting

- **CUDA out of memory inside a stage.** Raise `margin` or `min_reserve`, or set `max_memory` lower. Check that
  expandable segments are on: another library may have set `PYTORCH_CUDA_ALLOC_CONF` without them.
- **No speed-up.** Read `offload.last_call`. If `h2d_gib` stays at the size of all weights, the call signature
  changes on every call, or the budget leaves no room for resident blocks. Read `offload.describe()`: a component
  with a large `rest_gib` or a single huge unit needs explicit `units=` in `add_model`.
- **"Expected all tensors to be on the same device".** The model's code creates tensors on the device of
  `next(model.parameters())`, which can be the CPU, or it reads a unit's weights outside the unit's forward. Create
  such tensors on the execution device, or pass `units=` so that the weights belong to the remainder.

## Results

All numbers come from one RTX 4060 Ti 16 GB (PCIe 4.0 x8, power limit 150 W of 165 W, 62 GB host RAM) rented on
Vast.ai on 2026-11-11, with PyTorch 2.12.1 (CUDA 13.0), Diffusers 0.42.0.dev0 (875373ac), Transformers 5.22.2 and
accelerate 1.15.0. Times are the best of one to three images after a warm-up image, with `policy="belady"` and the
default margin.

### MiniMax-H3-Image OrbitQuant W4A8

[WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8](https://huggingface.co/WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8): a
13.9 GiB text encoder, a 9.5 GiB DiT and a 1.4 GiB VAE, 24.8 GiB of weights for a 16 GB card. The reference is
whole-model offload from the same memory-mapped files: each stage uploads its full component, 24.8 GiB per image.
Host copies are `host="mmap"`; "disk" drops the files from the page cache before every image.

| Image | Weights read from | Whole-model offload | offload-extra | Uploaded per image |
| --- | --- | ---: | ---: | ---: |
| 1024×1024 (native) | page cache | 6.89 s | **4.76 s** | 9.3 GiB |
| 1024×1024 (native) | disk | 14.80 s¹ | **7.42 s** | 9.3 GiB |
| 2048×2048 | page cache | 17.42 s | **14.83 s** | 9.8 GiB |
| 2048×2048 | disk | 24.14 s¹ | **17.63 s** | 9.8 GiB |
| 4096×4096 | page cache | 94.95 s¹ | **93.52 s** | 12.8 GiB |

¹ Measured on another RTX 4060 Ti instance on the same day with the same software. Its 2048×2048 page-cache time
was 17.40 s against 17.42 s on this one.

- offload-extra keeps the DiT (9.5 GiB), 0.8 GiB of the VAE and 3.0–3.5 GiB of the text encoder resident across
  images and streams the rest of the text encoder, which runs once per image. At 4096×4096 the larger activations
  leave no room for the text encoder, but the DiT's compute dominates the time.
- "native": `min_generation_area=1024` renders at 1024×1024; by default the pipeline renders at least 4 MP.
- The 1024×1024 and 2048×2048 images are bit-identical to whole-model offload (maximum pixel difference 0).
- NVML peak: 15.37 GiB against 14.76–14.81 GiB for whole-model offload; the package fills its budget. The process's
  anonymous memory peaked at 1.5 GiB, because the host copies are page cache.
- `host_embeddings=True` keeps the text encoder's 1.45 GiB token table on the host: 2048×2048 takes 14.78 s from the
  page cache and 17.24 s from disk, with 8.4 GiB uploaded per image.

### Sana 1600M on a simulated smaller card

[Efficient-Large-Model/Sana_1600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers),
fp16 variant loaded as bfloat16: a 4.9 GiB Gemma 2 text encoder, a 3.0 GiB DiT and a 0.6 GiB DC-AE decoder, 1024×1024,
20 steps. `benchmarks/bench_pipeline.py --simulate-gib X` first allocates a tensor that leaves
X GiB free. Every mode runs with expandable segments.

| Free memory | `enable_model_cpu_offload` | `enable_group_offload` (leaf level, stream) | offload-extra | Uploaded per image |
| --- | ---: | ---: | ---: | ---: |
| 8 GiB | 9.05 s | 8.97 s | **5.67 s** | 3.0 GiB |
| 4 GiB | out of memory | 8.99 s | **6.90 s** | 16.2 GiB |

The whole pipeline resident on the full card takes 5.47 s.

- All five offload runs give the same pixels (one SHA-256). The resident run on the full card differs from them
  (maximum 35 of 255, mean 0.23); the cause is not isolated.
- Group offload at block level fails on the DC-AE decoder in this Diffusers build (a convolution outside the blocks
  stays on the CPU), so the table uses leaf level.
- With the package's runtime switch alone (`PYTORCH_CUDA_ALLOC_CONF` empty), 8 GiB takes 5.66 s with the same
  pixels.
- With this configuration every path, the resident run included, renders a blocky background. The images serve
  only to compare the paths.

### Reproduce

The Sana table comes from `benchmarks/bench_pipeline.py` (`--mode` is `model`, `group`, `partial` or `resident`):

```bash
python benchmarks/bench_pipeline.py --model Efficient-Large-Model/Sana_1600M_1024px_diffusers --variant fp16 \
    --mode partial --simulate-gib 8 --images 3 --out results.jsonl
```

The MiniMax-H3-Image numbers come from a lab script that loads the release the same way as
[`examples/minimax_h3_image.py`](examples/minimax_h3_image.py).

### Tests

`pytest` on the same machine: 22 passed. They cover the scheduler (5), unit discovery (4), a toy three-stage pipeline
on the GPU (4: exact Belady transfer volume against LRU, memory-mapped host copies with page-cache drops, implicit
calls, `detach`), four tiny Diffusers pipelines (Flux, SD3, Stable Diffusion, SDXL) with `belady` and `lru`, each
bit-identical to a resident run, and Transformers `generate` on a tiny Llama: 24 greedy tokens identical to a
resident run. Speed on a real Transformers model is not measured yet.

## Limitations

- CUDA only. One process, one device.
- A unit must run through `module(...)` and its tensors must not be read by code outside its forward. Block classes
  satisfy this; pass `units=` to `add_model` for a model that does not.
- No `torch.compile` or CUDA graphs on offloaded modules: the weights change device between calls.
- Tensor subclasses with their own device logic (bitsandbytes `Params4bit`) are not supported.
- Uploads are synchronous; there is no prefetch yet.
- `learn_reserve` resets `torch.cuda` peak memory statistics at every hook event.
- A component that runs inside another component's forward is not supported.

## Adding an algorithm

Eviction decisions live in `src/offload_extra/scheduler.py`, which is pure Python and does not import torch. A
policy is a name in `POLICIES` and a branch in `Scheduler._victim`, which picks the resident unit to evict. It can
use the resident units in least-recently-used order (`self.resident`), the previous call's trace (`self.prev`) and
the distance to a unit's next use (`self.next_use(unit)`). Test a new
policy on a synthetic trace in `tests/test_scheduler.py`; no GPU is needed. `python -m pytest tests` runs the CPU
tests anywhere and the GPU tests on a CUDA machine.

## Related work

Diffusers model, sequential and group offloading move whole components or reload every group on every forward.
ComfyUI's dynamic VRAM (comfy-aimdo) loads weights on use and evicts partially across models by load recency.
This package uses no code from it.

## License

Apache-2.0
