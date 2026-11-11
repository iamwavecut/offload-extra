"""MiniMax-H3-Image OrbitQuant W4A8 on a 16 GB GPU with partial offload and memory-mapped host weights.

    hf download WaveCut/MiniMax-H3-Image-OrbitQuant-W4A8 --local-dir h3 --exclude 'assets/*'
    pip install "orbitquant[hf]" git+https://github.com/iamwavecut/offload-extra
    python examples/minimax_h3_image.py --release h3 --cache /fast/disk/h3-cache --size 2048 --images 3

The release loader builds each fused model on the GPU. Each model is registered right after it loads: its state is
written once to a safetensors file in ``--cache`` and the model's tensors become memory-mapped views of that file,
which frees its GPU memory before the next model loads. ``--cold`` drops the cache files from the page cache before
every image (a host without spare RAM): uploads then read the disk.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import offload_extra

ap = argparse.ArgumentParser()
ap.add_argument("--release", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--size", type=int, default=2048)
ap.add_argument("--min-generation-area", type=int, help="1024 for a native 1024x1024 image")
ap.add_argument("--attention", default="cudnn")
ap.add_argument("--images", type=int, default=3)
ap.add_argument("--policy", default="belady")
ap.add_argument("--host-embeddings", action="store_true")
ap.add_argument("--cold", action="store_true")
ap.add_argument("--prompt", default="A lighthouse on a cliff above a stormy sea at dusk, cinematic light")
ap.add_argument("--save")
args = ap.parse_args()

root = Path(args.release)
sys.path.insert(0, str(root))
import minimax_h3_image_orbitquant as release_loader  # noqa: E402
from diffusers import AutoencoderKLMiniMaxH3Image, ComponentsManager, MiniMaxH3Scheduler, ModularPipeline  # noqa: E402
from orbitquant.fused import release  # noqa: E402
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor  # noqa: E402

offload = offload_extra.PartialOffload("cuda", policy=args.policy, host="mmap", cache_dir=args.cache,
                                         host_embeddings=args.host_embeddings, timing=True)
t0 = time.perf_counter()
transformer = release_loader.load_transformer(root / "transformer", attention=args.attention)
release(transformer)  # the fused runtime's work buffers
offload.add_model("transformer", transformer, cache_key=str(root.resolve()))
torch.cuda.empty_cache()
text_encoder = release_loader.load_text_encoder(root / "text_encoder")
release(text_encoder.model.language_model)
offload.add_model("text_encoder", text_encoder, cache_key=str(root.resolve()))
torch.cuda.empty_cache()
vae = release_loader.speed_up_vae(
    AutoencoderKLMiniMaxH3Image.from_pretrained(str(root / "vae"), torch_dtype=torch.bfloat16))
offload.add_model("vae", vae, cache_key=str(root.resolve()))
pipe = ModularPipeline.from_pretrained(str(root), components_manager=ComponentsManager())
pipe.update_components(
    transformer=transformer, text_encoder=text_encoder, vae=vae,
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer")),
    processor=Qwen3VLProcessor.from_pretrained(str(root / "processor")),
    scheduler=MiniMaxH3Scheduler.from_pretrained(str(root / "scheduler")),
)
offload_extra.attach(pipe, offload)
print(json.dumps({"load_s": round(time.perf_counter() - t0, 1), "units": offload.describe()}), flush=True)

extra = {"min_generation_area": args.min_generation_area} if args.min_generation_area else {}


def generate(seed):
    return pipe(prompt=args.prompt, height=args.size, width=args.size, generator=torch.Generator().manual_seed(seed),
                output="images", **extra)[0]


with torch.inference_mode():
    generate(1)  # warm-up: records the trace and learns the stage reserves of this size
    for _ in range(args.images):
        if args.cold:
            offload.drop_page_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        image = generate(42)
        torch.cuda.synchronize()
        report = offload.last_call
        print(json.dumps({"seconds": round(time.perf_counter() - start, 2), "h2d_gib": report["h2d_gib"],
                          "upload_s": report["upload_s"],
                          "resident_gib": {k: v["resident_gib"] for k, v in report["stages"].items()}}), flush=True)
if args.save:
    image.save(args.save)
