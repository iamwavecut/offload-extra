"""Real Diffusers pipelines and a Transformers ``generate`` loop with tiny test checkpoints from the Hub (CUDA).

Every model is split into many small units (``min_unit_bytes=1``) and the budget holds about half of the weights,
so units are evicted and reloaded during every call. Outputs must equal the run with every model resident."""

import gc

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
np = pytest.importorskip("numpy")
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

import offload_extra  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.hub]
MIB = 1 << 20
PIPELINES = [
    ("hf-internal-testing/tiny-flux-pipe", {"num_inference_steps": 2, "max_sequence_length": 48}),
    ("hf-internal-testing/tiny-sd3-pipe", {"num_inference_steps": 2}),
    ("hf-internal-testing/tiny-stable-diffusion-torch", {"num_inference_steps": 2}),
    ("hf-internal-testing/tiny-sdxl-pipe", {"num_inference_steps": 2}),
]


def weight_bytes(modules):
    seen, total = set(), 0
    for module in modules:
        for tensor in [*module.parameters(), *module.buffers()]:
            if id(tensor) not in seen:
                seen.add(id(tensor))
                total += tensor.numel() * tensor.element_size()
    return total


def tight(total):
    return {"max_memory": total // 2 + 16 * MIB, "margin": 0, "min_reserve": MIB, "min_unit_bytes": 1}


@pytest.mark.parametrize("repo,kwargs", PIPELINES, ids=[r.split("/")[1] for r, _ in PIPELINES])
@pytest.mark.parametrize("policy", ["belady", "lru"])
def test_pipeline_matches_resident_run(repo, kwargs, policy):
    extra = {"safety_checker": None} if "stable-diffusion" in repo else {}
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, **extra)

    def run():
        return pipe("a red cube on a table", generator=torch.Generator("cpu").manual_seed(0), output_type="np",
                    **kwargs).images

    pipe.to("cuda")
    reference = run()
    pipe.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    modules = [m for m in pipe.components.values() if isinstance(m, torch.nn.Module)]
    offload = offload_extra.enable(pipe, "cuda", policy=policy, **tight(weight_bytes(modules)))
    offload.sched.reserve_margin = 0
    evictions = 0
    for _ in range(3):
        assert np.array_equal(run(), reference)
        evictions += sum(stage["evictions"] for stage in offload.last_call["stages"].values())
        assert offload.last_call["trace_len"] > 0
    assert evictions > 0
    offload_extra.disable(pipe)
    assert not hasattr(type(pipe), "_offload_extra_pipeline_base")


def test_generate_matches_resident_run():
    repo = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = transformers.AutoModelForCausalLM.from_pretrained(repo)
    ids = torch.tensor([[1, 306, 4658, 278, 6593, 310, 2834, 338]])
    settings = {"max_new_tokens": 24, "do_sample": False, "pad_token_id": 0}
    model.to("cuda")
    reference = model.generate(ids.cuda(), **settings)
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    offload = offload_extra.PartialOffload("cuda", **tight(weight_bytes([model])))
    offload.sched.reserve_margin = 0
    offload.add_model("llm", model)
    assert model.device == offload.device  # generate places its inputs on model.device
    for _ in range(2):
        out = model.generate(ids, **settings)  # host input ids: generate moves them to model.device
        assert torch.equal(out.cpu(), reference.cpu())
    assert offload.last_call is not None  # implicit calls: one per token step
    offload.detach()
    assert all(t.device.type == "cpu" for t in model.state_dict().values())
