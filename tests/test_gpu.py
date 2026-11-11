"""PartialOffload on a toy three-component pipeline (CUDA).

Encoder (6 blocks, 1 pass), denoiser (8 blocks, 4 passes), decoder (2 blocks, 1 pass); every block is 32 MiB and the
budget holds 14 of the 19 units (blocks and heads), so the call cycle does not fit."""

import gc

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
from torch import nn  # noqa: E402

from offload_extra import PartialOffload  # noqa: E402

pytestmark = pytest.mark.gpu
D, MIB = 4096, 1 << 20


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, D, dtype=torch.bfloat16)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class Stack(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self, n):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(n))
        self.head = nn.Linear(D, D, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.full((D,), 0.5, dtype=torch.bfloat16), persistent=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x) * self.scale


def build():
    torch.manual_seed(0)
    return {"enc": Stack(6), "den": Stack(8), "dec": Stack(2)}


def pipeline(models, x):
    h = models["enc"](x)
    for _ in range(4):
        h = models["den"](h)
    return models["dec"](h)


@pytest.fixture(scope="module")
def reference():
    x = torch.randn(4, D, dtype=torch.bfloat16, device="cuda")
    models = {k: m.cuda() for k, m in build().items()}
    with torch.inference_mode():
        out = pipeline(models, x)
    del models
    torch.cuda.empty_cache()
    return x, out


def offloaded(policy="belady", **options):
    gc.collect()
    torch.cuda.empty_cache()
    models = build()
    options = {"max_memory": (14 * 32 + 8) * MIB, "margin": 0, "min_reserve": 8 * MIB, "min_unit_bytes": MIB,
               **options}
    offload = PartialOffload("cuda", policy=policy, **options)
    offload.sched.reserve_margin = 4 * MIB
    for name, model in models.items():
        offload.add_model(name, model)
    return models, offload


def run_calls(models, offload, x, ref, calls=4, drop_cache=False):
    base = torch.cuda.memory_allocated()
    h2d = []
    with torch.inference_mode():
        for call in range(calls):
            if drop_cache and call >= 2:
                offload.drop_page_cache()
            with offload.call("toy"):
                out = pipeline(models, x)
            assert torch.equal(out, ref), f"call {call}: output differs from the resident run"
            held = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
            assert abs(held - offload.sched.used) <= 4 * MIB, (held, offload.sched.used)
            assert all(bool((m.scale.float() == 0.5).all()) for m in models.values())
            h2d.append(offload.last_call["h2d_gib"])
    return h2d


def test_units_are_discovered():
    models, offload = offloaded()
    # blocks by class, the 32 MiB head by size; the remainder keeps the 8 KiB heap buffer
    assert offload.describe() == {
        "enc": {"units": 7, "unit_gib": 0.219, "rest_gib": 0.0},
        "den": {"units": 9, "unit_gib": 0.281, "rest_gib": 0.0},
        "dec": {"units": 3, "unit_gib": 0.094, "rest_gib": 0.0},
    }
    offload.detach()


def test_belady_is_exact_and_copies_less_than_lru(reference):
    x, ref = reference
    models, offload = offloaded("belady")
    belady = run_calls(models, offload, x, ref)
    offload.detach()
    del models, offload
    models, offload = offloaded("lru")
    lru = run_calls(models, offload, x, ref)
    offload.detach()
    assert belady[-1] < 0.6 * lru[-1], (belady, lru)
    assert lru[-1] == pytest.approx(19 * 32 / 1024, abs=0.01)  # LRU reloads every unit every call


def test_mmap_host_and_page_cache_drop(reference, tmp_path):
    x, ref = reference
    models, offload = offloaded("belady", host="mmap", cache_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dec.safetensors", "den.safetensors", "enc.safetensors"]
    run_calls(models, offload, x, ref, drop_cache=True)
    offload.detach()


def test_implicit_calls_and_detach(reference):
    x, ref = reference
    models, offload = offloaded("belady")
    with torch.inference_mode():
        for _ in range(3):
            assert torch.equal(pipeline(models, x), ref)
    assert offload.last_call is not None and offload.last_call["trace_len"] > 0
    offload.detach()
    for model in models.values():
        assert all(t.device.type == "cpu" for t in model.state_dict().values())
        assert not hasattr(model, "_hf_hook")
        assert type(model) is Stack
