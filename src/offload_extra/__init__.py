"""Partial weight offload for multi-component Diffusers and Transformers models.

Keeps model blocks on the GPU across pipeline stages and calls, loads only the blocks that are missing, and evicts
by Belady's rule over the trace of the previous call.

    import offload_extra
    pipe = DiffusionPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)   # stays on the CPU
    offload = offload_extra.enable(pipe, "cuda")
    image = pipe(prompt).images[0]
    offload.last_call   # host-to-device GiB, loads, hits, resident GiB per component
"""

from importlib import import_module

__version__ = "0.1.0"
_EXPORTS = {
    "PartialOffload": "manager", "Scheduler": "scheduler", "attach": "pipeline", "call_signature": "pipeline",
    "disable": "pipeline", "discover_units": "units", "drop_page_cache": "host", "enable": "pipeline",
    "map_to_cache": "host",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):  # torch loads only when a torch-backed name is used
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'offload_extra' has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)
