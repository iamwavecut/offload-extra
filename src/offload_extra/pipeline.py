"""One call for a Diffusers pipeline (``DiffusionPipeline`` or ``ModularPipeline``)."""

from __future__ import annotations

import functools

from torch import nn

from .manager import PartialOffload

# keyword arguments that do not change the activation memory of a call
_NOT_SHAPE = {"num_inference_steps", "seed", "eta", "strength", "true_cfg_scale", "guidance_rescale",
              "callback_steps", "max_sequence_length", "return_dict", "output_type", "output"}
_CALL_CLASSES: dict[type, type] = {}


def call_signature(kwargs: dict) -> tuple:
    """The part of a pipeline call that sets its activation memory: integer, float and boolean arguments (image
    size, frame count, images per prompt, ...), the batch of a prompt list, whether guidance is on, tensor shapes."""
    items = []
    for key, value in sorted(kwargs.items()):
        if key in _NOT_SHAPE:
            continue
        if key == "guidance_scale" and isinstance(value, (int, float)):
            items.append((key, value > 1))
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            items.append((key, value))
        elif isinstance(value, (list, tuple)) and key.endswith("prompt"):
            items.append((key, len(value)))
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            items.append((key, tuple(value.shape)))
    return tuple(items)


def _with_calls(cls: type) -> type:
    if cls not in _CALL_CLASSES:
        original = cls.__call__

        @functools.wraps(original)
        def __call__(self, *args, **kwargs):
            offload = self.__dict__.get("_offload_extra")
            if offload is None:
                return original(self, *args, **kwargs)
            with offload.call(call_signature(kwargs)):
                return original(self, *args, **kwargs)

        _CALL_CLASSES[cls] = type(cls.__name__, (cls,), {
            "__call__": __call__, "__module__": cls.__module__, "__qualname__": cls.__qualname__,
            "_offload_extra_pipeline_base": cls,
        })
    return _CALL_CLASSES[cls]


def enable(pipe, device="cuda", *, components: list[str] | None = None, cache_key: str | None = None,
           **options) -> PartialOffload:
    """Enable partial offload on every ``nn.Module`` component of ``pipe`` (or on ``components``) and mark each
    ``pipe(...)`` as one call. Load the pipeline on the CPU and do not call ``pipe.to(device)``. ``options`` go to
    ``PartialOffload``. Returns the ``PartialOffload``; ``.last_call`` reports the latest call."""
    if "_offload_extra" in pipe.__dict__:
        raise ValueError("partial offload is already enabled on this pipeline")
    offload = PartialOffload(device, **options)
    for name, component in pipe.components.items():
        if components is not None and name not in components:
            continue
        if isinstance(component, nn.Module) and any(True for _ in component.parameters()):
            offload.add_model(name, component, cache_key=f"{cache_key}:{name}" if cache_key else None)
    return attach(pipe, offload)


def attach(pipe, offload: PartialOffload) -> PartialOffload:
    """Mark each ``pipe(...)`` as one call of ``offload``, whose components were registered with ``add_model``
    (for example one by one while loading, so that only one model is ever on the GPU)."""
    if "_offload_extra" in pipe.__dict__:
        raise ValueError("partial offload is already enabled on this pipeline")
    pipe.__dict__["_offload_extra"] = offload
    if not hasattr(type(pipe), "_offload_extra_pipeline_base"):
        pipe.__class__ = _with_calls(type(pipe))
    return offload


def disable(pipe) -> None:
    """Detach partial offload from ``pipe``: weights go to their host copies, hooks and classes are restored."""
    offload = pipe.__dict__.pop("_offload_extra", None)
    if offload is not None:
        offload.detach()
    base = getattr(type(pipe), "_offload_extra_pipeline_base", None)
    if base is not None:
        pipe.__class__ = base
