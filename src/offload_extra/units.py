"""Splitting a model into eviction units.

A unit is a submodule whose tensors move to the device together, right before the submodule's forward runs. A unit
must be called through ``module(...)`` (its forward hooks must fire) and must not have its tensors read by code
outside its own forward. Block classes satisfy both, so discovery prefers them:

1. Every module whose class name is listed in ``_no_split_modules`` or ``_repeated_blocks`` of a model in the
   component (Diffusers ``ModelMixin`` and Transformers ``PreTrainedModel`` declare them) becomes a unit, unless it
   is smaller than ``min_bytes``.
2. A remaining subtree with at least ``min_bytes`` of tensors that contains no such block becomes a unit when it is
   at most ``max_bytes``, or when it has no children with tensors (a large embedding table).
3. Containers (``ModuleList``, ``ModuleDict``) are never units; discovery descends into them.

Everything else (norms, small projections, rotary buffers, tensors shared between units) is the component's
remainder, which loads when the component's stage starts and stays pinned until the stage ends.
"""

from __future__ import annotations

from torch import nn

CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)


def own_tensors(module: nn.Module):
    """``(store, name, tensor)`` for the module's own parameters and buffers (not its children's)."""
    for store in (module._parameters, module._buffers):
        for name, tensor in store.items():
            if tensor is not None:
                yield store, name, tensor


def nbytes(tensor) -> int:
    return tensor.numel() * tensor.element_size()


def block_classes(component: nn.Module) -> set[str]:
    names: set[str] = set()
    for module in component.modules():
        for attr in ("_no_split_modules", "_repeated_blocks"):
            value = getattr(module, attr, None)
            if isinstance(value, (list, tuple, set)):
                names.update(str(v) for v in value)
    return names


def discover_units(component: nn.Module, *, min_bytes: int = 16 << 20, max_bytes: int = 2 << 30,
                   blocks: set[str] | None = None) -> list[tuple[str, nn.Module]]:
    """``(qualified name, module)`` of the eviction units of ``component`` (see the module docstring)."""
    blocks = block_classes(component) if blocks is None else blocks
    size: dict[int, int] = {}
    has_block: dict[int, bool] = {}

    def measure(module: nn.Module) -> tuple[set[int], bool]:
        seen = {id(t): nbytes(t) for _, _, t in own_tensors(module)}
        contains = False
        for child in module.children():
            child_seen, child_contains = measure(child)
            seen.update(child_seen)
            contains = contains or child_contains or type(child).__name__ in blocks
        size[id(module)] = sum(seen.values())
        has_block[id(module)] = contains
        return seen, contains

    measure(component)
    units: list[tuple[str, nn.Module]] = []

    def visit(module: nn.Module, prefix: str, root: bool) -> None:
        total = size[id(module)]
        if total == 0:
            return
        if not root and not isinstance(module, CONTAINERS) and total >= min_bytes:
            if type(module).__name__ in blocks:
                units.append((prefix, module))
                return
            leaf = not any(size[id(child)] for child in module.children())
            if not has_block[id(module)] and (total <= max_bytes or leaf):
                units.append((prefix, module))
                return
        for name, child in module.named_children():
            visit(child, f"{prefix}.{name}" if prefix else name, False)

    visit(component, "", True)
    return units
