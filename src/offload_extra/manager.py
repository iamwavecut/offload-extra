"""Partial weight residency for any set of PyTorch models that run one after another on one GPU.

``PartialOffload`` splits every registered model ("component") into eviction units (``units.py``), loads a unit
right before its forward and keeps it on the device afterwards. When a missing unit needs room, the scheduler
(``scheduler.py``) picks the victims; evicting swaps the unit's tensors back to their host copies (``host.py``), so it
costs no device-to-host copy.

Hooks:

* each unit: a forward pre-hook (load, pin) and a forward hook (unpin); nothing is evicted after a forward;
* each module that owns remainder tensors: a forward pre-hook that starts the component's stage;
* each component: an accelerate ``ModelHook`` with ``execution_device``. Diffusers pipelines read
  ``_execution_device`` from it, and ``apply_forward_hook`` (``vae.decode``) calls its ``pre_forward``.

``device`` of every Diffusers or Transformers model in a component reports the execution device, because some of
their code places inputs on ``model.device`` (Transformers ``generate``).

Calls: ``with offload.call(signature): ...`` marks one pipeline call; ``offload_extra.enable(pipe)`` does it for
every ``pipe(...)``. Without explicit calls, a new call starts when the first unit of the current one is used again
(one token step of ``generate``, one pipeline call).
"""

from __future__ import annotations

import contextlib
import os
import time
import warnings
from collections import Counter
from collections.abc import Callable, Hashable
from pathlib import Path

import torch
from accelerate.hooks import ModelHook, add_hook_to_module, remove_hook_from_module
from accelerate.utils import send_to_device
from torch import nn

from . import host as host_lib
from .scheduler import Scheduler
from .units import discover_units, nbytes, own_tensors

GIB = 1 << 30


def _enable_expandable_segments() -> None:
    """Set ``expandable_segments:True`` in the CUDA caching allocator unless the environment configures it."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "") + "," + os.environ.get("PYTORCH_ALLOC_CONF", "")
    if "expandable_segments" in conf:
        return
    # PyTorch 2.12 deprecates the CUDA-specific setter in favour of the accelerator one; both are private.
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None) or torch.cuda.memory._set_allocator_settings
    try:
        setter("expandable_segments:True")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"offload_extra: could not turn on expandable segments: {exc}", stacklevel=3)


class Unit:
    """Tensors that move together, as ``(store, name, host tensor)``."""

    def __init__(self, name: str, stage: str, tensors: list, modules: list[nn.Module]):
        self.name, self.stage, self.tensors, self.modules = name, stage, tensors, modules
        seen, self.nbytes = set(), 0
        for _, _, host in tensors:
            if host.data_ptr() not in seen:
                seen.add(host.data_ptr())
                self.nbytes += nbytes(host)
        # modules that drop device-side caches in _apply (for example OrbitQuantLinear)
        self.apply_modules = [m for m in modules if type(m)._apply is not nn.Module._apply]


class _StageHook(ModelHook):
    def __init__(self, offload: PartialOffload, stage: str):
        self.offload, self.stage = offload, stage
        self.execution_device = offload.device

    def pre_forward(self, module, *args, **kwargs):
        self.offload._touch(self.stage)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


_DEVICE_CLASSES: dict[tuple[type, str], type] = {}


def _has_device_property(cls: type) -> bool:
    return any(isinstance(vars(klass).get("device"), property) for klass in cls.__mro__)


def _with_device(cls: type, device: torch.device) -> type:
    """A subclass of ``cls`` whose ``device`` property returns ``device``."""
    key = (cls, str(device))
    if key not in _DEVICE_CLASSES:
        _DEVICE_CLASSES[key] = type(cls.__name__, (cls,), {
            "device": property(lambda self: device),
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "_offload_extra_base": cls,
        })
    return _DEVICE_CLASSES[key]


class PartialOffload:
    """Keep model blocks on the GPU across stages and calls; load only what is missing.

    Args:
        device: the execution device (CUDA).
        policy: ``"belady"`` (default) or ``"lru"``.
        max_memory: bytes for weights and activations; default: total device memory minus memory outside PyTorch's
            allocator, measured at the start of every call.
        margin: bytes kept free below ``max_memory`` (allocator fragmentation). Default 1 GiB.
        min_reserve: smallest activation reserve of a stage before its peak is learned. Default 1 GiB.
        host: ``"keep"`` (default; the CPU tensors the models hold) or ``"mmap"`` (safetensors files in
            ``cache_dir``, mapped; see ``host.py``).
        cache_dir: directory of the ``"mmap"`` files.
        min_unit_bytes, max_unit_bytes: unit discovery limits (``units.py``).
        host_embeddings: look token embeddings of at least ``min_unit_bytes`` up on the host and copy only the rows.
        learn_reserve: learn stage reserves from ``torch.cuda.max_memory_allocated`` (resets the peak statistics).
        release_runtimes: release OrbitQuant fused runtimes (``oq_runtime``) when their stage ends.
        timing: synchronize around uploads and report upload seconds.
        expandable_segments: turn on expandable segments in PyTorch's CUDA caching allocator, unless
            ``PYTORCH_CUDA_ALLOC_CONF`` already sets them. Evicted units leave freed blocks between resident ones;
            without expandable segments a large activation can fail to find a contiguous block although enough
            memory is free.
    """

    def __init__(self, device="cuda", *, policy: str = "belady", max_memory: int | None = None,
                 margin: int = GIB, min_reserve: int = GIB, host: str = "keep", cache_dir=None,
                 min_unit_bytes: int = 16 << 20, max_unit_bytes: int = 2 * GIB, host_embeddings: bool = False,
                 learn_reserve: bool = True, release_runtimes: bool = True, timing: bool = False,
                 expandable_segments: bool = True):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("partial offload needs a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if host not in ("keep", "mmap"):
            raise ValueError(f"unknown host mode {host!r}")
        if host == "mmap" and cache_dir is None:
            raise ValueError('host="mmap" needs cache_dir')
        if expandable_segments:
            _enable_expandable_segments()
        self.max_memory = max_memory
        self.margin = margin
        self.host = host
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.min_unit_bytes, self.max_unit_bytes = min_unit_bytes, max_unit_bytes
        self.host_embeddings = host_embeddings
        self.learn_reserve = learn_reserve
        self.release_runtimes = release_runtimes
        self.timing = timing
        self.sched = Scheduler(0, policy=policy, min_reserve=min_reserve)
        self.units: dict[str, Unit] = {}
        self.remainder: dict[str, Unit] = {}
        self.components: dict[str, nn.Module] = {}
        self.cache_files: list[str] = []
        self.stage: str | None = None
        self.upload_s: Counter = Counter()
        self.last_call: dict | None = None
        self.stage_exit: dict[str, list[Callable[[], None]]] = {}
        self._first_block: str | None = None
        self._explicit = False
        self._active = False
        self._foreign = 0
        self._weights = 0
        self._handles: list = []
        self._swapped: list[tuple[nn.Module, type]] = []
        self._patched_forward: list[nn.Module] = []
        self._tensor_owner: dict[int, str] = {}
        self.refresh_capacity()

    # ---- registration ------------------------------------------------------------------------------------------
    def add_model(self, name: str, model: nn.Module, *, units: list[tuple[str, nn.Module]] | None = None,
                  cache_key: str | None = None) -> None:
        """Register ``model`` as the component ``name``. ``units`` overrides discovery with ``(name, module)``
        pairs; ``cache_key`` (``"mmap"`` mode) is mixed into the cache fingerprint (for example a repo revision)."""
        if name in self.components:
            raise ValueError(f"component {name!r} is already registered")
        if any(model is c for c in self.components.values()):
            raise ValueError("this model is already registered as another component")
        if getattr(model, "_hf_hook", None) is not None:
            raise ValueError(f"{name}: the model already has an accelerate hook (another offload is enabled)")
        if self.host == "mmap":
            path = self.cache_dir / f"{name}.safetensors"
            host_lib.map_to_cache(model, path, cache_key)
            self.cache_files.append(str(path))
        else:
            host_lib.move_to_host(model)
        for sub in model.modules():
            for _, tname, tensor in own_tensors(sub):
                if tensor.is_meta:
                    raise ValueError(f"{name}: tensor {tname} is on the meta device; load the weights first")
                owner = self._tensor_owner.get(id(tensor))
                if owner is not None and owner != name:
                    raise NotImplementedError(f"{name} shares the tensor {tname} with component {owner}")
        for sub in model.modules():
            for _, _, tensor in own_tensors(sub):
                self._tensor_owner[id(tensor)] = name

        found = units if units is not None else discover_units(
            model, min_bytes=self.min_unit_bytes, max_bytes=self.max_unit_bytes)
        host_only: list[nn.Module] = []
        if self.host_embeddings:
            found, host_only = self._split_host_embeddings(model, found)
        # which unit claims each module and each tensor
        claimed: dict[int, str] = {}
        for uname, module in found:
            for sub in module.modules():
                claimed[id(sub)] = uname
        for module in host_only:
            for sub in module.modules():
                claimed[id(sub)] = "<host>"
        users: dict[int, set[str]] = {}
        for sub in model.modules():
            owner = claimed.get(id(sub), "<rest>")
            for _, _, tensor in own_tensors(sub):
                users.setdefault(id(tensor), set()).add(owner)
        shared = {tid for tid, owners in users.items() if len(owners) > 1}
        for uname, module in found:
            subs = list(module.modules())
            tensors = [(s, n, t.data if isinstance(t, nn.Parameter) else t)
                       for sub in subs for s, n, t in own_tensors(sub) if id(t) not in shared]
            unit = self._register(Unit(f"{name}.{uname}", name, tensors, subs))
            self._handles.append(module.register_forward_pre_hook(lambda _m, _a, unit=unit: self._enter(unit)))
            self._handles.append(module.register_forward_hook(lambda _m, _a, _o, unit=unit: self._exit(unit)))
        rest_modules = [m for m in model.modules() if id(m) not in claimed]
        rest = [(s, n, t.data if isinstance(t, nn.Parameter) else t)
                for sub in rest_modules for s, n, t in own_tensors(sub)]
        seen = {id(t) for _, _, t in rest}
        for sub in model.modules():  # shared tensors inside units move to the remainder
            for s, n, t in own_tensors(sub):
                if id(t) in shared and id(t) not in seen and claimed.get(id(sub)) != "<host>":
                    seen.add(id(t))
                    rest.append((s, n, t.data if isinstance(t, nn.Parameter) else t))
        self.remainder[name] = self._register(Unit(f"{name}.<rest>", name, rest, rest_modules))
        for module in rest_modules:
            if any(True for _ in own_tensors(module)):
                self._handles.append(module.register_forward_pre_hook(lambda _m, _a, stage=name: self._touch(stage)))
        self.components[name] = model
        self.stage_exit.setdefault(name, [])
        add_hook_to_module(model, _StageHook(self, name))
        for sub in model.modules():
            if _has_device_property(type(sub)) and not hasattr(type(sub), "_offload_extra_base"):
                self._swapped.append((sub, type(sub)))
                sub.__class__ = _with_device(type(sub), self.device)

    def _split_host_embeddings(self, model, found):
        tensors_seen: dict[int, int] = Counter()
        for sub in model.modules():
            for _, _, t in own_tensors(sub):
                tensors_seen[id(t)] += 1
        keep, host_only = [], []
        for uname, module in found:
            if (type(module) is nn.Embedding and module.max_norm is None
                    and tensors_seen[id(module.weight)] == 1 and nbytes(module.weight) >= self.min_unit_bytes):
                host_only.append(module)
                self._patch_embedding(module)
            else:
                keep.append((uname, module))
        return keep, host_only

    def _patch_embedding(self, embedding: nn.Embedding) -> None:
        weight, device = embedding.weight, self.device

        def forward(ids):
            return nn.functional.embedding(ids.cpu(), weight.data, embedding.padding_idx).to(device)

        embedding.forward = forward
        self._patched_forward.append(embedding)

    def _register(self, unit: Unit) -> Unit:
        self.units[unit.name] = unit
        self.sched.add_unit(unit.name, unit.nbytes, unit.stage)
        return unit

    def refresh_capacity(self) -> None:
        """Bytes for resident weights plus the running stage's reserve. Tensors that are alive at the start of a
        call and are not ours (latents, a KV cache, other models) are ``foreign``: they are subtracted here and from
        every observed peak."""
        self._foreign = max(0, torch.cuda.memory_allocated(self.device) - self.sched.used)
        if self.max_memory is not None:
            self.sched.capacity = int(self.max_memory) - self.margin
            return
        free, total = torch.cuda.mem_get_info(self.device)
        outside = total - free - torch.cuda.memory_reserved(self.device)
        self.sched.capacity = total - outside - self._foreign - self.margin

    # ---- calls -------------------------------------------------------------------------------------------------
    @contextlib.contextmanager
    def call(self, signature: Hashable = None):
        """One pipeline call. ``signature`` keys the learned activation reserves (for example the image size)."""
        if self._explicit:  # nested call: the outer one owns the trace
            yield self
            return
        self._finish_call()
        self._start_call(signature, explicit=True)
        try:
            yield self
        finally:
            self._finish_call()

    def _start_call(self, signature, explicit: bool) -> None:
        self.refresh_capacity()
        self.sched.begin_call(signature)
        self.upload_s = Counter()
        self._first_block = None
        self._explicit = explicit
        self._active = True
        self._open()

    def _finish_call(self, leave_stage: bool = True) -> None:
        if not self._active:
            return
        self._close()
        if leave_stage and self.stage is not None:
            self._leave_stage()
        self.last_call = self.report()
        self.sched.end_call()
        self._explicit = False
        self._active = False

    # ---- stages and units --------------------------------------------------------------------------------------
    def _touch(self, stage: str) -> None:
        if not self._active:
            self._start_call(None, explicit=False)
        if stage == self.stage:
            return
        self._close()
        if self.stage is not None:
            self._leave_stage()
        self.stage = stage
        for victim in self.sched.begin_stage(stage):
            self._evict(self.units[victim])
        self._use(self.remainder[stage])
        self._open()

    def _leave_stage(self) -> None:
        self.sched.unpin(self.remainder[self.stage].name)
        if self.release_runtimes:
            for module in self.components[self.stage].modules():
                runtime = module.__dict__.get("oq_runtime")
                if runtime is not None and hasattr(runtime, "release"):
                    runtime.release()
        for callback in self.stage_exit[self.stage]:
            callback()
        self.stage = None

    def _enter(self, unit: Unit) -> None:
        if not self._active:
            self._start_call(None, explicit=False)
        if not self._explicit:
            if self._first_block is None:
                self._first_block = unit.name
            elif unit.name == self._first_block:  # the cycle starts again: one implicit call ended
                self._finish_call(leave_stage=False)
                self._start_call(None, explicit=False)
                self._first_block = unit.name
        self._touch(unit.stage)
        self._close()
        self._use(unit)
        self._open()

    def _exit(self, unit: Unit) -> None:
        self._close()
        self.sched.unpin(unit.name)
        self._open()

    def _use(self, unit: Unit) -> None:
        hit, victims = self.sched.access(unit.name)
        for victim in victims:
            self._evict(self.units[victim])
        if not hit:
            self._load(unit)
        self.sched.pin(unit.name)

    # ---- peaks: non-weight memory between two hook events (the weights do not change in between) --------------
    def _open(self) -> None:
        if self.learn_reserve:
            self._weights = self.sched.used
            torch.cuda.reset_peak_memory_stats(self.device)

    def _close(self) -> None:
        if self.learn_reserve and self.stage is not None:
            peak = torch.cuda.max_memory_allocated(self.device)
            self.sched.observe(self.stage, peak - self._weights - self._foreign)

    # ---- tensor moves ------------------------------------------------------------------------------------------
    def _copy_in(self, unit: Unit) -> None:
        for store, name, host in unit.tensors:
            current = store[name]
            if current.device == self.device:
                continue
            if isinstance(current, nn.Parameter):
                current.data = host.to(self.device)
            else:
                store[name] = host.to(self.device)
        for module in unit.apply_modules:  # derived caches built before the first load
            module._apply(lambda tensor: tensor, recurse=False)

    def _load(self, unit: Unit) -> None:
        if self.timing:
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
        try:
            self._copy_in(unit)
        except torch.OutOfMemoryError:
            for victim in self.sched.release_all(keep=unit.name):
                self._evict(self.units[victim])
            torch.cuda.empty_cache()
            self._copy_in(unit)
        if self.timing:
            torch.cuda.synchronize(self.device)
            self.upload_s[unit.stage] += time.perf_counter() - start

    def _evict(self, unit: Unit) -> None:
        for store, name, host in unit.tensors:
            current = store[name]
            if isinstance(current, nn.Parameter):
                current.data = host
            else:
                store[name] = host
        for module in unit.apply_modules:
            module._apply(lambda tensor: tensor, recurse=False)

    # ---- control -----------------------------------------------------------------------------------------------
    def offload_all(self) -> None:
        """Evict every unit and free the cached device memory (for example before another process needs it)."""
        self._finish_call()
        for victim in self.sched.drop_all():
            self._evict(self.units[victim])
        torch.cuda.empty_cache()

    def drop_page_cache(self) -> None:
        """``"mmap"`` mode on Linux: drop the cache files from the page cache (next uploads read the disk)."""
        host_lib.drop_page_cache(self.cache_files)

    def detach(self) -> None:
        """Remove every hook, move all weights to their host copies and restore the model classes."""
        self.offload_all()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for model in self.components.values():
            remove_hook_from_module(model)
        for module, cls in self._swapped:
            module.__class__ = cls
        self._swapped.clear()
        for module in self._patched_forward:
            module.__dict__.pop("forward", None)
        self._patched_forward.clear()

    # ---- reporting ---------------------------------------------------------------------------------------------
    def report(self) -> dict:
        stages = {}
        resident = self.sched.resident_by_stage()
        for stage in self.components:
            stats = self.sched.stats.get(stage, Counter())
            row = {
                "h2d_gib": round(stats["h2d_bytes"] / GIB, 3),
                "loads": stats["loads"], "hits": stats["hits"], "evictions": stats["evictions"],
                "reserve_gib": round(self.sched.reserve(stage) / GIB, 2),
                "resident_units": resident.get(stage, {}).get("units", 0),
                "resident_gib": round(resident.get(stage, {}).get("bytes", 0) / GIB, 2),
            }
            if self.timing:
                row["upload_s"] = round(self.upload_s[stage], 3)
            stages[stage] = row
        out = {"policy": self.sched.policy, "signature": self.sched.signature,
               "capacity_gib": round(self.sched.capacity / GIB, 2),
               "h2d_gib": round(sum(s["h2d_gib"] for s in stages.values()), 3),
               "overcommits": self.sched.overcommits, "trace_len": len(self.sched.trace), "stages": stages}
        if self.timing:
            out["upload_s"] = round(sum(self.upload_s.values()), 3)
        return out

    def describe(self) -> dict:
        """Units per component: count and GiB, and the remainder size."""
        out = {}
        for stage in self.components:
            units = [u for u in self.units.values() if u.stage == stage and u is not self.remainder[stage]]
            out[stage] = {"units": len(units), "unit_gib": round(sum(u.nbytes for u in units) / GIB, 3),
                          "rest_gib": round(self.remainder[stage].nbytes / GIB, 3)}
        return out
