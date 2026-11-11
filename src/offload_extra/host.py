"""Host copies of the weights.

Evicting a unit never copies device memory back: the unit's tensors point at their host copies again and the device
copies are freed. Two kinds of host copy:

* ``"keep"``: the tensors the model already holds on the CPU (anonymous memory, or whatever the loader produced).
* ``"mmap"``: each component's state is written once to a safetensors file in a cache directory and the module's
  tensors become memory-mapped views of that file. The weights are then clean page cache: the kernel can drop them
  under memory pressure and reads them from disk again on the next upload. The host process itself holds almost no
  anonymous memory for weights.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import torch
from torch import nn

from .units import nbytes, own_tensors

CHUNK = 256 << 20
_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
    torch.bool: "BOOL",
}
for _name, _code in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2"), ("uint16", "U16"),
                     ("uint32", "U32"), ("uint64", "U64")):
    if hasattr(torch, _name):
        _DTYPES[getattr(torch, _name)] = _code
METADATA_KEY = "offload_extra_fingerprint"


def move_to_host(module: nn.Module) -> None:
    """``"keep"`` mode: make sure every tensor of ``module`` lives on the CPU."""
    if any(t.device.type != "cpu" for sub in module.modules() for _, _, t in own_tensors(sub)):
        module.to("cpu")


def _fingerprint(state: dict[str, torch.Tensor], key: str | None) -> str:
    """Names, shapes, dtypes and the first and last 4 KiB of every tensor (and ``key``, when given)."""
    digest = hashlib.sha256((key or "").encode())
    for name, tensor in state.items():
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}|".encode())
        flat = tensor.detach().reshape(-1).view(torch.uint8) if tensor.is_contiguous() else \
            tensor.detach().contiguous().reshape(-1).view(torch.uint8)
        digest.update(flat[:4096].cpu().numpy().tobytes())
        digest.update(flat[-4096:].cpu().numpy().tobytes())
    return digest.hexdigest()


def _header(path: Path) -> dict | None:
    try:
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            return json.loads(handle.read(length))
    except (OSError, ValueError, struct.error):
        return None


def save_streaming(state: dict[str, torch.Tensor], path: Path, metadata: dict[str, str] | None = None) -> None:
    """Write ``state`` as safetensors one bounded chunk at a time (no host copy of the whole model)."""
    header: dict = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for name, tensor in state.items():
        if tensor.dtype not in _DTYPES:
            raise TypeError(f"{name}: dtype {tensor.dtype} cannot be stored in safetensors")
        size = nbytes(tensor)
        header[name] = {"dtype": _DTYPES[tensor.dtype], "shape": list(tensor.shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for tensor in state.values():
            flat = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
            for start in range(0, flat.numel(), CHUNK):
                handle.write(flat[start : start + CHUNK].cpu().numpy().tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def map_to_cache(module: nn.Module, path: str | os.PathLike, key: str | None = None) -> bool:
    """``"mmap"`` mode: write the state of ``module`` to ``path`` unless a file with the same fingerprint exists,
    then make every tensor of the state a memory-mapped view of the file. Tied parameters stay tied. Buffers outside
    the state dict become small host copies. Returns True when the file was written."""
    from safetensors.torch import load_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = module.state_dict(keep_vars=True)
    first: dict[int, str] = {}
    aliases: dict[str, str] = {}
    unique: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if id(tensor) in first:
            aliases[name] = first[id(tensor)]
        else:
            first[id(tensor)] = name
            unique[name] = tensor.detach()
    fingerprint = _fingerprint(unique, key)
    header = _header(path)
    written = not (header and header.get("__metadata__", {}).get(METADATA_KEY) == fingerprint)
    if written:
        save_streaming(unique, path, {METADATA_KEY: fingerprint})
    del unique, state
    mapped = load_file(str(path), device="cpu")
    for alias, name in aliases.items():
        mapped[alias] = mapped[name]
    module.load_state_dict(mapped, strict=True, assign=True)
    for alias, name in aliases.items():  # assign made a new Parameter per name: tie them again
        owner, _, leaf = alias.rpartition(".")
        source_owner, _, source_leaf = name.rpartition(".")
        source = module.get_submodule(source_owner)
        target = module.get_submodule(owner)
        if source_leaf in source._parameters and leaf in target._parameters:
            target._parameters[leaf] = source._parameters[source_leaf]
        elif source_leaf in source._buffers and leaf in target._buffers:
            target._buffers[leaf] = source._buffers[source_leaf]
    for sub in module.modules():
        for name in sub._non_persistent_buffers_set:
            buffer = sub._buffers.get(name)
            if buffer is not None and buffer.device.type != "cpu":
                sub._buffers[name] = buffer.cpu()
    return written


def _file_mappings(path: str):
    path = os.path.realpath(path)
    out = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].strip() == path:
                low, high = (int(v, 16) for v in parts[0].split("-"))
                out.append((low, high))
    return out


def drop_page_cache(paths) -> None:
    """Linux only: drop the mapped cache files ``paths`` from this process and from the page cache, so the next read
    of their host tensors comes from disk (simulates a host without spare RAM). Only the file mappings are dropped;
    the weights are never written, so their private pages equal the file."""
    if not sys.platform.startswith("linux"):
        raise NotImplementedError("drop_page_cache needs Linux (/proc/self/maps, madvise, posix_fadvise)")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    madv_dontneed = 4
    for path in paths:
        for low, high in _file_mappings(path):
            if libc.madvise(ctypes.c_void_p(low), ctypes.c_size_t(high - low), madv_dontneed) != 0:
                raise OSError(ctypes.get_errno(), f"madvise failed for {path}")
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
