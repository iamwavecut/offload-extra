"""Residency bookkeeping: which units stay on the device and which leave (pure Python, no torch).

A pipeline call touches eviction units (blocks, layers, the rest of each component) in an order that repeats from
call to call: text encoder, denoiser blocks once per step, VAE decoder. The scheduler keeps units resident within a
byte budget, ``capacity - reserve(stage)``, and picks the victims when a missing unit needs room.

Policies:

* ``"belady"``: Belady's MIN with the future taken from the recorded trace of the previous call. The victim is the
  resident unit whose next use lies furthest ahead. A unit that is not used again in this call is ranked by its
  first use in the next call (the trace wraps). The first call has no trace and falls back to LRU.
* ``"lru"``: the least recently used unit leaves first. On a cyclic pipeline whose working set exceeds the budget
  it evicts every unit before its reuse; it is here for comparison.

The reserve of a stage is the device memory the stage needs besides resident weights (activations, work buffers).
It is learned per call signature (for example the image size) from observed peaks. For a signature it has not seen,
a stage reserves ``max(min_reserve, capacity - stage_bytes)``: the stage keeps at most its own weights resident, as
whole-model offload would.
"""

from __future__ import annotations

import bisect
from collections import Counter, OrderedDict
from collections.abc import Hashable

INF = float("inf")
POLICIES = ("belady", "lru")


class Scheduler:
    def __init__(self, capacity: int, *, policy: str = "belady", min_reserve: int = 1 << 30,
                 reserve_margin: int = 64 << 20):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")
        self.capacity = int(capacity)
        self.policy = policy
        self.min_reserve = int(min_reserve)
        self.reserve_margin = int(reserve_margin)
        self.size: dict[str, int] = {}
        self.stage_of: dict[str, str] = {}
        self.stage_bytes: Counter[str] = Counter()
        self.resident: OrderedDict[str, None] = OrderedDict()  # least recently used first
        self.used = 0
        self.pins: Counter[str] = Counter()
        self.prev: list[str] = []
        self.positions: dict[str, list[int]] = {}
        self.trace: list[str] = []
        self.cursor = 0
        self.learned: dict[tuple, int] = {}
        self.signature: Hashable = None
        self.stage: str | None = None
        self.overcommits = 0
        self.stats: dict[str, Counter] = {}

    # ---- setup -------------------------------------------------------------------------------------------------
    def add_unit(self, name: str, nbytes: int, stage: str) -> None:
        if name in self.size:
            raise ValueError(f"duplicate unit {name!r}")
        self.size[name] = int(nbytes)
        self.stage_of[name] = stage
        self.stage_bytes[stage] += int(nbytes)

    # ---- call and stage boundaries -----------------------------------------------------------------------------
    def begin_call(self, signature: Hashable = None) -> None:
        self.signature = signature
        self.trace = []
        self.cursor = 0
        self.stats = {}
        self.overcommits = 0

    def end_call(self) -> None:
        """Keep this call's trace as the forecast of the next call."""
        if self.trace:
            self.prev = self.trace
            self.positions = {}
            for index, unit in enumerate(self.prev):
                self.positions.setdefault(unit, []).append(index)
        self.trace = []
        self.cursor = 0

    def begin_stage(self, stage: str) -> list[str]:
        """Enter ``stage``; returns the units to evict so that the stage's reserve fits."""
        self.stage = stage
        return self._make_room(0, keep=None)

    # ---- accesses ----------------------------------------------------------------------------------------------
    def access(self, unit: str) -> tuple[bool, list[str]]:
        """Record a use of ``unit``. Returns ``(hit, victims)``: on a miss the caller evicts the victims, then loads
        the unit. The unit counts as resident from here on."""
        self.trace.append(unit)
        positions = self.positions.get(unit)
        if positions:
            index = bisect.bisect_left(positions, self.cursor)
            if index < len(positions):
                self.cursor = positions[index] + 1
        stats = self._stats(self.stage_of[unit])
        if unit in self.resident:
            self.resident.move_to_end(unit)
            stats["hits"] += 1
            return True, []
        victims = self._make_room(self.size[unit], keep=unit)
        self.resident[unit] = None
        self.used += self.size[unit]
        stats["loads"] += 1
        stats["h2d_bytes"] += self.size[unit]
        return False, victims

    def pin(self, unit: str) -> None:
        self.pins[unit] += 1

    def unpin(self, unit: str) -> None:
        if self.pins[unit] <= 1:
            self.pins.pop(unit, None)
        else:
            self.pins[unit] -= 1

    def release_all(self, keep: str | None = None) -> list[str]:
        """Evict every unpinned resident unit except ``keep`` (after an out-of-memory error)."""
        victims = [u for u in self.resident if u != keep and u not in self.pins]
        for unit in victims:
            self._drop(unit)
        return victims

    def drop_all(self) -> list[str]:
        """Evict every resident unit (offload everything)."""
        victims = list(self.resident)
        for unit in victims:
            self._drop(unit)
        self.pins.clear()
        return victims

    # ---- reserve -----------------------------------------------------------------------------------------------
    def reserve(self, stage: str | None = None) -> int:
        stage = stage or self.stage
        learned = self.learned.get((self.signature, stage))
        if learned is not None:
            return learned
        return max(self.min_reserve, self.capacity - self.stage_bytes.get(stage, 0))

    def observe(self, stage: str, non_weight_bytes: int) -> None:
        """A measured peak of non-weight device memory while ``stage`` ran."""
        key = (self.signature, stage)
        value = int(non_weight_bytes) + self.reserve_margin
        if value > self.learned.get(key, 0):
            self.learned[key] = value

    # ---- policy ------------------------------------------------------------------------------------------------
    def next_use(self, unit: str) -> float:
        positions = self.positions.get(unit)
        if not positions:
            return INF
        index = bisect.bisect_left(positions, self.cursor)
        if index < len(positions):
            return positions[index] - self.cursor
        return len(self.prev) - self.cursor + positions[0]  # first use in the next call

    def _victim(self, keep):
        candidates = [u for u in self.resident if u != keep and u not in self.pins]
        if not candidates:
            return None
        if self.policy == "lru" or not self.prev:
            return candidates[0]
        return max(candidates, key=lambda u: (self.next_use(u), self.size[u]))

    def _make_room(self, extra: int, keep) -> list[str]:
        budget = self.capacity - self.reserve()
        victims = []
        while self.used + extra > budget:
            victim = self._victim(keep)
            if victim is None:
                self.overcommits += 1
                break
            self._drop(victim)
            victims.append(victim)
        return victims

    def _drop(self, unit: str) -> None:
        del self.resident[unit]
        self.used -= self.size[unit]
        stats = self._stats(self.stage)
        stats["evictions"] += 1
        stats["evicted_bytes"] += self.size[unit]

    def _stats(self, stage) -> Counter:
        return self.stats.setdefault(stage, Counter())

    # ---- reporting ---------------------------------------------------------------------------------------------
    def resident_by_stage(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for unit in self.resident:
            row = out.setdefault(self.stage_of[unit], {"units": 0, "bytes": 0})
            row["units"] += 1
            row["bytes"] += self.size[unit]
        return out
