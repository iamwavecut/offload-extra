"""Scheduler tests (pure Python)."""

from offload_extra.scheduler import Scheduler

GIB = 1 << 30


def h3_stages():
    """Unit sizes (GiB) of MiniMax-H3-Image OrbitQuant W4A8 at 2048x2048, as in the byte model of the research."""
    te = [("te.embed", 1.449)] + [(f"te.layer{i}", 11.36 / 50) for i in range(50)]
    dit = [("dit.in", 0.065), ("dit.refiner", 0.359)]
    dit += [(f"dit.block{i}", 9.042 / 50) for i in range(50)] + [("dit.out", 0.012)]
    vae = [(f"vae.dec{i}", 0.975 / 24) for i in range(24)]
    return [("te", te, 1, 0.42), ("dit", dit, 6, 1.05), ("vae", vae, 2, 1.0)]


def run(policy, capacity_gib=15.0, calls=6):
    stages = h3_stages()
    s = Scheduler(int(capacity_gib * GIB), policy=policy, min_reserve=0, reserve_margin=0)
    for stage, units, _, _ in stages:
        for unit, gib in units:
            s.add_unit(unit, int(gib * GIB), stage)
    per_call = []
    for _ in range(calls):
        s.begin_call("2048")
        for stage, _, _, act in stages:
            s.learned[("2048", stage)] = int(act * GIB)
        for stage, units, passes, _ in stages:
            s.begin_stage(stage)
            for _ in range(passes):
                for unit, _ in units:
                    s.access(unit)
        per_call.append(sum(st["h2d_bytes"] for st in s.stats.values()) / GIB)
        s.end_call()
    return per_call, s


def test_belady_matches_the_byte_model():
    per_call, s = run("belady")
    assert abs(per_call[-1] - 9.32) < 0.1  # the research simulator: 9.32 GiB per call
    assert s.resident_by_stage()["dit"]["units"] == 53  # the whole denoiser stays resident


def test_lru_reloads_every_unit():
    per_call, _ = run("lru")
    assert abs(per_call[-1] - 23.26) < 0.05  # every used unit, every call


def test_first_call_reserve_keeps_only_the_stage():
    s = Scheduler(10 * GIB, min_reserve=1 * GIB)
    s.add_unit("a", 3 * GIB, "a")
    s.add_unit("b", 8 * GIB, "b")
    s.begin_call(None)
    assert s.reserve("a") == 7 * GIB  # capacity - stage bytes
    assert s.reserve("b") == 2 * GIB
    s.observe("a", 512 << 20)
    assert s.reserve("a") == (512 << 20) + s.reserve_margin


def test_next_use_wraps_to_the_next_call():
    s = Scheduler(GIB)
    for unit in "xyz":
        s.add_unit(unit, 1, "s")
    s.begin_call(None)
    for unit in "xyzx":
        s.access(unit)
    s.end_call()
    s.begin_call(None)
    s.access("x")
    s.access("y")
    assert s.next_use("z") == 0
    assert s.next_use("x") == 1
    s.access("z")
    s.access("x")
    assert s.next_use("y") == 1  # first use in the next call: position 1 of the trace


def test_pinned_units_are_never_victims():
    s = Scheduler(3, min_reserve=0, reserve_margin=0)
    for unit in "abc":
        s.add_unit(unit, 2, "s")
    s.begin_call(None)
    s.begin_stage("s")
    s.learned[(None, "s")] = 0
    s.access("a")
    s.pin("a")
    hit, victims = s.access("b")
    assert not hit and victims == []  # a is pinned: overcommit instead
    assert s.overcommits == 1
    s.unpin("a")
    hit, victims = s.access("c")
    assert victims == ["a", "b"]  # LRU order on the first call, until c fits


def test_shrink_evicts_the_latest_first_use_first():
    s = Scheduler(100, min_reserve=0, reserve_margin=0)
    for unit in "abcd":
        s.add_unit(unit, 10, "x")
    s.begin_call()
    s.begin_stage("x")
    for unit in "abcdbc":
        s.access(unit)
    s.end_call()
    s.pin("a")
    assert s.shrink(15) == ["d", "c", "b"]  # first uses in the next call: a 0, b 1, c 2, d 3; a is pinned
    assert s.used == 10 and list(s.resident) == ["a"]
    s.unpin("a")
    assert s.shrink(0) == ["a"] and s.used == 0
