"""Unit discovery tests (CPU)."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from offload_extra.units import discover_units  # noqa: E402

KIB = 1 << 10


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(256, 256)
        self.mlp = nn.Linear(256, 256)

    def forward(self, x):
        return self.mlp(self.attn(x))


class Model(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 256)  # 1000 KiB leaf
        self.blocks = nn.ModuleList(Block() for _ in range(4))  # 514 KiB each
        self.norm = nn.LayerNorm(256)  # 2 KiB
        self.head = nn.Sequential(nn.Linear(256, 64), nn.Linear(64, 8))  # 66 KiB


def names(units):
    return [name for name, _ in units]


def test_blocks_and_large_leaves_become_units():
    units = discover_units(Model(), min_bytes=100 * KIB)
    assert names(units) == ["embed", "blocks.0", "blocks.1", "blocks.2", "blocks.3"]


def test_small_modules_stay_in_the_remainder():
    units = discover_units(Model(), min_bytes=600 * KIB)
    assert names(units) == ["embed"]  # blocks are below min_bytes; norm and head are small


def test_size_rule_without_block_classes():
    model = Model()
    units = discover_units(model, min_bytes=50 * KIB, max_bytes=600 * KIB, blocks=set())
    # blocks (514 KiB) become units by size, the ModuleList itself never; head (66 KiB) is a unit too
    assert names(units) == ["embed", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "head"]


def test_large_subtree_is_split_further():
    model = nn.Sequential(nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 512)))  # 2 x 1026 KiB
    units = discover_units(model, min_bytes=100 * KIB, max_bytes=1500 * KIB, blocks=set())
    assert names(units) == ["0.0", "0.1"]
