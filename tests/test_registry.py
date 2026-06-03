"""Tests for NodeRegistry stable integer IDs."""
import json
import tempfile
from pathlib import Path

import pytest

from scgnn.data.registry import NodeID, NodeRegistry


def test_node_id_ordering():
    n1 = NodeID("USDC", "binance")
    n2 = NodeID("USDT", "binance")
    assert n1 < n2


def test_registry_deterministic():
    r1 = NodeRegistry.from_config(["USDT", "USDC"], ["kraken", "binance"])
    r2 = NodeRegistry.from_config(["USDC", "USDT"], ["binance", "kraken"])
    assert r1.node_strs() == r2.node_strs()


def test_registry_round_trip():
    reg = NodeRegistry.from_config(["USDC", "USDT", "DAI"], ["binance", "coinbase"])
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "registry.json"
        reg.save(p)
        reg2 = NodeRegistry.load(p)
    assert reg.node_strs() == reg2.node_strs()
    assert len(reg2) == 6


def test_idx_inverse():
    reg = NodeRegistry.from_config(["USDC"], ["binance", "kraken"])
    for i in range(len(reg)):
        node = reg.node(i)
        assert reg.idx(node) == i


def test_node_id_with_fee_tier():
    n = NodeID("USDC", "uniswap_v3", "500")
    assert str(n) == "USDC/uniswap_v3/500"
    assert NodeID.from_str("USDC/uniswap_v3/500") == n
