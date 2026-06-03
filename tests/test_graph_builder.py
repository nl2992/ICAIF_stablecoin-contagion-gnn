"""Unit tests for graph construction."""
import numpy as np
import pandas as pd
import pytest

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.graphs.builder import build_snapshot_graph, lead_lag_minutes
from scgnn.features.edge_features import rolling_correlation


def _dummy_returns(node_strs, n=500):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.001, (n, len(node_strs))), index=idx, columns=node_strs)


def test_node_registry_count():
    reg = NodeRegistry.from_config(["USDC", "USDT"], ["binance", "curve"])
    assert len(reg) == 4


def test_node_id_str():
    n = NodeID("USDC", "binance")
    assert str(n) == "USDC/binance"


def test_build_snapshot_graph_has_nodes():
    nodes = [NodeID("USDC", "binance"), NodeID("USDT", "binance"), NodeID("DAI", "binance")]
    node_strs = [str(n) for n in nodes]
    returns = _dummy_returns(node_strs)
    t = returns.index[-1]
    G = build_snapshot_graph(returns, nodes, t, corr_threshold=0.0)
    assert len(G.nodes) == 3


def test_rolling_correlation_bounds():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2023-01-01", periods=200, freq="1min", tz="UTC")
    s1 = pd.Series(rng.normal(0, 1, 200), index=idx)
    s2 = pd.Series(rng.normal(0, 1, 200), index=idx)
    r = rolling_correlation(s1, s2)
    assert -1.0 <= r <= 1.0


def test_lead_lag_returns_int():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=300, freq="1min", tz="UTC")
    s1 = pd.Series(rng.normal(0, 1, 300), index=idx)
    s2 = s1.shift(5).fillna(0)
    lag = lead_lag_minutes(s1, s2, max_lag=20)
    assert isinstance(lag, int)


def test_no_self_loops():
    nodes = [NodeID("USDC", "binance"), NodeID("USDT", "binance")]
    returns = _dummy_returns([str(n) for n in nodes])
    t = returns.index[-1]
    G = build_snapshot_graph(returns, nodes, t, corr_threshold=0.0)
    for u, v in G.edges():
        assert u != v
