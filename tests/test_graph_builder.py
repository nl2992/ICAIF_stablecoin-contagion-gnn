"""Unit tests for graph construction."""
import numpy as np
import pandas as pd
import pytest

from scgnn.graphs.builder import (
    NodeID,
    build_node_universe,
    build_snapshot_graph,
    lead_lag_minutes,
)


def _dummy_returns(nodes, n=500):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(rng.normal(0, 0.001, (n, len(nodes))), index=idx, columns=nodes)


def test_node_universe_count():
    assets = ["USDC", "USDT"]
    venues = ["binance", "curve"]
    nodes = build_node_universe(assets, venues)
    assert len(nodes) == 4


def test_node_id_str():
    n = NodeID("USDC", "binance")
    assert str(n) == "USDC/binance"


def test_build_snapshot_graph_has_nodes():
    nodes = [NodeID("USDC", "binance"), NodeID("USDT", "binance"), NodeID("DAI", "binance")]
    node_strs = [str(n) for n in nodes]
    returns = _dummy_returns(node_strs)
    t = returns.index[-1]
    G = build_snapshot_graph(returns, nodes, t, lookback="60min", corr_threshold=0.0)
    assert len(G.nodes) == 3


def test_lead_lag_returns_int():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=300, freq="1min", tz="UTC")
    s1 = pd.Series(rng.normal(0, 1, 300), index=idx)
    s2 = s1.shift(5).fillna(0)
    lag = lead_lag_minutes(s1, s2, max_lag=20)
    assert isinstance(lag, int)
