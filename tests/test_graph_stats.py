"""Tests for graph statistics."""
import networkx as nx
import pytest

from scgnn.graphs.stats import snapshot_stats, hub_nodes, is_expected_hub


def _simple_graph():
    G = nx.DiGraph()
    for node in ["USDC/binance", "USDT/binance", "DAI/coinbase", "FRAX/coinbase"]:
        G.add_node(node)
    G.add_edge("USDC/binance", "USDT/binance", weight=0.8, correlation=0.8)
    G.add_edge("USDC/binance", "DAI/coinbase", weight=0.6, correlation=0.6)
    G.add_edge("USDT/binance", "FRAX/coinbase", weight=0.5, correlation=0.5)
    return G


def test_snapshot_stats_counts():
    G = _simple_graph()
    stats = snapshot_stats(G)
    assert stats["n_nodes"] == 4
    assert stats["n_edges"] == 3


def test_snapshot_stats_density_range():
    G = _simple_graph()
    stats = snapshot_stats(G)
    assert 0.0 <= stats["density"] <= 1.0


def test_hub_nodes_returns_sorted():
    G = _simple_graph()
    hubs = hub_nodes(G, top_n=2)
    assert len(hubs) == 2
    # Should be sorted descending by betweenness
    assert hubs[0][1] >= hubs[1][1]


def test_is_expected_hub():
    assert is_expected_hub("USDC/binance")
    assert is_expected_hub("DAI/curve/3pool")
    assert not is_expected_hub("FRAX/kraken")
