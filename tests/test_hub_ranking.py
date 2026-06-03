"""Tests for hub ranking, stability, and export schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import pytest

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.hub.ranking import (
    compute_hub_scores,
    add_confidence_intervals,
    _normalize,
)
from scgnn.train.ensemble import compute_hub_stability, format_metric_with_ci
from scgnn.export.schema import validate_hub_ranking, validate_calibration, HUB_RANKING_REQUIRED


def _registry():
    return NodeRegistry.from_config(["USDC", "USDT", "DAI"], ["binance", "coinbase"])


def _simple_digraph(registry: NodeRegistry) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = registry.node_strs()
    for n in nodes:
        G.add_node(n)
    # USDC/binance as central hub
    for n in nodes[1:]:
        G.add_edge("USDC/binance", n, weight=0.8, correlation=0.8)
    return G


def test_hub_scores_shape():
    reg = _registry()
    G = _simple_digraph(reg)
    masks = {str(n): float(i) * 0.1 for i, n in enumerate(reg)}
    propagators = {"USDC/binance": 1, "USDT/binance": 1}
    df = compute_hub_scores(reg, masks, G, propagators)
    assert len(df) == len(reg)
    assert "hub_score" in df.columns
    assert "rank_full" in df.columns
    assert "rank_structural" in df.columns
    assert "hub_score_structural" in df.columns


def test_hub_scores_in_unit_interval():
    reg = _registry()
    G = _simple_digraph(reg)
    masks = {str(n): float(i + 1) * 0.2 for i, n in enumerate(reg)}
    df = compute_hub_scores(reg, masks, G, {})
    assert df["hub_score"].between(0.0, 1.0).all()


def test_hub_scores_rank_1_has_highest_score():
    reg = _registry()
    G = _simple_digraph(reg)
    masks = {str(n): float(i + 1) * 0.3 for i, n in enumerate(reg)}
    df = compute_hub_scores(reg, masks, G, {"USDC/binance": 1})
    rank1 = df[df["rank_full"] == 1].iloc[0]
    assert rank1["hub_score"] == df["hub_score"].max()


def test_normalize_bounds():
    s = pd.Series([0.0, 1.0, 2.0, 5.0])
    n = _normalize(s)
    assert abs(n.min()) < 1e-9
    assert abs(n.max() - 1.0) < 1e-9


def test_add_confidence_intervals():
    reg = _registry()
    G = _simple_digraph(reg)
    masks = {str(n): 0.5 for n in reg}
    df = compute_hub_scores(reg, masks, G, {})
    seeds = [{str(n): 0.5 + i * 0.05 for n in reg} for i in range(5)]
    df2 = add_confidence_intervals(df, seeds)
    assert "ci_lo" in df2.columns
    assert "ci_hi" in df2.columns
    assert (df2["ci_hi"] >= df2["ci_lo"]).all()


def test_hub_stability_perfect_agreement():
    scores = [{"A": 1.0, "B": 0.5, "C": 0.2}] * 5
    result = compute_hub_stability(scores)
    assert result["mean_rho"] == pytest.approx(1.0, abs=1e-6)


def test_hub_stability_random_is_low():
    rng = np.random.default_rng(0)
    scores = [{f"node_{i}": rng.random() for i in range(10)} for _ in range(5)]
    result = compute_hub_stability(scores)
    # Random rankings should have low correlation
    assert result["mean_rho"] < 0.8


def test_format_metric_with_ci():
    s = format_metric_with_ci(0.812, 0.798, 0.826)
    assert "0.812" in s
    assert "0.798" in s
    assert "0.826" in s


def test_validate_hub_ranking_missing_columns():
    df = pd.DataFrame({"node": ["USDC/binance"], "hub_score": [0.8]})
    missing = validate_hub_ranking(df)
    assert len(missing) > 0
    assert "rank" in missing


def test_validate_hub_ranking_complete():
    df = pd.DataFrame({col: ["x"] for col in HUB_RANKING_REQUIRED})
    missing = validate_hub_ranking(df)
    assert missing == []
