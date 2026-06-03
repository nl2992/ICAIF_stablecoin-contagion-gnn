"""
Graph statistics over time — for the positive-rate-over-time figure
and the stress-vs-calm snapshot comparison (Uniswap Fig 3/4 analogs).
"""
from __future__ import annotations

from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def snapshot_stats(G: nx.DiGraph) -> dict:
    """Per-snapshot structural metrics."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)
    # Use undirected version for degree stats
    UG = G.to_undirected()
    degrees = [d for _, d in UG.degree()]
    weights = [attr.get("weight", 0.0) for _, _, attr in G.edges(data=True)]
    return {
        "n_nodes": n,
        "n_edges": e,
        "density": density,
        "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
        "avg_weight": float(np.mean(weights)) if weights else 0.0,
        "max_weight": float(np.max(weights)) if weights else 0.0,
    }


def temporal_stats(
    snapshots: List[Tuple[pd.Timestamp, nx.DiGraph]],
) -> pd.DataFrame:
    """Compute stats for each snapshot and return a time-indexed DataFrame."""
    rows = []
    for t, G in snapshots:
        row = {"timestamp": t}
        row.update(snapshot_stats(G))
        rows.append(row)
    return pd.DataFrame(rows).set_index("timestamp")


def positive_rate_over_time(
    label_series: pd.Series,
    window: str = "24h",
) -> pd.Series:
    """Rolling positive rate — for the Uniswap Fig 4 analog."""
    return label_series.rolling(window, min_periods=1).mean().rename("positive_rate")


def hub_nodes(G: nx.DiGraph, top_n: int = 5) -> List[Tuple[str, float]]:
    """Return (node_str, betweenness) pairs for the top-N betweenness centrality nodes."""
    UG = G.to_undirected()
    bc = nx.betweenness_centrality(UG, weight="weight")
    return sorted(bc.items(), key=lambda x: x[1], reverse=True)[:top_n]


def is_expected_hub(node_str: str) -> bool:
    """Sanity check: is this node one of the known financial conduits?"""
    known_hubs = {"usdc", "usdt", "dai", "curve", "3pool"}
    node_lower = node_str.lower()
    return any(h in node_lower for h in known_hubs)
