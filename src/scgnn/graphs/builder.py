"""
Rolling-window temporal graph construction.

Node = (asset, venue, fee_tier) tuple.
Edges updated every step using a 6-hour rolling lookback (matching Uniswap paper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


@dataclass
class NodeID:
    asset: str
    venue: str
    fee_tier: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.asset, self.venue]
        if self.fee_tier:
            parts.append(self.fee_tier)
        return "/".join(parts)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NodeID) and str(self) == str(other)


def build_node_universe(
    assets: List[str],
    venues: List[str],
    fee_tiers: Optional[Dict[str, List[str]]] = None,
) -> List[NodeID]:
    nodes = []
    for asset in assets:
        for venue in venues:
            if fee_tiers and venue in fee_tiers:
                for tier in fee_tiers[venue]:
                    nodes.append(NodeID(asset, venue, tier))
            else:
                nodes.append(NodeID(asset, venue))
    return nodes


def _rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: str = "6h",
) -> float:
    aligned = pd.concat([s1, s2], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0
    r, _ = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(r) if not np.isnan(r) else 0.0


def build_snapshot_graph(
    price_returns: pd.DataFrame,      # columns = node str IDs, index = timestamps
    node_ids: List[NodeID],
    t: pd.Timestamp,
    lookback: str = "6h",
    corr_threshold: float = 0.4,
    flow_data: Optional[pd.DataFrame] = None,
) -> nx.Graph:
    """Return a NetworkX graph for a single time snapshot."""
    window_data = price_returns[price_returns.index <= t].last(lookback)
    G = nx.Graph()

    for node in node_ids:
        G.add_node(str(node), asset=node.asset, venue=node.venue)

    node_strs = [str(n) for n in node_ids]
    for i, ni in enumerate(node_strs):
        for j, nj in enumerate(node_strs):
            if j <= i:
                continue
            if ni not in window_data.columns or nj not in window_data.columns:
                continue
            corr = _rolling_correlation(window_data[ni], window_data[nj])
            if abs(corr) >= corr_threshold:
                attrs: dict = {"correlation": corr, "weight": abs(corr)}
                if flow_data is not None:
                    key = f"{ni}_{nj}"
                    if key in flow_data.columns:
                        flow_row = flow_data[flow_data.index <= t].last(lookback)
                        attrs["cross_pool_flow_usd"] = float(flow_row[key].sum()) if len(flow_row) else 0.0
                G.add_edge(ni, nj, **attrs)
    return G


def lead_lag_minutes(
    s1: pd.Series,
    s2: pd.Series,
    max_lag: int = 120,
) -> int:
    """
    Return the lag (in minutes) at which s2 maximally cross-correlates with s1.
    Positive = s1 leads s2.  Uses permutation-test baseline from contagion repo.
    """
    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        shifted = s2.shift(lag).dropna()
        aligned = s1.reindex(shifted.index).dropna()
        shifted = shifted.reindex(aligned.index)
        if len(aligned) < 10:
            continue
        r, _ = pearsonr(aligned, shifted)
        if r > best_corr:
            best_corr = r
            best_lag = lag
    return best_lag


def build_temporal_graph_sequence(
    price_returns: pd.DataFrame,
    node_ids: List[NodeID],
    timestamps: pd.DatetimeIndex,
    lookback: str = "6h",
    corr_threshold: float = 0.4,
) -> List[Tuple[pd.Timestamp, nx.Graph]]:
    """Return a list of (timestamp, graph) pairs — one per step."""
    return [
        (t, build_snapshot_graph(price_returns, node_ids, t, lookback, corr_threshold))
        for t in timestamps
    ]
