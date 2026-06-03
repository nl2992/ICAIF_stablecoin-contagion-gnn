"""
Rolling-window temporal directed graph construction.

Node = NodeID (asset, venue, fee_tier) with stable integer index from NodeRegistry.
Edges are directed (lead-lag asymmetric) and updated every hour using a 6-hour
rolling lookback window — matching the Uniswap paper's edge construction.

Edge direction convention:
  i → j  means "i leads j" (positive lead-lag) or "flow moves from i to j".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.features.edge_features import (
    rolling_correlation,
    cross_pool_flow,
    lead_lag_minutes,
    shared_lp_pct,
)
from scgnn.utils.time import last_window, last_window_series


def build_snapshot_graph(
    price_returns: pd.DataFrame,
    node_ids: List[NodeID],
    t: pd.Timestamp,
    lookback: str = "6h",
    corr_threshold: float = 0.4,
    flow_df: Optional[pd.DataFrame] = None,
    lp_sets: Optional[Dict[str, set]] = None,
    directed: bool = True,
) -> nx.DiGraph:
    """
    Build a directed graph snapshot for time t.

    Edges are added when |correlation| ≥ corr_threshold over the lookback window.
    Direction is set by the sign of lead_lag: if i leads j, edge goes i→j.
    Self-loops are excluded.
    """
    window = last_window(price_returns[price_returns.index <= t], lookback)
    G = nx.DiGraph() if directed else nx.Graph()

    for node in node_ids:
        G.add_node(str(node), asset=node.asset, venue=node.venue, fee_tier=node.fee_tier)

    node_strs = [str(n) for n in node_ids]
    for i, ni in enumerate(node_strs):
        for j, nj in enumerate(node_strs):
            if i == j:
                continue
            if ni not in window.columns or nj not in window.columns:
                continue
            si = window[ni].dropna()
            sj = window[nj].dropna()
            corr = rolling_correlation(si, sj)
            if abs(corr) < corr_threshold:
                continue

            ll = lead_lag_minutes(si, sj)
            # Direction: positive ll → ni leads nj → edge ni→nj
            if directed and ll < 0:
                continue  # nj leads ni; that edge is handled when i,j are swapped

            attrs: dict = {
                "correlation": corr,
                "weight": abs(corr),
                "lead_lag_min": ll,
            }
            if flow_df is not None:
                attrs["cross_pool_flow_usd_log"] = cross_pool_flow(flow_df, ni, nj)
            if lp_sets is not None:
                attrs["shared_lp_pct"] = shared_lp_pct(lp_sets, ni, nj)

            G.add_edge(ni, nj, **attrs)
    return G


def build_temporal_graph_sequence(
    price_returns: pd.DataFrame,
    node_ids: List[NodeID],
    timestamps: pd.DatetimeIndex,
    lookback: str = "6h",
    corr_threshold: float = 0.4,
    directed: bool = True,
) -> List[Tuple[pd.Timestamp, nx.DiGraph]]:
    """One snapshot per timestamp — the temporal graph sequence for an episode."""
    return [
        (t, build_snapshot_graph(price_returns, node_ids, t, lookback, corr_threshold, directed=directed))
        for t in timestamps
    ]


def graph_to_pyg(
    G: nx.DiGraph,
    registry: NodeRegistry,
    node_feature_matrix: np.ndarray,    # (N, F)
    edge_feature_dim: int = 4,
) -> "torch_geometric.data.Data":
    """Convert a NetworkX snapshot to a PyG Data object."""
    from torch_geometric.data import Data
    import torch

    node_strs = registry.node_strs()
    edges_src, edges_dst, edge_attrs = [], [], []
    for u, v, attr in G.edges(data=True):
        if u in node_strs and v in node_strs:
            edges_src.append(node_strs.index(u))
            edges_dst.append(node_strs.index(v))
            edge_attrs.append([
                attr.get("correlation", 0.0),
                attr.get("cross_pool_flow_usd_log", 0.0),
                float(attr.get("lead_lag_min", 0)),
                attr.get("shared_lp_pct", 0.0),
            ])

    x = torch.tensor(node_feature_matrix, dtype=torch.float32)
    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
