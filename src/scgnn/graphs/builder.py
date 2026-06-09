"""
Rolling-window temporal DIRECTED graph construction.

Design decisions (pre-registered):
  - Edges are DIRECTED: i → j means i leads j (positive lead-lag).
    Contagion has a direction (origin → victim); an undirected graph
    discards the lead-lag signal we went to the trouble of computing.
  - Rolling 6h lookback window, updated hourly.
  - Episode-boundary guard: the window is CLIPPED to episode_start so
    edges at the beginning of an episode cannot be built from the
    tail of the previous episode.
  - Candidate pairs: all (i, j) pairs for active nodes in the episode,
    i ≠ j; edge added only when |correlation| ≥ corr_threshold.

Edge direction preserved in PyG data object (asymmetric edge_index).
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
    episode_start: Optional[pd.Timestamp] = None,
    active_mask: Optional[Dict[str, bool]] = None,
) -> nx.DiGraph:
    """
    Build a directed graph snapshot for time t.

    Parameters
    ----------
    price_returns : DataFrame of 1-min log-returns (columns = node str IDs)
    node_ids      : node universe for this snapshot
    t             : current snapshot timestamp
    lookback      : rolling lookback window (default 6h)
    corr_threshold: minimum |correlation| to form an edge
    episode_start : if provided, the window is clipped to max(t-lookback, episode_start)
                    — prevents edges from bleeding across episode boundaries
    active_mask   : {node_str: bool} from compute_active_mask; inactive nodes
                    are added as isolated graph nodes (no edges)

    Edge direction
    --------------
    lead_lag_minutes(s_i, s_j) > 0  →  i leads j  →  edge i → j
    lead_lag_minutes(s_i, s_j) < 0  →  j leads i  →  edge j → i
    We only add each directed edge once (skip reverse duplicate).
    """
    # Episode-boundary guard: clip window start
    window_start = t - pd.Timedelta(lookback)
    if episode_start is not None:
        window_start = max(window_start, episode_start)

    window = price_returns[
        (price_returns.index >= window_start) & (price_returns.index <= t)
    ]

    G = nx.DiGraph() if directed else nx.Graph()

    for node in node_ids:
        ns = str(node)
        is_active = active_mask.get(ns, True) if active_mask else True
        G.add_node(ns, asset=node.asset, venue=node.venue, fee_tier=node.fee_tier,
                   active=is_active)

    node_strs = [str(n) for n in node_ids]

    for i, ni in enumerate(node_strs):
        for j, nj in enumerate(node_strs):
            if i >= j:
                continue   # process each unordered pair once

            # Skip inactive nodes
            if active_mask and not (active_mask.get(ni, True) and active_mask.get(nj, True)):
                continue

            if ni not in window.columns or nj not in window.columns:
                continue

            si = window[ni].dropna()
            sj = window[nj].dropna()
            if len(si) < 10 or len(sj) < 10:
                continue

            corr = rolling_correlation(si, sj)
            if abs(corr) < corr_threshold:
                continue

            # Directed edge: determine direction from lead-lag
            ll = lead_lag_minutes(si, sj)   # positive = ni leads nj

            attrs: dict = {
                "correlation": corr,
                "weight": abs(corr),
                "lead_lag_min": ll,
            }
            if flow_df is not None:
                attrs["cross_pool_flow_usd_log"] = cross_pool_flow(flow_df, ni, nj)
            if lp_sets is not None:
                attrs["shared_lp_pct"] = shared_lp_pct(lp_sets, ni, nj)

            if directed:
                if ll >= 0:
                    G.add_edge(ni, nj, **attrs)   # ni leads nj
                else:
                    # Reverse: nj leads ni; negate lead_lag for the reverse edge
                    attrs["lead_lag_min"] = -ll
                    G.add_edge(nj, ni, **attrs)
            else:
                G.add_edge(ni, nj, **attrs)

    return G


def graph_density_report(G: nx.DiGraph, episode_name: str = "") -> dict:
    """
    Report graph density after active-node masking.
    Warns when density is near zero (GNN message passing has nothing to aggregate).
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    active = sum(1 for _, d in G.nodes(data=True) if d.get("active", True))
    max_possible = active * (active - 1)   # directed, no self-loops
    density = e / max_possible if max_possible > 0 else 0.0
    report = {
        "episode": episode_name,
        "n_nodes_total": n,
        "n_nodes_active": active,
        "n_edges": e,
        "density": round(density, 4),
    }
    if density < 0.05 and active > 2:
        import warnings
        warnings.warn(
            f"Graph density {density:.4f} < 5% for episode '{episode_name}' "
            f"({active} active nodes, {e} edges). "
            "GNN message passing may have nothing to aggregate.",
            stacklevel=2,
        )
    return report


def build_temporal_graph_sequence(
    price_returns: pd.DataFrame,
    node_ids: List[NodeID],
    timestamps: pd.DatetimeIndex,
    lookback: str = "6h",
    corr_threshold: float = 0.4,
    directed: bool = True,
    episode_start: Optional[pd.Timestamp] = None,
    active_mask: Optional[Dict[str, bool]] = None,
) -> List[Tuple[pd.Timestamp, nx.DiGraph]]:
    """One directed snapshot per timestamp — the temporal graph sequence for an episode."""
    return [
        (t, build_snapshot_graph(
            price_returns, node_ids, t, lookback, corr_threshold,
            directed=directed,
            episode_start=episode_start,
            active_mask=active_mask,
        ))
        for t in timestamps
    ]


def graph_to_pyg(
    G: nx.DiGraph,
    registry: NodeRegistry,
    node_feature_matrix: np.ndarray,
    edge_feature_dim: int = 4,
) -> "torch_geometric.data.Data":
    """Convert a NetworkX directed snapshot to a PyG Data object. Direction preserved."""
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

    # Degree-preserving edge-rewiring null (referee test). When the env var
    # SCGNN_REWIRE_SEED is set, permute edge destinations to destroy the specific
    # directed lead-lag topology while preserving edge count, source out-degrees,
    # and the edge-attribute multiset. If the GAT's gain survives rewiring, it came
    # from having edges; if it collapses, it came from the genuine topology.
    import os as _os
    _rw = _os.environ.get("SCGNN_REWIRE_SEED")
    if _rw is not None and len(edges_dst) > 1:
        _rng = np.random.default_rng(int(_rw) + len(edges_dst) + int(sum(edges_src)))
        edges_dst = np.asarray(edges_dst)[_rng.permutation(len(edges_dst))].tolist()

    x = torch.tensor(node_feature_matrix, dtype=torch.float32)
    if edges_src:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, edge_feature_dim), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
