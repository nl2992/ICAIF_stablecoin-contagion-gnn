"""
Active-node filter.

The full node universe (8 assets × 5 venues × fee tiers) is too large
against only 7 episodes.  Most nodes will have zero liquidity in any given
episode — including them is empty padding that dilutes graph structure and
misleads GNNExplainer masks.

Filter rule (documented, pre-registered):
  A node (asset, venue, fee_tier) is ACTIVE in episode E if:
    (a) Price coverage ≥ min_coverage_pct (default 80%) over the episode window
    (b) At least one 1-min bar has volume > 0 in the episode window
    (c) The asset was listed before the episode start date
        (excludes UST and BUSD from episodes after their delist dates)

Mirrors the Uniswap paper's "top-10 pool selection" approach.

The filter is applied per-episode, so a node may be active in some episodes
and inactive in others.  The node registry stays fixed; active masks are
stored per episode in data/processed/active_nodes_{episode}.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.data.integrity import ASSET_STATUS_REGISTRY


def compute_active_mask(
    episode_window: pd.DataFrame,   # columns = node str IDs, index = 1-min timestamps
    episode_start: pd.Timestamp,
    registry: NodeRegistry,
    min_coverage_pct: float = 80.0,
    min_nonzero_volume_bars: int = 10,
) -> Dict[str, bool]:
    """
    Returns {node_str: is_active} for every node in the registry.

    Criteria:
      (a) Price coverage ≥ min_coverage_pct
      (b) At least min_nonzero_volume_bars with non-NaN, non-zero price
      (c) Not delisted before episode_start
    """
    active: Dict[str, bool] = {}

    # Pre-build delisted set for this episode's date
    delisted_at_episode = {
        s.asset for s in ASSET_STATUS_REGISTRY
        if s.delisted_date is not None and s.delisted_date <= episode_start
    }

    for node in registry:
        ns = str(node)

        # (c) Delist check
        if node.asset in delisted_at_episode:
            active[ns] = False
            continue

        if ns not in episode_window.columns:
            active[ns] = False
            continue

        col = episode_window[ns]

        # (a) Coverage
        coverage = float(col.notna().mean() * 100)
        if coverage < min_coverage_pct:
            active[ns] = False
            continue

        # (b) Volume proxy: at least N bars with price > 0 and non-NaN
        valid_bars = int((col.notna() & (col > 0)).sum())
        if valid_bars < min_nonzero_volume_bars:
            active[ns] = False
            continue

        active[ns] = True

    return active


def apply_active_mask(
    X: np.ndarray,                   # (T, N * F) flattened feature matrix
    active_mask: Dict[str, bool],
    registry: NodeRegistry,
    n_features_per_node: int,
) -> tuple[np.ndarray, List[str]]:
    """
    Zero-out feature columns for inactive nodes (rather than dropping, to keep
    fixed shape for the GNN).  Returns masked X and list of active node strings.
    """
    X = X.copy()
    active_nodes = []
    for i, node in enumerate(registry):
        ns = str(node)
        if not active_mask.get(ns, False):
            start = i * n_features_per_node
            end = start + n_features_per_node
            X[:, start:end] = 0.0
        else:
            active_nodes.append(ns)
    return X, active_nodes


def save_active_mask(
    active_mask: Dict[str, bool],
    episode_name: str,
    out_dir: Path = Path("data/processed"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"active_nodes_{episode_name}.json"
    with open(path, "w") as f:
        json.dump(active_mask, f, indent=2)


def load_active_mask(episode_name: str, out_dir: Path = Path("data/processed")) -> Dict[str, bool]:
    path = out_dir / f"active_nodes_{episode_name}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def active_node_summary(
    masks_per_episode: Dict[str, Dict[str, bool]],
    registry: NodeRegistry,
) -> pd.DataFrame:
    """
    Table of active/inactive status per episode × node.
    Flags nodes that are inactive in >50% of episodes.
    """
    rows = []
    for ep_name, mask in masks_per_episode.items():
        row = {"episode": ep_name}
        row.update({ns: mask.get(ns, False) for ns in registry.node_strs()})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("episode")

    # Activity rate per node
    rates = df.mean(axis=0).rename("activity_rate")
    low_activity = rates[rates < 0.5].index.tolist()
    if low_activity:
        print(f"[WARN] {len(low_activity)} nodes inactive in >50% of episodes: {low_activity[:5]}...")
    return df
