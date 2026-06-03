"""
Edge feature builders.

Edge features are computed for each (node_i, node_j, snapshot_t) triple
and stored as a dense matrix aligned with the graph's edge_index.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scgnn.utils.time import last_window, last_window_series


def rolling_correlation(
    s1: pd.Series,
    s2: pd.Series,
    window: str = "6h",
    min_periods: int = 10,
) -> float:
    """Pearson correlation over the most recent `window` of data."""
    combined = pd.concat([s1, s2], axis=1).dropna()
    if len(combined) < min_periods:
        return 0.0
    r, _ = pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(r) if not np.isnan(r) else 0.0


def cross_pool_flow(
    flow_df: pd.DataFrame,
    node_i: str,
    node_j: str,
    window: str = "6h",
) -> float:
    """
    Total USD flow from pool_i to pool_j within the rolling window.
    flow_df has columns like "USDC/curve_3pool→USDC/binance" (directional).
    """
    col = f"{node_i}→{node_j}"
    if col not in flow_df.columns:
        return 0.0
    return float(last_window_series(flow_df[col], window).sum())


def lead_lag_minutes(
    s1: pd.Series,
    s2: pd.Series,
    max_lag: int = 120,
    min_periods: int = 10,
) -> int:
    """
    Signed lead-lag: positive value means s1 leads s2 by that many minutes.
    Uses cross-correlation peak (matches contagion-repo permutation approach).
    """
    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        shifted = s2.shift(lag).dropna()
        aligned = s1.reindex(shifted.index).dropna()
        shifted = shifted.reindex(aligned.index)
        if len(aligned) < min_periods:
            continue
        r, _ = pearsonr(aligned.values, shifted.values)
        if not np.isnan(r) and r > best_corr:
            best_corr = r
            best_lag = lag
    return best_lag


def shared_lp_pct(
    lp_sets: Dict[str, set],
    node_i: str,
    node_j: str,
) -> float:
    """
    Fraction of liquidity providers shared between two pools.
    lp_sets[node] = set of LP wallet addresses.
    """
    if node_i not in lp_sets or node_j not in lp_sets:
        return 0.0
    a, b = lp_sets[node_i], lp_sets[node_j]
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def build_edge_feature_matrix(
    returns: pd.DataFrame,           # columns = node str IDs, index = 1-min timestamps
    node_pairs: List[Tuple[str, str]],
    t: pd.Timestamp,
    lookback: str = "6h",
    flow_df: Optional[pd.DataFrame] = None,
    lp_sets: Optional[Dict[str, set]] = None,
) -> np.ndarray:
    """
    Build an (E, D) edge feature matrix for a given snapshot at time t.
    node_pairs defines edge ordering (must match edge_index in the graph).
    Returns float32 array of shape (len(node_pairs), 4).
    """
    window = last_window(returns[returns.index <= t], lookback)
    rows = []
    for ni, nj in node_pairs:
        s1 = window.get(ni, pd.Series(dtype=float))
        s2 = window.get(nj, pd.Series(dtype=float))
        corr = rolling_correlation(s1, s2, window=lookback) if len(s1) and len(s2) else 0.0
        flow = cross_pool_flow(flow_df, ni, nj, lookback) if flow_df is not None else 0.0
        flow_log = np.log1p(abs(flow)) * np.sign(flow) if flow != 0.0 else 0.0
        ll = lead_lag_minutes(s1, s2) if len(s1) >= 10 and len(s2) >= 10 else 0
        lp = shared_lp_pct(lp_sets, ni, nj) if lp_sets is not None else 0.0
        rows.append([corr, float(flow_log), float(ll), float(lp)])
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 4), dtype=np.float32)
