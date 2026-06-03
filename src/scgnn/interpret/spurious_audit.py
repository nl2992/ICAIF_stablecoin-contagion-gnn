"""
Spurious correlation audit.

Flags hubs whose importance is explained primarily by raw volume/TVL rather
than network position.  These are "size hubs" not "contagion hubs" and should
be reported separately — they are prime suspects for divergence from causal hubs.

Algorithm:
  1. Rank nodes by hub_score.
  2. For each top-N hub, compute the partial correlation of hub_score with
     (a) volume/TVL features and (b) network features (betweenness, degree),
     after controlling for the other group.
  3. A node is flagged as "spurious" if corr(hub_score, volume) > corr(hub_score, network)
     AND the volume partial correlation exceeds a threshold.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


VOLUME_FEATURES = {"log_vol_1h", "log_vol_1h_lag1", "tvl_usd_log", "amihud"}
NETWORK_FEATURES = {"betweenness", "weighted_degree", "lop_wedge", "kyle_lambda"}
SPURIOUS_THRESHOLD = 0.7   # partial corr with volume > this → flag


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """
    Partial correlation of x and y controlling for z.
    Returns Pearson r of the residuals.
    """
    if len(x) < 5:
        return float("nan")
    # Regress x on z
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z).any(axis=1 if z.ndim > 1 else 0))
    if mask.sum() < 5:
        return float("nan")
    x_, y_, z_ = x[mask], y[mask], z[mask]
    try:
        coef_xz = np.linalg.lstsq(np.column_stack([z_, np.ones(len(z_))]), x_, rcond=None)[0]
        coef_yz = np.linalg.lstsq(np.column_stack([z_, np.ones(len(z_))]), y_, rcond=None)[0]
        resid_x = x_ - np.column_stack([z_, np.ones(len(z_))]) @ coef_xz
        resid_y = y_ - np.column_stack([z_, np.ones(len(z_))]) @ coef_yz
        r, _ = pearsonr(resid_x, resid_y)
        return float(r) if not np.isnan(r) else float("nan")
    except Exception:
        return float("nan")


def audit_spurious_hubs(
    hub_df: pd.DataFrame,
    feature_df: pd.DataFrame,    # rows = nodes, columns = feature names
    top_n: int = 10,
    volume_features: Optional[List[str]] = None,
    network_features: Optional[List[str]] = None,
    threshold: float = SPURIOUS_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag hubs whose ranking is driven by volume/TVL rather than network position.

    hub_df: the output of hub/ranking.py (columns: node, hub_score, ...)
    feature_df: per-node feature matrix (index = node str)

    Returns hub_df enriched with:
        corr_with_volume, corr_with_network, spurious_flag, audit_note
    """
    vol_feats = volume_features or [f for f in VOLUME_FEATURES if f in feature_df.columns]
    net_feats = network_features or [f for f in NETWORK_FEATURES if f in feature_df.columns]

    top_hubs = hub_df.head(top_n).copy()
    top_hubs = top_hubs.set_index("node")

    scores = top_hubs["hub_score"].reindex(feature_df.index, fill_value=0.0).values
    corr_vol_list, corr_net_list, spurious_list, notes_list = [], [], [], []

    for node in top_hubs.index:
        node_score = float(top_hubs.loc[node, "hub_score"])

        # Simple correlation of this node's feature values with hub scores across all nodes
        if vol_feats:
            vol_matrix = feature_df[vol_feats].values
            r_vol_vals = []
            for col in vol_feats:
                col_vals = feature_df[col].values
                if not np.isnan(col_vals).all() and np.std(col_vals) > 0:
                    r, _ = pearsonr(scores, col_vals) if len(scores) > 2 else (0.0, 1.0)
                    r_vol_vals.append(abs(r))
            corr_vol = float(np.mean(r_vol_vals)) if r_vol_vals else float("nan")
        else:
            corr_vol = float("nan")

        if net_feats:
            r_net_vals = []
            for col in net_feats:
                if col in feature_df.columns:
                    col_vals = feature_df[col].values
                    if not np.isnan(col_vals).all() and np.std(col_vals) > 0:
                        r, _ = pearsonr(scores, col_vals) if len(scores) > 2 else (0.0, 1.0)
                        r_net_vals.append(abs(r))
            corr_net = float(np.mean(r_net_vals)) if r_net_vals else float("nan")
        else:
            corr_net = float("nan")

        spurious = (
            not np.isnan(corr_vol) and
            not np.isnan(corr_net) and
            corr_vol > threshold and
            corr_vol > corr_net
        )
        note = ""
        if spurious:
            note = f"Volume/TVL correlation ({corr_vol:.2f}) dominates network ({corr_net:.2f}) — investigate"
        elif np.isnan(corr_vol) or np.isnan(corr_net):
            note = "Insufficient features for audit"

        corr_vol_list.append(corr_vol)
        corr_net_list.append(corr_net)
        spurious_list.append(spurious)
        notes_list.append(note)

    top_hubs["corr_with_volume"] = corr_vol_list
    top_hubs["corr_with_network"] = corr_net_list
    top_hubs["spurious_flag"] = spurious_list
    top_hubs["audit_note"] = notes_list
    return top_hubs.reset_index()


def print_audit_report(df: pd.DataFrame) -> None:
    flagged = df[df["spurious_flag"]]
    clean = df[~df["spurious_flag"]]
    print(f"\n=== Spurious Correlation Audit ===")
    print(f"Top-{len(df)} hubs examined: {len(clean)} clean, {len(flagged)} flagged\n")
    if not flagged.empty:
        print("FLAGGED (volume/TVL-dominated — weaker causal evidence):")
        print(flagged[["node", "hub_score", "rank", "corr_with_volume", "corr_with_network", "audit_note"]].to_string(index=False))
    print("\nCLEAN (network-position-driven):")
    print(clean[["node", "hub_score", "rank", "corr_with_volume", "corr_with_network"]].to_string(index=False))
