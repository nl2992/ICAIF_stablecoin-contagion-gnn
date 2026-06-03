"""
Empirical calibration targets for the ABM (Repo 2).

For each real episode × trigger asset, extract:
  - OU half-life (mean-reversion speed post-shock)
  - Propagation rho (fraction of node universe stressed)
  - Propagation lag (median minutes to first contagion node)
  - Peak depeg bps
  - Shock duration

These are the *empirical targets* the ABM must reproduce when calibrated.
Written to exports/calibration_v1_{episode_tag}.csv.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scgnn.features.node_features import ou_half_life
from scgnn.features.labels import _stress_indicator
from scgnn.export.schema import SCHEMA_VERSION, CALIBRATION_COLUMNS, validate_calibration


def compute_ou_half_life_episode(
    price: pd.Series,
    stress_start: pd.Timestamp,
    stress_end: pd.Timestamp,
    window_min: int = 240,
) -> tuple[float, float, float]:
    """
    Estimate OU half-life for the *recovery* phase of an episode
    (i.e., after the peak depeg, as price reverts to $1).

    Returns (mean_hl, ci_lo, ci_hi) in minutes via bootstrap.
    """
    window = price.loc[stress_start:stress_end]
    if len(window) < 30:
        return float("nan"), float("nan"), float("nan")

    # Full window estimate
    hl_series = ou_half_life(window, window=min(window_min, len(window) - 1))
    hl_vals = hl_series.dropna().values
    if len(hl_vals) == 0:
        return float("nan"), float("nan"), float("nan")

    mean_hl = float(np.median(hl_vals))
    # Bootstrap CI
    rng = np.random.default_rng(42)
    bootstrap = [float(np.median(rng.choice(hl_vals, size=len(hl_vals), replace=True))) for _ in range(500)]
    lo = float(np.percentile(bootstrap, 2.5))
    hi = float(np.percentile(bootstrap, 97.5))
    return mean_hl, lo, hi


def compute_propagation_stats(
    peg_deviations: Dict[str, pd.Series],
    trigger_node: str,
    threshold_bps: float,
    sustained_min: int = 10,
) -> dict:
    """
    Given peg deviations for all nodes during an episode, compute:
      - propagation_rho: fraction of nodes stressed (other than trigger)
      - propagation_lag_min: median lead-lag from trigger to first stress
      - peak_depeg_bps: maximum |deviation| observed across any node
    """
    n_nodes = len(peg_deviations)
    stressed_count = 0
    lag_minutes = []
    peak_bps = 0.0

    trigger_stress = None
    if trigger_node in peg_deviations:
        trigger_stress = _stress_indicator(
            peg_deviations[trigger_node], threshold_bps, sustained_min
        )
        trigger_onset = trigger_stress[trigger_stress == 1].index
        trigger_first = trigger_onset[0] if len(trigger_onset) else None
    else:
        trigger_first = None

    for node, dev in peg_deviations.items():
        peak_bps = max(peak_bps, float(dev.abs().max()))
        if node == trigger_node:
            continue
        stress = _stress_indicator(dev, threshold_bps, sustained_min)
        if stress.sum() > 0:
            stressed_count += 1
            if trigger_first is not None:
                onset = stress[stress == 1].index
                if len(onset):
                    lag = (onset[0] - trigger_first).total_seconds() / 60
                    lag_minutes.append(lag)

    return {
        "propagation_rho": stressed_count / max(n_nodes - 1, 1),
        "propagation_lag_min": float(np.median(lag_minutes)) if lag_minutes else float("nan"),
        "peak_depeg_bps": peak_bps,
    }


def build_calibration_table(
    episodes: list,
    price_grids: Dict[str, pd.DataFrame],     # {episode_name: aligned_price_df}
    threshold_map: Dict[str, float],           # {asset: threshold_bps}
    sustained_min: int = 10,
) -> pd.DataFrame:
    """
    Build the full calibration table for all real episodes.
    """
    rows = []
    for ep in episodes:
        if ep.name not in price_grids:
            continue
        grid = price_grids[ep.name]
        window = grid.loc[ep.start:ep.end]
        if window.empty:
            continue

        for col in window.columns:
            asset = col.split("/")[0]
            if asset != ep.trigger:
                continue  # only calibrate from the trigger asset per episode

            thr = threshold_map.get(asset, 25.0)
            price = window[col].dropna()
            dev = (price - 1.0) * 10_000

            hl, hl_lo, hl_hi = compute_ou_half_life_episode(price, ep.start, ep.end)
            prop_stats = compute_propagation_stats(
                {c: (window[c] - 1.0) * 10_000 for c in window.columns if c in window},
                trigger_node=col,
                threshold_bps=thr,
                sustained_min=sustained_min,
            )
            stress = _stress_indicator(dev, thr, sustained_min)
            shock_duration = int(stress.sum())

            rows.append({
                "episode": ep.name,
                "asset": asset,
                "venue": col.split("/")[1] if "/" in col else "unknown",
                "ou_half_life_min": hl,
                "ou_half_life_ci_lo": hl_lo,
                "ou_half_life_ci_hi": hl_hi,
                "propagation_rho": prop_stats["propagation_rho"],
                "propagation_lag_min": prop_stats["propagation_lag_min"],
                "peak_depeg_bps": prop_stats["peak_depeg_bps"],
                "shock_duration_min": shock_duration,
                "episode_tag": ep.name,
                "is_real": not ep.synthetic if hasattr(ep, "synthetic") else True,
            })

    df = pd.DataFrame(rows)
    missing = validate_calibration(df)
    if missing:
        raise ValueError(f"Calibration table missing required columns: {missing}")
    return df


def save_calibration(
    df: pd.DataFrame,
    episode_tag: str,
    out_dir: Path = Path("exports"),
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"calibration_v{SCHEMA_VERSION}_{episode_tag}.csv"
    df.to_csv(path, index=False)
    print(f"Calibration targets saved: {path}")
    return path
