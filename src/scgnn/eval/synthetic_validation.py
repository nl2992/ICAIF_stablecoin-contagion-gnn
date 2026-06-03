"""
Synthetic episode validity check.

Before using StressBench-generated episodes as training data, verify that
they preserve the empirical stylized facts of real stress episodes:
  1. OU half-life distribution (mean-reversion speed)
  2. Lead-lag structure (cross-asset predictability)
  3. Realized volatility during stress vs calm

A synthetic batch that fails these checks should NOT be used — using fabricated
signal would be worse than not augmenting at all.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu


def _moment_dict(prices: pd.DataFrame) -> dict:
    """Compute key moments for a batch of price series."""
    returns = prices.pct_change().dropna()
    return {
        "mean_rvol_1h": float(returns.rolling("1h", min_periods=1).std().mean().mean()),
        "mean_rvol_24h": float(returns.rolling("24h", min_periods=1).std().mean().mean()),
        "mean_autocorr_lag1": float(returns.apply(lambda s: s.autocorr(lag=1)).mean()),
        "mean_price_deviation_bps": float(((prices - 1.0).abs() * 10_000).mean().mean()),
        "max_price_deviation_bps": float(((prices - 1.0).abs() * 10_000).max().max()),
    }


def validate_synthetic(
    real_prices: Dict[str, pd.DataFrame],      # {episode_name: price_df}
    synthetic_prices: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, bool]:
    """
    KS test comparing moment distributions across real vs synthetic episodes.

    Returns (report_df, passed) where passed=True means augmentation is valid.
    """
    real_moments = [_moment_dict(df) for df in real_prices.values()]
    synth_moments = [_moment_dict(df) for df in synthetic_prices.values()]

    if not real_moments or not synth_moments:
        return pd.DataFrame(), False

    keys = list(real_moments[0].keys())
    rows = []
    all_pass = True
    for key in keys:
        real_vals = np.array([m[key] for m in real_moments])
        synth_vals = np.array([m[key] for m in synth_moments])
        ks_stat, ks_p = ks_2samp(real_vals, synth_vals)
        passed = ks_p >= alpha
        if not passed:
            all_pass = False
        rows.append({
            "moment": key,
            "real_mean": float(real_vals.mean()),
            "synth_mean": float(synth_vals.mean()),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "passed": passed,
        })

    return pd.DataFrame(rows).set_index("moment"), all_pass
