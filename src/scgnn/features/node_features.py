"""
Node feature builders (IAQF microstructure signals + lags).

All features are computed at 1-min resolution then down-sampled
as needed for hourly graph snapshots.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ------------------------------------------------------------------ core signals


def price_ratio(close: pd.Series, peg: float = 1.0) -> pd.Series:
    return (close / peg).rename("price_ratio")


def realized_vol(returns: pd.Series, window: str) -> pd.Series:
    return returns.rolling(window, min_periods=1).std().rename(f"rvol_{window}")


def log_volume(volume: pd.Series, window: str = "1h") -> pd.Series:
    return np.log1p(volume.rolling(window, min_periods=1).sum()).rename(f"log_vol_{window}")


def amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: str = "1h") -> pd.Series:
    ratio = returns.abs() / volume.replace(0, np.nan)
    return ratio.rolling(window, min_periods=1).mean().rename("amihud")


def kyle_lambda(returns: pd.Series, signed_volume: pd.Series, window: int = 60) -> pd.Series:
    """Rolling OLS slope of Δprice ~ signed_order_flow (Kyle 1985)."""
    results, idx = [], []
    for end in range(window, len(returns)):
        r = returns.iloc[end - window:end].values
        q = signed_volume.iloc[end - window:end].values
        mask = ~(np.isnan(r) | np.isnan(q) | (q == 0))
        if mask.sum() < 10:
            results.append(np.nan)
        else:
            slope, *_ = linregress(q[mask], r[mask])
            results.append(float(slope))
        idx.append(returns.index[end])
    return pd.Series(results, index=idx, name="kyle_lambda")


def ou_half_life(price: pd.Series, window: int = 1440) -> pd.Series:
    """Rolling OU mean-reversion half-life via log-lag OLS (IAQF pipeline)."""
    results, idx = [], []
    for end in range(window, len(price)):
        p = price.iloc[end - window:end]
        lag = p.shift(1).dropna()
        delta = p.diff().dropna()
        lag, delta = lag.align(delta, join="inner")
        if len(lag) < 30:
            results.append(np.nan)
            idx.append(price.index[end])
            continue
        slope, *_ = linregress(lag.values, delta.values)
        hl = -np.log(2) / slope if slope < 0 else np.nan
        results.append(hl)
        idx.append(price.index[end])
    return pd.Series(results, index=idx, name="ou_half_life")


def lop_wedge(venue_prices: Dict[str, pd.Series]) -> pd.Series:
    """Max cross-venue price spread for the same asset (law-of-one-price deviation)."""
    df = pd.DataFrame(venue_prices)
    return (df.max(axis=1) - df.min(axis=1)).rename("lop_wedge")


def tvl_log(tvl: pd.Series) -> pd.Series:
    return np.log1p(tvl).rename("tvl_usd_log")


# ------------------------------------------------------------------ lag wrapper


def add_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Append lagged copies of each column.  lag=5 → col_lag5 = col.shift(5).
    Mirrors the Uniswap node-feature design (t−5…t).
    NaN rows from shifting are forward-filled with the earliest valid value.
    """
    parts = [df]
    for lag in lags:
        lagged = df.shift(lag).add_suffix(f"_lag{lag}")
        parts.append(lagged)
    out = pd.concat(parts, axis=1)
    return out.bfill()


# ------------------------------------------------------------------ master builder


def build_node_feature_matrix(
    close: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
    signed_volume: pd.Series,
    venue_prices: Dict[str, pd.Series],
    ou_window: int = 1440,
    kyle_window: int = 60,
) -> pd.DataFrame:
    feats = pd.DataFrame(index=close.index)
    feats["price_ratio"] = price_ratio(close)
    feats["rvol_1h"] = realized_vol(returns, "1h")
    feats["rvol_24h"] = realized_vol(returns, "24h")
    feats["log_vol_1h"] = log_volume(volume, "1h")
    feats["amihud"] = amihud_illiquidity(returns, volume, "1h")

    kl = kyle_lambda(returns, signed_volume, kyle_window).reindex(feats.index)
    feats["kyle_lambda"] = kl.ffill().bfill().fillna(0.0)

    hl = ou_half_life(close, ou_window).reindex(feats.index)
    feats["ou_half_life"] = hl.ffill().bfill().fillna(0.0)

    feats["lop_wedge"] = lop_wedge(venue_prices).reindex(feats.index).fillna(0.0)
    return feats.fillna(0.0)
