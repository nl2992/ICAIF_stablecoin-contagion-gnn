"""
Node feature builders.

All signals drawn from IAQF 2026 microstructure pipeline:
price_ratio, realized_vol, log_volume, Amihud, Kyle's lambda, OU half-life, LOP wedge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress


def price_ratio(close: pd.Series, peg: float = 1.0) -> pd.Series:
    return (close / peg).rename("price_ratio")


def realized_vol(returns: pd.Series, window: str) -> pd.Series:
    return returns.rolling(window).std().rename(f"rvol_{window}")


def log_volume(volume: pd.Series, window: str = "1h") -> pd.Series:
    return np.log1p(volume.rolling(window).sum()).rename(f"log_vol_{window}")


def amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: str = "1h") -> pd.Series:
    """Amihud (2002): |r_t| / volume_t, rolling mean."""
    ratio = returns.abs() / volume.replace(0, np.nan)
    return ratio.rolling(window).mean().rename("amihud")


def kyle_lambda(returns: pd.Series, signed_volume: pd.Series, window: int = 60) -> pd.Series:
    """
    Kyle (1985) lambda: OLS slope of price-change on signed order flow.
    Estimated on a rolling window of `window` 1-min bars.
    """
    results = []
    idx = []
    for end in range(window, len(returns)):
        r = returns.iloc[end - window:end].values
        q = signed_volume.iloc[end - window:end].values
        mask = ~(np.isnan(r) | np.isnan(q))
        if mask.sum() < 10:
            results.append(np.nan)
        else:
            slope, *_ = linregress(q[mask], r[mask])
            results.append(slope)
        idx.append(returns.index[end])
    return pd.Series(results, index=idx, name="kyle_lambda")


def ou_half_life(price: pd.Series, window: int = 1440) -> pd.Series:
    """
    OU mean-reversion half-life via log-lag regression (IAQF pipeline).
    Returns half-life in minutes for each rolling window endpoint.
    """
    results, idx = [], []
    for end in range(window, len(price)):
        p = price.iloc[end - window:end]
        lag = p.shift(1).dropna()
        delta = p.diff().dropna()
        lag, delta = lag.align(delta, join="inner")
        if len(lag) < 30:
            results.append(np.nan)
        else:
            slope, intercept, *_ = linregress(lag.values, delta.values)
            if slope < 0:
                hl = -np.log(2) / slope
            else:
                hl = np.nan
            results.append(hl)
        idx.append(price.index[end])
    return pd.Series(results, index=idx, name="ou_half_life")


def lop_wedge(prices: dict[str, pd.Series]) -> pd.Series:
    """
    Law-of-one-price wedge: max cross-venue spread for the same asset.
    prices = {venue_name: 1-min close series}
    """
    df = pd.DataFrame(prices)
    spread = df.max(axis=1) - df.min(axis=1)
    return spread.rename("lop_wedge")


def build_node_feature_matrix(
    close: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
    signed_volume: pd.Series,
    venue_prices: dict[str, pd.Series],
    ou_window: int = 1440,
    kyle_window: int = 60,
) -> pd.DataFrame:
    feats = pd.DataFrame(index=close.index)
    feats["price_ratio"] = price_ratio(close)
    feats["rvol_1h"] = realized_vol(returns, "1h")
    feats["rvol_24h"] = realized_vol(returns, "24h")
    feats["log_vol_1h"] = log_volume(volume, "1h")
    feats["amihud"] = amihud_illiquidity(returns, volume, "1h")
    feats["kyle_lambda"] = kyle_lambda(returns, signed_volume, kyle_window).reindex(feats.index)
    feats["ou_half_life"] = ou_half_life(close, ou_window).reindex(feats.index)
    feats["lop_wedge"] = lop_wedge(venue_prices).reindex(feats.index)
    return feats
