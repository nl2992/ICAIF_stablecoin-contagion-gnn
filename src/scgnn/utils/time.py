"""Pandas 3.x-compatible time-window helpers (DataFrame.last() was removed)."""
from __future__ import annotations

import pandas as pd


def last_window(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Return rows from df within the last `window` duration (e.g. '6h', '30min')."""
    if df.empty:
        return df
    cutoff = df.index[-1] - pd.Timedelta(window)
    return df[df.index >= cutoff]


def last_window_series(s: pd.Series, window: str) -> pd.Series:
    if s.empty:
        return s
    cutoff = s.index[-1] - pd.Timedelta(window)
    return s[s.index >= cutoff]
