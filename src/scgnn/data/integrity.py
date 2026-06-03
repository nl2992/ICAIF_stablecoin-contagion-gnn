"""
Data integrity checks for the stablecoin episode dataset.

Covers three concerns flagged in the roadmap review:

1. DEAD / DELISTED ASSETS
   BUSD stopped minting Feb 2023 (Paxos); UST collapsed May 2022.
   Their price → 0 or → NaN is a data artifact, not a contagion signal.
   We detect this and exclude affected nodes from the episode feature matrix.

2. 2018 FEATURE-SUPPORT MISMATCH
   The USDT_Oct2018 episode predates:
     - Uniswap v1 (mainnet Nov 2018) → all DEX features are N/A
     - DeFi TVL measurement → TVL features are N/A
     - Curve (launched 2020) → all Curve features are N/A
   Only CEX OHLCV features are valid for this episode.
   We flag this and produce a "reduced feature set" version for 2018.

3. PER-EPISODE MISSINGNESS REPORT
   For each episode × feature, report % coverage and flag sparse ones.
   Cross-venue features (LOP wedge, shared-LP %) will be near-zero
   for 2018 and early episodes.

IMPUTATION POLICY (documented here, not implicit):
  - Forward-fill gaps ≤ max_forward_fill_min (default 5)
  - Gaps > 5 min → NaN; do NOT extrapolate
  - Training matrices: feature-wise median imputation from TRAIN SET ONLY
    (fit on train, apply to val/test — no leakage)
  - Structural zeros (LOP wedge for single-venue episodes) → 0.0, not imputed
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ dead asset registry

@dataclass
class AssetStatus:
    asset: str
    delisted_date: Optional[pd.Timestamp]     # None = still active
    delist_reason: str = ""
    price_floor_bps: float = 0.0              # price below this = delisting artifact

ASSET_STATUS_REGISTRY: List[AssetStatus] = [
    AssetStatus(
        "UST",
        pd.Timestamp("2022-05-13", tz="UTC"),
        "Terra/Luna collapse; UST ceased to trade as stablecoin",
        price_floor_bps=100.0,
    ),
    AssetStatus(
        "BUSD",
        pd.Timestamp("2023-02-13", tz="UTC"),
        "NYDFS ordered Paxos to stop minting; gradual wind-down",
        price_floor_bps=50.0,   # below 50 bps depeg is likely delisting artifact
    ),
]

DELISTED_ASSETS: Set[str] = {s.asset for s in ASSET_STATUS_REGISTRY if s.delisted_date}


def is_delisting_artifact(
    price: pd.Series,
    asset: str,
    threshold_zero_pct: float = 0.30,
) -> bool:
    """
    Return True if price series appears to be a delisting artifact
    (>30% of values are 0 or NaN after the known delisting date).
    """
    status = next((s for s in ASSET_STATUS_REGISTRY if s.asset == asset), None)
    if status is None or status.delisted_date is None:
        return False
    after_delist = price[price.index >= status.delisted_date]
    if len(after_delist) == 0:
        return False
    zero_or_nan = (after_delist.isna() | (after_delist == 0)).mean()
    return float(zero_or_nan) > threshold_zero_pct


# ------------------------------------------------------------------ 2018 feature support

# Features that are NOT available for pre-DeFi episodes (before 2019-01-01)
DEFI_ERA_FEATURES = {
    "tvl_usd_log",
    "lop_wedge",          # requires ≥2 venues; CEX-only in 2018
    "shared_lp_pct",      # requires DEX LP data
    "cross_pool_flow_usd_log",   # requires DEX pool data
    "ou_half_life",       # requires sufficient history (≥24h rolling)
}

PRE_DEFI_CUTOFF = pd.Timestamp("2019-01-01", tz="UTC")


def get_available_features(
    episode_start: pd.Timestamp,
    all_features: List[str],
) -> List[str]:
    """
    Return the subset of features that are valid for this episode's time period.
    For pre-2019 episodes, DeFi-derived features are excluded.
    """
    if episode_start < PRE_DEFI_CUTOFF:
        available = [f for f in all_features if not any(d in f for d in DEFI_ERA_FEATURES)]
        excluded = [f for f in all_features if any(d in f for d in DEFI_ERA_FEATURES)]
        if excluded:
            logger.warning(
                "Episode pre-dates DeFi era (%s < %s). "
                "Excluding %d features: %s",
                episode_start.date(), PRE_DEFI_CUTOFF.date(),
                len(excluded), excluded[:5],
            )
        return available
    return all_features


def episode_feature_support_report(
    episodes: list,
    all_features: List[str],
) -> pd.DataFrame:
    """
    Table of feature availability per episode.
    Rows = episodes, columns = features, values = True/False.
    """
    rows = []
    for ep in episodes:
        available = set(get_available_features(ep.start, all_features))
        row = {"episode": ep.name, "era": "pre-DeFi" if ep.start < PRE_DEFI_CUTOFF else "DeFi"}
        for f in all_features:
            row[f] = f in available
        rows.append(row)
    return pd.DataFrame(rows).set_index("episode")


# ------------------------------------------------------------------ missingness report

def per_episode_missingness(
    episode_windows: Dict[str, pd.DataFrame],   # {ep_name: aligned price/feature DataFrame}
    all_features: List[str],
) -> pd.DataFrame:
    """
    Per-episode × feature missingness (% NaN).
    Flags any feature with > 50% missing in any episode.
    """
    rows = []
    for ep_name, df in episode_windows.items():
        row = {"episode": ep_name}
        for feat in all_features:
            if feat in df.columns:
                pct = float(df[feat].isna().mean() * 100)
            else:
                pct = 100.0   # entirely absent
            row[feat] = round(pct, 1)
        rows.append(row)
    result = pd.DataFrame(rows).set_index("episode")
    # Flag matrix
    flag = result.applymap(lambda v: "SPARSE" if v > 50 else ("OK" if v < 10 else "PARTIAL"))
    return result, flag


# ------------------------------------------------------------------ imputation

class TrainFitImputer:
    """
    Feature-wise median imputer that fits on train only and transforms val/test.
    Prevents leakage from test-set statistics into training normalization.
    """

    def __init__(self):
        self._medians: Optional[np.ndarray] = None
        self._structural_zeros: Set[int] = set()   # columns that are structural zeros

    def fit(self, X_train: np.ndarray, structural_zero_cols: Optional[List[int]] = None) -> "TrainFitImputer":
        with np.errstate(all="ignore"):   # suppress all-NaN slice warning for structural-zero cols
            self._medians = np.nanmedian(X_train, axis=0)
        self._structural_zeros = set(structural_zero_cols or [])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._medians is None:
            raise RuntimeError("Fit the imputer on train data first.")
        X = X.copy().astype(float)
        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col])
            if not nan_mask.any():
                continue
            if col in self._structural_zeros:
                X[nan_mask, col] = 0.0   # structural zero (e.g. LOP wedge for single-venue)
            else:
                X[nan_mask, col] = self._medians[col]
        return X

    def fit_transform(self, X_train: np.ndarray, **kwargs) -> np.ndarray:
        return self.fit(X_train, **kwargs).transform(X_train)
