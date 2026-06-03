"""
Real-data dataset builder.

Replaces the synthetic stub with genuine Binance + Coinbase 1-minute data for the
fetchable 2022-2023 stress episodes.  Produces a *coherent* dataset where the
tabular ladder and the GNN see the **same samples** (one per (hourly snapshot,
node)), so the GNN-vs-tabular comparison is fair.

Outputs (under data/processed/)
-------------------------------
node_registry.json          stable node ordering (asset/venue)
feature_names.json          F per-node feature names
tabular/X_{split}.npy       (M, F) per-(snapshot,node) feature rows
tabular/y_{split}_h{h}.npy  (M,)   onset label at horizon h
tabular/meta_{split}.parquet episode / timestamp / node / snapshot for each row
graphs/{episode}.pkl        per-snapshot PyG-ready tensors + per-horizon labels
base_rates.csv              positive rate per episode x horizon
dataset_manifest.json       shapes + provenance summary
../data_card.md             human-readable provenance (written by CLI)

Design decisions match the pre-registered config:
- hourly snapshots (graph.step_minutes), 6h rolling directed edges
- onset labels with origin exclusion + pre-existing-stress mask
- asset-class bps thresholds, sustained 10 min
- chronological LOEO-ready episode tagging (no shuffle)
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scgnn.data.realfetch import SYMBOL_MAP, fetch_series
from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.data.windows import load_episodes
from scgnn.features.labels import _stress_indicator
from scgnn.graphs.builder import build_snapshot_graph, graph_to_pyg

logger = logging.getLogger(__name__)

# All candidate nodes (asset, venue) we attempt to fetch.
CANDIDATE_NODES: List[Tuple[str, str]] = list(SYMBOL_MAP.keys())

# Trigger asset -> origin node string (excluded from labels/samples).
ORIGIN = {
    "USDC_SVB": "USDC/binance",
    "FRAX_SVB": "FRAX/binance",
    "DAI_FTX": "DAI/coinbase",
    "BUSD_winddown": "BUSD/binance",
    "UST_Terra": "UST/binance",
    "USDT_May2022": "USDT/coinbase",
    "USDT_Oct2018": "USDT/coinbase",
}

ASSET_CLASS = {
    "USDC": "fiat_backed", "USDT": "fiat_backed", "TUSD": "fiat_backed",
    "PYUSD": "fiat_backed", "BUSD": "fiat_backed", "USDP": "fiat_backed",
    "FDUSD": "fiat_backed", "DAI": "crypto_backed", "FRAX": "crypto_backed",
    "UST": "synthetic", "USDe": "synthetic",
}

FEATURE_BASE = ["price_ratio", "rvol_1h", "rvol_24h", "log_vol_1h",
                "amihud", "kyle_lambda", "ou_half_life", "lop_wedge"]


# --------------------------------------------------------------------- features
def _rolling_slope(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of y on x (cov/var), vectorised."""
    cov = y.rolling(window, min_periods=window // 2).cov(x)
    var = x.rolling(window, min_periods=window // 2).var()
    return cov / var.replace(0, np.nan)


def hourly_node_features(close: pd.Series, volume: pd.Series,
                         hourly_idx: pd.DatetimeIndex,
                         ou_window: int = 1440, kyle_window: int = 60) -> pd.DataFrame:
    """Compute the 8 base features at 1-min then sample on the hourly grid."""
    close = close.astype(float)
    volume = volume.astype(float).fillna(0.0)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    signed_vol = volume * np.sign(returns)

    f = pd.DataFrame(index=close.index)
    f["price_ratio"] = close
    f["rvol_1h"] = returns.rolling(60, min_periods=10).std()
    f["rvol_24h"] = returns.rolling(1440, min_periods=60).std()
    f["log_vol_1h"] = np.log1p(volume.rolling(60, min_periods=1).sum())
    f["amihud"] = (returns.abs() / volume.replace(0, np.nan)).rolling(60, min_periods=10).mean()
    f["kyle_lambda"] = _rolling_slope(returns, signed_vol, kyle_window)
    slope = _rolling_slope(close.diff(), close.shift(1), ou_window)
    hl = -np.log(2) / slope
    f["ou_half_life"] = hl.where(slope < 0, np.nan).clip(0, 1e5)
    f["lop_wedge"] = 0.0  # single venue per asset; kept for schema parity
    f = f.ffill().bfill().fillna(0.0)
    return f.reindex(hourly_idx, method="ffill").fillna(0.0)


def add_snapshot_lags(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    parts = [df]
    for lag in lags:
        parts.append(df.shift(lag).add_suffix(f"_lag{lag}"))
    return pd.concat(parts, axis=1).bfill().fillna(0.0)


def feature_names(lags: List[int]) -> List[str]:
    names = list(FEATURE_BASE)
    for lag in lags:
        names += [f"{b}_lag{lag}" for b in FEATURE_BASE]
    return names


# --------------------------------------------------------------------- episode build
def build_episode(ep, registry: NodeRegistry, cfg: dict, cache_dir: Path) -> Optional[dict]:
    """Fetch + featurise + label + graph one episode.  Returns a dict or None."""
    baseline_days = cfg["data"].get("pre_event_baseline_days", 3)
    ep_start = pd.Timestamp(ep.start); ep_end = pd.Timestamp(ep.end)
    if ep_start.tz is None:
        ep_start = ep_start.tz_localize("UTC"); ep_end = ep_end.tz_localize("UTC")
    start = ep_start - pd.Timedelta(days=baseline_days)
    end = ep_end + pd.Timedelta(days=1)
    lags = cfg["features"]["node_lags"]
    step = cfg["graph"]["step_minutes"]
    horizons = cfg["labels"]["horizons_min"]
    thr_map = cfg["labels"]["thresholds_bps"]
    sustained = cfg["labels"]["sustained_min"]

    # 1-min close grid (true-USD) per candidate node
    usdt_usd = None
    raw_close, raw_vol = {}, {}
    for asset, venue in CANDIDATE_NODES:
        df = fetch_series(asset, venue, start, end, cache_dir)
        if df.empty or len(df) < 200:
            continue
        raw_close[f"{asset}/{venue}"] = df["close"]
        raw_vol[f"{asset}/{venue}"] = df["volume"]
        if asset == "USDT" and venue == "coinbase":
            usdt_usd = df["close"]

    if not raw_close:
        logger.warning("episode %s: no data fetched", ep.name)
        return None

    grid_1m = pd.date_range(start, end, freq="1min", tz="UTC")
    # Stablecoin price is piecewise-constant between trades: ffill across illiquid
    # interior minutes (large limit), but never fill BEFORE first / AFTER last trade.
    price_ffill = cfg["data"].get("price_ffill_limit_min", 360)
    close_df = pd.DataFrame(index=grid_1m)
    vol_df = pd.DataFrame(index=grid_1m)
    for k in raw_close:
        close_df[k] = raw_close[k].reindex(grid_1m).ffill(limit=price_ffill)
        vol_df[k] = raw_vol[k].reindex(grid_1m).fillna(0.0)

    # USDT/USD peg conversion for Binance (USDT-quoted) nodes
    if usdt_usd is not None:
        usdt_ref = usdt_usd.reindex(grid_1m).ffill().bfill().fillna(1.0)
    else:
        usdt_ref = pd.Series(1.0, index=grid_1m)
    for k in close_df.columns:
        if k.endswith("/binance"):
            close_df[k] = close_df[k] * usdt_ref  # asset/USDT * USDT/USD = asset/USD

    # coverage gate — computed over the EVENT window only (not the pre-baseline,
    # which several assets lack due to listing/relisting history, e.g. Binance's
    # 2022 BUSD auto-conversion that delisted USDC/TUSD/USDP until 2023-03-11).
    event_slice = close_df.loc[ep_start:ep_end]
    cov = event_slice.notna().mean() * 100
    active_cols = [c for c in close_df.columns if cov[c] >= cfg["data"]["min_coverage_pct"]]
    if not active_cols:
        logger.warning("episode %s: no node passes coverage", ep.name)
        return None
    close_df = close_df[active_cols]
    returns_1m = np.log(close_df).diff().fillna(0.0)

    # hourly snapshot grid
    hourly_idx = pd.date_range(close_df.index[0].ceil("h"), close_df.index[-1].floor("h"),
                               freq=f"{step}min", tz="UTC")
    if len(hourly_idx) < 5:
        return None

    # per-node hourly features
    feat_cols = feature_names(lags)
    N = len(registry)
    node_strs = registry.node_strs()
    S = len(hourly_idx)
    X_grid = np.zeros((S, N, len(feat_cols)), dtype=np.float32)
    active = np.zeros((S, N), dtype=bool)
    dev_bps_1m = {}  # node -> bps deviation series (1-min) for labels

    for col in active_cols:
        feats = hourly_node_features(close_df[col], vol_df[col], hourly_idx)
        feats = add_snapshot_lags(feats, lags)[feat_cols]
        j = node_strs.index(col)
        X_grid[:, j, :] = feats.values.astype(np.float32)
        active[:, j] = True
        dev_bps_1m[col] = (close_df[col] - 1.0) * 1e4

    # onset labels per horizon at hourly snapshots
    def thr_for(col: str) -> float:
        return float(thr_map[ASSET_CLASS.get(col.split("/")[0], "fiat_backed")])

    stressed_1m = {c: _stress_indicator(dev_bps_1m[c], thr_for(c), sustained) for c in active_cols}
    labels = {h: np.zeros((S, N), dtype=np.int8) for h in horizons}
    origin = ORIGIN.get(ep.name)
    for col in active_cols:
        j = node_strs.index(col)
        s_ind = stressed_1m[col]
        # future stress onset within horizon, evaluated at each hourly t
        calm_now = (s_ind.reindex(hourly_idx, method="ffill").fillna(0) == 0).astype(int)
        for h in horizons:
            fut = s_ind.rolling(h, min_periods=1).max().shift(-h).fillna(0)
            fut_h = (fut.reindex(hourly_idx, method="ffill").fillna(0) > 0).astype(int)
            y = (calm_now.values * fut_h.values).astype(np.int8)
            if origin == col:
                y[:] = 0  # origin exclusion
            labels[h][:, j] = y

    # graph snapshots (directed, 6h rolling correlation + lead-lag)
    lookback = f"{cfg['graph']['lookback_hours']}h"
    corr_thr = cfg["graph"]["edge_corr_threshold"]
    node_ids = [NodeID.from_str(s) for s in active_cols]
    snapshots = []
    for si, t in enumerate(hourly_idx):
        amask = {c: bool(active[si, node_strs.index(c)]) for c in active_cols}
        G = build_snapshot_graph(returns_1m, node_ids, t, lookback=lookback,
                                 corr_threshold=corr_thr, directed=True,
                                 episode_start=close_df.index[0], active_mask=amask)
        data = graph_to_pyg(G, registry, X_grid[si], edge_feature_dim=4)
        snapshots.append({
            "t": t.isoformat(),
            "edge_index": data.edge_index.numpy().astype(np.int64),
            "edge_attr": data.edge_attr.numpy().astype(np.float32),
        })

    # 1-min peg deviation (bps) per active node + epoch-ms index — consumed by the
    # hub export (propagator labels) and the ABM (OU / peak-depeg calibration targets).
    index_1m_ms = (close_df.index.asi8 // 1_000_000).astype("int64")
    dev_bps_out = {c: dev_bps_1m[c].values.astype(np.float32) for c in active_cols}

    return {
        "episode": ep.name, "trigger": ep.trigger, "trigger_type": ep.trigger_type,
        "split": ep.split, "origin": origin, "node_strs": node_strs,
        "hourly_idx": [t.isoformat() for t in hourly_idx],
        "X": X_grid, "active": active, "labels": labels,
        "snapshots": snapshots, "feature_names": feat_cols,
        "n_active_nodes": len(active_cols), "active_node_strs": active_cols,
        "dev_bps_1m": dev_bps_out, "index_1m_ms": index_1m_ms,
    }


# --------------------------------------------------------------------- assemble
def assemble(cfg: dict, out_dir: Path, cache_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tabular").mkdir(exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)

    episodes = load_episodes(Path(cfg["episodes_file"]))
    registry = NodeRegistry([NodeID.from_str(f"{a}/{v}") for a, v in CANDIDATE_NODES])
    registry.save(out_dir / "node_registry.json")
    lags = cfg["features"]["node_lags"]
    feat_cols = feature_names(lags)
    (out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2))
    horizons = cfg["labels"]["horizons_min"]

    built: Dict[str, dict] = {}
    for ep in episodes:
        logger.info("building episode %s ...", ep.name)
        b = build_episode(ep, registry, cfg, cache_dir)
        if b is None:
            logger.warning("  dropped (no/low data)")
            continue
        built[ep.name] = b
        with open(out_dir / "graphs" / f"{ep.name}.pkl", "wb") as fh:
            pickle.dump(b, fh)
        logger.info("  ok: %d active nodes, %d snapshots", b["n_active_nodes"], len(b["hourly_idx"]))

    # tabular rows per split: one (snapshot, node) per active, non-origin node
    splits = {"train": [], "val": [], "test": []}
    for name, b in built.items():
        splits[b["split"]].append(name)

    manifest = {"provenance": "real: Binance(USDT) + Coinbase(USD), 1-min; USDT/USD peg-adjusted",
                "feature_dim": len(feat_cols), "n_nodes": len(registry),
                "episodes_built": list(built.keys()), "splits": splits, "horizons": horizons}
    base_rate_rows = []

    for split, names in splits.items():
        Xs, ys = [], {h: [] for h in horizons}
        meta_rows = []
        for name in names:
            b = built[name]
            S, N = b["active"].shape
            for si in range(S):
                for j in range(N):
                    if not b["active"][si, j]:
                        continue
                    if b["node_strs"][j] == b["origin"]:
                        continue
                    Xs.append(b["X"][si, j])
                    for h in horizons:
                        ys[h].append(int(b["labels"][h][si, j]))
                    meta_rows.append({"episode": name, "t": b["hourly_idx"][si],
                                      "node": b["node_strs"][j], "snapshot": si})
        if not Xs:
            continue
        X = np.stack(Xs).astype(np.float32)
        np.save(out_dir / "tabular" / f"X_{split}.npy", X)
        for h in horizons:
            yarr = np.array(ys[h], dtype=np.int8)
            np.save(out_dir / "tabular" / f"y_{split}_h{h}.npy", yarr)
        pd.DataFrame(meta_rows).to_parquet(out_dir / "tabular" / f"meta_{split}.parquet")
        manifest[f"{split}_samples"] = int(X.shape[0])

        # per episode x horizon base rates
        meta_df = pd.DataFrame(meta_rows)
        for name in names:
            mask = (meta_df["episode"] == name).values
            for h in horizons:
                yh = np.array(ys[h])[mask]
                base_rate_rows.append({"episode": name, "horizon_min": h,
                                       "n": int(mask.sum()), "n_pos": int(yh.sum()),
                                       "pos_rate": round(float(yh.mean()), 5) if mask.sum() else 0.0})

    pd.DataFrame(base_rate_rows).to_csv(out_dir / "base_rates.csv", index=False)
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
