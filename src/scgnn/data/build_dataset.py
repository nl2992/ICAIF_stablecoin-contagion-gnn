"""
Master dataset assembly script.

Reads configs/experiment.yaml and data/episodes.yaml, fetches raw data,
builds feature matrices and labels, and writes train/val/test splits to
data/processed/ as numpy arrays + a dataset manifest.

Usage:
    python -m scgnn.data.build_dataset [--config configs/experiment.yaml] [--stub]

--stub: use synthetic price data instead of live API calls (for CI / smoke testing).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve src/ on path regardless of install state
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.utils.seeds import set_all_seeds
from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.data.windows import load_episodes, episodes_by_split, build_episode_window, compute_availability_matrix, save_availability_matrix
from scgnn.data.fetch import fetch_ccxt_ohlcv, resample_to_common_grid, coverage_pct
from scgnn.features.node_features import build_node_feature_matrix, add_lags
from scgnn.features.edge_features import build_edge_feature_matrix
from scgnn.features.labels import make_onset_labels, base_rate_table


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_stub_price_grid(registry: NodeRegistry, start: pd.Timestamp, end: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic 1-min prices for all nodes — used for CI and dev."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", tz="UTC")
    n, m = len(idx), len(registry)
    prices = 1.0 + rng.normal(0, 0.0005, (n, m))
    # Inject a depeg for one node in the middle
    mid = n // 2
    prices[mid:mid + 30, 0] += 0.005
    return pd.DataFrame(prices, index=idx, columns=registry.node_strs())


def fetch_episode_prices(
    episode,
    registry: NodeRegistry,
    cfg: dict,
    raw_dir: Path,
    stub: bool = False,
) -> pd.DataFrame:
    """Return a 1-min price DataFrame for one episode's full window."""
    if stub:
        return make_stub_price_grid(registry, episode.full_window_start, episode.end)

    series = {}
    for node in registry:
        venue = node.venue
        if venue in ("binance", "coinbase", "kraken"):
            symbol = f"{node.asset}/USDT" if node.asset != "USDT" else "USDT/USDC"
            try:
                df = fetch_ccxt_ohlcv(
                    venue,
                    symbol,
                    episode.full_window_start,
                    episode.end,
                    raw_dir / venue,
                    max_forward_fill_min=cfg["data"]["max_forward_fill_min"],
                )
                if not df.empty:
                    series[str(node)] = df["close"]
            except Exception as exc:
                logger.warning("Skipping %s/%s: %s", venue, symbol, exc)
    return resample_to_common_grid(series, max_forward_fill_min=cfg["data"]["max_forward_fill_min"])


def assemble_split(
    episodes,
    registry: NodeRegistry,
    cfg: dict,
    raw_dir: Path,
    stub: bool,
    horizons: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Build stacked feature matrix X and label arrays for a list of episodes."""
    X_parts, y_parts = [], {h: [] for h in horizons}

    for ep in episodes:
        logger.info("Processing episode: %s", ep.name)
        price_grid = fetch_episode_prices(ep, registry, cfg, raw_dir, stub)
        window, cov = build_episode_window(ep, price_grid, cfg["data"]["min_coverage_pct"])

        if cov < cfg["data"]["min_coverage_pct"]:
            logger.warning("Skipping %s (coverage %.1f%%)", ep.name, cov)
            continue

        # Per-node feature matrices, then stack
        node_feats = []
        peg_devs = {}
        for node in registry:
            col = str(node)
            if col not in window.columns or window[col].isna().all():
                # Pad with zeros for missing nodes — flagged in availability matrix
                nf = pd.DataFrame(np.zeros((len(window), 8)), index=window.index)
            else:
                close = window[col].ffill().bfill()
                returns = close.pct_change().fillna(0)
                vol = pd.Series(np.abs(close.diff().fillna(0)) * 1e6 + 1, index=close.index)
                nf = build_node_feature_matrix(
                    close=close, volume=vol, returns=returns,
                    signed_volume=vol * np.sign(returns),
                    venue_prices={node.venue: close},
                )
                nf = add_lags(nf, lags=cfg["features"]["node_lags"])
                peg_devs[col] = (close - 1.0) * 10_000

            node_feats.append(nf.values)

        # Shape: (T, N*F)
        T = min(len(nf) for nf in node_feats)
        X_ep = np.concatenate([nf[-T:] for nf in node_feats], axis=1)
        X_parts.append(X_ep)

        # Labels per horizon
        asset_type_map = {
            **{a: "fiat_backed" for a in cfg["assets"]["fiat_backed"]},
            **{a: "crypto_backed" for a in cfg["assets"]["crypto_backed"]},
            **{a: "synthetic" for a in cfg["assets"]["synthetic"]},
        }
        thr = cfg["labels"]["thresholds_bps"]
        for node in registry:
            col = str(node)
            if col in peg_devs:
                asset_type = asset_type_map.get(node.asset, "fiat_backed")
                peg_devs[col] = peg_devs[col]   # already bps

        for h in horizons:
            y_ep_dict = make_onset_labels(
                peg_deviations=peg_devs,
                horizon_min=h,
                thresholds_bps={col: thr[asset_type_map.get(col.split("/")[0], "fiat_backed")]
                                for col in peg_devs},
                sustained_min=cfg["labels"]["sustained_min"],
            )
            y_ep = np.stack([y_ep_dict.get(col, np.zeros(T)) for col in [str(n) for n in registry]], axis=1)[-T:]
            y_parts[h].append(y_ep)

    if not X_parts:
        return np.empty((0,)), {h: np.empty((0,)) for h in horizons}

    X = np.concatenate(X_parts, axis=0)
    y = {h: np.concatenate(y_parts[h], axis=0) for h in horizons}
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--stub", action="store_true", help="Use synthetic data (no API calls)")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_all_seeds(cfg["seed"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")

    all_assets = (
        cfg["assets"]["fiat_backed"]
        + cfg["assets"]["crypto_backed"]
        + cfg["assets"]["synthetic"]
    )
    registry = NodeRegistry.from_config(
        assets=all_assets,
        venues=cfg["venues"]["cex"],
    )
    registry.save(out_dir / "node_registry.json")
    logger.info("Node registry: %d nodes", len(registry))

    episodes = load_episodes(Path(cfg["episodes_file"]))
    split = episodes_by_split(episodes)
    horizons = cfg["labels"]["horizons_min"]

    manifest = {}
    for split_name, eps in split.items():
        if not eps:
            continue
        logger.info("Assembling %s split (%d episodes)...", split_name, len(eps))
        X, y = assemble_split(eps, registry, cfg, raw_dir, args.stub, horizons)
        np.save(out_dir / f"X_{split_name}.npy", X)
        for h in horizons:
            np.save(out_dir / f"y_{split_name}_h{h}.npy", y[h])
        manifest[split_name] = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]) if X.ndim > 1 else 0,
            "episodes": [ep.name for ep in eps],
            "horizons": horizons,
        }
        logger.info("%s: X=%s", split_name, X.shape)
        for h in horizons:
            pos_rate = float(y[h].mean()) if y[h].size > 0 else 0.0
            logger.info("  horizon %d min: positive rate=%.3f", h, pos_rate)

    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Dataset manifest written to %s", out_dir / "dataset_manifest.json")


if __name__ == "__main__":
    main()
