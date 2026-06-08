"""
Plan C — On-chain proxy feature engineering.

Computes 3 proxy features that approximate on-chain signals using existing
microstructure data. These serve as approximations until real Dune/The-Graph
exports are available; the code structure is identical to what would be used
with real on-chain CSVs.

Proxy definitions:
  1. pool_imbalance_proxy   : signed price-ratio spread between USDC and USDT
                              per snapshot; for other nodes, their own deviation
                              from mean across all nodes (cross-asset Z-score)
  2. redemption_pressure    : amihud × log_vol_1h — high market impact on high
                              volume signals forced liquidation / redemption flow
  3. cex_netflow_proxy      : rolling change in log_vol_1h (positive = inflow spike,
                              negative = outflow / volume drop)

These augment the existing 48-dim node feature vector → 51-dim.

Usage:
  python features/onchain_proxies.py --output data/processed/onchain_proxy_features.parquet
  python features/onchain_proxies.py --retrain --model gat --horizon 1440

Outputs:
  data/processed/onchain_proxy_features.parquet   (episode, snapshot, node) → 3 new features
  results/eval/onchain_augmented_h{H}.csv          model comparison with/without
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_episode, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402

# Indices of features in the 48-dim vector
FEAT_PRICE_RATIO = 0    # price_ratio (index 0)
FEAT_AMIHUD = 4         # amihud
FEAT_LOG_VOL = 3        # log_vol_1h

# Node names for pool_imbalance: USDC is anchor
USDC_NODE = "USDC"
USDT_NODE = "USDT"


def compute_onchain_proxies(episode_name: str) -> pd.DataFrame:
    """
    Compute 3 proxy on-chain features for all (snapshot, node) pairs in an episode.
    Returns a long-format DataFrame with columns:
        episode, snapshot, node, pool_imbalance_proxy, redemption_pressure, cex_netflow_proxy
    """
    b = load_episode(episode_name)
    node_strs = b["node_strs"]
    X = b["X"]              # (S, N, F)
    active = b["active"]    # (S, N)
    S, N, F = X.shape

    # Locate USDC and USDT node indices for pool_imbalance
    usdc_idx = node_strs.index(USDC_NODE) if USDC_NODE in node_strs else -1
    usdt_idx = node_strs.index(USDT_NODE) if USDT_NODE in node_strs else -1

    rows = []
    prev_log_vol = None

    for si in range(S):
        x_snap = X[si]  # (N, F)
        price_ratios = x_snap[:, FEAT_PRICE_RATIO]  # (N,)
        amihud_vals = x_snap[:, FEAT_AMIHUD]        # (N,)
        log_vol_vals = x_snap[:, FEAT_LOG_VOL]      # (N,)

        # 1. pool_imbalance_proxy: signed USDC-USDT spread, else cross-node Z-score
        if usdc_idx >= 0 and usdt_idx >= 0:
            anchor_spread = price_ratios[usdc_idx] - price_ratios[usdt_idx]
        else:
            anchor_spread = 0.0
        active_mask = active[si]
        mean_pr = price_ratios[active_mask].mean() if active_mask.any() else 0.0
        std_pr = price_ratios[active_mask].std() + 1e-8 if active_mask.any() else 1.0

        # 2. redemption_pressure: amihud × log_vol (high-impact high-volume = pressure)
        redemption = amihud_vals * log_vol_vals

        # 3. cex_netflow_proxy: rolling log_vol change
        if prev_log_vol is not None:
            netflow = log_vol_vals - prev_log_vol
        else:
            netflow = np.zeros(N)
        prev_log_vol = log_vol_vals.copy()

        for j, node in enumerate(node_strs):
            if not active_mask[j]:
                continue
            # Pool imbalance: USDC and USDT get the anchor spread; others get Z-score
            if node in (USDC_NODE, USDT_NODE):
                pool_imb = float(anchor_spread)
            else:
                pool_imb = (price_ratios[j] - mean_pr) / std_pr

            rows.append({
                "episode": episode_name,
                "snapshot": si,
                "node": node,
                "pool_imbalance_proxy": float(pool_imb),
                "redemption_pressure": float(np.clip(redemption[j], -10, 10)),
                "cex_netflow_proxy": float(np.clip(netflow[j], -5, 5)),
            })

    return pd.DataFrame(rows)


def build_all_proxies(episodes: List[str]) -> pd.DataFrame:
    """Build proxy features for a list of episodes."""
    parts = [compute_onchain_proxies(ep) for ep in episodes]
    return pd.concat(parts, ignore_index=True)


def augment_tabular(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame,
                    proxy_df: pd.DataFrame) -> np.ndarray:
    """
    Append the 3 proxy features to X.
    meta must have columns: episode, snapshot, node.
    Returns X_aug of shape (M, F+3).
    """
    proxy_df = proxy_df.set_index(["episode", "snapshot", "node"])
    new_cols = ["pool_imbalance_proxy", "redemption_pressure", "cex_netflow_proxy"]
    extra = np.zeros((len(X), 3), dtype=np.float32)
    meta = meta.reset_index(drop=True)
    for i, row in meta.iterrows():
        key = (row["episode"], int(row["snapshot"]), row["node"])
        if key in proxy_df.index:
            vals = proxy_df.loc[key, new_cols].values.astype(np.float32)
            extra[i] = vals
    return np.concatenate([X, extra], axis=1)


def compare_with_onchain_proxies(feat_names: list, horizon: int, seed: int = 42) -> pd.DataFrame:
    """
    Train XGBoost and GAT with and without proxy on-chain features on held-out SVB test.
    Returns comparison DataFrame.
    """
    from run_benchmark import cluster_of
    all_eps = list_episodes()
    train_eps = [e for e in all_eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")]
    val_eps = [e for e in all_eps if cluster_of(e) == "FTX_2022"]
    test_eps = [e for e in all_eps if cluster_of(e) == "SVB_2023"]

    Xtr, ytr, mtr = tabular_from_episodes(train_eps, horizon, feat_names)
    Xte, yte, mte = tabular_from_episodes(test_eps, horizon, feat_names)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    # Build proxy features
    proxy_tr = build_all_proxies(train_eps)
    proxy_te = build_all_proxies(test_eps)
    Xtr_aug = augment_tabular(Xtr, ytr, mtr, proxy_tr)
    Xte_aug = augment_tabular(Xte, yte, mte, proxy_te)

    rows = {}
    # XGBoost without proxy
    m = make_xgboost(scale_pos_weight=spw)
    m.fit(Xtr, ytr, verbose=False)
    rows["xgboost_baseline"] = full_report(yte, m.predict_proba(Xte)[:, 1])

    # XGBoost with proxy
    m2 = make_xgboost(scale_pos_weight=spw)
    m2.fit(Xtr_aug, ytr, verbose=False)
    rows["xgboost_onchain"] = full_report(yte, m2.predict_proba(Xte_aug)[:, 1])

    return pd.DataFrame({k: {m: v for m, v in r.items() if m in ("pr_auc", "roc_auc", "weighted_f1")}
                         for k, r in rows.items()}).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/processed/onchain_proxy_features.parquet")
    ap.add_argument("--retrain", action="store_true",
                    help="Also retrain XGBoost with proxy features and report PR-AUC comparison")
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_eps = list_episodes()
    print(f"Computing on-chain proxy features for {len(all_eps)} episodes...")
    proxy_df = build_all_proxies(all_eps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_df.to_parquet(out_path, index=False)
    print(f"Saved {len(proxy_df)} rows to {out_path}")
    print(proxy_df.describe())

    if args.retrain:
        feat_names = load_feature_names()
        print(f"\n=== Retraining with on-chain proxies (h={args.horizon}) ===")
        Path("results/eval").mkdir(parents=True, exist_ok=True)
        comp = compare_with_onchain_proxies(feat_names, args.horizon, args.seed)
        out_csv = f"results/eval/onchain_augmented_h{args.horizon}.csv"
        comp.to_csv(out_csv)
        print(comp.round(4).to_string())
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
