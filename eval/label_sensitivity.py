"""
Label-threshold sensitivity sweep.

Trains XGBoost (fastest reliable non-trivial model) at each threshold and
reports PR-AUC + top-10 hub rankings.  Hub rankings must be stable across
thresholds — if they reshuffle, the result is threshold-tuned.

Thresholds swept: [10, 25, 50] bps  (fiat-backed; crypto/synth scaled ×3/×2).
Horizons: all four from configs/experiment.yaml.

Outputs:
  results/sensitivity/pr_auc_by_threshold.csv
  results/sensitivity/hub_rank_spearman.csv    ← key stability check
  results/sensitivity/fig_sensitivity.png

Usage:
    python eval/label_sensitivity.py [--config configs/experiment.yaml] [--stub]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.utils.seeds import set_all_seeds
from scgnn.features.labels import make_onset_labels, base_rate_table
from scgnn.eval.metrics import full_report
from scgnn.models.classical import make_xgboost
from scgnn.features.labels import class_weights

SWEEP_THRESHOLDS_FIAT = [10, 25, 50]   # bps
SUSTAINED_MIN = 10


def _scale_thresholds(fiat_bps: int, node_strs: list[str]) -> dict[str, float]:
    return {
        n: fiat_bps * (3 if n.split("/")[0] in ("DAI", "FRAX") else
                       2 if n.split("/")[0] in ("USDe", "USDE") else 1)
        for n in node_strs
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--out_dir", default="results/sensitivity")
    p.add_argument("--stub", action="store_true")
    return p.parse_args()


def _stub_data(n=400, n_nodes=6, seed=0):
    rng = np.random.default_rng(seed)
    prices = 1.0 + rng.normal(0, 0.001, (n, n_nodes))
    prices[200:230, 0] += 0.006   # inject depeg on node 0
    node_strs = [f"USDC/binance", f"USDT/binance", f"DAI/coinbase",
                 f"FRAX/coinbase", f"TUSD/kraken", f"USDe/binance"]
    idx = pd.date_range("2023-03-10", periods=n, freq="1min", tz="UTC")
    return {ns: pd.Series((prices[:, i] - 1) * 10_000, index=idx)
            for i, ns in enumerate(node_strs)}


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_all_seeds(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = Path("data/processed")

    if args.stub:
        peg_devs = _stub_data()
        node_strs = list(peg_devs.keys())
        origin = "USDC/binance"
    else:
        X_train = np.load(processed / "X_train.npy")
        X_test  = np.load(processed / "X_test.npy")
        node_strs = pd.read_json(processed / "node_registry.json")["node"].tolist()
        origin = "USDC/binance"   # USDC_SVB test episode trigger

    rows = []
    hub_rankings: dict[int, list[str]] = {}   # {threshold: ranked node list}

    for thr_fiat in SWEEP_THRESHOLDS_FIAT:
        thr_map = _scale_thresholds(thr_fiat, node_strs)

        if args.stub:
            labels = make_onset_labels(peg_devs, args.horizon, thr_map, SUSTAINED_MIN, origin)
            # Build a flat feature matrix from deviations for XGB
            idx = list(peg_devs.values())[0].index
            X = np.column_stack([peg_devs[n].values for n in node_strs])
            y = np.stack([labels[n] for n in node_strs], axis=1).ravel()
            X_flat = np.tile(X, (1, 1)).reshape(-1, len(node_strs))
            n_split = int(len(idx) * 0.7)
            X_tr, y_tr = X_flat[:n_split], y[:n_split * len(node_strs)]
            X_te, y_te = X_flat[n_split:], y[n_split * len(node_strs):]
        else:
            y_tr = np.load(processed / f"y_train_h{args.horizon}_thr{thr_fiat}.npy").ravel()
            y_te = np.load(processed / f"y_test_h{args.horizon}_thr{thr_fiat}.npy").ravel()
            X_tr, X_te = X_train, X_test

        if y_tr.sum() == 0 or y_te.sum() == 0:
            print(f"[WARN] threshold={thr_fiat} bps: no positives — skipping")
            continue

        cw = class_weights(pd.Series(y_tr))
        spw = cw.get(0, 1.0) / cw.get(1, 1.0)
        xgb = make_xgboost(scale_pos_weight=spw)
        xgb.fit(X_tr, y_tr)
        probs = xgb.predict_proba(X_te)[:, 1]
        report = full_report(y_te, probs)

        rows.append({
            "threshold_bps_fiat": thr_fiat,
            "positive_rate_train": round(float(y_tr.mean()), 4),
            "positive_rate_test": round(float(y_te.mean()), 4),
            "pr_auc": round(report["pr_auc"], 4),
            "roc_auc": round(report["roc_auc"], 4),
            "weighted_f1": round(report["weighted_f1"], 4),
        })

        # Hub ranking from XGB feature importance (stub: use gain as proxy for node importance)
        scores = xgb.get_booster().get_score(importance_type="gain")
        # Aggregate score per node (each node contributes multiple features)
        node_score = {}
        for n in node_strs:
            n_idx = node_strs.index(n)
            relevant = {k: v for k, v in scores.items() if k.startswith(f"f{n_idx}")}
            node_score[n] = sum(relevant.values())
        ranked = sorted(node_score.keys(), key=lambda x: node_score[x], reverse=True)
        hub_rankings[thr_fiat] = ranked

    # ── Results table ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out / "pr_auc_by_threshold.csv", index=False)
        print("\nPR-AUC by threshold:")
        print(df.to_string(index=False))

    # ── Hub-ranking stability across thresholds ───────────────────────────
    if len(hub_rankings) >= 2:
        from scipy.stats import spearmanr
        thresholds_run = list(hub_rankings.keys())
        all_nodes = hub_rankings[thresholds_run[0]]
        spear_rows = []
        for i, t1 in enumerate(thresholds_run):
            for j, t2 in enumerate(thresholds_run):
                if j <= i:
                    continue
                r1 = hub_rankings[t1]
                r2 = hub_rankings[t2]
                shared = [n for n in r1 if n in r2]
                if len(shared) < 3:
                    continue
                rank1 = [r1.index(n) for n in shared]
                rank2 = [r2.index(n) for n in shared]
                rho, pval = spearmanr(rank1, rank2)
                spear_rows.append({
                    "thr_a": t1, "thr_b": t2,
                    "spearman_rho": round(float(rho), 3),
                    "p_value": round(float(pval), 4),
                    "stable": rho > 0.7,
                })
        stab_df = pd.DataFrame(spear_rows)
        stab_df.to_csv(out / "hub_rank_spearman.csv", index=False)
        print("\nHub rank Spearman ρ across thresholds (>0.7 = stable):")
        print(stab_df.to_string(index=False))

    # ── Figure ───────────────────────────────────────────────────────────
    if not df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["threshold_bps_fiat"], df["pr_auc"], marker="o", label="PR-AUC", color="#b2182b")
        ax.plot(df["threshold_bps_fiat"], df["weighted_f1"], marker="s", label="Weighted F1", color="#4393c3")
        ax2 = ax.twinx()
        ax2.bar(df["threshold_bps_fiat"], df["positive_rate_test"], alpha=0.15,
                color="gray", width=6, label="Positive rate (test)")
        ax.set_xlabel("Depeg threshold (bps, fiat-backed)")
        ax.set_ylabel("Performance metric")
        ax2.set_ylabel("Positive rate", color="gray")
        ax.set_title("Label sensitivity sweep — XGBoost")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out / "fig_sensitivity.png", dpi=150)
        plt.close(fig)
        print(f"\nFigure saved: {out}/fig_sensitivity.png")


if __name__ == "__main__":
    main()
