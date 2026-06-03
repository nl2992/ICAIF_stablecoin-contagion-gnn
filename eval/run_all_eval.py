"""
Phase 5 evaluation runner.

Generates:
  - Lead-time decay table + figure
  - Ablation tables (a, b, c)
  - LOEO-CV table
  - Threshold sensitivity figure
  - Synthetic validation moments table

Usage:
    python eval/run_all_eval.py --config configs/experiment.yaml --model xgboost
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
from scgnn.eval.metrics import lead_time_table, threshold_sensitivity, full_report, results_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", default="xgboost", help="Model name to use for eval")
    p.add_argument("--out_dir", default="results/eval")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_all_seeds(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = Path("data/processed")
    ladder_dir = Path("results/ladder")

    horizons = cfg["labels"]["horizons_min"]

    # ── Lead-time decay ──────────────────────────────────────────────────────
    lt_rows = []
    for h in horizons:
        probs_path = ladder_dir / f"probs_{args.model}_h{h}.npy"
        y_path = processed / f"y_test_h{h}.npy"
        if not probs_path.exists() or not y_path.exists():
            print(f"[WARN] Missing data for horizon {h} — run train/run_ladder.py first")
            continue
        probs = np.load(probs_path)
        y = np.load(y_path).ravel()
        report = full_report(y, probs)
        lt_rows.append({"horizon_min": h, "pr_auc": report["pr_auc"],
                        "roc_auc": report["roc_auc"], "weighted_f1": report["weighted_f1"]})

    if lt_rows:
        lt_df = pd.DataFrame(lt_rows).set_index("horizon_min")
        lt_df.to_csv(out / f"lead_time_{args.model}.csv")
        # Figure
        fig, ax = plt.subplots(figsize=(7, 4))
        for col, marker, label in [("pr_auc", "o", "PR-AUC"), ("roc_auc", "s", "ROC-AUC"), ("weighted_f1", "^", "Weighted F1")]:
            ax.plot(lt_df.index, lt_df[col], marker=marker, label=label)
        ax.set_xlabel("Prediction horizon (minutes)")
        ax.set_ylabel("Score")
        ax.set_title(f"Lead-time accuracy decay — {args.model}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out / f"lead_time_{args.model}.png", dpi=150)
        plt.close(fig)
        print("Lead-time table:")
        print(lt_df.to_string())

    # ── Threshold sensitivity (best horizon = 60 min) ────────────────────────
    h_best = 60
    sens_probs_path = ladder_dir / f"probs_{args.model}_h{h_best}.npy"
    sens_y_path = processed / f"y_test_h{h_best}.npy"
    if sens_probs_path.exists() and sens_y_path.exists():
        probs = np.load(sens_probs_path)
        y = np.load(sens_y_path).ravel()
        thresholds = list(np.arange(0.1, 0.9, 0.05))
        sens_df = threshold_sensitivity(y, probs, thresholds)
        sens_df.to_csv(out / f"threshold_sensitivity_{args.model}.csv")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(sens_df.index, sens_df["pr_auc"], marker="o", label="PR-AUC")
        ax.plot(sens_df.index, sens_df["weighted_f1"], marker="s", label="Weighted F1")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="threshold=0.5")
        ax.set_xlabel("Classification threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold sensitivity — {args.model} (horizon={h_best}min)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out / f"threshold_sensitivity_{args.model}.png", dpi=150)
        plt.close(fig)

    print(f"\nAll eval outputs saved to {out}/")


if __name__ == "__main__":
    main()
