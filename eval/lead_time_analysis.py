"""
Lead-time analysis: accuracy vs prediction horizon for the best model.

Usage:
    python eval/lead_time_analysis.py --model graphsage --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scgnn.eval.metrics import full_classification_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", default="graphsage")
    p.add_argument("--out_dir", default="results/lead_time")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    horizons = cfg["labels"]["horizon_minutes"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    processed = Path("data/processed")
    rows = []
    for h in horizons:
        X_test = np.load(processed / "X_test.npy")
        y_test = np.load(processed / f"y_test_h{h}.npy")
        probs_path = processed / f"probs_{args.model}_h{h}.npy"
        if not probs_path.exists():
            print(f"[WARN] Missing predictions for horizon={h}, skipping.")
            continue
        probs = np.load(probs_path)
        report = full_classification_report(y_test, probs)
        rows.append({"horizon_min": h, **report})

    if not rows:
        print("No predictions found. Run train scripts first.")
        return

    df = pd.DataFrame(rows).set_index("horizon_min")
    df.to_csv(out / f"lead_time_{args.model}.csv")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.index, df["weighted_f1"], marker="o", label="Weighted F1")
    ax.plot(df.index, df["roc_auc"], marker="s", label="ROC-AUC")
    ax.set_xlabel("Prediction horizon (minutes)")
    ax.set_ylabel("Score")
    ax.set_title(f"Lead-time accuracy decay — {args.model}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / f"lead_time_{args.model}.png", dpi=150)
    print(f"Lead-time analysis saved to {out}")


if __name__ == "__main__":
    main()
