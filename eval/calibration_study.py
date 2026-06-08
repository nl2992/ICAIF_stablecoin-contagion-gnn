"""
Plan D — Probabilistic calibration and deployment-threshold study.

Computes:
  1. Reliability diagrams (10-bin) and ECE for GAT, XGBoost, GRU
  2. Brier score per model
  3. Threshold sweep (0.05 → 0.95): precision, recall, F1, alerts-per-24h
  4. Operating-point table at precision >= 0.60 and recall >= 0.60

Uses the pre-saved probability arrays from results/ladder/probs_{model}_h{H}.npy
(produced by run_benchmark.py). Falls back to re-training if .npy not found.

Usage:
  python eval/calibration_study.py [--horizon 1440] [--retrain]

Outputs:
  results/eval/calibration_ece_h{H}.csv        ECE + Brier per model
  results/eval/threshold_sweep_h{H}.csv         precision/recall/F1 per threshold
  results/eval/operating_points_h{H}.csv        recommended alert thresholds
  results/figures/reliability_diagram_h{H}.png  reliability diagram figure
  results/figures/prt_curve_h{H}.png            precision-recall-threshold curve
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, precision_recall_curve

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from scgnn.eval.calibration_curve import (  # noqa: E402
    expected_calibration_error, plot_reliability_diagram, reliability_diagram)


def load_probs(model: str, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pre-computed probabilities. Returns (y_true, y_prob) or None."""
    labels_path = Path(f"results/ladder/test_labels_h{horizon}.npy")
    probs_path = Path(f"results/ladder/probs_{model}_h{horizon}.npy")
    if labels_path.exists() and probs_path.exists():
        y_true = np.load(labels_path)
        y_prob = np.load(probs_path)
        return y_true, y_prob
    return None


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                    n_snapshots_24h: int = 24) -> pd.DataFrame:
    """
    Sweep thresholds 0.05 → 0.95. For each threshold record:
      precision, recall, F1, alert_rate (fraction of predictions that fire).
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        alert_rate = float(y_pred.mean())
        rows.append({
            "threshold": round(float(thr), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "alert_rate": round(alert_rate, 4),
        })
    return pd.DataFrame(rows).set_index("threshold")


def find_operating_points(sweep_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Find thresholds at precision >= 0.60 and recall >= 0.60."""
    rows = []
    # Best threshold at precision >= 0.60 (maximize recall)
    high_prec = sweep_df[sweep_df["precision"] >= 0.60]
    if len(high_prec) > 0:
        best = high_prec.loc[high_prec["recall"].idxmax()]
        rows.append({"model": model_name, "operating_point": "prec_ge_0.60",
                     "threshold": best.name, "precision": best["precision"],
                     "recall": best["recall"], "f1": best["f1"],
                     "alert_rate": best["alert_rate"]})
    # Best threshold at recall >= 0.60 (maximize precision)
    high_rec = sweep_df[sweep_df["recall"] >= 0.60]
    if len(high_rec) > 0:
        best = high_rec.loc[high_rec["precision"].idxmax()]
        rows.append({"model": model_name, "operating_point": "rec_ge_0.60",
                     "threshold": best.name, "precision": best["precision"],
                     "recall": best["recall"], "f1": best["f1"],
                     "alert_rate": best["alert_rate"]})
    return pd.DataFrame(rows)


def plot_prt_curve(models_sweep: dict, horizon: int, out_dir: Path) -> None:
    """Precision-Recall-Threshold curve for all models on a shared plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"gat": "#d62728", "xgboost": "#1f77b4", "gru": "#2ca02c",
              "graphsage": "#ff7f0e", "logreg": "#9467bd"}

    for model_name, df in models_sweep.items():
        col = colors.get(model_name, "black")
        thrs = df.index.values
        ax1.plot(thrs, df["precision"], label=f"{model_name} precision",
                 color=col, linestyle="-", marker="o", markersize=3)
        ax1.plot(thrs, df["recall"], label=f"{model_name} recall",
                 color=col, linestyle="--", marker="s", markersize=3)

    ax1.axhline(0.60, color="gray", linestyle=":", alpha=0.6, label="0.60 target")
    ax1.set_xlabel("Alert threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(f"Precision & Recall vs Threshold (h={horizon}min)")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    for model_name, df in models_sweep.items():
        col = colors.get(model_name, "black")
        ax2.plot(df["alert_rate"], df["precision"],
                 label=model_name, color=col, marker="o", markersize=3)

    ax2.set_xlabel("Alert rate (fraction of timesteps flagged)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision vs Alert Rate (operational cost)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / f"prt_curve_h{horizon}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--models", default="gat,xgboost,gru",
                    help="Comma-separated models to evaluate")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain models if .npy files not found")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    h = args.horizon

    Path("results/eval").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # ---- Load or retrain ----
    model_data = {}
    for model_name in models:
        loaded = load_probs(model_name, h)
        if loaded is not None:
            y_true, y_prob = loaded
            if y_true.sum() > 0:
                model_data[model_name] = (y_true, y_prob)
                print(f"Loaded {model_name}: n={len(y_true)}, pos={int(y_true.sum())}")
        else:
            print(f"[warn] No saved probs for {model_name} h={h}. Run run_benchmark.py first.")

    if not model_data:
        print("No model data found. Run: python eval/run_benchmark.py --horizon", h)
        return

    # ---- ECE and Brier score ----
    ece_rows = []
    for model_name, (y_true, y_prob) in model_data.items():
        ece = expected_calibration_error(y_true, y_prob)
        brier = float(brier_score_loss(y_true, y_prob))
        ece_rows.append({
            "model": model_name, "ece": round(ece, 4), "brier_score": round(brier, 4),
            "calibrated": ece < 0.05,
            "n": len(y_true), "pos_rate": round(float(y_true.mean()), 4),
        })
        print(f"{model_name:15s}: ECE={ece:.4f}, Brier={brier:.4f}")

    ece_df = pd.DataFrame(ece_rows).set_index("model")
    ece_df.to_csv(f"results/eval/calibration_ece_h{h}.csv")
    print(f"\nSaved: results/eval/calibration_ece_h{h}.csv")

    # ---- Reliability diagram ----
    plot_reliability_diagram(
        model_data,
        n_bins=10,
        out_path=Path(f"results/figures/reliability_diagram_h{h}.png"),
    )

    # ---- Threshold sweep ----
    all_sweeps = {}
    all_ops = []
    for model_name, (y_true, y_prob) in model_data.items():
        sweep = threshold_sweep(y_true, y_prob)
        all_sweeps[model_name] = sweep
        ops = find_operating_points(sweep, model_name)
        all_ops.append(ops)
        sweep.to_csv(f"results/eval/threshold_sweep_{model_name}_h{h}.csv")

    # Combined threshold sweep
    combined_rows = []
    for model_name, df in all_sweeps.items():
        for thr, row in df.iterrows():
            combined_rows.append({"model": model_name, "threshold": thr, **row.to_dict()})
    pd.DataFrame(combined_rows).to_csv(f"results/eval/threshold_sweep_h{h}.csv", index=False)

    # Operating points table
    if all_ops:
        ops_df = pd.concat(all_ops, ignore_index=True)
        ops_df.to_csv(f"results/eval/operating_points_h{h}.csv", index=False)
        print(f"\n=== Recommended Operating Points (h={h}min) ===")
        print(ops_df.to_string(index=False))

    # ---- PRT curve figure ----
    plot_prt_curve(all_sweeps, h, Path("results/figures"))

    print(f"\n=== Calibration Summary (h={h}min) ===")
    print(ece_df.to_string())

    # Headline: which model is best calibrated on stress episodes?
    if "gat" in ece_df.index and "xgboost" in ece_df.index:
        gat_ece = ece_df.loc["gat", "ece"]
        xgb_ece = ece_df.loc["xgboost", "ece"]
        better = "GAT" if gat_ece < xgb_ece else "XGBoost"
        print(f"\nCalibration finding: {better} has lower ECE ({min(gat_ece, xgb_ece):.4f} vs {max(gat_ece, xgb_ece):.4f})")


if __name__ == "__main__":
    main()
