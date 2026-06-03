"""
Probability calibration and reliability diagrams.

A contagion early-warning tool that isn't calibrated is operationally useless:
if the model says P=0.8 but actual frequency is 0.3, risk managers will
miscalibrate their responses.

We report:
  1. Reliability diagram (calibration curve) per model
  2. Expected Calibration Error (ECE) — single scalar for paper table
  3. Isotonic regression post-hoc calibration (if ECE > 0.05, apply and report)

Calibration is computed on the VAL SET (not test), then evaluated on test.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",   # "uniform" or "quantile"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data.

    Returns (bin_centers, fraction_positive, bin_counts).
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_prob, quantiles)
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    bin_ids = np.digitize(y_prob, bins[1:-1])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    fraction_positive = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_ids == b
        bin_counts[b] = mask.sum()
        if bin_counts[b] > 0:
            fraction_positive[b] = y_true[mask].mean()

    return bin_centers, fraction_positive, bin_counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted mean |fraction_positive - bin_mean_prob|."""
    bins_c, frac_pos, counts = reliability_diagram(y_true, y_prob, n_bins)
    n = len(y_true)
    ece = 0.0
    for b in range(len(bins_c)):
        if counts[b] > 0:
            conf = bins_c[b]
            ece += (counts[b] / n) * abs(frac_pos[b] - conf)
    return float(ece)


def fit_isotonic_calibration(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
):
    """Fit isotonic regression calibrator on validation set."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs_cal, y_cal)
    return ir


def plot_reliability_diagram(
    models: dict,           # {model_name: (y_true, y_prob)}
    n_bins: int = 10,
    out_path: Optional[Path] = None,
) -> None:
    """
    Plot reliability diagrams for multiple models on one figure.
    models = {"xgboost": (y_true, y_prob), "graphsage": (y_true, y_prob), ...}
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), squeeze=False)

    for ax, (name, (y_true, y_prob)) in zip(axes[0], models.items()):
        bins_c, frac_pos, counts = reliability_diagram(y_true, y_prob, n_bins)
        ece = expected_calibration_error(y_true, y_prob, n_bins)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        mask = counts > 0
        ax.plot(bins_c[mask], frac_pos[mask], "o-", color="#b2182b",
                label=f"{name}\nECE={ece:.3f}", linewidth=1.5, markersize=5)

        # Histogram of predicted probabilities (right y-axis)
        ax2 = ax.twinx()
        ax2.bar(bins_c, counts / counts.sum(), width=1.0 / n_bins,
                alpha=0.15, color="steelblue", align="center")
        ax2.set_ylabel("Fraction of samples", color="steelblue", fontsize=7)
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Reliability — {name}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Reliability diagram saved: {out_path}")
    plt.close(fig)


def calibration_summary_table(
    models: dict,   # {model_name: (y_true, y_prob)}
    n_bins: int = 10,
) -> pd.DataFrame:
    """ECE and calibration verdict for each model — for paper table."""
    rows = []
    for name, (y_true, y_prob) in models.items():
        ece = expected_calibration_error(y_true, y_prob, n_bins)
        rows.append({
            "model": name,
            "ece": round(ece, 4),
            "calibrated": ece < 0.05,
            "action": "OK" if ece < 0.05 else "Apply isotonic recalibration",
        })
    return pd.DataFrame(rows).set_index("model")
