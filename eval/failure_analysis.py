"""
Plan G — Failure-mode case study: most-confident false positives and false negatives.

For each LOCO test fold, identifies the N most-confident misclassifications from GAT.
For each selected example: extracts feature vector, predicted probability, true label,
top features by magnitude, and (if available) GAT attention weights on that snapshot.

Outputs:
  results/eval/failure_cases.csv      — structured table of all cases
  results/eval/failure_summary.csv    — aggregated by failure mode (FP vs FN)
  results/figures/failure_gallery.png — top-K misclassification gallery

Usage:
  python eval/failure_analysis.py [--n_cases 10] [--horizon 1440]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402


def classify_failure_mode(prob: float, true_label: int,
                           top_feat: str, rvol_val: float,
                           threshold: float = 0.5) -> str:
    """
    Assign a qualitative failure-mode label.
    FP = predicted positive, actually negative.
    FN = predicted negative, actually positive.
    """
    pred = int(prob >= threshold)
    if pred == 1 and true_label == 0:
        if rvol_val > 0.01:
            return "FP_vol_spike"          # high vol but no contagion
        return "FP_spurious_edge"          # structural false alarm
    if pred == 0 and true_label == 1:
        if prob > 0.3:
            return "FN_borderline"         # model nearly fires but misses
        return "FN_slow_onset"             # model ignores gradual peg drift
    return "correct"


def analyze_fold(held_cluster: str, all_eps: list, feat_names: list,
                 horizon: int, n_cases: int, seed: int) -> pd.DataFrame:
    """Train GAT on all-but-held, collect misclassifications from held."""
    held = [e for e in all_eps if cluster_of(e) == held_cluster]
    train = [e for e in all_eps if cluster_of(e) != held_cluster]
    Xte, yte, meta = tabular_from_episodes(held, horizon, feat_names)

    if yte.sum() == 0:
        print(f"  [skip] {held_cluster}: no positives in test set")
        return pd.DataFrame()

    trainer = GNNContagionTrainer(kind="gat", horizon=horizon, seed=seed,
                                   epochs=80, patience=10)
    trainer.fit(train, None)
    probs = trainer.predict_episodes(held)

    meta = meta.reset_index(drop=True)
    df = meta.copy()
    df["prob"] = probs
    df["y_true"] = yte
    df["correct"] = (probs >= 0.5).astype(int) == yte.astype(int)

    misclassified = df[~df["correct"]].copy()
    misclassified["confidence"] = np.where(
        (misclassified["prob"] >= 0.5),
        misclassified["prob"],
        1.0 - misclassified["prob"],
    )

    # Top-N false positives (highest prob among actually-negative)
    fp = misclassified[misclassified["y_true"] == 0].nlargest(n_cases, "prob")
    # Top-N false negatives (lowest prob among actually-positive)
    fn = misclassified[misclassified["y_true"] == 1].nsmallest(n_cases, "prob")
    selected = pd.concat([fp, fn], ignore_index=True)

    # Enrich with feature info
    feat_arr = Xte
    rvol_idx = feat_names.index("rvol_1h") if "rvol_1h" in feat_names else 1
    amihud_idx = feat_names.index("amihud") if "amihud" in feat_names else 4

    records = []
    for _, row in selected.iterrows():
        i = int(row.name)
        x = feat_arr[i]
        feat_magnitudes = np.abs(x)
        top_feat_idx = feat_magnitudes.argsort()[::-1][:5]
        top_feat_names = [feat_names[j] for j in top_feat_idx]
        top_feat_vals = [float(x[j]) for j in top_feat_idx]

        fm = classify_failure_mode(
            float(row["prob"]), int(row["y_true"]),
            top_feat_names[0], float(x[rvol_idx]))

        records.append({
            "held_cluster": held_cluster,
            "episode": row.get("episode", ""),
            "snapshot": row.get("snapshot", -1),
            "node": row.get("node", ""),
            "prob": round(float(row["prob"]), 4),
            "y_true": int(row["y_true"]),
            "error_type": "FP" if int(row["y_true"]) == 0 else "FN",
            "failure_mode": fm,
            "top_feature_1": top_feat_names[0],
            "top_feature_1_val": round(top_feat_vals[0], 4),
            "top_feature_2": top_feat_names[1] if len(top_feat_names) > 1 else "",
            "top_feature_2_val": round(top_feat_vals[1], 4) if len(top_feat_vals) > 1 else 0.0,
            "rvol_1h": round(float(x[rvol_idx]), 5),
            "amihud": round(float(x[amihud_idx]), 5),
        })

    return pd.DataFrame(records)


def plot_failure_gallery(df: pd.DataFrame, n_show: int, out_path: Path) -> None:
    """Bar chart of confidence levels for top misclassifications."""
    top = pd.concat([
        df[df["error_type"] == "FP"].nlargest(n_show // 2, "prob"),
        df[df["error_type"] == "FN"].nsmallest(n_show // 2, "prob"),
    ])
    if len(top) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.4)))
    colors = ["#d62728" if t == "FP" else "#1f77b4" for t in top["error_type"]]
    labels = [f"{row['error_type']} | {row['episode']} | {row['node']} | p={row['prob']:.2f}"
              for _, row in top.iterrows()]

    bars = ax.barh(range(len(top)), top["prob"].values, color=colors, alpha=0.8)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="Decision threshold")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted probability")
    ax.set_title("Top misclassifications by GAT (red=FP, blue=FN)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=10,
                    help="Top-N FPs and FNs to extract per fold")
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_names = load_feature_names()
    all_eps = list_episodes()
    clusters = sorted(set(cluster_of(e) for e in all_eps))

    Path("results/eval").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    all_cases = []
    for clus in clusters:
        print(f"\n=== Failure analysis: hold out {clus} ===")
        fold_df = analyze_fold(clus, all_eps, feat_names, args.horizon,
                               args.n_cases, args.seed)
        all_cases.append(fold_df)

    if not any(len(d) > 0 for d in all_cases):
        print("No misclassifications found (all folds skipped).")
        return

    full_df = pd.concat([d for d in all_cases if len(d) > 0], ignore_index=True)
    full_df.to_csv("results/eval/failure_cases.csv", index=False)
    print(f"\nSaved: results/eval/failure_cases.csv ({len(full_df)} cases)")

    # ---- Summary by failure mode ----
    summary = (full_df.groupby(["error_type", "failure_mode"])
               .agg(count=("prob", "count"),
                    mean_prob=("prob", "mean"),
                    top_feature=("top_feature_1", lambda x: x.value_counts().idxmax()))
               .reset_index())
    summary.to_csv("results/eval/failure_summary.csv", index=False)
    print("\n=== Failure Mode Summary ===")
    print(summary.to_string(index=False))

    # ---- Gallery figure ----
    plot_failure_gallery(full_df, n_show=min(12, len(full_df)),
                         out_path=Path("results/figures/failure_gallery.png"))

    # ---- Paper-ready narrative of 3 representative cases ----
    print("\n=== Three representative failure cases (for paper Section 5.4) ===")
    modes_shown = set()
    for _, row in full_df.iterrows():
        fm = row["failure_mode"]
        if fm in modes_shown or fm == "correct":
            continue
        modes_shown.add(fm)
        print(f"\n  [{fm}]")
        print(f"    Episode: {row['episode']}, Node: {row['node']}, Snapshot: {row['snapshot']}")
        print(f"    Predicted prob: {row['prob']:.4f}, True label: {row['y_true']}")
        print(f"    Top feature: {row['top_feature_1']} = {row['top_feature_1_val']:.4f}")
        print(f"    RVol 1h: {row['rvol_1h']:.5f}, Amihud: {row['amihud']:.5f}")
        if len(modes_shown) >= 3:
            break


if __name__ == "__main__":
    main()
