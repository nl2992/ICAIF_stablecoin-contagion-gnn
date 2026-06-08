"""
Plan B — LOCO runner with --episode-set flag for sparse-episode sensitivity analysis.

Compares LOCO PR-AUC on:
  all_7   : all 7 episodes (default)
  stable_5: the 5 high-quality episodes (exclude BUSD_winddown, USDT_Oct2018)

Usage:
  python scripts/run_loco.py --episode_set stable_5 --model gat --horizon 1440
  python scripts/run_loco.py --compare_sets all_7,stable_5

Outputs:
  results/eval/loco_{episode_set}_h{H}.csv   per-fold PR-AUC
  results/eval/loco_stability_comparison.csv  side-by-side comparison (if --compare_sets)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402
from scgnn.utils.seeds import set_all_seeds  # noqa: E402

STABLE_5 = {"USDC_SVB", "FRAX_SVB", "DAI_FTX", "UST_Terra", "USDT_May2022"}

CLUSTERS_ALL7 = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}

CLUSTERS_STABLE5 = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022",
}


def get_episode_set(name: str) -> tuple[list, dict]:
    """Return (episodes, cluster_map) for a given set name."""
    all_eps = list_episodes()
    if name == "all_7":
        return all_eps, CLUSTERS_ALL7
    if name == "stable_5":
        eps = [e for e in all_eps if e in STABLE_5]
        return eps, CLUSTERS_STABLE5
    raise ValueError(f"Unknown episode set: {name}. Choose 'all_7' or 'stable_5'.")


def run_loco(episodes: list, cluster_map: dict, feat_names: list,
             model_kind: str, horizon: int, seed: int = 42) -> pd.DataFrame:
    """Run LOCO evaluation and return per-fold results DataFrame."""
    clusters = sorted(set(cluster_map[e] for e in episodes))
    rows = []

    for held_cluster in clusters:
        held = [e for e in episodes if cluster_map.get(e) == held_cluster]
        train = [e for e in episodes if cluster_map.get(e) != held_cluster]
        if not train or not held:
            continue

        Xtr, ytr, _ = tabular_from_episodes(train, horizon, feat_names)
        Xte, yte, _ = tabular_from_episodes(held, horizon, feat_names)
        pos_rate = float(yte.mean()) if len(yte) > 0 else 0.0
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

        row = {"held_cluster": held_cluster, "episodes": "+".join(held),
               "n_test": len(yte), "pos_rate": round(pos_rate, 4)}

        # XGBoost baseline
        m = make_xgboost(scale_pos_weight=spw)
        m.fit(Xtr, ytr, verbose=False)
        xgb_pr = full_report(yte, m.predict_proba(Xte)[:, 1])["pr_auc"] if yte.sum() > 0 else float("nan")
        row["xgboost"] = round(xgb_pr, 5)

        # GNN model
        if model_kind in ("gat", "graphsage"):
            set_all_seeds(seed)
            tr = GNNContagionTrainer(kind=model_kind, horizon=horizon, seed=seed,
                                     epochs=80, patience=10)
            tr.fit(train, None)
            p = tr.predict_episodes(held)
            gnn_pr = full_report(yte, p)["pr_auc"] if yte.sum() > 0 else float("nan")
            row[model_kind] = round(gnn_pr, 5)
            row["margin_vs_xgb"] = round(gnn_pr - xgb_pr, 5) if not (np.isnan(gnn_pr) or np.isnan(xgb_pr)) else float("nan")

        rows.append(row)
        print(f"  {held_cluster}: n={len(yte)}, pos_rate={pos_rate:.3f}, xgb={xgb_pr:.4f}, {model_kind}={gnn_pr:.4f}")

    df = pd.DataFrame(rows).set_index("held_cluster")
    valid = df[df["pos_rate"] > 0]
    if len(valid) > 0:
        mean_row = valid.select_dtypes("number").mean()
        mean_row.name = "MEAN"
        df = pd.concat([df, mean_row.to_frame().T])
    return df


def compare_sets(model_kind: str, horizon: int, seed: int, feat_names: list) -> pd.DataFrame:
    """Run both episode sets and produce a side-by-side comparison."""
    results = {}
    for set_name in ["all_7", "stable_5"]:
        episodes, cluster_map = get_episode_set(set_name)
        print(f"\n=== Episode set: {set_name} ({len(episodes)} episodes) ===")
        df = run_loco(episodes, cluster_map, feat_names, model_kind, horizon, seed)
        results[set_name] = df

    # Align on common index for comparison
    all_idx = sorted(set(results["all_7"].index) | set(results["stable_5"].index))
    comparison_rows = []
    for idx in all_idx:
        row = {"fold": idx}
        for set_name, df in results.items():
            if idx in df.index:
                for col in ["xgboost", model_kind, "margin_vs_xgb", "pos_rate"]:
                    if col in df.columns:
                        row[f"{set_name}_{col}"] = df.loc[idx, col]
        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows).set_index("fold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_set", default="stable_5", choices=["all_7", "stable_5"])
    ap.add_argument("--compare_sets", action="store_true",
                    help="Run both sets and produce side-by-side comparison")
    ap.add_argument("--model", default="gat", choices=["gat", "graphsage", "xgboost"])
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feat_names = load_feature_names()
    Path("results/eval").mkdir(parents=True, exist_ok=True)

    if args.compare_sets:
        print("\n=== SPARSE-EPISODE SENSITIVITY (Plan B) ===")
        comp = compare_sets(args.model, args.horizon, args.seed, feat_names)
        out = f"results/eval/loco_stability_comparison_h{args.horizon}.csv"
        comp.to_csv(out)
        print(f"\n{comp.round(4).to_string()}")
        print(f"\nSaved: {out}")

        # Report the key numbers
        for set_name in ["all_7", "stable_5"]:
            col_m = f"{set_name}_{args.model}"
            col_x = f"{set_name}_xgboost"
            if col_m in comp.columns and "MEAN" in comp.index:
                gnn_mean = comp.loc["MEAN", col_m]
                xgb_mean = comp.loc["MEAN", col_x]
                print(f"\n{set_name}: LOCO mean PR-AUC = {gnn_mean:.4f} ({args.model}), "
                      f"{xgb_mean:.4f} (xgboost), margin = {gnn_mean - xgb_mean:+.4f}")
    else:
        episodes, cluster_map = get_episode_set(args.episode_set)
        print(f"\n=== LOCO | set={args.episode_set} | model={args.model} | h={args.horizon} ===")
        print(f"Episodes: {episodes}")
        df = run_loco(episodes, cluster_map, feat_names, args.model, args.horizon, args.seed)
        out = f"results/eval/loco_{args.episode_set}_h{args.horizon}.csv"
        df.to_csv(out)
        print(f"\n{df.round(4).to_string()}")
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
