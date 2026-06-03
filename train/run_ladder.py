"""
Full model ladder — entry point for Phase 4.

Runs: majority → persistence → logreg → XGBoost → (LSTM) → GraphSAGE → GAT
Saves per-model predictions and the main results table.

Usage:
    python train/run_ladder.py --config configs/experiment.yaml --horizon 60
    python train/run_ladder.py --config configs/experiment.yaml --horizon 60 --stub
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.utils.seeds import set_all_seeds
from scgnn.eval.metrics import full_report, results_table
from scgnn.features.labels import class_weights
from scgnn.models.baselines import MajorityClassifier, PersistenceClassifier
from scgnn.models.classical import make_logreg, make_xgboost, xgb_gain_ranking
from scgnn.interpret.explainability import plot_xgb_importance


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_split(processed: Path, split: str, horizon: int):
    X = np.load(processed / f"X_{split}.npy")
    y = np.load(processed / f"y_{split}_h{horizon}.npy")
    if y.ndim > 1:
        y = y.ravel()
    return X, y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--out_dir", default="results/ladder")
    p.add_argument("--stub", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    set_all_seeds(cfg["seed"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = Path("data/processed")

    if not (processed / f"X_train.npy").exists():
        stub_flag = "--stub" if args.stub else ""
        print(f"[ERROR] Run first: python -m scgnn.data.build_dataset {stub_flag}")
        sys.exit(1)

    X_train, y_train = _load_split(processed, "train", args.horizon)
    X_val, y_val = _load_split(processed, "val", args.horizon)
    X_test, y_test = _load_split(processed, "test", args.horizon)
    print(f"Data: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
    print(f"Positive rate — train:{y_train.mean():.3f} val:{y_val.mean():.3f} test:{y_test.mean():.3f}")

    feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
    cw = class_weights(pd.Series(y_train))
    spw = cw.get(0, 1.0) / cw.get(1, 1.0) if 1 in cw else 1.0

    reports = {}
    probs_out = {}

    # Baselines
    maj = MajorityClassifier().fit(X_train, y_train)
    maj_probs = maj.predict(X_test).astype(float)
    reports["majority"] = full_report(y_test, maj_probs)
    probs_out["majority"] = maj_probs

    # Persistence: last feature is current label by convention
    # (we use predicted prob = the first feature, price_ratio, as a proxy)
    persist_probs = np.abs(X_test[:, 0] - 1.0)  # |price_ratio - 1| as persistence proxy
    persist_probs /= persist_probs.max() + 1e-9
    reports["persistence"] = full_report(y_test, persist_probs)
    probs_out["persistence"] = persist_probs

    # Logistic regression
    lr = make_logreg()
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    reports["logreg"] = full_report(y_test, lr_probs)
    probs_out["logreg"] = lr_probs

    # XGBoost
    xgb = make_xgboost(scale_pos_weight=spw)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    reports["xgboost"] = full_report(y_test, xgb_probs)
    probs_out["xgboost"] = xgb_probs

    # Save XGBoost feature importance
    imp_df = xgb_gain_ranking(xgb, feature_names, topk=cfg["interpretability"]["xgb_gain_topk"])
    imp_df.to_csv(out / f"xgb_importance_h{args.horizon}.csv")
    plot_xgb_importance(imp_df, out_path=out / f"xgb_importance_h{args.horizon}.png")

    # GNN models — only run if PyG is available
    try:
        from scgnn.models.gnn import GraphSAGEContagion, GATContagion
        print("[INFO] PyG available — GNN results need graph dataset; skipping for now.")
        print("       Run train/run_gnn.py after building graph snapshots.")
    except ImportError:
        print("[WARN] torch-geometric not installed; skipping GNN models.")

    # Persist predictions for lead-time analysis
    for model_name, probs in probs_out.items():
        np.save(out / f"probs_{model_name}_h{args.horizon}.npy", probs)

    # Main results table
    tbl = results_table(reports, metrics=["pr_auc", "roc_auc", "weighted_f1", "precision", "recall"])
    print("\n=== Main Results Table ===")
    print(tbl.to_string())
    tbl.to_csv(out / f"results_table_h{args.horizon}.csv")

    with open(out / f"reports_h{args.horizon}.json", "w") as f:
        # Exclude confusion_matrix for JSON serialisation
        clean = {k: {m: v for m, v in r.items() if m != "confusion_matrix"} for k, r in reports.items()}
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out}/")


if __name__ == "__main__":
    main()
