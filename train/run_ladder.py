"""
Run the full model ladder: baselines → LR → XGB → LSTM → GraphSAGE → GAT.

Usage:
    python train/run_ladder.py --config configs/experiment.yaml --horizon 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scgnn.eval.metrics import full_classification_report, lead_time_decay
from scgnn.features.labels import make_labels, class_weights
from scgnn.models.baselines import MajorityClassifier, PersistenceClassifier
from scgnn.models.classical import make_logreg, make_xgboost, xgb_feature_importance
from scgnn.interpret.explainability import plot_xgb_importance


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--horizon", type=int, default=60, help="Prediction horizon in minutes")
    p.add_argument("--out_dir", default="results/ladder")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    processed = Path("data/processed")
    if not processed.exists():
        print("[ERROR] Run data pipeline first: python data/build_dataset.py")
        sys.exit(1)

    X_train = np.load(processed / "X_train.npy")
    y_train = np.load(processed / f"y_train_h{args.horizon}.npy")
    X_val = np.load(processed / "X_val.npy")
    y_val = np.load(processed / f"y_val_h{args.horizon}.npy")
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / f"y_test_h{args.horizon}.npy")

    results = {}

    for name, clf in [
        ("majority", MajorityClassifier()),
        ("persistence", PersistenceClassifier()),
        ("logreg", make_logreg()),
    ]:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_test)[:, 1]
        else:
            probs = clf.predict(X_test).astype(float)
        results[name] = full_classification_report(y_test, probs)
        print(f"{name}: {results[name]}")

    cw = class_weights(pd.Series(y_train))
    spw = cw.get(0, 1.0) / cw.get(1, 1.0)
    xgb = make_xgboost(scale_pos_weight=spw)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    probs = xgb.predict_proba(X_test)[:, 1]
    results["xgboost"] = full_classification_report(y_test, probs)
    print(f"xgboost: {results['xgboost']}")

    feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
    importance = xgb_feature_importance(xgb, feature_names, topk=cfg["interpretability"]["xgb_gain_topk"])
    plot_xgb_importance(importance, out_path=out / "xgb_importance.png")

    with open(out / f"ladder_h{args.horizon}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
