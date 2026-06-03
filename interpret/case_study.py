"""
Phase 6 — Single-prediction trace case study.

Traces the model's prediction for USDC/SVB episode:
  - Which node was first predicted to propagate?
  - Which features drove that prediction?
  - What was the predicted probability vs actual outcome?

Usage:
    python interpret/case_study.py --config configs/experiment.yaml
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

from scgnn.utils.seeds import set_all_seeds
from scgnn.interpret.explainability import trace_single_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", default="xgboost")
    p.add_argument("--horizon", type=int, default=60)
    p.add_argument("--out_dir", default="results/interpret")
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

    X_test_path = processed / "X_test.npy"
    y_test_path = processed / f"y_test_h{args.horizon}.npy"
    probs_path = ladder_dir / f"probs_{args.model}_h{args.horizon}.npy"

    if not X_test_path.exists() or not probs_path.exists():
        print("[ERROR] Run train/run_ladder.py first to generate predictions.")
        sys.exit(1)

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path).ravel()
    probs = np.load(probs_path)

    n_features = X_test.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_features)]

    # Find the earliest true-positive prediction (first correctly caught onset)
    tp_mask = (probs >= 0.5) & (y_test == 1)
    if not tp_mask.any():
        print("[WARN] No true positives found in test set.")
        idx = int(np.argmax(probs))   # fallback: highest-confidence prediction
    else:
        idx = int(np.argmax(tp_mask))

    # Try to load SHAP values if pre-computed
    shap_path = ladder_dir / f"shap_{args.model}_h{args.horizon}.npy"
    shap_vals = np.load(shap_path)[idx] if shap_path.exists() else None

    trace_single_prediction(
        model=None,
        X_node=X_test[idx],
        feature_names=feature_names,
        node_str=f"test_node_{idx}",
        t=pd.Timestamp("2023-03-10", tz="UTC"),   # USDC_SVB start
        true_label=int(y_test[idx]),
        predicted_prob=float(probs[idx]),
        shap_values=shap_vals,
        out_path=out / f"case_study_{args.model}_h{args.horizon}",
    )


if __name__ == "__main__":
    main()
