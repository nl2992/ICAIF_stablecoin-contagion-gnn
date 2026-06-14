"""Label-shuffle null: confirm PR-AUC 0.447 is not a class-imbalance artifact.

Randomly permutes training labels (100 seeds) and trains XGBoost on each shuffled
dataset. Evaluates on the real SVB test labels. Under the null, all PR-AUC values
should cluster at the base rate (0.29). The observed GAT PR-AUC 0.447 is then
compared to the null distribution as a one-sided empirical p-value.

Output: results/eval/label_shuffle_null.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from run_loco import get_episode_set  # noqa: E402
from scgnn.data.dataset import load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

N_PERMS = 100
HZN = 1440
# Known GAT headline PR-AUC from multiseed_summary_h1440.json headline_SVB
GAT_PR_AUC = 0.4465


def main() -> None:
    feat = load_feature_names()
    episodes, cmap = get_episode_set("all_7")
    held = [e for e in episodes if cmap.get(e) == "SVB_2023"]
    train = [e for e in episodes if cmap.get(e) != "SVB_2023"]

    Xtr, ytr, _ = tabular_from_episodes(train, HZN, feat)
    Xte, yte, _ = tabular_from_episodes(held, HZN, feat)
    ytr = np.asarray(ytr).ravel()
    yte = np.asarray(yte).ravel()
    base_rate = float(yte.mean())
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    null_pr_aucs: list[float] = []
    rng = np.random.default_rng(0)
    for i in range(N_PERMS):
        y_shuf = rng.permutation(ytr)
        xgb = make_xgboost(scale_pos_weight=spw)
        xgb.fit(Xtr, y_shuf, verbose=False)
        p = xgb.predict_proba(Xte)[:, 1]
        null_pr_aucs.append(float(average_precision_score(yte, p)))
        if (i + 1) % 20 == 0:
            print(f"  perm {i+1}/{N_PERMS}: mean null PR-AUC so far = {np.mean(null_pr_aucs):.4f}")

    null_arr = np.array(null_pr_aucs)
    emp_p = float((null_arr >= GAT_PR_AUC).mean())
    result = {
        "n_perms": N_PERMS,
        "horizon_min": HZN,
        "base_rate": round(base_rate, 4),
        "null_pr_auc_mean": round(float(null_arr.mean()), 4),
        "null_pr_auc_std": round(float(null_arr.std()), 4),
        "null_pr_auc_min": round(float(null_arr.min()), 4),
        "null_pr_auc_max": round(float(null_arr.max()), 4),
        "gat_pr_auc": GAT_PR_AUC,
        "empirical_p_value": emp_p,
        "note": (
            "XGBoost trained on randomly permuted training labels, evaluated on real test labels. "
            "Under H0 (no signal), PR-AUC should equal the base rate. "
            "GAT headline PR-AUC from multiseed_summary_h1440.json (5-seed mean, SVB cluster)."
        ),
    }
    out = _ROOT / "results" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "label_shuffle_null.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
