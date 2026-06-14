"""Multi-seed precision@k on the held-out SVB cluster (5-seed version of run_precision_at_k.py).

The paper argues seed noise matters for the headline PR-AUC (a single-seed run understated GAT at 0.29
vs the 5-seed 0.447), yet the operational precision@k table was reported from a single seed. This closes
that gap: it recomputes GAT precision@k over the SAME five seeds (0-4) used for the multi-seed PR-AUC and
reports mean +/- std. XGBoost is deterministic (one fit). Pipeline is otherwise identical to
run_precision_at_k.py.

Output: results/eval/precision_at_k_svb_multiseed.json
"""
from __future__ import annotations
import json, sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
from run_loco import get_episode_set
from scgnn.data.dataset import tabular_from_episodes, load_feature_names
from scgnn.models.classical import make_xgboost
from scgnn.models.gnn_trainer import GNNContagionTrainer
from scgnn.utils.seeds import set_all_seeds

HZN = 1440
SEEDS = [0, 1, 2, 3, 4]   # same seeds as the multi-seed PR-AUC (multiseed_h1440.csv)


def precision_at_k(y, p, ks):
    order = np.argsort(-p)
    return {k: float(y[order[:min(k, len(order))]].mean()) for k in ks}


def main():
    feat = load_feature_names()
    episodes, cmap = get_episode_set("all_7")
    held = [e for e in episodes if cmap.get(e) == "SVB_2023"]
    train = [e for e in episodes if cmap.get(e) != "SVB_2023"]
    Xtr, ytr, _ = tabular_from_episodes(train, HZN, feat)
    Xte, yte, _ = tabular_from_episodes(held, HZN, feat)
    yte = np.asarray(yte).ravel()
    base = float(yte.mean()); n_pos = int(yte.sum())
    ks = [10, 25, 50, n_pos]

    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    xgb = make_xgboost(scale_pos_weight=spw); xgb.fit(Xtr, ytr, verbose=False)
    p_xgb = xgb.predict_proba(Xte)[:, 1]
    xgb_pk = precision_at_k(yte, p_xgb, ks)

    per_seed = {k: [] for k in ks}
    for s in SEEDS:
        set_all_seeds(s)
        tr = GNNContagionTrainer(kind="gat", horizon=HZN, seed=s, epochs=80, patience=10)
        tr.fit(train, None)
        p = np.asarray(tr.predict_episodes(held)).ravel()
        pk = precision_at_k(yte, p, ks)
        for k in ks:
            per_seed[k].append(pk[k])
        print(f"seed {s}: " + " ".join(f"p@{k}={pk[k]:.3f}" for k in ks))

    gat = {str(k): {"mean": round(float(np.mean(per_seed[k])), 4),
                    "std": round(float(np.std(per_seed[k])), 4),
                    "per_seed": [round(v, 4) for v in per_seed[k]]} for k in ks}
    res = {"n_test": int(len(yte)), "base_rate": round(base, 4), "n_positives": n_pos,
           "seeds": SEEDS, "ks": ks, "gat_precision_at_k_multiseed": gat,
           "xgboost_precision_at_k": {str(k): round(xgb_pk[k], 4) for k in ks},
           "note": "GAT precision@k over 5 seeds (0-4, same as multiseed PR-AUC); XGBoost deterministic."}
    out = _ROOT / "results/eval"; out.mkdir(parents=True, exist_ok=True)
    (out / "precision_at_k_svb_multiseed.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(gat, indent=2))


if __name__ == "__main__":
    main()
