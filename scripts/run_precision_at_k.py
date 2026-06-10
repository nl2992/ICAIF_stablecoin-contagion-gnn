"""Precision@k under an alert budget on the held-out SVB cluster.

Reviewer point: PR-AUC 0.447 at a 0.293 base rate is ~1.5x lift, and a risk desk
consumes a ranked alert list, not an AUC. This reports precision@k (the form a
desk uses) for the GAT vs the XGBoost baseline vs the base rate, on the held-out
SVB cluster at 24h, reusing the exact LOCO training pipeline.

Output: results/eval/precision_at_k_svb.json
"""
from __future__ import annotations
import json, sys, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src")); sys.path.insert(0, str(_ROOT / "scripts"))
from run_loco import get_episode_set, CLUSTERS_ALL7
from scgnn.data.dataset import tabular_from_episodes, load_feature_names
from scgnn.models.classical import make_xgboost
from scgnn.models.gnn_trainer import GNNContagionTrainer
from scgnn.utils.seeds import set_all_seeds

HZN = 1440; SEED = 42


def precision_at_k(y, p, ks):
    order = np.argsort(-p)
    out = {}
    for k in ks:
        kk = min(k, len(order))
        out[k] = round(float(y[order[:kk]].mean()), 4) if kk else float("nan")
    return out


def main():
    feat = load_feature_names()
    episodes, cmap = get_episode_set("all_7")
    held = [e for e in episodes if cmap.get(e) == "SVB_2023"]
    train = [e for e in episodes if cmap.get(e) != "SVB_2023"]
    Xtr, ytr, _ = tabular_from_episodes(train, HZN, feat)
    Xte, yte, _ = tabular_from_episodes(held, HZN, feat)
    yte = np.asarray(yte).ravel()
    base = float(yte.mean()); n_pos = int(yte.sum())

    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    xgb = make_xgboost(scale_pos_weight=spw); xgb.fit(Xtr, ytr, verbose=False)
    p_xgb = xgb.predict_proba(Xte)[:, 1]

    set_all_seeds(SEED)
    tr = GNNContagionTrainer(kind="gat", horizon=HZN, seed=SEED, epochs=80, patience=10)
    tr.fit(train, None); p_gat = np.asarray(tr.predict_episodes(held)).ravel()

    ks = [10, 25, 50, n_pos]
    res = {
        "n_test": int(len(yte)), "base_rate": round(base, 4), "n_positives": n_pos,
        "gat_precision_at_k": precision_at_k(yte, p_gat, ks),
        "xgboost_precision_at_k": precision_at_k(yte, p_xgb, ks),
        "ks": ks,
        "note": "Held-out SVB cluster, 24h. precision@k = fraction of the top-k highest-scored "
                "(node,snapshot) alerts that are true contagion onsets; k=n_positives is the oracle budget.",
    }
    out = _ROOT / "results/eval"; out.mkdir(parents=True, exist_ok=True)
    (out / "precision_at_k_svb.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
