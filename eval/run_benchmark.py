"""
Unified model-ladder benchmark — the headline experiment.

Runs the full ladder on IDENTICAL samples (one per active, non-origin (snapshot,node)):
    majority -> persistence -> logreg -> xgboost -> gru(sequence) -> graphsage -> gat

Two protocols:
  - pooled : train/val/test per episodes.yaml split (all horizons)
  - loeo   : leave-one-episode-out over every built episode (primary horizon)

Writes:
  results/ladder/pooled_results_h{H}.csv         models x metrics, per horizon
  results/ladder/probs_{model}_h{H}.npy          test-set probabilities (pooled)
  results/ladder/test_labels_h{H}.npy            aligned test labels
  results/eval/loeo_h{H}.csv                      per-fold PR-AUC for every model
  results/eval/loeo_verdict_h{H}.json            pre-registered GNN-vs-XGB verdict

Usage:
  python eval/run_benchmark.py --horizon 60
  python eval/run_benchmark.py --all-horizons
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

from scgnn.data.dataset import (  # noqa: E402
    episode_split_map, list_episodes, load_feature_names, tabular_from_episodes)
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_logreg, make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402
from scgnn.utils.seeds import set_all_seeds  # noqa: E402

TABULAR = ["logreg", "xgboost"]
GNN = ["graphsage", "gat"]

# Episode CLUSTERS — same-period episodes must never split across train/test.
# (FRAX_SVB depegged *because* USDC did, same March-2023 window -> co-leakage;
#  USDT_May2022 was the Terra-collapse contagion to USDT, same May-2022 window.)
CLUSTERS = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}


def cluster_of(ep: str) -> str:
    return CLUSTERS.get(ep, ep)


# --------------------------------------------------------------- model fns
def _tabular_probs(kind, Xtr, ytr, Xval, yval, Xte, spw):
    if kind == "logreg":
        m = make_logreg(); m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    if kind == "xgboost":
        m = make_xgboost(scale_pos_weight=spw)
        es = [(Xval, yval)] if len(yval) else None
        m.fit(Xtr, ytr, eval_set=es, verbose=False)
        return m.predict_proba(Xte)[:, 1], m
    raise ValueError(kind)


def _gru_probs(feat_names, train_names, val_names, test_names, horizon, seed=42):
    """Per-(episode,node) GRU over snapshot sequences. Returns probs in test row order."""
    import torch
    import torch.nn as nn
    set_all_seeds(seed)
    F = len(feat_names)

    def seqs(names):
        Xtr, ytr, meta = tabular_from_episodes(names, horizon, feat_names)
        groups = []
        if len(meta) == 0:
            return groups
        meta = meta.reset_index(drop=True)
        for (ep, node), idx in meta.groupby(["episode", "node"]).groups.items():
            idx = sorted(idx, key=lambda i: meta.loc[i, "snapshot"])
            groups.append((Xtr[idx], ytr[idx], list(idx)))
        return groups

    class GRUClf(nn.Module):
        def __init__(self, f, h=48):
            super().__init__()
            self.gru = nn.GRU(f, h, batch_first=True)
            self.head = nn.Linear(h, 1)

        def forward(self, x):
            o, _ = self.gru(x)
            return self.head(o).squeeze(-1)

    model = GRUClf(F)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_groups = seqs(train_names)
    pos = sum(float(y.sum()) for _, y, _ in train_groups)
    tot = sum(len(y) for _, y, _ in train_groups)
    pw = torch.tensor([(tot - pos) / max(pos, 1.0)])
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(40):
        model.train()
        np.random.shuffle(train_groups)
        for Xg, yg, _ in train_groups:
            if len(yg) < 2:
                continue
            opt.zero_grad()
            xb = torch.tensor(Xg, dtype=torch.float32).unsqueeze(0)
            logit = model(xb).squeeze(0)
            loss = lossfn(logit, torch.tensor(yg, dtype=torch.float32))
            loss.backward(); opt.step()
    # predict test in tabular row order
    Xte, yte, meta = tabular_from_episodes(test_names, horizon, feat_names)
    probs = np.zeros(len(yte))
    model.eval()
    meta = meta.reset_index(drop=True)
    with torch.no_grad():
        for (ep, node), idx in meta.groupby(["episode", "node"]).groups.items():
            idx_sorted = sorted(idx, key=lambda i: meta.loc[i, "snapshot"])
            xb = torch.tensor(Xte[idx_sorted], dtype=torch.float32).unsqueeze(0)
            p = torch.sigmoid(model(xb).squeeze(0)).numpy()
            for k, i in enumerate(idx_sorted):
                probs[i] = p[k]
    return probs


def run_partition(feat_names, train_names, val_names, test_names, horizon, seed=42):
    """Return {model: probs_on_test}, y_test, and fitted xgb (for importance)."""
    set_all_seeds(seed)
    Xtr, ytr, _ = tabular_from_episodes(train_names, horizon, feat_names)
    Xval, yval, _ = tabular_from_episodes(val_names, horizon, feat_names) if val_names else (np.empty((0, len(feat_names))), np.empty(0), None)
    Xte, yte, _ = tabular_from_episodes(test_names, horizon, feat_names)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    out = {}
    # baselines
    out["majority"] = np.full(len(yte), ytr.mean())
    out["persistence"] = np.abs(Xte[:, 0] - 1.0)  # |price_ratio-1| proxy
    if out["persistence"].max() > 0:
        out["persistence"] = out["persistence"] / out["persistence"].max()
    # tabular
    out["logreg"] = _tabular_probs("logreg", Xtr, ytr, Xval, yval, Xte, spw)
    xgb_probs, xgb_model = _tabular_probs("xgboost", Xtr, ytr, Xval, yval, Xte, spw)
    out["xgboost"] = xgb_probs
    # sequence
    try:
        out["gru"] = _gru_probs(feat_names, train_names, val_names, test_names, horizon, seed)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] gru failed: {e}")
    # GNN
    for kind in GNN:
        try:
            tr = GNNContagionTrainer(kind=kind, horizon=horizon, seed=seed,
                                     epochs=80, patience=10)
            tr.fit(train_names, val_names if val_names else None)
            out[kind] = tr.predict_episodes(test_names)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {kind} failed: {e}")
    return out, yte, xgb_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--all-horizons", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    feat_names = load_feature_names()
    eps = list_episodes()
    clusters = sorted(set(cluster_of(e) for e in eps))
    horizons = cfg["labels"]["horizons_min"] if args.all_horizons else [args.horizon]
    Path("results/ladder").mkdir(parents=True, exist_ok=True)
    Path("results/eval").mkdir(parents=True, exist_ok=True)

    # Headline held-out test = SVB cluster (the marquee crisis); train = all other
    # clusters; val = FTX cluster.  This is leakage-safe (no same-window episode in
    # both train and test).
    test = [e for e in eps if cluster_of(e) == "SVB_2023"]
    val = [e for e in eps if cluster_of(e) == "FTX_2022"]
    train = [e for e in eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")]
    print(f"clusters={clusters}\nheadline: train={train} val={val} test={test}")

    metrics = ["pr_auc", "roc_auc", "weighted_f1", "positive_rate", "n"]
    for h in horizons:
        print(f"\n=== HEADLINE (held-out SVB) h={h} ===")
        probs, yte, xgb_model = run_partition(feat_names, train, val, test, h, args.seed)
        rows = {}
        for model, p in probs.items():
            rep = full_report(yte, p)
            rows[model] = {k: rep[k] for k in metrics}
            np.save(f"results/ladder/probs_{model}_h{h}.npy", p)
        np.save(f"results/ladder/test_labels_h{h}.npy", yte)
        tbl = pd.DataFrame(rows).T
        tbl.to_csv(f"results/ladder/pooled_results_h{h}.csv")
        print(tbl.round(4).to_string())
        if xgb_model is not None and h == args.horizon:
            import pickle
            with open("results/ladder/xgb_model_h%d.pkl" % h, "wb") as f:
                pickle.dump((xgb_model, feat_names), f)

    # ---- leave-one-CLUSTER-out, primary horizon ----
    h = args.horizon
    print(f"\n=== LEAVE-ONE-CLUSTER-OUT h={h} ===")
    loeo_pr, loeo_roc = [], []
    for held_cluster in clusters:
        held = [e for e in eps if cluster_of(e) == held_cluster]
        tr = [e for e in eps if cluster_of(e) != held_cluster]
        probs, yte, _ = run_partition(feat_names, tr, [], held, h, args.seed)
        rpr = {"held_cluster": held_cluster, "episodes": "+".join(held),
               "n_test": len(yte), "pos_rate": round(float(yte.mean()), 4)}
        rroc = dict(rpr)
        for model, p in probs.items():
            rep = full_report(yte, p)
            rpr[model] = rep["pr_auc"]; rroc[model] = rep["roc_auc"]
        loeo_pr.append(rpr); loeo_roc.append(rroc)
        print("  PR ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rpr.items()})
    loeo = pd.DataFrame(loeo_pr).set_index("held_cluster")
    loeo.loc["MEAN"] = loeo.select_dtypes("number").mean()
    loeo.to_csv(f"results/eval/loeo_h{h}.csv")
    pd.DataFrame(loeo_roc).set_index("held_cluster").to_csv(f"results/eval/loeo_roc_h{h}.csv")
    print(loeo.round(3).to_string())

    # ---- pre-registered verdict (PR-AUC, GNN vs XGB across folds with positives) ----
    folds = loeo.drop(index="MEAN")
    valid = folds[folds["pos_rate"] > 0]
    verdict = {"protocol": "leave-one-cluster-out", "primary_metric": "pr_auc"}
    for gnn in GNN:
        if gnn in valid and "xgboost" in valid:
            margin = (valid[gnn] - valid["xgboost"]).dropna()
            verdict[gnn] = {
                "mean_margin_vs_xgb": round(float(margin.mean()), 4),
                "folds_win_ge_0.05": int((margin >= 0.05).sum()),
                "n_folds": int(len(margin)),
                "PASS_preregistered": bool(margin.mean() >= 0.05 and (margin >= 0.05).sum() >= max(1, int(0.7 * len(margin)))),
            }
    verdict["interpretation"] = (
        "If PASS=false, execute the pre-registered honest-null branch: report tabular as "
        "competitive and use the graph as a structural lens (hub ranking) validated causally "
        "by the ABM.")
    Path(f"results/eval/loeo_verdict_h{h}.json").write_text(json.dumps(verdict, indent=2))
    print("\nVERDICT:", json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
