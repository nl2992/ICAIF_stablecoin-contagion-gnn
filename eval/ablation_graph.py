"""
Ablation: does the GRAPH structure add anything over per-node features?

Three rungs on identical samples, held-out SVB cluster @ h=1440, averaged over seeds:
  - node-only tabular   : XGBoost (no graph at all)
  - GNN, edges removed  : GraphSAGE/GAT with empty edge_index -> per-node MLP (no msg passing)
  - GNN, real edges     : GraphSAGE/GAT with the temporal directed graph

The (real-edges - edges-removed) gap is the marginal contribution of the graph, holding the
GNN architecture fixed. Writes results/eval/ablation_graph.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402

H = 1440
SEEDS = [0, 1, 2, 3, 4]


def main():
    feat = load_feature_names()
    eps = list_episodes()
    train = [e for e in eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")]
    val = [e for e in eps if cluster_of(e) == "FTX_2022"]
    test = [e for e in eps if cluster_of(e) == "SVB_2023"]

    Xtr, ytr, _ = tabular_from_episodes(train, H, feat)
    Xte, yte, _ = tabular_from_episodes(test, H, feat)
    base = float(yte.mean())

    rows = []
    # node-only tabular (deterministic)
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = make_xgboost(scale_pos_weight=spw); m.fit(Xtr, ytr, verbose=False)
    pr = full_report(yte, m.predict_proba(Xte)[:, 1])["pr_auc"]
    rows.append({"rung": "node-only (XGBoost)", "edges": "n/a", "pr_auc_mean": round(pr, 4),
                 "pr_auc_std": 0.0})

    for kind in ["graphsage", "gat"]:
        for ablate in [True, False]:
            scores = []
            for s in SEEDS:
                tr = GNNContagionTrainer(kind=kind, horizon=H, seed=s, epochs=80,
                                         patience=10, ablate_edges=ablate)
                tr.fit(train, val)
                p = tr.predict_episodes(test)
                scores.append(full_report(yte, p)["pr_auc"])
            rows.append({"rung": f"{kind} ({'no edges' if ablate else 'real edges'})",
                         "edges": "removed" if ablate else "real",
                         "pr_auc_mean": round(float(np.mean(scores)), 4),
                         "pr_auc_std": round(float(np.std(scores)), 4)})
            print(rows[-1])

    df = pd.DataFrame(rows)
    df["base_rate"] = round(base, 4)
    df.to_csv("results/eval/ablation_graph.csv", index=False)
    print("\n=== ABLATION (held-out SVB @24h) ===")
    print(df.to_string(index=False))
    for kind in ["graphsage", "gat"]:
        try:
            real = df[df["rung"] == f"{kind} (real edges)"]["pr_auc_mean"].iloc[0]
            noedge = df[df["rung"] == f"{kind} (no edges)"]["pr_auc_mean"].iloc[0]
            print(f"GRAPH contribution ({kind}): {real - noedge:+.4f} PR-AUC (real - no-edges)")
        except IndexError:
            pass


if __name__ == "__main__":
    main()
