"""
Interpretability + hub export — the depth layer and the hand-off to the ABM.

Produces:
  results/interpret/xgb_importance_h{H}.csv        microstructure feature importance
  results/interpret/gnn_node_importance_{ep}.csv   per-node GNN saliency
  exports/hub_ranking_v1_{ep}.{csv,json}           hub ranking (ABM contract)
  exports/spurious_hub_{ep}.json                    named spurious-hub candidate

Hub score reuses scgnn.hub.ranking (structural vs full variants, non-circular
propagator labels).  The spurious hub = high structural centrality but
propagator_label == 0 (central / correlated, yet did NOT causally propagate) —
this is exactly the node the ABM's counterfactual is designed to expose.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.data.dataset import load_episode  # noqa: E402
from scgnn.data.registry import NodeID, NodeRegistry  # noqa: E402
from scgnn.hub.ranking import (compute_hub_scores, compute_propagator_labels,  # noqa: E402
                               save_hub_ranking)
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402


def xgb_importance(horizon: int) -> pd.DataFrame:
    p = Path(f"results/ladder/xgb_model_h{horizon}.pkl")
    if not p.exists():
        print("[warn] xgb model not found; run eval/run_benchmark.py first")
        return pd.DataFrame()
    model, feat_names = pickle.load(open(p, "rb"))
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    # map f0.. to feature names
    rows = []
    for k, v in score.items():
        idx = int(k[1:]) if k.startswith("f") else None
        name = feat_names[idx] if idx is not None and idx < len(feat_names) else k
        rows.append({"feature": name, "gain": v})
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    # also aggregate by base feature (strip _lag*)
    df["base"] = df["feature"].str.replace(r"_lag\d+", "", regex=True)
    agg = df.groupby("base")["gain"].sum().sort_values(ascending=False)
    Path("results/interpret").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"results/interpret/xgb_importance_h{horizon}.csv", index=False)
    agg.to_csv(f"results/interpret/xgb_importance_by_base_h{horizon}.csv")
    print("Top microstructure features (by aggregated gain):")
    print(agg.head(8).to_string())
    return df


def gnn_node_saliency(trainer: GNNContagionTrainer, b: dict, horizon: int) -> dict:
    """
    Occlusion-based node influence (forward-only; no autograd -> no GAT CPU segfault).

    For each node k we zero its features (toward the $1 peg / neutral state) across all
    snapshots and measure the drop in total predicted contagion probability over the OTHER
    evaluated nodes.  A large drop means k drives its neighbours' predicted stress via
    message passing — exactly the "propagation influence" a contagion hub should have.
    """
    import torch
    from scgnn.data.dataset import to_pyg_snapshots
    model = trainer.model
    model.eval()
    node_strs = b["node_strs"]
    infl = {ns: 0.0 for ns in node_strs}
    snaps = list(to_pyg_snapshots(b, horizon))
    with torch.no_grad():
        for si, d in snaps:
            if d.eval_mask.sum() == 0:
                continue
            base = torch.sigmoid(model(d.x, d.edge_index, d.edge_attr).squeeze(-1))
            base_tot = float(base[d.eval_mask].sum())
            for k in range(len(node_strs)):
                xk = d.x.clone()
                xk[k] = 0.0
                # neutralise this node only; measure effect on the OTHER eval nodes
                other = d.eval_mask.clone()
                other[k] = False
                if other.sum() == 0:
                    continue
                occ = torch.sigmoid(model(xk, d.edge_index, d.edge_attr).squeeze(-1))
                drop = float(base[other].sum() - occ[other].sum())
                infl[node_strs[k]] += abs(drop)
    return infl


def aggregate_graph(b: dict) -> nx.DiGraph:
    """Episode-level directed contagion graph: union of all hourly snapshots, edge
    weight = mean |correlation| over the snapshots where the edge appeared.  A single
    snapshot is too sparse for betweenness; the aggregate captures persistent conduits."""
    node_strs = b["node_strs"]
    acc = {}  # (u,v) -> [sum_w, count]
    for snap in b["snapshots"]:
        ei = snap["edge_index"]; ea = snap["edge_attr"]
        for k in range(ei.shape[1]):
            u, v = node_strs[ei[0, k]], node_strs[ei[1, k]]
            w = abs(float(ea[k, 0])) if ea.shape[0] > k else 1.0
            s, c = acc.get((u, v), (0.0, 0))
            acc[(u, v)] = (s + w, c + 1)
    G = nx.DiGraph()
    for ns in node_strs:
        G.add_node(ns)
    n_snap = max(len(b["snapshots"]), 1)
    for (u, v), (s, c) in acc.items():
        G.add_edge(u, v, weight=s / c, persistence=c / n_snap)
    return G


def peak_snapshot(b: dict) -> int:
    """Snapshot with the largest cross-node peg stress (sum |dev| at active nodes)."""
    # approximate using hourly price_ratio feature (index 0) deviation
    X = b["X"]; active = b["active"]
    dev = np.abs(X[:, :, 0] - 1.0) * active
    return int(dev.sum(axis=1).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--episode", default="USDC_SVB")
    ap.add_argument("--horizon", type=int, default=1440)  # graph signal lives at 24h
    ap.add_argument("--kind", default="gat")              # GAT is the pre-registered winner
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    thr_map = cfg["labels"]["thresholds_bps"]
    sustained = cfg["labels"]["sustained_min"]

    # 1) XGBoost feature importance
    xgb_importance(args.horizon)

    # 2) train GNN on all-but-episode, explain on the episode
    from scgnn.data.dataset import list_episodes
    eps = list_episodes()
    train_names = [e for e in eps if e != args.episode]
    tr = GNNContagionTrainer(kind=args.kind, horizon=args.horizon, seed=args.seed,
                             epochs=100, patience=12).fit(train_names)
    b = load_episode(args.episode)
    sal = gnn_node_saliency(tr, b, args.horizon)
    Path("results/interpret").mkdir(parents=True, exist_ok=True)
    pd.Series(sal, name="gnn_saliency").sort_values(ascending=False).to_csv(
        f"results/interpret/gnn_node_importance_{args.episode}.csv")

    # 3) propagator labels from 1-min deviations (non-circular)
    idx = pd.to_datetime(b["index_1m_ms"], unit="ms", utc=True)
    peg_dev = {ns: pd.Series(b["dev_bps_1m"][ns], index=idx) for ns in b["active_node_strs"]}
    thr = {ns: float(thr_map[{"USDC": "fiat_backed", "USDT": "fiat_backed", "TUSD": "fiat_backed",
                              "BUSD": "fiat_backed", "USDP": "fiat_backed", "FDUSD": "fiat_backed",
                              "DAI": "crypto_backed", "FRAX": "crypto_backed",
                              "UST": "synthetic"}.get(ns.split("/")[0], "fiat_backed")])
           for ns in b["active_node_strs"]}
    prop = compute_propagator_labels(peg_dev, b["origin"], thr, sustained)

    # 4) hub ranking on the peak-stress snapshot
    registry = NodeRegistry([NodeID.from_str(s) for s in b["node_strs"]])
    G = aggregate_graph(b)
    hub_df = compute_hub_scores(registry, sal, G, prop)
    hub_df["episode_tag"] = args.episode
    hub_df["is_real"] = True
    # schema parity: ensure 'rank' column the ABM expects
    hub_df["rank"] = hub_df["rank_full"]
    for c in ["ci_lo", "ci_hi", "ci_std"]:
        if c not in hub_df.columns:
            hub_df[c] = hub_df["hub_score"]
    save_hub_ranking(hub_df, args.episode)

    # 5) spurious-hub candidate: central but did NOT propagate
    cand = hub_df[(hub_df["betweenness"] > hub_df["betweenness"].median())
                  & (hub_df["propagator_label"] == 0)]
    if len(cand):
        spurious = cand.sort_values("betweenness", ascending=False).iloc[0]
        out = {"episode": args.episode, "spurious_hub": spurious["node"],
               "betweenness": float(spurious["betweenness"]),
               "betweenness_rank": int(hub_df["betweenness"].rank(ascending=False)[spurious.name]),
               "propagator_label": 0,
               "gnn_saliency": float(spurious["gnn_mask_sum"]),
               "rationale": ("High structural centrality / correlation but did not enter "
                             "new stress within 24h of the origin onset — a correlational hub "
                             "that the ABM counterfactual should reveal as non-causal.")}
    else:
        out = {"episode": args.episode, "spurious_hub": None,
               "rationale": "No central non-propagator found; all central nodes propagated."}
    Path(f"exports/spurious_hub_{args.episode}.json").write_text(json.dumps(out, indent=2))
    print("\nSpurious-hub candidate:", json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
