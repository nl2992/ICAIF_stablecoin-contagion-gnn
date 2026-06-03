"""
Hub export entry point — produces the Repo 2 bridge artifacts.

Reads results/hub/ (GNNExplainer masks, per-seed hub scores) and
results/ladder/ (propagator labels from test-set predictions), then:
  1. Builds hub rankings for each real episode separately
  2. Builds an aggregated synthetic ranking
  3. Writes exports/hub_ranking_v1_{episode}.csv + .json
  4. Writes exports/calibration_v1_{episode}.csv
  5. Writes exports/schema_v1.json

Usage:
    python scripts/export_hubs.py --config configs/experiment.yaml [--stub]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.utils.seeds import set_all_seeds
from scgnn.data.registry import NodeRegistry
from scgnn.data.windows import load_episodes, episodes_by_split
from scgnn.hub.ranking import compute_hub_scores, add_confidence_intervals, save_hub_ranking
from scgnn.train.ensemble import compute_hub_stability
from scgnn.export.schema import write_schema_doc, validate_hub_ranking
from scgnn.export.calibration import save_calibration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out_dir", default="exports")
    return p.parse_args()


def _stub_hub_scores(registry: NodeRegistry, seed: int) -> dict:
    """Generate synthetic hub scores for stub/CI mode."""
    rng = np.random.default_rng(seed)
    return {str(n): float(rng.random()) for n in registry}


def _stub_graph(registry: NodeRegistry) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = registry.node_strs()
    for n in nodes:
        G.add_node(n)
    # Connect first node (USDC/binance) as hub
    for n in nodes[1:3]:
        G.add_edge(nodes[0], n, weight=0.8, correlation=0.8)
    return G


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_all_seeds(cfg["seed"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = Path("data/processed")
    hub_dir = Path("results/hub")

    registry_path = processed / "node_registry.json"
    if not registry_path.exists():
        print("[ERROR] Run make dataset first.")
        sys.exit(1)

    registry = NodeRegistry.load(registry_path)
    episodes = load_episodes(Path(cfg["episodes_file"]))
    split = episodes_by_split(episodes)

    # Write schema doc first
    write_schema_doc(out_dir)

    # ── Per-episode hub rankings ──────────────────────────────────────────
    for ep in episodes:
        print(f"\nProcessing hub ranking for: {ep.name}")

        if args.stub:
            # Stub mode: synthetic hub scores across 5 seeds
            seeds_scores = [_stub_hub_scores(registry, s) for s in [42, 137, 271, 503, 789]]
            gnn_masks = {str(n): np.mean([ss.get(str(n), 0.0) for ss in seeds_scores]) for n in registry}
            G = _stub_graph(registry)
            propagators = {str(n): int(gnn_masks[str(n)] > 0.5) for n in registry}
        else:
            # Real mode: load pre-computed GNNExplainer masks and propagator labels
            mask_path = hub_dir / f"gnn_masks_{ep.name}.json"
            prop_path = hub_dir / f"propagators_{ep.name}.json"
            graph_path = hub_dir / f"graph_{ep.name}.pkl"

            if not mask_path.exists():
                print(f"  [WARN] No GNN masks for {ep.name} — skipping (run train/run_gnn.py first)")
                continue

            with open(mask_path) as f:
                gnn_masks = json.load(f)
            with open(prop_path) as f:
                propagators = json.load(f)
            import pickle
            with open(graph_path, "rb") as f:
                G = pickle.load(f)

            seeds_scores = []
            for seed_file in sorted(hub_dir.glob(f"hub_scores_{ep.name}_seed*.json")):
                with open(seed_file) as f:
                    seeds_scores.append(json.load(f))

        df = compute_hub_scores(registry, gnn_masks, G, propagators)

        if len(seeds_scores) >= 2:
            df = add_confidence_intervals(df, seeds_scores)
            stability = compute_hub_stability(seeds_scores)
        else:
            stability = None

        df["episode_tag"] = ep.name
        df["is_real"] = True

        # Validate schema
        missing = validate_hub_ranking(df)
        if missing:
            print(f"  [WARN] Missing schema columns for {ep.name}: {missing}")

        save_hub_ranking(df, ep.name, out_dir, stability=stability)

        # Print top-5
        print(f"  Top-5 hubs for {ep.name}:")
        print(df.head(5)[["node", "hub_score", "rank"]].to_string(index=False))

    # ── Aggregated all-real ranking ───────────────────────────────────────
    all_csvs = list(out_dir.glob("hub_ranking_v1_*.csv"))
    real_csvs = [f for f in all_csvs if "synthetic" not in f.name and "all_real" not in f.name]
    if real_csvs:
        all_real = pd.concat([pd.read_csv(f) for f in real_csvs])
        agg = all_real.groupby("node").agg(
            hub_score=("hub_score", "mean"),
            betweenness=("betweenness", "mean"),
            gnn_mask_sum=("gnn_mask_sum", "mean"),
            propagator_label=("propagator_label", "max"),
            asset=("asset", "first"),
            venue=("venue", "first"),
            fee_tier=("fee_tier", "first"),
        ).reset_index()
        agg = agg.sort_values("hub_score", ascending=False).reset_index(drop=True)
        agg["rank"] = agg.index + 1
        agg["episode_tag"] = "all_real"
        agg["is_real"] = True
        save_hub_ranking(agg, "all_real", out_dir)
        print(f"\nAggregated all-real hub ranking saved.")

    print(f"\nAll exports written to {out_dir}/")


if __name__ == "__main__":
    main()
