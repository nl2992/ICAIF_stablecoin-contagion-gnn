"""
Generate hub centrality report: which stablecoins/pools are contagion hubs?

Usage:
    python interpret/hub_report.py --snapshot data/processed/graph_snapshot.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scgnn.interpret.explainability import hub_centrality_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", default="data/processed/graph_snapshot.pkl")
    p.add_argument("--labels", default="data/processed/node_labels.pkl")
    p.add_argument("--out_dir", default="results/interpret")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.snapshot, "rb") as f:
        G = pickle.load(f)
    with open(args.labels, "rb") as f:
        labels = pickle.load(f)

    df = hub_centrality_report(G, labels, top_n=20)
    csv_path = out / "hub_report.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string())
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
