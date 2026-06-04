"""Additional explanatory figures for the expanded paper."""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_ROOT / "src"))
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)


def fig_network_snapshot():
    """The contagion graph at peak SVB stress: nodes coloured by depeg, directed lead-lag edges."""
    b = pickle.load(open("data/processed/graphs/USDC_SVB.pkl", "rb"))
    nodes = b["node_strs"]; X = b["X"]; active = b["active"]
    dev = np.abs(X[:, :, 0] - 1.0) * active
    si = int(dev.sum(axis=1).argmax())                       # peak-stress snapshot
    snap = b["snapshots"][si]; ei = snap["edge_index"]; ea = snap["edge_attr"]
    # aggregate edges across all snapshots for a readable, persistent graph
    acc = {}
    for s in b["snapshots"]:
        e = s["edge_index"]
        for k in range(e.shape[1]):
            acc[(e[0, k], e[1, k])] = acc.get((e[0, k], e[1, k]), 0) + 1
    G = nx.DiGraph()
    asset = lambda n: nodes[n].split("/")[0]
    act_idx = [j for j in range(len(nodes)) if active[:, j].any()]
    for j in act_idx:
        peak = float(np.abs(X[:, j, 0] - 1.0).max())
        G.add_node(asset(j), depeg=peak)
    nsnap = len(b["snapshots"])
    for (u, v), c in acc.items():
        if c / nsnap > 0.05 and u in act_idx and v in act_idx:
            G.add_edge(asset(u), asset(v), w=c / nsnap)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pos = nx.circular_layout(G)
    depg = [G.nodes[n]["depeg"] for n in G.nodes]
    nc = nx.draw_networkx_nodes(G, pos, node_size=1700, node_color=depg, cmap="OrRd",
                                vmin=0, vmax=max(depg) if depg else 1, edgecolors="k", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    if G.edges:
        ws = [G[u][v]["w"] for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, width=[1 + 3 * w for w in ws], edge_color="#555",
                               arrowsize=16, arrowstyle="-|>",
                               connectionstyle="arc3,rad=0.08", ax=ax, node_size=1700)
    plt.colorbar(nc, ax=ax, label="peak |depeg| during episode", fraction=0.046)
    ax.set_title("Stablecoin contagion network during USDC/SVB\n"
                 "(node colour = how far it depegged; arrows = directed lead-lag)")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig6_network_snapshot.png", dpi=200); plt.close(fig)
    print("fig6 ok")


def fig_multiseed_ci():
    """Headline GAT-vs-baselines on held-out SVB @24h, mean +/- std over 5 seeds."""
    df = pd.read_csv("results/eval/multiseed_h1440.csv")
    sub = df[df["setup"] == "headline_SVB"]
    models = [m for m in ["xgboost", "graphsage", "gat"] if m in df.columns]
    means = [sub[m].mean() for m in models]; stds = [sub[m].std() for m in models]
    base = sub["base_rate"].mean()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"xgboost": "#2166ac", "graphsage": "#d6604d", "gat": "#b2182b"}
    ax.bar(range(len(models)), means, yerr=stds, capsize=5,
           color=[colors[m] for m in models], edgecolor="k", linewidth=0.5)
    ax.axhline(base, ls="--", color="k", label=f"base rate ({base:.2f})")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m.upper() if m == "gat" else m.capitalize() for m in models])
    ax.set_ylabel("PR-AUC (held-out SVB, 24h)")
    ax.set_title("Day-ahead contagion prediction: graph attention wins\n(mean ± std over 5 seeds)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig7_multiseed_ci.png", dpi=200); plt.close(fig)
    print("fig7 ok")


def fig_roc_heatmap():
    """ROC-AUC for every model x horizon (base-rate-free skill map)."""
    rows = {}
    for h in [30, 60, 240, 1440]:
        p = Path(f"results/ladder/pooled_results_h{h}.csv")
        if p.exists():
            d = pd.read_csv(p, index_col=0); rows[h] = d["roc_auc"]
    M = pd.DataFrame(rows)
    order = [m for m in ["majority","persistence","logreg","xgboost","gru","graphsage","gat"] if m in M.index]
    M = M.loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    im = ax.imshow(M.values, cmap="RdYlGn", vmin=0.35, vmax=0.7, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(["30m","1h","4h","24h"])
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels([m.upper() if m in ("gru","gat") else m.capitalize() for m in M.index])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="ROC-AUC", fraction=0.046)
    ax.set_title("Ranking skill (ROC-AUC) by model × horizon\nskill emerges only for graph models at 24h")
    fig.tight_layout(); fig.savefig(OUT / "fig8_roc_heatmap.png", dpi=200); plt.close(fig)
    print("fig8 ok")


if __name__ == "__main__":
    fig_network_snapshot(); fig_multiseed_ci(); fig_roc_heatmap()
    print("extra GNN figures ->", OUT)
