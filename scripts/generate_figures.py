"""
Generate paper figures from results/ + exports/ artefacts (real-data pipeline).

  fig1_leadtime_decay.png   PR-AUC vs horizon — the lead-time story (graph wins at 24h)
  fig2_loeo_bars.png        leave-one-cluster-out PR-AUC by model @ h1440
  fig3_feature_importance.png  XGBoost gain by base microstructure feature
  fig4_hub_ranking.png      GNN influence vs betweenness, propagators vs spurious (BUSD)
  fig5_depeg_paths.png      real 1-min peg paths during USDC/SVB (data realism)

Usage: python scripts/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style as ps
ps.apply()

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = [30, 60, 240, 1440]
HLAB = ["30m", "1h", "4h", "24h"]
COL = ps.LADDER


def fig_leadtime():
    """Two-panel lead-time: ROC-AUC (base-rate-free) and PR-AUC lift over base rate.
    Absolute PR-AUC rises mechanically with the base rate, so these honest metrics are
    used instead of raw PR-AUC."""
    roc, lift = {}, {}
    for h in HORIZONS:
        p = Path(f"results/ladder/pooled_results_h{h}.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        base = float(df["positive_rate"].iloc[0])
        for m in df.index:
            roc.setdefault(m, []).append((h, df.loc[m, "roc_auc"]))
            lift.setdefault(m, []).append((h, df.loc[m, "pr_auc"] - base))
    fig, axes = plt.subplots(1, 2, figsize=ps.WIDE)
    for ax, data, ylab, ttl, ref in [
        (axes[0], roc, "ROC-AUC (held-out SVB)", "Ranking skill by horizon", 0.5),
        (axes[1], lift, "PR-AUC lift over base rate", "Precision lift by horizon", 0.0)]:
        for m in ["xgboost", "logreg", "gru", "graphsage", "gat"]:
            if m not in data:
                continue
            xs, ys = zip(*data[m])
            lw = 2.6 if m in ("gat", "graphsage") else 1.4
            ax.plot(xs, ys, marker="o", color=COL.get(m, "k"), lw=lw,
                    label=m.upper() if m in ("gat", "gru") else m.capitalize(),
                    markersize=7 if m in ("gat", "graphsage") else 5)
        ax.axhline(ref, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_xscale("log"); ax.set_xticks(HORIZONS); ax.set_xticklabels(HLAB)
        ax.set_xlabel("Contagion prediction horizon"); ax.set_ylabel(ylab)
        ax.set_title(ttl)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig1_leadtime_decay.png"); plt.close(fig)
    print("fig1 ok")


def fig_loeo():
    p = Path("results/eval/loeo_h1440.csv")
    if not p.exists():
        return
    df = pd.read_csv(p, index_col=0)
    df = df[~df.index.isin(["MEAN"])]
    models = [m for m in ["xgboost", "logreg", "gru", "graphsage", "gat"] if m in df.columns]
    folds = df.index.tolist()
    x = np.arange(len(folds)); w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=ps.WIDE)
    for i, m in enumerate(models):
        ax.bar(x + i * w, df[m].values, w, label=m.upper() if m == "gat" else m, color=COL.get(m, "k"))
    if "pos_rate" in df.columns:
        ax.plot(x + 0.4, df["pos_rate"].values, "k--", marker="x", label="base rate")
    ax.set_xticks(x + 0.4); ax.set_xticklabels(folds, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC"); ax.set_title("Leave-one-cluster-out PR-AUC at 24 hours")
    ax.legend(fontsize=8, ncol=3); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig2_loeo_bars.png"); plt.close(fig)
    print("fig2 ok")


def fig_features():
    p = Path("results/interpret/xgb_importance_by_base_h1440.csv")
    if not p.exists():
        return
    s = pd.read_csv(p, index_col=0).iloc[:, 0].sort_values()
    fig, ax = plt.subplots(figsize=ps.SINGLE)
    ax.barh(s.index, s.values, color=ps.BLUE)
    ax.set_xlabel("XGBoost gain (summed over lags)")
    ax.set_title("Microstructure precursors of contagion onset")
    fig.tight_layout(); fig.savefig(OUT / "fig3_feature_importance.png"); plt.close(fig)
    print("fig3 ok")


def fig_hub():
    p = Path("exports/hub_ranking_v1_USDC_SVB.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["gnn_mask_sum"] > 0].copy()
    spurious = json.loads(Path("exports/spurious_hub_USDC_SVB.json").read_text()).get("spurious_hub")
    origin = "USDC/binance"  # the SVB episode origin (excluded from propagator labels)
    fig, ax = plt.subplots(figsize=ps.TALL)
    for _, r in df.iterrows():
        if r["node"] == origin:
            c, mark = ps.BLUE, "*"
        elif r["node"] == spurious:
            c, mark = ps.RED, "X"
        elif r["propagator_label"] == 1:
            c, mark = ps.GREEN, "o"
        else:
            c, mark = ps.GREY, "s"
        ax.scatter(r["betweenness"], r["gnn_mask_sum"], s=240 if mark == "*" else 160,
                   color=c, marker=mark, zorder=3)
        ax.annotate(r["node"].split("/")[0], (r["betweenness"], r["gnn_mask_sum"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Betweenness centrality (structural)")
    ax.set_ylabel("GNN occlusion influence (predictive)")
    ax.set_title("Hub map: BUSD is central and influential but does not propagate")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor=ps.BLUE, label="origin (USDC)", markersize=14),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ps.GREEN, label="true propagator", markersize=10),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ps.GREY, label="non-propagator", markersize=10),
        Line2D([0], [0], marker="X", color="w", markerfacecolor=ps.RED, label="spurious hub", markersize=11),
    ], fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "fig4_hub_ranking.png"); plt.close(fig)
    print("fig4 ok")


def fig_depeg():
    from scgnn.data.dataset import load_episode
    b = load_episode("USDC_SVB")
    idx = pd.to_datetime(b["index_1m_ms"], unit="ms", utc=True)
    fig, ax = plt.subplots(figsize=ps.WIDE)
    # Muted navy/grey family for the non-origin pegs; the origin is the one maroon accent.
    _muted = ["#26425a", "#5b7fa6", "#9a9a9a", "#45617a", "#7a8a99", "#8c7d77"]
    _ci = 0
    for ns in b["active_node_strs"]:
        price = b["dev_bps_1m"][ns] / 1e4 + 1.0
        if ns == b["origin"]:
            col, lw = ps.RED, 2.2
        else:
            col, lw = _muted[_ci % len(_muted)], 1.1
            _ci += 1
        ax.plot(idx, price, lw=lw, color=col,
                label=ns.split("/")[0] + (" (origin)" if ns == b["origin"] else ""))
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_ylabel("price (USD)"); ax.set_title("USDC/SVB one-minute peg paths")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(OUT / "fig5_depeg_paths.png"); plt.close(fig)
    print("fig5 ok")


if __name__ == "__main__":
    fig_leadtime(); fig_loeo(); fig_features(); fig_hub(); fig_depeg()
    print("All figures ->", OUT)
