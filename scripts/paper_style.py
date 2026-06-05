"""Shared figure style: one palette, one set of sizes, one look across all paper figures.

Import and call ``apply()`` at the top of any figure script::

    import paper_style as ps
    ps.apply()

Colours are semantic and identical across both papers so the two PDFs read as one body
of work: green = causal / origin / good, red = spurious hub (BUSD) / attention model,
blue = primary baseline, orange = temporal model, grey = no skill.
"""
from __future__ import annotations

import matplotlib as mpl

# --- semantic palette -------------------------------------------------------
BLUE = "#2166ac"     # primary baseline / neutral series
LBLUE = "#67a9cf"    # secondary baseline
GREEN = "#1b7837"    # causal / origin / good outcome
RED = "#b2182b"      # spurious hub (BUSD) / attention model
SALMON = "#d6604d"   # graph-sage / weaker graph model
ORANGE = "#e08214"   # temporal (GRU) / accent
GREY = "#9aa0a6"     # no-skill / non-propagator
LGREY = "#cccccc"    # trivial baseline
INK = "#222222"

SEQ_CMAP = "YlOrRd"  # single sequential map (loss / heat)
DIV_CMAP = "RdYlGn"  # single diverging map (skill: green = good)

# model ladder, coherent light-to-dark ramp ending on the winning attention model
LADDER = {
    "majority": GREY, "persistence": LGREY, "logreg": LBLUE,
    "xgboost": BLUE, "gru": ORANGE, "graphsage": SALMON, "gat": RED,
}

# --- standard figure sizes (inches) -----------------------------------------
SINGLE = (5.2, 3.5)   # a \columnwidth figure
TALL = (5.2, 4.3)     # a \columnwidth scatter / map
WIDE = (9.4, 3.7)     # a figure* spanning the text width


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0,
        "legend.frameon": False, "figure.autolayout": False,
    })
