"""Shared figure style: one palette, one set of sizes, one look across all paper figures.

Import and call ``apply()`` at the top of any figure script::

    import paper_style as ps
    ps.apply()

Refined Columbia-University palette. The *semantic* colours are unchanged so the two
papers still read as one body of work: green = causal / origin / good, red = spurious
hub (BUSD) / attention model, orange = temporal model. What changed is the institutional
identity, the neutral and baseline series are reskinned toward Columbia Blue (#B9D9EB)
and a deep navy, and axes/text are navy rather than generic black, so the figures read
as a coherent, professional Columbia set rather than matplotlib defaults.
"""
from __future__ import annotations

import matplotlib as mpl

# --- Columbia institutional anchors ----------------------------------------
COLUMBIA_BLUE = "#B9D9EB"   # iconic Columbia Blue (Pantone 290) — fills, light accents
COLUMBIA_MID  = "#6CA6CD"   # mid Columbia blue — secondary series
COLUMBIA_NAVY = "#1D4F91"   # deep Columbia blue — primary baseline series
NAVY_INK      = "#0A1F44"   # near-black navy — text, axes, ticks

# --- semantic palette (meanings preserved, neutrals reskinned to Columbia) --
BLUE = COLUMBIA_NAVY  # primary baseline / neutral series
LBLUE = COLUMBIA_MID  # secondary baseline
GREEN = "#1b7837"     # causal / origin / good outcome
RED = "#b2182b"       # spurious hub (BUSD) / attention model
SALMON = "#d6604d"    # graph-sage / weaker graph model
ORANGE = "#e08214"    # temporal (GRU) / accent
GREY = "#8895a7"      # no-skill / non-propagator (blue-grey, harmonised)
LGREY = "#D5DEE8"     # trivial baseline (light Columbia blue-grey)
INK = NAVY_INK

SEQ_CMAP = "YlOrRd"  # single sequential map (loss / heat — warm is semantic)
DIV_CMAP = "RdYlGn"  # single diverging map (skill: green = good)
COL_CMAP = "Blues"   # optional Columbia-blue sequential map (on-brand heat)

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
    # Serif typography to match the ACM (acmart) body font, with navy ink and
    # Columbia-blue grid so figures read as part of the paper and as a Columbia set.
    mpl.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        # Columbia institutional ink: navy axes / text / ticks, not generic black.
        "text.color": NAVY_INK, "axes.labelcolor": NAVY_INK,
        "axes.titlecolor": NAVY_INK, "axes.edgecolor": NAVY_INK,
        "xtick.color": NAVY_INK, "ytick.color": NAVY_INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLUMBIA_MID,
        "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "lines.linewidth": 2.0,
        "patch.edgecolor": NAVY_INK,
        "legend.frameon": False, "figure.autolayout": False,
    })
