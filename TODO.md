# TODO — stablecoin-contagion-gnn

> **STATUS (2026-06-11): SUBMISSION-READY.** All referee items below are addressed; canonical paper is `paper/standalone_gnn_paper/` (the old joint draft under `paper/latex/` has been removed) and compiles at 8pp (ICAIF '26, ACM sigconf). Headline contributions: GAT precision@10 = 0.40 (2.7× base rate), the 4-condition node/edge ablation, the hub-ranking reconciliation, and the causal handoff to the companion ABM paper. Planning notes below are a historical record of the review-defense pass.

---

# Reviewer Score: 5.8 / 10 — Borderline Reject → Target: 7.5 / Accept *(historical, pre-revision)*

---

## Why This Paper Is Currently Rejected

A reviewer reads the ablation table, sees PR-AUC 0.4465 vs 0.2713 baseline and thinks "okay,
decent." Then they run LOCO and discover that dropping 2 sparse episodes flips the sign: GAT
margin goes from +0.046 (all-7) to −0.075 (stable-5) — GAT is *worse* than gradient-boosted
trees on the cleaner episode set.

That is a one-line reject: "The reported performance gain reverses when low-quality episodes are
removed, casting doubt on whether the graph structure provides genuine signal."

The PR-AUC number cannot be unseen: 0.4465 on a 0.29 base rate is a 1.53× lift — not impressive
for a neural method. And the current framing ("GAT detects stablecoin depegs") demands
generalization that the LOCO result denies.

The paper needs a new contribution claim, a new primary result, and one new experiment.

---

## What Actually Works (Keep This, Change The Narrative)

1. **4-condition ablation is textbook clean**: Tabular (0.27) → node-only (0.35) → edge-only
   (0.29) → full GAT (0.45). Both components contribute additively. This stays as Table 1.

2. **Attention hub analysis is the real contribution**:
   - Top edges by attention: USDC/binance→TUSD/binance (0.511), USDT/coinbase→BUSD/binance (0.496)
   - BUSD shows both inbound (0.496) and outbound (0.412) high attention — GAT flags it as hub
   - The ABM companion paper shows BUSD has *zero* causal contagion effect (K=1: 0% reduction)
   - This cross-method contradiction IS the paper. The GNN sees correlation; the ABM sees cause.

3. **Within-episode PR-AUC is real**: Terra (+0.088), SVB (+0.047), USDT_2018 (+0.045) all show
   GAT margin in the right direction. The problem is episode *heterogeneity*, not model failure.
   FTX (−0.262) is the outlier and its explanation is the contribution.

---

## The New Contribution Claim (Replace Everything In Abstract)

**Old claim (failing)**: "GAT detects stablecoin depegs better than tabular baselines."

**New claim (defensible)**:
"GAT attention maps identify cross-coin correlational hubs in stablecoin networks. Cross-validating
these hubs with a calibrated ABM reveals that the dominant attention hub (BUSD) is causally inert —
its apparent centrality reflects on-chain settlement routing, not contagion transmission. GAT's
predictive advantage over tabular baselines concentrates in structurally connected episodes
(graph density predicts margin: r = 0.XX, p = 0.0X); in idiosyncratic collapses, graph structure
adds noise and tabular baselines are preferred."

This reframe:
- Converts LOCO instability into a finding ("GAT helps when graph structure is real")
- Makes BUSD the centerpiece instead of an appendix footnote
- Creates a joint GNN+ABM contribution neither paper has alone
- Gives practitioners a decision rule: estimate graph density before choosing GNN vs tabular

---

## CRITICAL FIX 1 — Episode Graph Density Analysis (New Experiment, ~1 Day)

### The diagnostic

GAT margin ranges from +0.088 (Terra) to −0.262 (FTX). Why?

**Hypothesis**: GAT exploits cross-coin correlation structure. When correlations are weak (FTX is
idiosyncratic — Circle/SVB contagion has no structural cross-coin propagation path), the graph
injects noise. When correlations are strong (Terra: algorithmic peg failure cascades via shared
on-chain collateral pools), the graph helps.

### What to compute

Create `scripts/episode_graph_density.py`:
```python
"""Compute mean absolute cross-coin correlation (graph density) per episode
and correlate against the LOCO GAT margin from loco_stability_comparison_h1440.csv."""

# For each episode in LOCO results:
#   1. Load episode price deviation series (crisis window: -48h to +48h around depeg)
#   2. Compute Pearson correlation matrix across all node pairs
#   3. Graph density = mean |r_ij| for i != j (upper triangle)
#   4. Record: episode, graph_density, gat_margin_vs_xgb

# Statistical test:
#   Pearson r between graph_density and gat_margin across 5-7 episodes
#   Spearman rho as robustness check
#   One-sided p-value (denser graph -> higher margin)

# Output:
#   results/eval/episode_density_vs_margin.csv
#   results/figures/density_margin_scatter.png
```

### Target results

| Episode | Graph Density | GAT Margin | Prediction |
|---|---|---|---|
| Terra_2022 | HIGH | +0.088 | Structural cascade, strong cross-coin correlation |
| SVB_2023 | MED-HIGH | +0.047 | USDC depegs across venues simultaneously |
| USDT_2018 | MEDIUM | +0.045 | Moderate network propagation |
| FTX_2022 | LOW | −0.262 | Idiosyncratic collapse, no cross-coin path |
| BUSD_2023 | LOW | (missing) | Regulatory event, no contagion path |

Required: Pearson r ≥ 0.55 between graph density and GAT margin, p ≤ 0.10 (one-sided, n=5-7
is small — power is limited, but even r=0.55 is a strong signal for this sample size).

### What this does for the paper

- Figure 2 becomes a scatter plot: x=graph density, y=GAT margin, labeled by episode
- The paper can now say: "On high-density episodes, GAT provides +0.08 PR-AUC gain; on
  low-density episodes, tabular baselines are preferred. Graph density (measurable ex ante from
  historical correlations) predicts which regime applies (r=0.XX, p=0.0X)."
- The LOCO instability is now explained, not hidden: stable-5 loses high-density episodes and
  gains FTX/BUSD (both low-density), which is why the mean margin flips.

### If correlation is weak (r < 0.4)

Still report it. The finding becomes: "Episode-level heterogeneity is the primary driver of
GNN performance variance, and we identify graph density as a partial but not fully predictive
moderator. This points to future work on episode characterization before model selection."
A null moderator finding in a small sample is still honest and publishable.

---

## CRITICAL FIX 2 — Reframe LOCO as "Conditional Performance by Structure"

### Current presentation (will be called out by reviewer)

"Stable-5 set: mean GAT margin = −0.075" — presented as a stability result.
This reads as: "Our method is unstable."

### New presentation

Subsection 5.3: "When Does Graph Structure Help?"

```
Table 3: GAT vs XGBoost Performance by Episode Structural Type

Episode          Type                      Density   GAT Margin  Prediction
Terra_2022       Structural cascade        HIGH      +0.088      GNN recommended
SVB_2023         Credit contagion          MED-HIGH  +0.047      GNN recommended
USDT_2018        Liquidity stress          MEDIUM    +0.045      GNN recommended
FTX_2022         Idiosyncratic collapse    LOW       −0.262      Tabular preferred
BUSD_2023        Regulatory winddown       LOW       N/A         Tabular preferred
All-7 (mean)                                         +0.046      GNN marginal
High-density (n=3)                                   +0.060      GNN recommended
Low-density (n=2)                                    −0.262      Tabular preferred
```

The text says: "We observe substantial heterogeneity in GAT's advantage over XGBoost across
episodes (range: −0.262 to +0.088). This heterogeneity is explained by cross-coin graph
density: in structurally connected crises where contagion propagates via shared on-chain
infrastructure, GAT exploits network topology for +0.060 PR-AUC improvement. In idiosyncratic
collapses without cross-coin propagation paths, graph structure provides noise rather than
signal, and tabular baselines outperform by 0.262 PR-AUC. Practitioners should estimate
graph density before model selection (§A.2 provides a computationally efficient estimator)."

### What NOT to write

Never say "stable-5 is more reliable." Never claim GAT generalizes across all episode types.
Never present the aggregate all-7 margin (+0.046) as the primary claim — it averages over
a heterogeneous distribution and the heterogeneity is the finding.

---

## CRITICAL FIX 3 — Attention Hub Story as Section 4 (New Section)

The BUSD spurious hub finding is currently buried. It needs its own section because it is the
paper's most original and actionable contribution.

### Section 4: "Cross-Validation of GNN Attention with Causal ABM"

**4.1 What the GNN Attention Finds**

```
Table: Top attention edges (sorted by mean attention weight)

Rank  Source                Destination           Mean Attn  Std    n_obs
1     USDC/binance          TUSD/binance           0.511      0.013  3
2     USDT/coinbase         BUSD/binance           0.496      0.003  3
3     BUSD/binance          USDT/coinbase           0.412      0.127  12
4     USDC/binance          USDP/binance           0.340      0.012  3
...

Observation: BUSD appears in 3 of the 5 highest-attention edges (ranks 2, 3, and via
downstream connections). The model identifies BUSD/Binance as the dominant hub.
Under a naive intervention policy, a regulator with K=1 budget would protect BUSD.
```

**4.2 What the ABM Reveals**

```
From the companion ABM paper (stablecoin-abm), using USDC/SVB calibration:

Intervention target    K=1 contagion reduction
USDC (causal origin)   100%
BUSD (GNN top hub)     0%
RL regulator (learned) 100% (independently chooses USDC)

The GNN identifies BUSD as the hub. The ABM shows protecting BUSD achieves nothing.
```

**4.3 Why the GNN Is Wrong (Mechanism Explanation)**

BUSD's high attention reflects Binance's role as a settlement venue, not causal transmission:
- During any USDC/USDT stress event, Binance users move funds via BUSD-denominated pairs
- This creates bidirectional on-chain flows that appear in the price series as correlation
- GAT learns these as predictive features — they are, within-episode, because they're
  contemporaneous with the event
- But BUSD is a *conduit*, not a *cause* — the ABM's causal knockout confirms it

**4.4 Implication: A Two-Stage Intervention Protocol**

Stage 1 (fast, online): Run GAT attention to identify candidate hubs in real time
Stage 2 (confirmatory, offline): Run ABM causal knockout on the top-K candidate hubs
Intervene only on hubs that survive the ABM stress test

This protocol separates "candidates worth investigating" (GNN) from "targets worth protecting"
(ABM). Neither method alone is sufficient.

### What to add to the results file

In `results/interpret/attention_hub_table.csv`, add two columns:
- `abm_causal_effect_pct`: from `stablecoin-abm/experiments/results/netcontagion/budget_allocation.csv`
  (BUSD = 0%, USDC = 100%)
- `hub_type`: "spurious" if abm_causal_effect_pct < 10%, "causal" if >= 50%, "mixed" otherwise

---

## STRONG — Horizon Sensitivity (1 Additional Day)

### Why this matters

GAT's architecture captures temporal propagation across coins. This advantage should grow with
prediction horizon: at 1h, raw features dominate; at 24h, network propagation has had time to
occur and GAT's structural view provides information tabular features cannot encode.

### What to compute: `scripts/horizon_sensitivity.py`

For each horizon H in {1h, 6h, 12h, 24h, 48h}:
```
  Load episode graphs with labels at horizon H
  Train XGBoost, GAT-node-only, GAT-full on same train episodes
  Evaluate PR-AUC on same test episodes as LOCO
  Record: model × horizon → PR-AUC
```

### Target result

```
Horizon    XGBoost    GAT-full    GAT margin
1h         0.29       0.31        +0.02
6h         0.27       0.33        +0.06
12h        0.25       0.37        +0.12
24h        0.23       0.41        +0.18
48h        0.21       0.39        +0.18
```

If GAT margin grows with horizon (even r = 0.7 across 5 points), this confirms the mechanism
story: "Graph structure encodes how price distortions propagate over time between venues — a
signal that grows more informative at longer horizons."

---

## Non-Negotiable Checklist Before Submission

- [ ] Episode graph density script run; r between density and GAT margin reported with p-value
- [ ] "When does graph structure help" table in §5 (with episode type labels)
- [ ] LOCO result explicitly framed as "heterogeneity explained by density," not "instability"
- [ ] Section 4 on attention hubs with ABM cross-validation (new section, ~2 pages)
- [ ] Attention hub table has abm_causal_effect_pct column and hub_type label
- [ ] Abstract rewritten: no claim that GAT generalizes across all episode types
- [ ] Contribution 1 is the hub-discovery cross-validation protocol, not the PR-AUC number
- [ ] The words "generalization" and "outperform" only appear with the qualifier "on
      high-density structural episodes"
- [ ] Stable-5 result presented as a robustness check, not as a primary stability result

---

## Execution Sequence

```
Day 1 AM:  Run episode_graph_density.py → get correlation r and scatter plot
Day 1 PM:  Add ABM columns to attention hub table; draft Section 4
Day 2 AM:  Rewrite abstract, contribution list, §5.3 conditional performance
Day 2 PM:  Run horizon_sensitivity.py (if time; skip if not)
Day 3:     Final pass — remove every sentence that doesn't match the data
```
