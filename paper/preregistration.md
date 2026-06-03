# Pre-registration: stablecoin-contagion-gnn

**Committed before any results are computed.**
Any deviation from these criteria must be documented in the paper,
not silently revised after seeing results.

---

## Success criteria (binary — pre-registered)

### Primary: GNN adds beyond tabular baseline (the "Not enough AI" challenge)
**SUCCESS** if GraphSAGE or GAT achieves PR-AUC ≥ 0.05 above XGBoost on the
LOEO test, averaged across all 7 folds AND observed in at least 5 of 7 folds.

**FAILURE** if the margin is < 0.05 or reverses in >2 folds.
Honest response to failure: report XGBoost as the recommended model;
retain GNN as a structural analysis tool for hub ranking even if it
does not win the prediction task.

### Secondary: Hub stability across seeds
**SUCCESS** if mean Spearman ρ (hub_score across 5 seeds) ≥ 0.70 for
real-episode rankings.

**FAILURE** if ρ < 0.70. Response: report hub_score_structural (centrality
only, no GNN) as the primary hub artifact; relegate full composite to appendix.

### Tertiary: Hub-ranking threshold robustness
**SUCCESS** if Spearman ρ(ranking at 25 bps, ranking at 25±15 bps) ≥ 0.70.

**FAILURE**: thresholds are load-bearing; report sensitivity honestly,
present all three threshold arms in the main table.

---

## Pre-registered narrative framings

Two possible outcomes, one clean story each:

### If GNN > tabular (success):
"Graph structure carries signal beyond individual-node features. USDC/binance
and Curve 3pool appear as the principal contagion conduits — they sit centrally
in the lead-lag graph and their GNNExplainer masks dominate.  The ABM should
assign elevated propagation probability to these nodes."

### If GNN ≤ tabular (honest null):
"The tabular baseline (XGBoost) is competitive with the GNN on individual-node
features alone, suggesting that cross-node graph structure does not add
predictive signal beyond microstructure signals when conditioned per-node.
We retain the graph framework for its structural interpretability (hub ranking),
not for prediction. The hub ranking uses betweenness centrality (structural-only
variant) as the primary artifact sent to the ABM."

---

## LOEO success criterion (per-fold)

Report the full per-fold table. The aggregate mean is informative but
individual-fold results are the headline: a model that succeeds on 6/7 folds
but fails on the one "out-of-distribution" episode type is a clearly different
result from one that succeeds on all 7.

**Pre-registered failure threshold**: if PR-AUC on the test fold is BELOW
the majority-class baseline on more than 2 of 7 folds, the model does not
generalize and the paper must frame results in terms of the folds where it does.

---

## Synthetic augmentation contingency

Run `synthetic_stress_test()` before using synthetics.

| Gap (within-synth vs real PR-AUC) | Action |
|---|---|
| < 0.05 | Synthetics behave like real data → use with equal weight |
| 0.05–0.20 | Moderate difference → down-weight synthetics by 0.5 in class weighting |
| > 0.20 | Synthetics too easy → real-only as PRIMARY reported result; synthetic augmentation reported separately as "sensitivity" |

In all cases: **report real-only PR-AUC as a first-class metric**, not buried
in the appendix.

---

## Stamp

This file was committed on: 2025-06-03
It must not be modified after any model has been run.
