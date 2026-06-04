# From Correlation to Causation: Predicting Stablecoin Contagion with Temporal GNNs and Validating the Hubs with a Calibrated Agent-Based Model

*Working draft — ICAIF-style. Spans two repositories: `stablecoin-contagion-gnn` (prediction)
and `stablecoin-abm` (causal validation + policy). All numbers are from the committed,
reproducible runs; see each repo's `RESULTS.md`.*

## Abstract

Graph neural networks are increasingly used to identify the "hubs" that drive financial
contagion, but a hub that is *correlated* with propagation need not *cause* it — the
explainable-AI literature calls this the spurious-correlation problem, and acting on it
misallocates scarce regulatory intervention. We study stablecoin depeg contagion across
seven real crisis episodes (2022–2023) using minute-level Binance and Coinbase data. We
(i) train a leakage-safe temporal GNN that predicts day-ahead cross-asset contagion —
graph attention (GAT) beats every per-node tabular baseline by **+0.18 PR-AUC** on the
held-out SVB cluster, with the directed lead-lag *edges* contributing **+0.10** of that
margin; (ii) build a reduced-form networked-contagion agent-based model, calibrated **4/4**
to the empirical moments, that performs per-node *causal* knockout counterfactuals; and
(iii) join the two. The GNN's top hub during the SVB crisis is **BUSD**, yet its causal
effect on contagion is **exactly zero** (it co-moved during its regulatory wind-down without
transmitting) — a spurious hub the ABM refutes, robust to ±30% calibration uncertainty.
Across crises the correlational ranking is *unreliable*: it matches causal importance in
some episodes and diverges in others, so the causal test is necessary. Finally, a budget-
constrained regulator who protects the GNN's correlational hub achieves **0%** contagion
reduction, while protecting the ABM's causal pick achieves **100%**; a PPO regulator given
only network features independently rediscovers the causal targets and ignores the spurious
hub. Transparency about *causation*, not just prediction, is what makes contagion analytics
safe to act on.

## 1. Introduction

Stablecoins are now systemic: the GENIUS Act (US) and MiCA (EU) both make reserve adequacy
and contagion containment central. A natural analytics pipeline trains a GNN on the
cross-asset network and reports the "hubs" that drive depeg propagation. But GNN hub scores
are *correlational*: betweenness centrality and learned attention reward nodes that *move
with* a crisis, which is not the same as nodes that *cause* it. A regulator who reads a
correlational hub ranking and backstops the top node may spend the entire intervention
budget on a venue that transmits nothing.

We make the prediction→causation gap concrete and then close it. Our contributions:

1. A leakage-safe temporal-GNN contagion predictor on **real** multi-venue data, with an
   honest lead-time finding and an ablation isolating the graph's contribution (§4–5).
2. A reduced-form, **4/4-calibrated** networked-contagion ABM that answers per-node *causal*
   counterfactuals an observational GNN cannot (§6).
3. The **join**: the GNN's top hub is causally inert (spurious) in the marquee crisis;
   correlational rankings are unreliable across crises; the result is robust (§7).
4. The **policy** consequence: correlational hubs misallocate intervention budget, shown by
   an intervention sweep and a PPO regulator that rediscovers the causal targets (§8).

## 2. Related work

*GNNs for on-chain contagion / link prediction* (e.g. ICAIF'25 Uniswap-v3 bridge-swap GNN)
establish temporal directed graphs of pools/venues and predict propagation, but evaluate
prediction, not causal hub importance. *Agent-based models of crypto/LOB markets* (ICAIF'24
liquidity-spoofing; JaxMARL-HFT) note ABMs are hard to calibrate and use them for
mechanism/policy. *Intrinsic interpretability / XAI* (ProtoHedge; post-hoc-explanation
critiques) motivate causal, not just attributive, explanations. We connect these: a GNN
predicts, an ABM causally validates, and the disagreement is the finding.

## 3. Data

Real 1-minute OHLCV: **Binance** (USDT-quoted: USDC, TUSD, USDP, FRAX, BUSD, FDUSD, UST) and
**Coinbase** (USD-quoted: DAI, USDT), USDT/USD peg-adjusted so a USDT depeg is not hidden.
Seven episodes (UST/Terra, FTX/DAI, BUSD wind-down, USDC/SVB, FRAX/SVB, USDT 2018/2022). One
sample per (hourly snapshot, active non-origin node); tabular and graph models see identical
samples. **Leakage control** is load-bearing: FRAX/SVB and USDC/SVB are the same March-2023
window, so naively splitting them inflates XGBoost to PR-AUC = 1.0; we hold same-window
episodes out together as *clusters*. (`data/data_card.md`.)

## 4. Part I — Predicting contagion (GNN)

A model ladder (majority, persistence, logistic, XGBoost, GRU, GraphSAGE, GAT) on identical
samples, leakage-safe held-out SVB cluster and leave-one-cluster-out.

**Lead-time.** Contagion onset is unpredictable at ≤4h (every model ≈ base rate) but
predictable at **24h**. Because PR-AUC rises mechanically with the base rate, we report
ROC-AUC (base-rate-free) and lift: at 24h only the graph models clear ROC-AUC 0.5 and
positive lift (GraphSAGE ROC **0.65**).

**Headline (5-seed).** On the held-out SVB cluster at 24h, **GAT = 0.447 ± 0.016 PR-AUC,
+0.175 ± 0.016 over XGBoost** (which sits at the base rate 0.29). Leave-one-cluster-out
margins are smaller but positive (GAT +0.08 Terra, +0.03 FTX); robust to dropping the
single-positive USDT-2018 fold (GAT +0.083).

**Ablation (the graph's contribution).** Holding architecture fixed and removing all edges:
node-only XGBoost 0.271 → GNN-no-edges 0.35–0.39 (the *architecture*) → GNN-real-edges. The
directed lead-lag **edges add +0.100 PR-AUC for GAT** (0.347→0.447) but only +0.009 for
GraphSAGE — the contagion signal lives in the edges and is what *attention* exploits.

## 5. Interpretation and the hub ranking (the hand-off)

Top microstructure precursors (XGBoost gain): OU half-life and 24h realized vol. The hub
ranking combines GNN occlusion-influence, betweenness, and a non-circular propagator label
computed from raw prices. For USDC/SVB it flags **BUSD** as the #1 hub (highest betweenness
0.071 and GNN influence 12.3) — the input to the causal test. We also export per-episode
empirical moments (OU half-life, peak depeg, propagation ρ) for ABM calibration.

## 6. Part II — Causal validation (ABM)

An AMM-only market is *bimodal* (it resists a shock or collapses), so it cannot smoothly hit
a ~14% depeg and fails the calibration gate. We instead use a **networked-contagion engine**:
per-asset near-critical mean-reverting peg dynamics with a **directed transmission matrix W**
estimated from the real 1-min lead-lag (net-directionalised to separate transmitters from
co-movers). It calibrates **4/4** to USDC/SVB (contagion magnitude 0.1376 → 0.02% error,
crisis half-life 116 → exact, baseline vol and cross-venue ρ within tolerance).

**Causal knockout.** Protecting a node (holding it at peg) and measuring the change in
contagion to the *other* victims (deterministic propagation, fixed measure set) isolates its
causal effect. Protecting the origin USDC removes **100%** of contagion (sanity). **BUSD's
causal Δ = 0** with out-transmission 0 — the GNN's #1 hub is causally inert. Spearman between
GNN predicted importance and ABM causal effect is **−0.77**.

## 7. The join, generalized and stress-tested

**Across crises** (3 contagion-producing episodes): the GNN's top hub *equals* the causal
driver in Terra (1/3) but *diverges* with a spurious hub in SVB (BUSD) and USDT-May-2022
(2/3). Correlational hub rankings are therefore **unreliable** — sometimes right, sometimes
badly wrong — which is precisely why a causal test is required.

**Robustness.** Under ±30% perturbation of (coupling, κ, W) over 60 draws, USDC stays the top
causal node **100%** of the time and BUSD stays inert **100%** — the headline is not a
calibration artifact.

## 8. Part III — Policy: correlational hubs misallocate budget

**Intervention sweep.** Targeted protection of USDC → 100% reduction; reserve-strengthening
USDC ×20 → 89%; circuit breaker (cap 2%) → 85%; redemption gating (coupling→0) → 100%.
Protecting or strengthening **BUSD → 0%** at every intensity. A welfare-by-node matrix shows
protecting BUSD is *identical* to no intervention, while protecting USDC saves all victims.

**The budget result.** A regulator who can backstop one venue and picks by the GNN's
correlational ranking protects BUSD → **0%** contagion reduction; picking by the ABM's causal
ranking protects USDC → **100%**.

**RL regulator.** A PPO agent given only transmission-network features (no causal labels)
learns to allocate its reserve budget USDC 1.0 / DAI 0.48 / **BUSD 0.0**, achieving **93.7%**
reduction — independently rediscovering the causal targets and ignoring the spurious hub.

## 9. Limitations

n = 7 episodes; two are too thin (1–3 nodes, ~0 contagion) to analyse causally. The
transmission matrix W is estimated from 1-min lead-lag, which can over-credit thinly traded
coins as "leaders," so the load-bearing claim is the **BUSD divergence** (predicted #1,
causal 0), not the full ranking. The networked-contagion engine is a reduced form for the
causal analysis; the full AMM agent model remains for microstructure studies. Short-horizon
contagion is genuinely unpredictable here — an honest negative that scopes the claim.

## 10. Conclusion

A GNN can predict day-scale stablecoin contagion, and graph attention adds real signal — but
its hub rankings are correlational and, in the marquee SVB crisis, its top hub (BUSD) causes
*no* contagion. A calibrated ABM supplies the missing causal test, and the disagreement
matters: acting on the correlational hub wastes 100% of an intervention budget that the
causal target would have spent to fully contain the cascade. For reserve-transparency and
intervention design under the GENIUS Act and MiCA, contagion analytics should report
*causation*, validated, not correlation alone.

## Reproducibility

```bash
# GNN (repo: stablecoin-contagion-gnn)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/build_real_dataset.py
... python eval/run_benchmark.py --all-horizons ; --horizon 1440
... python eval/robustness_multiseed.py ; python eval/ablation_graph.py ; python eval/polish_results.py
... python interpret/run_interpret.py --horizon 1440 --kind gat ; python scripts/export_calibration.py

# ABM (repo: stablecoin-abm)
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/run_netcontagion_join.py
... python scripts/run_intervention_sweep.py ; python scripts/run_multi_episode_join.py
... python scripts/run_robustness_welfare.py ; python scripts/run_rl_regulator.py
```
Figures and tables: `*/results/figures/`, `*/experiments/results/netcontagion/`.
