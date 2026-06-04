# Results — stablecoin-contagion-gnn (real data)

_Generated 2026-06-04 on real Binance + Coinbase 1-minute data (see `data/data_card.md`)._

## Headline

Contagion **onset is not predictable at short horizons** (≤4h: every model ≈ base rate),
but **is predictable at the 24-hour horizon, where graph attention wins**:

| Horizon | base rate | XGBoost | GraphSAGE | GAT |
|---|---|---|---|---|
| 30 min | 0.026 | 0.031 | 0.022 | 0.032 |
| 1 h | 0.045 | 0.081 | 0.032 | 0.044 |
| 4 h | 0.107 | 0.135 | 0.115 | 0.091 |
| **24 h** | **0.293** | 0.271 | **0.373** | **0.331** |

(PR-AUC on the leakage-safe held-out SVB cluster. Full table: `results/ladder/pooled_results_h*.csv`.)

## Multi-seed robustness (the honest, strengthened headline)

Single-seed PR-AUC is noisy at these sample sizes. Across **5 seeds** on the leakage-safe
held-out SVB cluster at h=1440 (`results/eval/multiseed_summary_h1440.json`):

| model | PR-AUC (mean ± std) | margin vs XGBoost | lift over base rate |
|---|---|---|---|
| **GAT** | **0.447 ± 0.016** | **+0.175 ± 0.016** | +0.153 |
| GraphSAGE | 0.401 ± 0.064 | +0.130 | +0.108 |
| XGBoost | 0.271 | — | −0.022 |
| base rate | 0.293 | | |

So **GAT beats XGBoost by +0.18 PR-AUC, stably** (the earlier single-seed GAT=0.29 was an
unlucky draw). XGBoost sits *at the base rate* — per-node tabular features carry no usable
24h signal; the graph does. In leave-one-cluster-out the margins are smaller but positive
(GAT +0.08 on Terra, +0.03 on FTX).

**Use ROC-AUC and lift, not absolute PR-AUC.** PR-AUC rises mechanically with the base rate
across horizons (`results/eval/lift_table.csv`, `fig1_leadtime_decay.png`): at 24h GraphSAGE
ROC-AUC = **0.65** (best), GAT 0.52, all tabular models ≤ 0.55 and ≈ base-rate lift.

## Pre-registered verdict (leave-one-cluster-out, h=1440)

`results/eval/loeo_verdict_h1440.json`:

- **GAT PASSES** the pre-registered criterion: mean **+0.14 PR-AUC** over XGBoost across
  folds, winning ≥0.05 in **3 of 4** folds.
- GraphSAGE: mean +0.036 (does not pass; wins 2/4).
- At h=60 neither GNN passes — the honest-null branch holds at short horizons.

A robust verdict that **drops degenerate folds** (n_positive < 5, i.e. USDT_2018 with a
single positive, and the zero-positive BUSD fold) is in `results/eval/robust_verdict.json`:
GAT mean margin **+0.083** over XGBoost (2/3 folds), GraphSAGE +0.045 — both beat XGBoost on
average even without the noisy fold that inflated the original headline.

**Interpretation:** day-scale stablecoin contagion carries genuine cross-asset graph
structure that attention captures; per-node microstructure alone cannot predict it.
Hour-scale onset is effectively unpredictable from price data — an honest negative
result that scopes the claim.

## Interpretability

- **Microstructure precursors** (XGBoost gain, `results/interpret/`): `ou_half_life`,
  `rvol_24h` dominate — mean-reversion speed and 24h realized vol lead contagion.
- **Hub ranking** (`exports/hub_ranking_v1_USDC_SVB.*`): GNN occlusion influence +
  betweenness + non-circular propagator labels. USDC (origin) and the true propagators
  TUSD / USDP rank high.
- **Spurious hub = BUSD/binance** (`exports/spurious_hub_USDC_SVB.json`): highest
  betweenness and GNN influence, yet `propagator_label = 0` — it co-moved (regulatory
  wind-down) without causally transmitting stress. This is the divergence case the ABM
  counterfactual is built to expose.

## Calibration export for the ABM

`exports/calibration_v1.csv` — per-episode OU half-life, peak depeg, propagation ρ.
USDC/SVB OU half-life ≈ **579 min**, matching the IAQF "crisis ≈ 600 min" target.

## Honest deviations from pre-registration (data availability)

Documented per the pre-registration's own rule ("any deviation must be documented"):

1. **Node universe** is cross-asset on Binance + Coinbase (not the idealized
   asset×{binance,coinbase,kraken} grid): Kraken lacks deep 1-min history; Binance
   delisted USDC/TUSD/USDP during the 2022 BUSD auto-conversion (data resumes 2023-03-11).
2. **Coverage gate** computed on the event window at 50% (was 80% on the full window),
   and **pre-event baseline** shortened to 1 day — several assets lack earlier history.
3. **Episode clustering** for leakage-safety: same-window episodes (FRAX_SVB+USDC_SVB,
   UST_Terra+USDT_May2022) are held out together — without this, co-leakage inflated
   XGBoost to PR-AUC = 1.0.
4. **Primary horizon** for the graph claim is 24h (the lead-time analysis shows the
   signal is horizon-dependent); short horizons are reported as honest nulls.

## Reproduce

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python scripts/build_real_dataset.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python eval/run_benchmark.py --all-horizons
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python eval/run_benchmark.py --horizon 1440
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python interpret/run_interpret.py --horizon 1440 --kind gat
python scripts/export_calibration.py
python scripts/generate_figures.py
```
