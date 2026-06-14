# Claims and Evidence

What the paper argues, and for every headline number, the exact committed file it comes from. This is
the map to check the results against the code.

## The narrative

When one stablecoin breaks its dollar peg, which of the others follow, and how soon? We treat this as
prediction on a temporal graph: nodes are (asset, venue) pairs, edges are directed lead-lag links read
off minute-level prices, and the structure changes every hour. We run a full model ladder — trivial
baselines, gradient-boosted trees, a GRU, GraphSAGE, and a graph-attention network (GAT) — over seven
real crisis episodes (2018–2023), under a leakage-safe protocol that never lets a model train and test
on the same crisis window.

Three findings carry the paper. First, contagion onset is essentially unpredictable at horizons of a
few hours and becomes predictable only at one day; the graph models are the first to clear chance, and
the GAT attains the highest PR-AUC. Second, a four-condition ablation separates the neural architecture
from the network itself: directed lead-lag edges add +0.10 PR-AUC for the GAT and almost nothing for
GraphSAGE, so the signal lives in attention's ability to weight the few informative neighbours rather
than in extra model capacity. Third, the model exports an interpretable per-crisis hub ranking — but
that ranking is correlational, and we hand it to a companion agent-based model that overturns its
top hub causally.

The contribution is the leakage-safe evaluation protocol and the correlational-to-causal handoff, not a
new architecture. The scope is stated plainly: the cross-crisis advantage is conditional on including
the two sparse episodes (the leave-one-cluster-out mean reverses on the stable-five subset), and seven
episodes on two exchanges cannot characterise all future stress.

## Where each number lives

| Claim | Number | File | Field / row |
|---|---|---|---|
| GAT vs XGBoost, held-out SVB 24h | PR-AUC 0.447±0.016, margin +0.175; XGB 0.271 | `results/eval/multiseed_summary_h1440.json` | `headline_SVB.gat.pr_auc_mean`=0.4465, `margin_vs_xgb_mean`=0.1752, `xgboost_pr_auc_mean`=0.2713 |
| Five seeds, all beat XGBoost (sign-test p=0.031) | {0.432, 0.453, 0.447, 0.432, 0.469} | `results/eval/multiseed_h1440.csv` | `headline_SVB` rows, `gat` column (5/5 > XGB → 0.5^5≈0.031) |
| "Initial single-seed run put GAT at only 0.29" | 0.2875 | `results/ladder/pooled_results_h1440.csv` | GAT row (the single-seed ladder run, below the 5-seed range) |
| Full ladder table | majority/persistence/logreg/XGB/GRU/SAGE/GAT | `results/eval/multiseed_summary_h1440.json`, `results/eval/lift_table.csv` | per-model PR-AUC / ROC-AUC / lift |
| Lead-time: skill appears only at 24h | ROC-AUC and PR-lift by horizon | `results/eval/lead_time_presaved.csv`, `results/eval/lift_table.csv` | 30m/1h/4h/24h rows; GraphSAGE ROC 0.651 @24h; logreg PR-AUC 0.357 @24h |
| Four-condition ablation; edges add +0.100, node feats +0.153 | (1)0.271 (2)0.347 (3)0.293 (4)0.447 | `results/eval/ablation_4condition.csv` | `delta_vs_xgboost`; (4)-(2)=+0.100, (4)-(3)=+0.153 |
| Edges add +0.100 (GAT) vs +0.009 (SAGE) | 0.4465-0.3465 vs 0.4008-0.3916 | `results/eval/ablation_graph.csv` | `gat (real edges)` / `gat (no edges)`; `graphsage (real edges)` / `graphsage (no edges)` |
| On-chain proxy features add only +0.005 | 0.27031-0.26545 | `results/eval/onchain_augmented_h1440.csv` | `xgboost_onchain` − `xgboost_baseline` |
| Degree-preserving rewiring null → base rate | SVB 0.294±0.024 (20 rewirings); FTX 0.164 vs base 0.163 | `results/eval/edge_rewiring_null.json` | per-fold rewired vs real/base rate |
| LOCO (5-seed) | FTX 0.150/+0.029, Terra 0.217/+0.082, SVB 0.447/+0.175, mean +0.095 | `results/eval/multiseed_summary_h1440.json` | `loeo_FTX` / `loeo_Terra` / `headline_SVB` |
| FRAX/SVB fold (co-located, shown separately) | XGB 0.850, GAT 0.426, margin −0.424 | `results/eval/loco_frax_standalone_h1440.csv` | the single row |
| Episode-selection sensitivity (all-7 vs stable-5) | all-7 +0.046, stable-5 −0.075 | `results/eval/loco_stability_comparison_h1440.csv` | `MEAN` row, `all_7_*` / `stable_5_*` |
| Density–GAT-margin correlation (underpowered) | Pearson r=0.80, p=0.21, n=4 | `results/eval/graph_density_summary.json` | `pearson_r`=0.7953, `pearson_p`=0.2047, `n_valid_for_corr`=4 |
| Precision@k under an alert budget | @10 0.40, @25 0.32, oracle(545) 0.33 vs XGB 0/0/0.19 | `results/eval/precision_at_k_svb.json` | `gat_precision_at_k`, `xgboost_precision_at_k`, `n_positives`=545 |
| Calibration: raw poor, isotonic repairs it | raw ECE 0.23–0.26; GAT recalibrated ≈0.02 | `results/eval/calibration_ece_h1440.csv` (raw), `results/eval/calibration_recalibrated_h1440.csv` (out-of-sample isotonic) | `gat` raw 0.2567; recalibrated xfit 0.0197 |
| TGN-lite adds little | +0.008 mean, 4 of 8 fold-seed wins | `results/eval/tgn_verdict_h1440.json`, `results/eval/tgn_vs_gat_h1440.csv` | `mean_delta_tgn_minus_gat`=0.0083, `tgn_wins_folds_seeds`=4/8 |
| Hub ranking (correlational) and attention | BUSD 1.00 / USDC 0.85 / TUSD 0.53 (GNNExplainer); USDC→TUSD ᾱ=0.51 (1.9×) | `exports/hub_ranking_v1_USDC_SVB.csv` (`norm_gnn`), `results/interpret/attention_hub_table.csv` (`mean_attn`) | BUSD/USDC/TUSD `norm_gnn`; USDC→TUSD 0.5107 vs network avg ≈0.27 |
| Causal Δ column (BUSD 0%, USDC 100%, DAI 98%) | from the companion ABM | companion `stablecoin-abm` `intervention_sweep.csv` | cross-referenced; see that repo |

Base rate for the held-out SVB cluster at 24h is 0.293 throughout. All numbers regenerate from the
committed scripts under `scripts/` and `src/scgnn/`.
