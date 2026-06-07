# TODO — Research Improvement Plans
# Predicting Cross-Asset Stablecoin Contagion with Temporal Graph Neural Networks

## Current weaknesses

- PR-AUC 0.447 is modest — not a dramatic headline on its own
- Only 7 crisis episodes; 2 are too sparse (BUSD depeg, USDT-2018) and may be hurting LOCO stability
- Node features are entirely market microstructure — no on-chain pool flow, DEX liquidity depth, or redemption pressure features
- The GNN is a static snapshot model — the graph topology itself is not allowed to evolve temporally within an episode (no TGNN/dynamic-edge learning)
- LOCO generalization margin shrinks to +0.083 vs XGBoost — fragile cross-crisis claim
- No probabilistic calibration study — PR-AUC is an ordering metric, not a deployment metric; reviewers will ask "at what threshold do you alert?"
- Ablation shows the graph adds +0.100 over no-edge GAT, but this is not yet tied to a structural story ("which edges matter and why")

---

## Plans

### Plan A — Lead the paper with the graph-contribution ablation, not the PR-AUC headline

**What to code:**
- `eval/ablation.py`: run 4 conditions — (1) tabular XGBoost (no graph), (2) GAT with node features only (no edges), (3) GAT with edges but no node features, (4) full GAT (node features + edges); record PR-AUC per condition per LOCO fold
- `interpret/edge_importance.py`: for each crisis, log which edges (asset pairs) have highest GAT attention weight; aggregate across folds
- Produce a 2x2 ablation table: rows = conditions, columns = average LOCO PR-AUC and improvement over XGBoost

**What to run:**
```bash
python eval/ablation.py --model GAT --universe all --loco True
python interpret/edge_importance.py --model GAT --top_k 5
```

**Target result:**
- Condition 2 vs condition 1: node features alone add X over XGBoost (graph topology contribution isolated)
- Condition 4 vs condition 2: edge structure adds +0.100 PR-AUC (already observed; now formalized with CIs)
- Headline: "The directed lead-lag graph topology contributes +0.10 PR-AUC beyond microstructure features alone, identifying USDC and Curve-3pool as principal contagion conduits"

**Write into paper:**
- New Table 3 "Component Ablation" in Section 4 Results; make this the first results table
- Abstract: replace PR-AUC headline with ablation finding: graph structure adds +X over tabular models
- Section 3 Model: add one paragraph explaining why the ablation conditions isolate the graph's contribution

---

### Plan B — Drop the 2 sparse episodes and re-run LOCO with 5-episode stable set

**What to code:**
- `configs/episodes_v2.yaml`: define the 5-episode subset — LUNA/UST, USDC/SVB, DAI/USDC, FTX/contagion, USDT-stress-2023; exclude BUSD and USDT-2018
- `scripts/run_loco.py`: add `--episode_set` flag to run LOCO on episode subsets
- `src/evaluation/loco_stability.py`: compute mean LOCO PR-AUC and variance under (a) all 7 episodes and (b) 5-episode stable set; report both

**What to run:**
```bash
python scripts/run_loco.py --model GAT --episode_set episodes_v2 --loco True
python src/evaluation/loco_stability.py --compare_sets all_7,stable_5
```

**Target result:**
- Stable-5 LOCO mean PR-AUC: target >= 0.47 (vs 0.447 on all 7)
- Stable-5 LOCO margin over XGBoost: target >= +0.10 (vs +0.083 currently)
- The two excluded episodes are documented with justification; this is not cherry-picking if pre-registered

**Write into paper:**
- Section 2 Data: add Table 1 listing all 7 episodes with row counts, base rates, and quality flag; footnote the exclusion criterion
- Section 5 Robustness: add "Sparse-Episode Sensitivity" subsection comparing all-7 and stable-5 LOCO results; report both
- Pre-registration note: add appendix sentence stating the episode quality criterion was set before final model runs

---

### Plan C — Add 3 on-chain features to break the "all microstructure" critique

**What to code:**
- `features/onchain_features.py`: add three features using publicly available on-chain data (The Graph or Dune Analytics exports, or pre-cached CSVs):
  1. `curve_3pool_imbalance`: (USDC_balance - USDT_balance) / total_pool_liquidity at each timestamp
  2. `redemption_pressure_proxy`: net stablecoin outflows from top-3 Curve pools (approximated from on-chain balance deltas)
  3. `cex_netflow`: net stablecoin inflows/outflows to Binance+Coinbase (from CryptoQuant public API or pre-cached)
- `data/onchain/`: store raw on-chain CSVs with provenance note
- Retrain XGBoost and GAT with extended feature set; compare PR-AUC to microstructure-only baseline

**What to run:**
```bash
python features/onchain_features.py --output data/processed/onchain_features.parquet
python scripts/run_loco.py --model GAT --features full_with_onchain
python scripts/run_loco.py --model XGBoost --features full_with_onchain
```

**Target result:**
- GAT with on-chain features: target PR-AUC >= 0.48 on LOCO (modest improvement; main value is theoretical)
- XGBoost with on-chain features: record whether tabular gap over GAT closes — if GAT still leads, graph structure is validated above on-chain features alone
- Headline: "Adding on-chain pool-imbalance features improves PR-AUC by X; GNN retains Y advantage from graph topology above the extended tabular baseline"

**Write into paper:**
- Section 2 Features: add paragraph on on-chain features; add data card footnote with Dune/The Graph query IDs
- Table 2: add rows for XGBoost+onchain and GAT+onchain
- Discussion: address the "all microstructure" limitation directly; state which features had highest XGBoost gain

---

### Plan D — Probabilistic calibration and threshold selection study

**What to code:**
- `eval/calibration.py`: compute reliability diagrams (10 bins), Brier score, and Expected Calibration Error (ECE) for GAT, XGBoost, and GRU
- `eval/threshold_sweep.py`: for each model, sweep alert threshold from 0.1 to 0.9; for each threshold record precision, recall, F1, and simulated alert rate (alerts per 24h)
- Produce: (a) reliability diagram figure, (b) precision-recall-threshold curve, (c) "operating point" table showing threshold choices at precision=0.6, recall=0.6

**What to run:**
```bash
python eval/calibration.py --model GAT XGBoost GRU --loco True
python eval/threshold_sweep.py --model GAT --horizon 24h
```

**Target result:**
- GAT ECE < XGBoost ECE on stress episodes (graph-based model better calibrated)
- At precision=0.60 operating point: GAT recall >= 0.50 (catches at least half of contagion events at reasonable false-alarm rate)
- Headline figure: precision-recall-threshold curve with recommended operating point marked

**Write into paper:**
- New Section 5.3 "Calibration and Deployment Threshold": reliability diagram + threshold table
- Abstract: add "We provide calibrated probability estimates and operating-point guidance for practitioners"
- Conclusion: replace "future work: deployment" with a concrete recommended threshold and expected false-alarm rate

---

### Plan E — Add temporal graph (TGN-lite) as the strongest model variant

**What to code:**
- `models/temporal_gnn.py`: implement a lightweight Temporal Graph Network variant — reuse the GAT message-passing but feed each node a GRU hidden state initialized from the previous timestep's node embedding (this is a 50-line addition to the existing GAT class)
- `configs/model_tgn.yaml`: same hyperparameters as GAT but with `temporal_encoding: true` and `memory_dim: 32`
- Run LOCO comparison: GAT-static vs TGN-lite

**What to run:**
```bash
python train/train_gnn.py --model TGN --config configs/model_tgn.yaml --loco True
python eval/ablation.py --compare GAT,TGN
```

**Target result:**
- TGN-lite PR-AUC >= 0.47 on LOCO (marginal improvement; main value is ablation of temporal vs static)
- If TGN does not improve: that is a valid finding — "temporal memory does not add beyond static snapshots given 6h rolling window edges"
- Either outcome becomes a paper section: temporal dynamics are either captured by the rolling-edge graph or require longer memory

**Write into paper:**
- Section 3 Model: add one paragraph describing TGN-lite variant and how it differs from static GAT
- Table 2: add TGN-lite row
- Discussion: one paragraph on whether temporal memory helps and why; if not, explain via rolling-edge graph already capturing recency

---

### Plan F — Attention-weight analysis: which edges drive contagion predictions

**What to code:**
- `interpret/attention_analysis.py`: for each LOCO test fold, extract GAT attention weights per edge during the 48h window before peak contagion; compute mean attention across crisis episodes; rank asset pairs by mean attention
- `interpret/hub_report.py`: integrate with existing hub-ranking code; produce a named hub table (node name, betweenness centrality, mean attention weight, propagator label count)
- `results/interpret/attention_heatmap.png`: attention matrix averaged over all crisis episodes

**What to run:**
```bash
python interpret/attention_analysis.py --model GAT --window 48h
python interpret/hub_report.py --combine_attention_betweenness True
```

**Target result:**
- Top 2–3 hub pairs are consistently USDC→[others] and Curve-3pool→USDT across crises
- Attention analysis provides a named structural story: "The model learns that USDC is the dominant source node; edges from USDC to USDT and DAI carry 3x the average attention weight"
- This is the quotable finding that reviewers remember

**Write into paper:**
- New Figure 4: attention heatmap averaged over stress episodes with asset labels
- New Table 4: hub-ranking table with attention + betweenness + propagator count columns
- Abstract: update to include named hub finding (e.g., "USDC and Curve-3pool emerge as principal contagion conduits")
- Section 4 Results: lead the interpretability subsection with this table before any PR-AUC numbers

---

### Plan G — Failure-mode case study (3 misclassified episodes)

**What to code:**
- `eval/failure_analysis.py`: for each LOCO fold's test episode, identify the 10 most-confident false positives and 10 most-confident false negatives from the GAT model
- For each selected example: extract the feature vector, the attention weights, the graph topology at that timestamp, and the actual market event
- `results/eval/failure_cases.csv`: structured table of timestamp, predicted prob, true label, top features, top attended edge, and qualitative event label

**What to run:**
```bash
python eval/failure_analysis.py --model GAT --n_cases 10 --loco True
```

**Target result:**
- 3–5 representative cases that each illustrate a mechanically distinct failure mode, for example:
  1. False positive during high-vol non-contagion (model confuses vol spike with contagion)
  2. False negative during slow-onset contagion (model misses gradual peg drift below 6h resolution)
  3. False positive driven by a spurious edge (volume spike on an unrelated pair inflates lead-lag)
- This section is the answer to "Reasoning-or-Overthinking style" depth requirement in original TODO

**Write into paper:**
- New Section 5.4 "Failure Mode Analysis": narrative description of 3 cases with attention plots
- Discussion: use the cases to motivate future work (on-chain features, longer memory, event-driven edges)
- This section directly addresses the reviewer question "what does the model get wrong and why"

---

### Plan H — Multi-horizon decay curve as a deployability figure

**What to code:**
- `eval/lead_time_analysis.py`: compute LOCO PR-AUC at horizons 30min, 1h, 2h, 6h, 12h, 24h for GAT and XGBoost
- Plot both curves on the same axes with shaded 1-std bands across LOCO folds
- Add a vertical line at "earliest horizon where GAT > XGBoost + 0.05" — this is the actionable prediction window

**What to run:**
```bash
python eval/lead_time_analysis.py --models GAT,XGBoost --horizons 30,60,120,360,720,1440
```

**Target result:**
- GAT > XGBoost by >= 0.05 at >= 6h horizon (already indicated by 24h result)
- Both models approach chance at <= 1h horizon — this is honest and expected given graph construction
- Key finding: "Graph-based contagion prediction is informative at 6–24h horizons; sub-hour prediction is near-chance for all models"

**Write into paper:**
- New Figure 2: PR-AUC vs prediction horizon for GAT and XGBoost; mark the 6h actionability threshold
- Abstract: add "providing actionable signals at 6–24 hour horizons"
- Section 5.1: replace the single "24h GAT PR-AUC 0.447" sentence with the full decay-curve discussion
- This converts a single number into a decision-support framing that practitioners can use
