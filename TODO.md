# TODO — stablecoin-contagion-gnn

Goal: turn the (excellent) scaffold into a paper that rivals the ICAIF'25 Uniswap
bridge-swap GNN paper on **narrative, breadth, and depth of findings**.
Template paper: Zhou, Liu, Brini — *GNN for Bridge Swap Link Prediction in Uniswap v3*.

The architecture is done. **What is missing is executed results.** Everything below
is dependency-ordered. Do not reorder the gates.

---

## P0 — Verify the foundation is real (blocks everything)

- [ ] **Confirm data provenance.** Establish whether `data/processed/*.npy` is real
      fetched OHLCV or synthetic placeholders. A reviewer's first question. Document
      the answer in `data/data_card.md` with source, date pulled, and row counts.
- [ ] **Run `pytest tests/` green** and record the leakage-test results in the paper
      appendix (you already have `test_leakage`, `test_loeo_leakage` — these are a
      selling point; surface them).
- [ ] **Sanity-check label base rates** per episode and per horizon (30/60/240/1440).
      Severe imbalance is expected; record it — it justifies PR-AUC + weighted-F1.

## P1 — Produce the headline result (the model ladder)

- [ ] Run the **full ladder** (Majority → Persistence → LogReg → XGBoost → LSTM →
      GraphSAGE → GAT) across all 4 horizons. Populate `results/ladder/`.
- [ ] Run **LOEO** (leave-one-episode-out) — the per-fold PR-AUC table is your
      headline, exactly as pre-registered. Populate `results/eval/`.
- [ ] Evaluate against the **pre-registered success criterion**: GNN ≥ XGBoost + 0.05
      PR-AUC on ≥5/7 folds. Write the verdict down **before** polishing prose.
- [ ] Add a **calibration curve + Brier score** per model (`eval/calibration_curve.py`
      exists — wire it in). The Uniswap paper does not have this; it's a depth edge.

## P2 — Breadth (what turns one result into a paper)

- [ ] **Lead-time decay analysis** (`eval/lead_time_analysis.py`): PR-AUC vs horizon
      curve. "How early can we see contagion?" is a money figure.
- [ ] **Ablations** (`eval/ablation.py`): node-features-only vs +edge-features vs
      +graph-structure. This is the literal answer to "does the graph add anything?"
- [ ] **Label-threshold sensitivity** (`eval/label_sensitivity.py`): re-run hub ranking
      at 25 ± 15 bps; report Spearman ρ (pre-registered tertiary criterion).
- [ ] **Multi-seed stability** (`train/ensemble.py`): 5 seeds, report mean ± std and the
      hub-ranking Spearman ρ (pre-registered secondary criterion, threshold 0.70).
- [ ] **Synthetic-vs-real gap test** (`eval/synthetic_validation.py`): run the
      pre-registered contingency table; report real-only PR-AUC as first-class.

## P3 — Depth (what wins over "fine" papers)

- [ ] **Feature + node importance** (mirror Uniswap §4): XGBoost gain ranking +
      GNNExplainer masks. Headline claim: *which* nodes/features drive contagion.
      Populate `results/interpret/`.
- [ ] **Hub ranking with confidence** (`hub/ranking.py`, `interpret/hub_report.py`):
      betweenness × propagator-label, with the structural-only fallback ready per
      pre-registration. Export via `export_hubs.py`. This is the artifact the ABM consumes.
- [ ] **Spurious-correlation audit** (`interpret/spurious_audit.py`): explicitly test the
      volume/TVL confound. This is the hand-off to the ABM's divergence case study —
      name the candidate spurious hub here.
- [ ] **Confusion-matrix / failure-mode section** (Reasoning-or-Overthinking style):
      pick 3–5 representative misclassifications; explain each mechanically.

## P4 — Sharpen the thesis (ProtoHedge lesson: one number)

- [ ] Land a **single quotable sentence**: e.g. "graph structure adds X PR-AUC over
      microstructure features alone, and identifies USDC/Curve-3pool as the principal
      conduits Y minutes before peak spread." Fill X, Y from real results.
- [ ] If the GNN does **not** beat tabular: execute the pre-registered honest-null
      narrative. A clean null + structural-interpretability story still publishes.

## P5 — Paper + release

- [ ] Generate all figures (`scripts/generate_figures.py`) into `results/figures/`.
- [ ] Write the paper from `paper/` skeleton; fold in `threats_to_validity.md`.
- [ ] **Release the dataset + schema** (`exports/schema_v1.json` already drafted) —
      the Uniswap paper's "we release our dataset" line is part of why it won.
- [ ] Reproducibility appendix: commands, seeds, data hashes, env.

---

## Definition of done (reviewer-facing)
1. Per-fold LOEO table exists and the pre-registered verdict is stated plainly.
2. Ablation isolates the graph's marginal contribution.
3. Hub ranking is stable across seeds (ρ reported) and exported for the ABM.
4. At least one named spurious-hub candidate handed to stablecoin-abm.
5. Dataset + code released; tests green; threats-to-validity addressed.
