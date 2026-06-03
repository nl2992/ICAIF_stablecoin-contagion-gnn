# HOW-TO — stablecoin-contagion-gnn

Concrete execution guide for `TODO.md`. Each block = commands + what "good" looks like
+ the failure mode to watch. Run from repo root with the dev env installed:

```bash
pip install -e ".[dev]"
pytest tests/          # must be green before trusting any result
```

---

## P0 — Verify the foundation

**Data provenance.** Open `src/scgnn/data/fetch.py` and confirm whether it hits a real
API (CCXT/Binance/DeFiLlama) or fabricates series. Then:
```bash
python -c "import numpy as np; X=np.load('data/processed/X_train.npy'); \
print(X.shape, X.mean(), X.std(), np.isnan(X).mean())"
```
- *Good:* finite, non-degenerate stats; NaN fraction ~0; shape matches manifest.
- *Bad:* constant columns or all-NaN ⇒ fetch is stubbed; re-pull real data first.
Write the verdict + source URLs + pull date into `data/data_card.md`.

**Label base rates.**
```bash
python - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob('data/processed/y_*_h*.npy')):
    y=np.load(f); print(os.path.basename(f), 'pos_rate=%.4f'%y.mean(), 'n=%d'%len(y))
PY
```
Record these; they justify PR-AUC over accuracy and the weighted-F1 tuning choice.

---

## P1 — Headline: model ladder + LOEO

```bash
# Full ladder, all horizons -> results/ladder/
python train/run_ladder.py --horizon 30
python train/run_ladder.py --horizon 60
python train/run_ladder.py --horizon 240
python train/run_ladder.py --horizon 1440

# LOEO per-fold (the headline table) -> results/eval/
python eval/run_all_eval.py    # or invoke src/scgnn/eval/loeo.py directly
```
- *Good:* every fold logs PR-AUC for every model; a per-fold CSV lands in `results/eval/`.
- *Verdict gate (pre-registered):* compute `mean(GNN_PR_AUC - XGB_PR_AUC)` across folds
  and the count of folds where margin ≥ 0.05. If `mean ≥ 0.05 AND folds ≥ 5` ⇒ success
  framing; else ⇒ honest-null framing. **Decide here, then write prose.**
- *Failure mode:* a single huge fold (USDC_SVB is the held-out test) dominating the mean.
  Report per-fold, never just the average.

**Calibration:** after the ladder, run `src/scgnn/eval/calibration_curve.py` on the best
model; save reliability diagram + Brier score. This is a depth edge over the template paper.

---

## P2 — Breadth

```bash
python eval/lead_time_analysis.py --model graphsage     # PR-AUC vs horizon curve
python -m scgnn.eval.ablation                            # node / +edge / +structure
python eval/label_sensitivity.py                         # 25 ± 15 bps hub ρ
python -m scgnn.train.ensemble --seeds 5                 # mean±std + hub Spearman ρ
python -m scgnn.eval.synthetic_validation               # real-vs-synth gap table
```
- *Ablation is the most important breadth result:* it is the literal answer to the
  ICAIF reviewer's "is the graph doing anything, or is it the features?" Present it as a
  table: PR-AUC for {node-only, +edge, +structure}. The delta is your contribution size.
- *Multi-seed:* if hub-ranking Spearman ρ < 0.70, fall back to `hub_score_structural`
  (betweenness only) as the primary artifact — pre-registered, so it's honest not ad hoc.

---

## P3 — Depth (interpretability)

```bash
python -m scgnn.interpret.explainability     # XGBoost gain + GNNExplainer masks
python interpret/hub_report.py               # betweenness × propagator -> hub ranking
python scripts/export_hubs.py                # -> exports/hub_ranking_v1_*.{json,csv}
python -m scgnn.interpret.spurious_audit     # volume/TVL confound test
```
- *Feature/node importance:* mirror the Uniswap paper — produce a top-20 feature bar and
  name the top contagion-conduit nodes. The claim "USDC/binance and Curve-3pool are the
  principal conduits" must be backed by both GNNExplainer mass AND centrality.
- *Spurious audit ⇒ hand-off:* identify the node whose importance is highest *but* whose
  signal is explained by raw volume/TVL. Name it explicitly — the ABM repo's divergence
  case study intervenes on exactly this node. Put the name in the exported hub JSON.
- *Failure modes:* pull 3–5 misclassified node-windows; for each, plot the price/vol path
  and write one sentence on why the model erred (Reasoning-or-Overthinking §6 style).

---

## P4 — Thesis

Fill this template sentence with real numbers and put it in the abstract:
> "On leave-one-episode-out evaluation, graph structure adds **X** PR-AUC over
> microstructure features alone and flags **{nodes}** as the principal contagion conduits
> **Y** minutes ahead of peak cross-venue spread."

If X ≤ 0: pivot to the pre-registered null — "tabular is competitive per-node; the graph's
value is structural interpretability (stable hub ranking), validated causally downstream."

---

## P5 — Figures, paper, release

```bash
python scripts/generate_figures.py     # -> results/figures/
```
Build: (1) LOEO per-fold bar, (2) lead-time decay curve, (3) ablation table-figure,
(4) feature-importance bar, (5) hub network with conduits highlighted, (6) calibration
reliability diagram. Write the paper from `paper/`, fold in `threats_to_validity.md`,
release `exports/` + `data_card.md`, and add the reproducibility appendix (seeds, hashes,
exact commands above).

**The export contract is the bridge to the ABM** — keep `exports/hub_ranking_v1_*.json`
schema-stable; the ABM's `counterfactual/hub_loader.py` reads it.
