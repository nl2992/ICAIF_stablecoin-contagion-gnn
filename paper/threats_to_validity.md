# Threats to Validity

## 1. Sample size: n = 7 real episodes

**Threat**: Seven real stress events is an extremely small sample for a supervised
learning claim.  Any single episode can dominate the LOEO aggregate metric.
With 7 episodes and 4 trigger types, the cross-type generalization test is
essentially n=1 per type.

**Mitigation**:
- Strict LOEO — per-fold results reported individually, never pooled only.
- Bootstrap CIs on every Spearman ρ (n=7 makes the asymptotic approximation invalid).
- Synthetic augmentation (StressBench, 200 episodes) is validated against real moments
  and reported separately — never used to inflate the headline number.
- We explicitly pre-register that failure to generalize is a valid result.

## 2. Synthetic learnability

**Threat**: Block-resampled synthetic episodes may be unrealistically learnable
(the injected shock is a clean step function; real crises are messier).
If a trivial baseline scores high on synthetics but poorly on real episodes,
augmentation is fabricating signal.

**Mitigation**:
- `synthetic_stress_test()` computes within-synth vs cross (synth→real) PR-AUC gap.
- Gap > 0.2 triggers the contingency protocol (real-only primary result).
- Synthetic validation KS test on OU half-life, rvol, autocorr, depeg moments.

## 3. Threshold degrees of freedom

**Threat**: The pre-registered depeg thresholds (25/75/50 bps) were chosen before
modelling, but they inevitably reflect prior knowledge about what constitutes a
"real" stress event.  A reviewer may argue they were informed by looking at the data.

**Mitigation**:
- All thresholds are committed in `configs/experiment.yaml` before any model is run.
- Sensitivity sweep across ±15 bps relative to each class threshold.
- Uniform-threshold arm (25 bps for all assets) reported alongside per-class.
- Hub stability measured across all three threshold arms (Spearman ρ table).

## 4. Propagator-label circularity residual risk

**Threat**: `compute_propagator_labels()` uses raw peg deviations, which are
highly correlated with the input features fed to the GNN.  The propagator label
is not causally independent of the features.

**Mitigation**:
- Propagator labels are computed from price data only (no model output) —
  this is tested explicitly in `test_propagator_label.py`.
- The deeper check: regress propagator_label on the feature matrix; if R² > 0.9,
  the label is essentially a function of the features and the classification task
  is near-tautological.  This regression is run and reported.
- Structural hub variant (betweenness only) is always reported alongside the
  composite — if structural ≈ composite (Spearman ρ > 0.9), the GNN contributes nothing.

## 5. Calibration-on-crisis limits

**Threat**: Probability calibration (ECE, reliability diagram) is computed on the
val fold.  In crisis settings, the true positive rate is highly non-stationary —
a model calibrated on one type of crisis may be badly calibrated on another.

**Mitigation**:
- ECE reported per fold (not just globally).
- Isotonic calibration is applied fold-internally (using `LOEOSafeTransformer`).
- We note explicitly that calibration on tail events is an open problem and that
  the reliability diagram should be interpreted alongside the LOEO table.

## 6. 2018 USDT episode: feature-support mismatch

**Threat**: The USDT_Oct2018 episode predates Uniswap v1, Curve, and reliable
on-chain DeFi data.  Features computed from DEX pools are all NaN for this
episode.  Pooling it with 2022–23 episodes means the model sees different
feature supports per episode.

**Mitigation**:
- `get_available_features(episode_start)` excludes DEX/TVL/LOP features for
  pre-2019 episodes; feature support is documented per-episode.
- Imputer fills structural zeros (not median) for features absent due to era.
- We run a "2018-excluded" LOEO arm and report whether results differ.

## 7. Dead-asset delisting artifacts

**Threat**: BUSD and UST price series going to zero after delisting can be
misread as extreme stress events.  Including them would create false positives.

**Mitigation**:
- `is_delisting_artifact()` detects price→0 patterns after the known delist date.
- For UST specifically: the terminal collapse (May 2022) is a REAL event,
  not an artifact.  The function uses a 30-day window AFTER delist date to
  distinguish "gradual wind-down to zero" from "crisis trajectory."
- Active-node filter drops dead assets from any episode where they are delisted.

## 8. Small-sample Spearman correlations

**Threat**: With n=7 episodes, Spearman ρ has very wide CIs.  A ρ=0.7 could
be consistent with ρ=0.2–0.95 at the 95% level.

**Mitigation**:
- Bootstrap CIs on every Spearman ρ (1000 bootstrap resamples).
- We report ρ with CI explicitly and do not over-claim from point estimates.
- Hub stability is also assessed across seeds (5 runs × node-level), giving
  more data points than the 7-episode correlation.
