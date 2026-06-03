"""
Leave-One-Episode-Out (LOEO) cross-validation.

With only 7 real episodes, LOEO is the ONLY honest generalization claim.
A pooled train/val/test split hides the risk of fitting to one crisis type.

Protocol:
  For each episode k ∈ {1..7}:
    - Train on real episodes {1..7} \ {k}  (+ synthetic if enabled)
    - Test on episode k
    - Report PR-AUC, weighted-F1 per held-out episode

Additionally reports:
  1. Per-trigger-type breakdown (algo / fiat_bank / fiat_regulatory / crypto_backed)
  2. Real-only vs real+synthetic performance gap
  3. Trivial-baseline stress test: does majority-class score suspiciously
     high on synthetics?  If yes, synthetics are too easy.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scgnn.data.windows import Episode
from scgnn.eval.metrics import full_report
from scgnn.models.baselines import MajorityClassifier


def loeo_cv(
    episodes: List[Episode],
    feature_fn: Callable[[List[Episode]], Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable,
    horizon_min: int,
    synthetic_fn: Optional[Callable[[], Tuple[np.ndarray, np.ndarray]]] = None,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Strict leave-one-episode-out cross-validation over real episodes.

    feature_fn(episode_list) → (X, y) for those episodes combined.
    model_factory() → fresh unfitted sklearn-compatible model.
    synthetic_fn()  → (X_synth, y_synth)  added to train only if provided.

    Returns DataFrame with one row per held-out episode + aggregate.
    """
    rows = []
    all_probs, all_labels = [], []

    for i, held_out in enumerate(episodes):
        train_eps = [ep for j, ep in enumerate(episodes) if j != i]
        if not train_eps:
            continue

        X_train, y_train = feature_fn(train_eps)
        X_test, y_test = feature_fn([held_out])

        if y_train.ravel().sum() == 0:
            print(f"[WARN] LOEO: no positives in train for held_out={held_out.name} — skipping")
            continue

        # Optionally augment with synthetics
        if synthetic_fn is not None:
            X_synth, y_synth = synthetic_fn()
            X_train = np.concatenate([X_train, X_synth], axis=0)
            y_train = np.concatenate([y_train.ravel(), y_synth.ravel()])

        y_train = y_train.ravel()
        y_test_flat = y_test.ravel()

        if y_test_flat.sum() == 0:
            print(f"[WARN] LOEO: no positives in test for held_out={held_out.name}")
            pr_auc, roc_auc, wf1 = float("nan"), float("nan"), float("nan")
            maj_pr_auc = float("nan")
        else:
            model = model_factory()
            model.fit(X_train, y_train)
            probs = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
                     else model.predict(X_test).astype(float))
            report = full_report(y_test_flat, probs.ravel())
            pr_auc, roc_auc, wf1 = report["pr_auc"], report["roc_auc"], report["weighted_f1"]

            # Majority baseline for this episode
            maj = MajorityClassifier().fit(X_train, y_train)
            maj_probs = maj.predict(X_test.reshape(len(y_test_flat), -1)).astype(float)
            maj_pr_auc = full_report(y_test_flat, maj_probs)["pr_auc"]

            all_probs.append(probs.ravel())
            all_labels.append(y_test_flat)

        rows.append({
            "held_out_episode": held_out.name,
            "trigger": held_out.trigger,
            "trigger_type": held_out.trigger_type,
            "split": held_out.split,
            "n_train_positive": int(y_train.sum()),
            "n_test": int(len(y_test_flat)) if y_test.size > 0 else 0,
            "n_test_positive": int(y_test_flat.sum()) if y_test.size > 0 else 0,
            "positive_rate_test": round(float(y_test_flat.mean()), 4) if y_test.size > 0 else float("nan"),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "weighted_f1": wf1,
            "majority_pr_auc": maj_pr_auc if y_test_flat.sum() > 0 else float("nan"),
            "gap_vs_majority": pr_auc - maj_pr_auc if not (np.isnan(pr_auc) or np.isnan(maj_pr_auc)) else float("nan"),
        })

    df = pd.DataFrame(rows)

    # Aggregate row
    if not df.empty:
        numeric_cols = ["pr_auc", "roc_auc", "weighted_f1", "gap_vs_majority"]
        agg = {"held_out_episode": "MEAN", "trigger": "—", "trigger_type": "—", "split": "—"}
        for col in numeric_cols:
            vals = df[col].dropna()
            agg[col] = round(float(vals.mean()), 4) if len(vals) else float("nan")
        agg["n_test"] = int(df["n_test"].sum())
        agg["n_test_positive"] = int(df["n_test_positive"].sum())
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    return df.set_index("held_out_episode")


def synthetic_stress_test(
    synthetic_feature_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    real_feature_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable,
    n_runs: int = 5,
    seed: int = 42,
) -> dict:
    """
    Stress test: does majority-class score suspiciously high on synthetics?

    Train on synthetic, test on synthetic (within-distribution).
    Train on synthetic, test on real (out-of-distribution).
    A large gap (>0.2 PR-AUC) means synthetics are too easy / unrealistically learnable.

    Returns {"within_synth_pr_auc", "on_real_pr_auc", "gap", "verdict"}.
    """
    from scgnn.utils.seeds import set_all_seeds
    rng = np.random.default_rng(seed)

    X_synth, y_synth = synthetic_feature_fn()
    X_real, y_real = real_feature_fn()

    within_scores, cross_scores = [], []

    for run in range(n_runs):
        run_seed = int(rng.integers(0, 2**31))
        set_all_seeds(run_seed)

        # Within-synthetic split
        n = len(y_synth)
        idx = rng.permutation(n)
        split = int(n * 0.7)
        tr_idx, te_idx = idx[:split], idx[split:]
        m_w = model_factory()
        m_w.fit(X_synth[tr_idx], y_synth[tr_idx])
        if y_synth[te_idx].sum() > 0:
            probs_w = m_w.predict_proba(X_synth[te_idx])[:, 1]
            within_scores.append(full_report(y_synth[te_idx], probs_w)["pr_auc"])

        # Cross: train synth, test real
        m_c = model_factory()
        m_c.fit(X_synth, y_synth)
        if y_real.sum() > 0:
            probs_c = m_c.predict_proba(X_real)[:, 1]
            cross_scores.append(full_report(y_real, probs_c)["pr_auc"])

    within = float(np.mean(within_scores)) if within_scores else float("nan")
    cross = float(np.mean(cross_scores)) if cross_scores else float("nan")
    gap = within - cross if not (np.isnan(within) or np.isnan(cross)) else float("nan")

    verdict = "OK" if np.isnan(gap) or gap < 0.20 else "WARN: synthetics may be too easy"
    return {
        "within_synth_pr_auc": round(within, 4),
        "on_real_pr_auc": round(cross, 4),
        "gap": round(gap, 4),
        "verdict": verdict,
    }
