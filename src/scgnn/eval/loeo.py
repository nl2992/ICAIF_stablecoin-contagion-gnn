"""
Leave-one-episode-out (LOEO) cross-validation.

For each episode in the pool, train on all others, evaluate on the left-out one.
This is the strongest generalization test available given the small episode count.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from scgnn.data.windows import Episode
from scgnn.eval.metrics import full_report, results_table


def loeo_cv(
    episodes: List[Episode],
    feature_fn: Callable[[List[Episode]], Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable,
    horizon_min: int,
    n_bootstrap: int = 0,
) -> pd.DataFrame:
    """
    Run LOEO-CV.

    feature_fn(episode_list) → (X, y) for those episodes combined.
    model_factory() → a fresh, unfitted sklearn-compatible model.

    Returns a DataFrame with one row per left-out episode + aggregate stats.
    """
    rows = []
    all_probs, all_labels = [], []

    for i, held_out in enumerate(episodes):
        train_eps = [ep for j, ep in enumerate(episodes) if j != i]
        if not train_eps:
            continue

        X_train, y_train = feature_fn(train_eps)
        X_test, y_test = feature_fn([held_out])

        if y_train.ravel().sum() == 0 or y_test.ravel().sum() == 0:
            # Skip degenerate folds with no positives
            continue

        model = model_factory()
        model.fit(X_train, y_train.ravel())
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test).astype(float)

        report = full_report(y_test.ravel(), probs.ravel())
        rows.append({
            "held_out_episode": held_out.name,
            "trigger_type": held_out.trigger_type,
            "n_test": int(y_test.ravel().shape[0]),
            "pr_auc": report["pr_auc"],
            "roc_auc": report["roc_auc"],
            "weighted_f1": report["weighted_f1"],
        })
        all_probs.append(probs.ravel())
        all_labels.append(y_test.ravel())

    df = pd.DataFrame(rows)
    if not df.empty:
        # Aggregate row
        agg = {
            "held_out_episode": "MEAN",
            "trigger_type": "-",
            "n_test": int(df["n_test"].sum()),
            "pr_auc": float(df["pr_auc"].mean()),
            "roc_auc": float(df["roc_auc"].mean()),
            "weighted_f1": float(df["weighted_f1"].mean()),
        }
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    return df.set_index("held_out_episode")
