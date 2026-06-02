"""Logistic regression and XGBoost tabular models (flattened node features)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(class_weight: str = "balanced") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000, C=1.0)),
    ])


def make_xgboost(
    n_estimators: int = 500,
    max_depth: int = 6,
    scale_pos_weight: Optional[float] = None,
    seed: int = 42,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=-1,
    )


def xgb_feature_importance(model: xgb.XGBClassifier, feature_names: list[str], topk: int = 20) -> dict:
    scores = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    return dict(ranked)
