"""Majority-class and persistence baselines."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        vals, counts = np.unique(y, return_counts=True)
        self.majority_ = vals[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


class PersistenceClassifier(BaseEstimator, ClassifierMixin):
    """Predict that the label at t+Δ equals the label at t."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        # Expects X to include the current label as the last column.
        return X[:, -1].astype(int)
