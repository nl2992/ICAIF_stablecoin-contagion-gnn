"""
LOEO-safe transformer wrapper.

Every fit/transform component used inside LOEO (imputer, scaler, isotonic
calibrator) MUST be refitted on each fold's training episodes only.
Fitting on global data (which includes the held-out episode) is a hard error.

Design
------
Each training sample is tagged with its source episode index (episode_tag).
The LOEOFold context records which episode index is held out.
LOEOSafeTransformer intercepts .fit() and raises LeakageError if any row
with the held-out episode tag is passed — no silent failure possible.

Usage pattern (enforced in every eval pipeline)
------------------------------------------------
    fold = LOEOFold(held_out_idx=2, all_episode_tags=sample_tags)

    scaler     = LOEOSafeTransformer(StandardScaler(), fold)
    imputer    = LOEOSafeTransformer(TrainFitImputer(), fold)

    X_train_sc = scaler.fit_transform(X_train, episode_tags=train_tags)
    X_test_sc  = scaler.transform(X_test)   # no episode_tags needed

    # This would raise LeakageError:
    # scaler.fit_transform(X_all, episode_tags=all_tags)

Isotonic calibrator rule
------------------------
The calibrator must be fit on the VALIDATION split (not test).
LOEOFold.validate_calibrator_fit() enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


# ------------------------------------------------------------------ exceptions


class LeakageError(RuntimeError):
    """Raised when a transformer is fit on data that includes held-out samples."""


# ------------------------------------------------------------------ fold context


@dataclass
class LOEOFold:
    """Context object for one LOEO fold."""
    held_out_episode_name: str          # name of the held-out episode
    train_episode_names: List[str]      # names of training episodes
    val_episode_name: Optional[str] = None   # name of fold-internal validation episode

    def assert_no_held_out(self, episode_tags: List[str], context: str = "") -> None:
        """
        Raise LeakageError if the held-out episode appears in episode_tags.
        Call this before every .fit() inside the LOEO loop.
        """
        if self.held_out_episode_name in episode_tags:
            raise LeakageError(
                f"LEAKAGE: held-out episode '{self.held_out_episode_name}' "
                f"present in fit data{' (' + context + ')' if context else ''}. "
                f"Refit transformers on train fold only."
            )

    def assert_is_val_not_test(self, episode_tags: List[str], context: str = "") -> None:
        """
        Raise LeakageError if the test (held-out) episode appears in calibrator fit data.
        The calibrator must be fit on the fold-internal val split, not the test fold.
        """
        self.assert_no_held_out(episode_tags, context=f"calibrator fit {context}")
        if self.val_episode_name and self.val_episode_name not in episode_tags:
            # Warn (not error) if val data isn't being used — could be intentional
            import warnings
            warnings.warn(
                f"Calibrator fit does not include validation episode "
                f"'{self.val_episode_name}' — ensure val data is included.",
                stacklevel=2,
            )

    def assert_train_only_stats(self, episode_tags: List[str], context: str = "") -> None:
        """
        Raise LeakageError if standardization/imputation uses non-training data.
        """
        non_train = [t for t in set(episode_tags)
                     if t not in self.train_episode_names
                     and t != self.val_episode_name]
        if non_train:
            raise LeakageError(
                f"LEAKAGE: feature statistics computed on non-training episodes "
                f"{non_train}{' (' + context + ')' if context else ''}."
            )


# ------------------------------------------------------------------ safe transformer


class LOEOSafeTransformer:
    """
    Wraps any sklearn-compatible transformer, enforcing LOEO-clean fitting.

    .fit() or .fit_transform() REQUIRES episode_tags.
    If episode_tags includes the held-out episode → LeakageError.
    .transform() does NOT require episode_tags (inference is always safe).
    """

    def __init__(self, transformer: Any, fold: LOEOFold, role: str = "transformer"):
        self._inner = transformer
        self._fold = fold
        self._role = role
        self._fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            episode_tags: Optional[List[str]] = None, **kwargs) -> "LOEOSafeTransformer":
        if episode_tags is None:
            raise LeakageError(
                f"LOEOSafeTransformer({self._role}).fit() called without episode_tags. "
                f"You must pass episode_tags to prove no held-out data is included."
            )
        if self._role == "calibrator":
            self._fold.assert_is_val_not_test(episode_tags, context=self._role)
        else:
            self._fold.assert_no_held_out(episode_tags, context=self._role)
            self._fold.assert_train_only_stats(episode_tags, context=self._role)

        if y is not None:
            self._inner.fit(X, y, **kwargs)
        else:
            self._inner.fit(X, **kwargs)
        self._fitted = True
        return self

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                      episode_tags: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        self.fit(X, y, episode_tags=episode_tags, **kwargs)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                f"LOEOSafeTransformer({self._role}) has not been fitted. "
                f"Call .fit() with episode_tags first."
            )
        return self._inner.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """For calibrators that use predict() rather than transform()."""
        if not self._fitted:
            raise RuntimeError(f"LOEOSafeTransformer({self._role}) not fitted.")
        return self._inner.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"LOEOSafeTransformer({self._role}) not fitted.")
        return self._inner.predict_proba(X)

    @property
    def inner(self):
        return self._inner


# ------------------------------------------------------------------ LOEO-safe pipeline builder


def make_loeo_safe_components(fold: LOEOFold) -> dict:
    """
    Return a dict of LOEOSafeTransformer instances for the standard pipeline.
    Use these inside every LOEO fold — never reuse across folds.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.isotonic import IsotonicRegression
    from scgnn.data.integrity import TrainFitImputer

    return {
        "scaler":     LOEOSafeTransformer(StandardScaler(),      fold, role="scaler"),
        "imputer":    LOEOSafeTransformer(TrainFitImputer(),     fold, role="imputer"),
        "calibrator": LOEOSafeTransformer(IsotonicRegression(out_of_bounds="clip"),
                                          fold, role="calibrator"),
    }
