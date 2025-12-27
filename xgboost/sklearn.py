"""Minimal sklearn-compatible shims for the local xgboost stub."""
from __future__ import annotations

import numpy as _np

IS_QT_XGBOOST_STUB = True


class XGBClassifier:
    """Small stand-in for :class:`xgboost.sklearn.XGBClassifier`.

    The implementation only preserves the constructor signature used during
    tests and provides `fit`, `predict`, and `predict_proba` methods so that
    pickled models depending on the sklearn wrapper can be deserialised
    without importing the real xgboost distribution.
    """

    QT_IS_STUB = True

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        verbosity: int = 0,
        objective: str | None = None,
        booster: str | None = None,
        **_: object,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self._positive_prob: float | None = None

    def __getstate__(self):  # pragma: no cover - invoked via pickle
        return self.__dict__

    def __setstate__(self, state):  # pragma: no cover - invoked via pickle
        self.__dict__.update(state)
        if "_positive_prob" not in self.__dict__:
            self._positive_prob = None

    def fit(self, X, y):
        _ = _np.asarray(X)
        y_arr = _np.asarray(y, dtype=float)
        if y_arr.size == 0:
            self._positive_prob = 0.0
        else:
            mean_val = float(_np.nanmean(y_arr))
            self._positive_prob = float(_np.clip(mean_val, 0.0, 1.0))
        return self

    def predict_proba(self, X):
        prob = getattr(self, "_positive_prob", None)
        if prob is None:
            prob = 0.5
            self._positive_prob = prob
        X_arr = _np.asarray(X)
        prob_pos = _np.full((X_arr.shape[0],), prob)
        prob_neg = 1.0 - prob_pos
        return _np.vstack([prob_neg, prob_pos]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


__all__ = ["XGBClassifier"]
