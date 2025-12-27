"""Core helpers for the lightweight xgboost stub."""
from __future__ import annotations

import numpy as _np


class Booster:
    """Placeholder for :class:`xgboost.core.Booster` used in pickled models."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params = params or {}
        self._model_data = None

    def load_model(self, buffer: object) -> None:  # pragma: no cover - simple shim
        self._model_data = buffer

    def predict(self, data, **_: object):
        """Return zeros; real inference is not required in tests."""
        rows = getattr(data, "num_row", None)
        if callable(rows):
            count = int(rows())
        else:
            arr = _np.asarray(getattr(data, "data", data))
            count = int(arr.shape[0]) if arr.ndim > 0 else 1
        return _np.zeros(count, dtype=float)


class DMatrix:
    """Very small stand-in for :class:`xgboost.core.DMatrix`."""

    def __init__(self, data, label=None, **_: object) -> None:
        self.data = _np.asarray(data)
        self.label = None if label is None else _np.asarray(label)

    def num_row(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim > 0 else 1


__all__ = ["Booster", "DMatrix"]
