"""Lightweight xgboost shim that prefers the real package when available."""

from __future__ import annotations

import importlib as _importlib
import os as _os
import sys as _sys
from pathlib import Path as _Path
from types import ModuleType as _ModuleType
from typing import Optional as _Optional


def _should_force_stub() -> bool:
    value = _os.getenv("QT_FORCE_XGBOOST_STUB", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_native_xgboost() -> _Optional[_ModuleType]:
    if _should_force_stub():
        return None

    module_name = __name__
    repo_root = _Path(__file__).resolve().parents[1]
    original_sys_path = list(_sys.path)
    filtered_sys_path = []
    for entry in original_sys_path:
        candidate = entry or "."
        try:
            resolved = _Path(candidate).resolve()
        except Exception:
            filtered_sys_path.append(entry)
            continue
        if resolved == repo_root:
            continue
        filtered_sys_path.append(entry)

    stub_module = _sys.modules.get(module_name)
    try:
        _sys.modules.pop(module_name, None)
        _sys.path = filtered_sys_path
        return _importlib.import_module(module_name)
    except Exception:
        return None
    finally:
        _sys.path = original_sys_path
        if module_name not in _sys.modules and stub_module is not None:
            _sys.modules[module_name] = stub_module


_NATIVE_XGBOOST = _load_native_xgboost()

if _NATIVE_XGBOOST is not None:
    globals().update(vars(_NATIVE_XGBOOST))
    __all__ = getattr(_NATIVE_XGBOOST, "__all__", [name for name in vars(_NATIVE_XGBOOST) if not name.startswith("_")])
else:
    import numpy as _np

    IS_QT_XGBOOST_STUB = True

    class XGBRegressor:
        """Simplified drop-in replacement for the real XGBRegressor.

        The stub keeps the public constructor arguments used in the codebase and
        implements `fit`/`predict` with trivial behaviour so tests can verify the
        control flow without requiring the native xgboost dependency.
        """

        def __init__(self, n_estimators: int = 100, max_depth: int = 3, verbosity: int = 0) -> None:
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.verbosity = verbosity
            self._mean_target: float | None = None

        def fit(self, X, y):
            _ = _np.asarray(X)
            y_arr = _np.asarray(y, dtype=float)
            self._mean_target = float(_np.nanmean(y_arr)) if y_arr.size else 0.0
            return self

        def predict(self, X):
            if self._mean_target is None:
                raise RuntimeError("XGBRegressor must be fitted before predict")
            X_arr = _np.asarray(X)
            return _np.full((X_arr.shape[0],), self._mean_target)

    from . import core  # noqa: F401
    from . import sklearn  # noqa: F401  # Ensure sklearn shims register

    __all__ = ["XGBRegressor", "core", "sklearn", "IS_QT_XGBOOST_STUB"]
