"""Lightweight JSON cache helpers for persisting API responses."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from backend.utils.telemetry import record_cache_hit, record_cache_miss, record_cache_write
except ModuleNotFoundError:  # pragma: no cover - fallback for legacy import paths
    from utils.telemetry import record_cache_hit, record_cache_miss, record_cache_write


def load_json(path: Path) -> Optional[Any]:
    """Return cached JSON data if present and readable."""
    label = _cache_label(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        record_cache_miss(label)
        return None
    except json.JSONDecodeError:
        record_cache_miss(label)
        return None
    except Exception:
        record_cache_miss(label)
        raise
    record_cache_hit(label)
    return payload


def save_json(path: Path, payload: Any) -> None:
    """Persist JSON serialisable payload, creating parent directories as needed."""
    label = _cache_label(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    record_cache_write(label)


def _json_default(value: Any) -> Any:
    """Fallback encoder used by ``save_json`` for datetime-like objects."""
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _cache_label(path: Path) -> str:
    """Normalise cache label for telemetry."""

    return path.stem or str(path)
