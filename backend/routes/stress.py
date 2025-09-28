from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter()

ROOT = Path(__file__).resolve().parents[2]


def _resolve_aggregated_path() -> Path:
    path = os.environ.get("STRESS_AGGREGATED_PATH")
    if path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
    base = os.environ.get("STRESS_ARTIFACT_DIR")
    if base:
        candidate = Path(base)
        candidate = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        return candidate / "aggregated.json"
    return ROOT / "artifacts" / "stress" / "aggregated.json"


def _normalize_counts(raw: Dict[str, Any] | None) -> Dict[str, int]:
    counts = {"ok": 0, "fail": 0, "skipped": 0, "error": 0}
    if not isinstance(raw, dict):
        return counts
    for key in counts:
        value = raw.get(key)
        if isinstance(value, (int, float)):
            counts[key] = int(value)
    return counts


def _count_task(task: str, runs: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"ok": 0, "fail": 0, "skipped": 0, "error": 0}
    for item in runs:
        summary = item.get("summary") or {}
        state = summary.get(task)
        if state == 0:
            counts["ok"] += 1
        elif state == "skipped" or state is None:
            counts["skipped"] += 1
        elif state == "error":
            counts["error"] += 1
        else:
            try:
                counts["ok" if int(state) == 0 else "fail"] += 1
            except Exception:
                counts["error"] += 1
    return counts


def _trend(task: str, runs: List[Dict[str, Any]]) -> List[int]:
    values: List[int] = []
    for item in runs:
        summary = item.get("summary") or {}
        state = summary.get(task)
        if state == 0:
            values.append(2)
        elif state == "skipped" or state is None:
            values.append(1)
        elif state == "error":
            values.append(-1)
        else:
            try:
                values.append(2 if int(state) == 0 else 0)
            except Exception:
                values.append(0)
    return values


def _load_aggregated(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@router.get("/summary")
async def stress_summary() -> Dict[str, Any]:
    agg_path = _resolve_aggregated_path()
    try:
        data = _load_aggregated(agg_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No aggregated.json at {agg_path}") from None

    runs: List[Dict[str, Any]] = data.get("runs") or []
    stats: Dict[str, Any] = data.get("stats") or {}

    duration_stats = stats.get("duration_sec") or {}
    durations = [float(r.get("total_duration")) for r in runs if r.get("total_duration") is not None]
    if not duration_stats:
        if durations:
            duration_stats = {
                "min": min(durations),
                "max": max(durations),
                "avg": sum(durations) / len(durations),
            }
        else:
            duration_stats = {"min": None, "max": None, "avg": None}
    else:
        duration_stats = {
            "min": duration_stats.get("min"),
            "max": duration_stats.get("max"),
            "avg": duration_stats.get("avg"),
        }

    iterations = stats.get("iterations") or len(runs)
    task_stats = stats.get("tasks") or {}
    task_names = list(task_stats.keys())
    if not task_names:
        task_names = sorted({name for run in runs for name in (run.get("summary") or {}).keys()})

    tasks_payload = []
    for name in task_names:
        counts = _normalize_counts(task_stats.get(name))
        if sum(counts.values()) == 0:
            counts = _count_task(name, runs)
        total = counts["ok"] + counts["fail"] + counts["skipped"] + counts["error"]
        denom = iterations or total
        pass_rate = (counts["ok"] / denom * 100.0) if denom else 0.0
        tasks_payload.append(
            {
                "name": name,
                "counts": counts,
                "pass_rate": pass_rate,
                "trend": _trend(name, runs),
            }
        )

    payload = {
        "status": "ok",
        "source": str(agg_path),
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "iterations": iterations,
        "duration": duration_stats,
        "totals": {"runs": len(runs)},
        "duration_series": durations,
        "tasks": tasks_payload,
        "recent_runs": [
            {
                "iteration": run.get("iteration"),
                "summary": run.get("summary"),
                "total_duration": run.get("total_duration"),
                "details": run.get("details"),
            }
            for run in runs[-25:]
        ],
    }
    return payload
