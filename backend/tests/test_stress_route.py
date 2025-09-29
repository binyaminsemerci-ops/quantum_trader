import json
from pathlib import Path

import pytest


def _write_aggregated(tmp_path: Path) -> Path:
    data = {
        "started_at": "2025-09-28T00:00:00Z",
        "finished_at": "2025-09-28T01:00:00Z",
        "runs": [
            {
                "iteration": 1,
                "summary": {
                    "pytest": 0,
                    "backtest": 0,
                    "frontend_tests": 0,
                },
                "total_duration": 45.5,
            },
            {
                "iteration": 2,
                "summary": {
                    "pytest": 1,
                    "backtest": 0,
                    "frontend_tests": "skipped",
                },
                "total_duration": 52.0,
            },
            {
                "iteration": 3,
                "summary": {
                    "pytest": 0,
                    "backtest": "error",
                    "frontend_tests": 0,
                },
                "total_duration": 40.0,
            },
        ],
        "stats": {
            "iterations": 3,
            "tasks": {
                "pytest": {"ok": 2, "fail": 1, "skipped": 0, "error": 0},
                "backtest": {"ok": 2, "fail": 0, "skipped": 0, "error": 1},
                "frontend_tests": {"ok": 2, "fail": 0, "skipped": 1, "error": 0},
            },
            "duration_sec": {"min": 40.0, "max": 52.0, "avg": (45.5 + 52.0 + 40.0) / 3},
        },
    }
    dest = tmp_path / "aggregated.json"
    dest.write_text(json.dumps(data), encoding="utf-8")
    return dest


def test_stress_summary_missing(client, tmp_path, monkeypatch):
    agg_path = tmp_path / "aggregated.json"
    monkeypatch.setenv("STRESS_AGGREGATED_PATH", str(agg_path))

    resp = client.get("/stress/summary")

    assert resp.status_code == 404


def test_stress_summary_success(client, tmp_path, monkeypatch):
    agg_path = _write_aggregated(tmp_path)
    monkeypatch.setenv("STRESS_AGGREGATED_PATH", str(agg_path))

    resp = client.get("/stress/summary")

    assert resp.status_code == 200
    payload = resp.json()

    assert payload["iterations"] == 3
    assert payload["totals"]["runs"] == 3
    assert payload["duration"]["min"] == pytest.approx(40.0)
    assert payload["duration"]["max"] == pytest.approx(52.0)
    assert payload["duration"]["avg"] == pytest.approx((45.5 + 52.0 + 40.0) / 3)

    tasks = {task["name"]: task for task in payload["tasks"]}
    assert tasks["pytest"]["counts"]["fail"] == 1
    assert tasks["pytest"]["pass_rate"] == pytest.approx((2 / 3) * 100.0)
    assert len(tasks["pytest"]["trend"]) == 3

    assert payload["duration_series"] == pytest.approx([45.5, 52.0, 40.0])
    assert len(payload["recent_runs"]) == 3
    assert payload["recent_runs"][0]["iteration"] == 1
