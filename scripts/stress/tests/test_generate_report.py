import json
from pathlib import Path
import importlib.util


def load_report_module():
    mod_path = Path(__file__).resolve().parents[1] / "generate_report.py"
    spec = importlib.util.spec_from_file_location("stress.generate_report", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_report_uses_override(tmp_path, monkeypatch):
    outdir = tmp_path / "custom"
    outdir.mkdir(parents=True)
    aggregated = {
        "runs": [
            {
                "iteration": 1,
                "summary": {
                    "pytest": 0,
                    "backtest": "error",
                    "frontend_tests": 0,
                },
                "total_duration": 1.5,
            },
            {
                "iteration": 2,
                "summary": {
                    "pytest": "skipped",
                    "backtest": 0,
                    "frontend_tests": 1,
                },
                "total_duration": 2.5,
            },
        ],
        "stats": {
            "iterations": 2,
            "tasks": {
                "pytest": {"ok": 1, "fail": 0, "skipped": 1, "error": 0},
                "backtest": {"ok": 1, "fail": 0, "skipped": 0, "error": 1},
                "frontend_tests": {"ok": 1, "fail": 1, "skipped": 0, "error": 0},
            },
            "duration_sec": {"min": 1.5, "max": 2.5, "avg": 2.0},
        },
    }
    (outdir / "aggregated.json").write_text(json.dumps(aggregated), encoding="utf-8")
    monkeypatch.setenv("STRESS_REPORT_OUTDIR", str(outdir))

    load_report_module()
    html = (outdir / "report.html").read_text(encoding="utf-8")        
    assert "Summary" in html
    assert "Summary (percentages)" in html
    assert "Trend" in html
    # Ensure sparkline SVG is rendered
    assert "<svg" in html
    # Expect percentage values (rough check for % symbol)
    assert "%" in html
