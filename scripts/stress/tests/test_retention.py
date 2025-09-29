import os
import sys
import json
from pathlib import Path
import importlib.util


def load_retention_module():
    mod_path = Path(__file__).resolve().parents[1] / "retention.py"
    spec = importlib.util.spec_from_file_location("stress.retention", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prune_iteration_jsons_keeps_latest(tmp_path):
    mod = load_retention_module()
    # Create sample files with different mtimes
    for idx in range(5):
        f = tmp_path / f"iter_{idx:04d}.json"
        f.write_text(json.dumps({"iteration": idx}))
        os.utime(f, (idx + 1, idx + 1))

    removed = mod.prune_iteration_jsons(tmp_path, keep=2)
    assert removed == 3
    remaining = sorted(p.name for p in tmp_path.glob("iter_*.json"))
    assert remaining == ["iter_0003.json", "iter_0004.json"]


def test_retention_warns_when_threshold_exceeded(tmp_path, monkeypatch, capsys):
    mod = load_retention_module()
    for idx in range(4):
        f = tmp_path / f"iter_{idx:04d}.json"
        f.write_text("{}")
        os.utime(f, (idx + 1, idx + 1))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    called = {}

    class DummyResp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=10):
        called['req'] = req
        return DummyResp()

    monkeypatch.setenv("STRESS_PRUNE_ALERT_WEBHOOK", "https://example.com/webhook")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    monkeypatch.setattr(sys, "argv", [
        "retention.py",
        "--keep",
        "1",
        "--outdir",
        str(tmp_path),
        "--warn-over",
        "2",
    ])

    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "::warning" in out
    assert 'req' in called
