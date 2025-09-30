import json
from pathlib import Path
import types
import shutil


def test_run_frontend_repeats_monkeypatched(tmp_path, monkeypatch):
    """Monkeypatch the build/run functions so we don't need Docker, then run once and
    validate the aggregated JSON schema is produced under artifacts/stress.
    """
    repo_root = Path(__file__).resolve().parents[3]
    outdir = repo_root / "artifacts" / "stress"
    # ensure clean
    if outdir.exists():
        for f in outdir.iterdir():
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
    else:
        outdir.mkdir(parents=True)

    # load harness module by path (harness.py lives in scripts/stress)
    harness_path = Path(__file__).resolve().parents[1] / "harness.py"
    spec_name = "scripts.stress.harness.test"
    import importlib.util
    spec = importlib.util.spec_from_file_location(spec_name, str(harness_path))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    # monkeypatch build_frontend_image and run_frontend_image
    def fake_build(fe_dir=None, img_name="quantum_trader_frontend_test:latest"):
        return 0, "built (fake)"

    def fake_run(img_name="quantum_trader_frontend_test:latest", timeout=600):
        return 0, "fake-stdout\nOK", ""

    monkeypatch.setattr(harness, "build_frontend_image", fake_build)
    monkeypatch.setattr(harness, "run_frontend_image", fake_run)

    # run one iteration
    harness.run_frontend_repeats(count=1, start_at=1)

    agg = outdir / "frontend_aggregated.json"
    assert agg.exists(), "aggregated file should be created"
    j = json.loads(agg.read_text(encoding="utf-8"))
    assert "runs" in j and isinstance(j["runs"], list)
    assert len(j["runs"]) == 1
    r = j["runs"][0]
    assert r.get("iteration") == 1
    assert "summary" in r
    # details should be short string or None
    assert isinstance(r.get("details"), (str, type(None)))
