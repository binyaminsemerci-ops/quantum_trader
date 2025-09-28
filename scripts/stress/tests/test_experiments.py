
import importlib.util
import sys
from pathlib import Path


def _load_experiments_module():
    module_path = Path(__file__).resolve().parents[1] / "experiments.py"
    spec = importlib.util.spec_from_file_location("scripts.stress.experiments._test", str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_normalize_extra_deps_variants():
    experiments = _load_experiments_module()
    assert experiments.normalize_extra_deps(["playwright", "@types/node"]) == "playwright @types/node"
    assert experiments.normalize_extra_deps("  vitest ") == "vitest"
    assert experiments.normalize_extra_deps(None) == ""


def test_build_experiments_matrix_generates_unique_envs():
    experiments = _load_experiments_module()
    config = {
        "node_images": ["node:18-bullseye-slim", "node:20-bullseye-slim"],
        "extra_npm_deps": ["", "playwright@1.39.0"],
        "count": 2,
        "prefer_docker": True,
    }
    matrix = experiments.build_experiments(config, override_count=None)
    assert len(matrix) == 4
    image_tags = {exp.env["STRESS_FRONTEND_IMAGE"] for exp in matrix}
    assert len(image_tags) == 4
    for exp in matrix:
        assert exp.count == 2
        assert exp.env.get("STRESS_PREFER_DOCKER") == "1"
        assert exp.env["STRESS_OUTDIR"].startswith("artifacts/stress/experiments/")


def test_build_experiments_explicit_respects_override_count():
    experiments = _load_experiments_module()
    config = {
        "experiments": [
            {
                "name": "custom-node",
                "frontend_base_image": "node:22-alpine",
                "extra_npm_deps": ["vitest"],
                "count": 5,
            }
        ],
        "count": 1,
    }
    result = experiments.build_experiments(config, override_count=3)
    assert len(result) == 1
    exp = result[0]
    assert exp.name == "custom-node"
    assert exp.count == 3  # override should win
    assert exp.env["STRESS_FRONTEND_BASE_IMAGE"] == "node:22-alpine"
    assert exp.env["STRESS_FRONTEND_EXTRA_NPM_DEPS"] == "vitest"
    assert exp.env["STRESS_OUTDIR"].endswith("/custom-node")
