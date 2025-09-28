"""Run stress harness experiments across node image / dependency matrices.

This helper automates running `harness.py` multiple times with different
frontend Docker base images and optional extra npm dependencies. Results are
written under `artifacts/stress/experiments/<name>` and summarised in
`artifacts/stress/experiments/index.json`.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "stress" / "harness.py"
DEFAULT_CONFIG = {
    "node_images": [
        "node:20-bullseye-slim",
    ],
    "extra_npm_deps": [""],
    "count": 1,
    "prefer_docker": True,
}


@dataclass
class Experiment:
    name: str
    env: Dict[str, str]
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def slugify(value: str, fallback: str = "default") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return cleaned or fallback


def normalize_extra_deps(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Iterable):
        parts: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
    return str(value)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return DEFAULT_CONFIG.copy()
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_str_path(value: str) -> str:
    return value.replace("\\", "/")


def build_experiments(config: Dict[str, Any], override_count: Optional[int]) -> List[Experiment]:
    experiments: List[Experiment] = []
    base_env: Dict[str, str] = {str(k): str(v) for k, v in (config.get("env") or {}).items()}
    prefer_docker_default = bool(config.get("prefer_docker", True))
    default_count = override_count or int(config.get("count", 1))

    explicit = config.get("experiments") or []
    if explicit:
        for item in explicit:
            name = item.get("name")
            if not name:
                raise ValueError("Experiment entries must define a 'name'.")
            env = base_env.copy()
            experiment_meta: Dict[str, Any] = {}

            node_image = item.get("frontend_base_image")
            if node_image:
                env["STRESS_FRONTEND_BASE_IMAGE"] = str(node_image)
                experiment_meta["frontend_base_image"] = str(node_image)

            extra_deps = normalize_extra_deps(item.get("extra_npm_deps"))
            if extra_deps:
                env["STRESS_FRONTEND_EXTRA_NPM_DEPS"] = extra_deps
            experiment_meta["extra_npm_deps"] = extra_deps

            env.update({str(k): str(v) for k, v in (item.get("env") or {}).items()})

            slug = slugify(name)
            env.setdefault("STRESS_FRONTEND_IMAGE", f"quantum_trader_frontend_test:{slug}")
            env.setdefault("STRESS_OUTDIR", ensure_str_path(str(Path("artifacts/stress/experiments") / slug)))
            if item.get("prefer_docker", prefer_docker_default):
                env.setdefault("STRESS_PREFER_DOCKER", "1")

            count = override_count or int(item.get("count", default_count))
            experiments.append(Experiment(name=name, env=env, count=count, metadata=experiment_meta))
        return experiments

    node_images = config.get("node_images") or [None]
    extra_deps_options = config.get("extra_npm_deps") or [""]
    combos = list(itertools.product(node_images, extra_deps_options))
    for node_image, extra in combos:
        env = base_env.copy()
        parts: List[str] = []
        experiment_meta: Dict[str, Any] = {}

        if node_image:
            node_image_str = str(node_image)
            env["STRESS_FRONTEND_BASE_IMAGE"] = node_image_str
            experiment_meta["frontend_base_image"] = node_image_str
            parts.append(f"node-{slugify(node_image_str)}")
        else:
            parts.append("node-default")

        extra_deps = normalize_extra_deps(extra)
        if extra_deps:
            env["STRESS_FRONTEND_EXTRA_NPM_DEPS"] = extra_deps
            parts.append(f"deps-{slugify(extra_deps, 'custom')}")
        else:
            parts.append("deps-baseline")
        experiment_meta["extra_npm_deps"] = extra_deps

        slug = "__".join(parts)
        env.setdefault("STRESS_FRONTEND_IMAGE", f"quantum_trader_frontend_test:{slug}")
        env.setdefault("STRESS_OUTDIR", ensure_str_path(str(Path("artifacts/stress/experiments") / slug)))
        if prefer_docker_default:
            env.setdefault("STRESS_PREFER_DOCKER", "1")

        count = default_count
        experiments.append(Experiment(name=slug, env=env, count=count, metadata=experiment_meta))
    return experiments


def summarize_experiment(outdir: Path) -> Dict[str, Any]:
    aggregated = outdir / "aggregated.json"
    if not aggregated.exists():
        raise FileNotFoundError(f"Missing aggregated.json in {outdir}")
    with aggregated.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    stats = data.get("stats") or {}
    tasks = {}
    iterations = stats.get("iterations") or len(data.get("runs") or [])
    raw_tasks = stats.get("tasks") or {}
    for task_name, counts in raw_tasks.items():
        ok = int(counts.get("ok") or 0)
        denom = iterations or max(1, ok + int(counts.get("fail") or 0) + int(counts.get("error") or 0) + int(counts.get("skipped") or 0))
        pass_rate = (ok * 100.0 / denom) if denom else 0.0
        tasks[task_name] = {
            "counts": counts,
            "pass_rate": pass_rate,
        }
    return {
        "finished_at": data.get("finished_at"),
        "iterations": iterations,
        "duration": stats.get("duration_sec") or {},
        "tasks": tasks,
        "runs": len(data.get("runs") or []),
    }


def run_experiment(exp: Experiment, dry_run: bool = False) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(exp.env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("STRESS_EXPERIMENT", exp.name)

    outdir_value = env.get("STRESS_OUTDIR", "artifacts/stress")
    outdir = Path(outdir_value)
    if not outdir.is_absolute():
        outdir = (ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(HARNESS), "--count", str(exp.count)]
    print(f"\n==> Running {exp.name} (count={exp.count})")
    print("    docker image:", env.get("STRESS_FRONTEND_IMAGE"))
    print("    base image:", env.get("STRESS_FRONTEND_BASE_IMAGE", "<default>"))
    if env.get("STRESS_FRONTEND_EXTRA_NPM_DEPS"):
        print("    extra npm deps:", env["STRESS_FRONTEND_EXTRA_NPM_DEPS"])
    if dry_run:
        print("    [dry-run] would execute:", " ".join(cmd))
        return {
            "name": exp.name,
            "status": "skipped",
            "metadata": exp.metadata,
            "env": exp.env,
            "count": exp.count,
        }

    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    duration = time.time() - start
    outdir_str = str(outdir.relative_to(ROOT)) if str(outdir).startswith(str(ROOT)) else str(outdir)
    result: Dict[str, Any] = {
        "name": exp.name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "seconds": duration,
        "metadata": exp.metadata,
        "env": exp.env,
        "count": exp.count,
        "outdir": outdir_str,
    }
    if proc.returncode == 0:
        try:
            summary = summarize_experiment(outdir)
            result.update(summary)
        except FileNotFoundError as exc:
            result["status"] = "missing-artifacts"
            result["error"] = str(exc)
    else:
        result["error"] = f"harness exited with code {proc.returncode}"
    return result


def write_summary(results: List[Dict[str, Any]]) -> None:
    experiments_dir = ROOT / "artifacts" / "stress" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    index_path = experiments_dir / "index.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": results,
    }
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSummary written to {index_path.relative_to(ROOT)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run stress harness experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "stress" / "experiments.json"),
        help="Path to experiments JSON config",
    )
    parser.add_argument("--count", type=int, default=None, help="Override iteration count for every experiment")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing harness")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    experiments = build_experiments(config, args.count)
    if not experiments:
        print("No experiments configured.")
        return 0

    results = []
    for exp in experiments:
        try:
            result = run_experiment(exp, dry_run=args.dry_run)
        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "name": exp.name,
                "status": "error",
                "error": str(exc),
                "metadata": exp.metadata,
                "env": exp.env,
                "count": exp.count,
            }
        results.append(result)

    if not args.dry_run:
        write_summary(results)
    else:
        print("\nDry run complete; no summary written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
