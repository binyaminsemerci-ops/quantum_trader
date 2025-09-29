def run_frontend_repeats(count=100, start_at=1):
    """
    Run the frontend Docker test helper many times and record results.
    Creates per-run files: artifacts/stress/frontend_iter_0001.json
    and an aggregated file: artifacts/stress/frontend_aggregated.json
    """
    outdir = resolve_outdir()
    img_name = frontend_image_name()
    fe_dir = ROOT / "frontend"

    agg = {"runs": [], "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())}
    end = start_at + count - 1

    code, out = build_frontend_image(fe_dir=fe_dir, img_name=img_name)
    print('build->', code)
    if code != 0:
        print(out)
        raise SystemExit(code)

    for i in range(start_at, end + 1):
        print(f"Running frontend iteration {i}/{end}")
        start_time = time.time()
        try:
            rc, sout, serr = run_frontend_image(img_name=img_name)
            duration = time.time() - start_time
            res = {
                "cmd": ["docker", "run", "--rm", img_name],
                "returncode": rc,
                "stdout": sout,
                "stderr": serr,
                "duration": duration,
            }
        except Exception as exc:
            duration = time.time() - start_time
            res = {
                "cmd": ["docker", "run", "--rm", img_name],
                "error": str(exc),
                "duration": duration,
            }
        payload = {"iteration": i, "result": res}
        p = outdir / f"frontend_iter_{i:04d}.json"
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        summary = None
        if isinstance(res, dict):
            if "returncode" in res:
                summary = res.get("returncode")
            elif res.get("error") is not None:
                summary = "error"
            elif res.get("skipped") is not None:
                summary = "skipped"
        details = None
        if isinstance(res, dict):
            stdout = res.get("stdout") or ""
            stderr = res.get("stderr") or ""
            for line in list(stdout.splitlines()) + list(stderr.splitlines()):
                s = line.strip()
                if s:
                    details = s[:300]
                    break
            if not details and res.get("error"):
                details = str(res.get("error"))[:300]
        agg["runs"].append(
            {
                "iteration": i,
                "summary": summary,
                "details": details,
                "duration": res.get("duration"),
            }
        )

    agg["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    out_path = outdir / "frontend_aggregated.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
    print(f"Wrote {out_path}")

import subprocess
import sys
import json
import os
from pathlib import Path
import time
import random
import shutil


ROOT = Path(__file__).resolve().parents[2]


def resolve_outdir(create: bool = True) -> Path:
    raw = os.environ.get("STRESS_OUTDIR")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
    else:
        candidate = ROOT / "artifacts" / "stress"
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


OUTDIR = resolve_outdir()


def frontend_image_name() -> str:
    return os.environ.get("STRESS_FRONTEND_IMAGE", "quantum_trader_frontend_test:latest")


def frontend_dockerfile_path(fe_dir: Path) -> Path:
    override = os.environ.get("STRESS_FRONTEND_DOCKERFILE")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (ROOT / candidate).resolve()
        return candidate
    return fe_dir / "Dockerfile.test"


def frontend_build_args() -> list[str]:
    args: list[str] = []
    base_image = os.environ.get("STRESS_FRONTEND_BASE_IMAGE")
    if base_image:
        args.extend(["--build-arg", f"BASE_IMAGE={base_image}"])
    extra_deps = os.environ.get("STRESS_FRONTEND_EXTRA_NPM_DEPS")
    if extra_deps:
        args.extend(["--build-arg", f"EXTRA_NPM_DEPS={extra_deps}"])
    return args


def run_cmd(cmd, cwd=None, timeout=300, env=None, retries=0, retry_delay=1):
    """Run a subprocess command. On exceptions (like timeouts or file not
    found) retry up to `retries` times with `retry_delay` seconds between
    attempts. Returns a dict with stdout/stderr/returncode/duration or an
    error field when an exception occurred.
    """
    attempt = 0
    while True:
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd or ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            duration = time.time() - start
            return {
                "cmd": cmd,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "duration": duration,
            }
        except Exception as e:
            duration = time.time() - start
            attempt += 1
            err = {"cmd": cmd, "error": str(e), "duration": duration}
            if attempt > retries:
                return err
            time.sleep(retry_delay)


def git_commit_hash():
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
        return p.stdout.strip()
    except Exception:
        return None


def build_frontend_image(fe_dir=None, img_name=None):
    fe_dir = fe_dir or (ROOT / "frontend")
    image = img_name or frontend_image_name()
    dockerfile = frontend_dockerfile_path(fe_dir)
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        image,
    ]
    cmd.extend(frontend_build_args())
    cmd.append(str(fe_dir))
    # If image exists and DOCKER_FORCE_BUILD not set, skip build
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0 and os.environ.get("DOCKER_FORCE_BUILD") != "1":
        return 0, "image exists, skipping build\n"
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")


def run_frontend_image(img_name=None, timeout=600):
    image = img_name or frontend_image_name()
    cmd = ["docker", "run", "--rm", image]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='replace')
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def has_exe(name: str) -> bool:
    """Return True if executable `name` is available on PATH."""
    return shutil.which(name) is not None


def iteration(i, seed=None):
    now = time.time()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    out = {"iteration": i, "timestamp": ts, "seed": seed, "results": []}

    # Set reproducible environment for child processes
    env = os.environ.copy()
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
        env["STRESS_RUN_SEED"] = str(seed)

    # 1) run pytest quick
    out["results"].append(
        {
            "name": "pytest",
            "result": run_cmd(["pytest", "-q"], timeout=120, env=env),
        }
    )

    # 2) run backtest/train script (if exists)
    bt_script = ROOT / "main_train_and_backtest.py"
    if bt_script.exists():
        out["results"].append(
            {
                "name": "backtest",
                "result": run_cmd(["python", str(bt_script)], timeout=300, env=env),
            }
        )
    else:
        out["results"].append({"name": "backtest", "result": {"skipped": True}})

    # 3) frontend tests if node/npx present and node_modules available
    fe_dir = ROOT / "frontend"
    frontend_entry = {"name": "frontend_tests", "result": None}
    img_name = frontend_image_name()
    # Decide whether to use Docker for frontend tests
    prefer_docker = (
        os.environ.get("STRESS_PREFER_DOCKER") == "1"
        or sys.platform.startswith("win")
        or not has_exe("node")
        or not has_exe("npx")
    )

    if prefer_docker:
        # If docker is available, run frontend tests inside docker image
        if has_exe("docker"):
            # If main prebuilt the image it will have set DOCKER_BUILT flag in env
            # but we also handle case where build failed by checking env var
            docker_build_failed = os.environ.get("STRESS_DOCKER_BUILD_FAILED")
            if docker_build_failed:
                frontend_entry["result"] = {"error": f"docker build failed: {docker_build_failed}"}
            else:
                rc, sout, serr = run_frontend_image()
                frontend_entry["result"] = {
                    "cmd": ["docker", "run", "--rm", img_name],
                    "returncode": rc,
                    "stdout": sout,
                    "stderr": serr,
                }
        else:
            frontend_entry["result"] = {
                "skipped": True,
                "reason": "docker not found (and docker preferred)",
            }
    elif (fe_dir / "node_modules").exists():
        frontend_entry["result"] = run_cmd(
            ["npx", "vitest", "--run"],
            cwd=fe_dir,
            timeout=120,
            env=env,
        )
    else:
        frontend_entry["result"] = {
            "skipped": True,
            "reason": "frontend node_modules not present",
        }
    out["results"].append(frontend_entry)

    # compute total duration for the iteration
    try:
        out["total_duration"] = sum(
            (
                r["result"].get("duration", 0)
                if isinstance(r.get("result"), dict)
                else 0
            )
            for r in out["results"]
        )
    except Exception:
        out["total_duration"] = None

    # Save per-iteration JSON
    outpath = OUTDIR / f"iter_{i:04d}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


def main(count=1, start_at=1, resume=False, max_retries: int = 0, retry_delay: int = 1):
    global OUTDIR
    OUTDIR = resolve_outdir()
    aggregated = {
        "runs": [],
        "git_hash": git_commit_hash(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }

    def _summarize_result(rr):
        """Return a simple summary value for a command result dict:
        - integer returncode if present
        - string 'skipped' if skipped flag is present
        - string 'error' if an error field is present
        - otherwise None
        """
        if not rr:
            return None
        if isinstance(rr, dict):
            if "returncode" in rr:
                return rr.get("returncode")
            if rr.get("skipped") is not None:
                return "skipped"
            if rr.get("error") is not None:
                return "error"
        return None

    def _summary_details(rr):
        """Return a short one-line summary message for a command result dict."""
        if not rr:
            return None
        if isinstance(rr, dict):
            if rr.get("returncode") is not None:
                # try to capture first non-empty stdout/stderr line
                out = (rr.get("stdout") or "").splitlines()
                err = (rr.get("stderr") or "").splitlines()
                for line in out + err:
                    s = line.strip()
                    if s:
                        return s[:300]
                return f"returncode={rr.get('returncode')}"
            if rr.get("skipped") is not None:
                return "skipped"
            if rr.get("error") is not None:
                return str(rr.get("error"))[:300]
        return None
    i = start_at
    end = start_at + count - 1

    # If Docker is likely to be used for frontend tests, attempt a one-time build
    prebuild_docker = (
        os.environ.get("STRESS_PREFER_DOCKER") == "1"
        or sys.platform.startswith("win")
        or not has_exe("node")
        or not has_exe("npx")
    ) and has_exe("docker")
    if prebuild_docker:
        code, out = build_frontend_image()
        if code != 0:
            # record build failure in env so iterations can surface a clear error
            os.environ["STRESS_DOCKER_BUILD_FAILED"] = out
            print("Docker build failed:")
            print(out)
        else:
            print("Docker image ready")
    while i <= end:
        outpath = OUTDIR / f"iter_{i:04d}.json"
        if resume and outpath.exists():
            print(f"Skipping iteration {i} (existing result) - resume enabled")
            # load summary from file
            try:
                with open(outpath, "r", encoding="utf-8") as fh:
                    j = json.load(fh)
                    aggregated["runs"].append(
                        {
                            "iteration": i,
                            "summary": {
                                r["name"]: _summarize_result(r.get("result"))
                                for r in j["results"]
                            },
                            "summary_details": {
                                r["name"]: _summary_details(r.get("result"))
                                for r in j["results"]
                            },
                            "total_duration": j.get("total_duration"),
                        }
                    )
            except Exception:
                aggregated["runs"].append({"iteration": i, "summary": {}})
            i += 1
            continue

        print(f"Starting iteration {i}/{end}")
        # deterministic seed per iteration
        seed = random.randint(0, 2**31 - 1)
        # pass retries into specific commands via environment or call pattern
        res = iteration(i, seed=seed)
        # Normalize summary safely (returncode if present, else skipped or None)
        summary = {}
        summary_details = {}
        for r in res["results"]:
            name = r.get("name")
            rr = r.get("result")
            summary[name] = _summarize_result(rr)
            summary_details[name] = _summary_details(rr)
        aggregated["runs"].append(
            {
                "iteration": i,
                "summary": summary,
                "summary_details": summary_details,
                "total_duration": res.get("total_duration"),
            }
        )
        i += 1

    aggregated["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    # compute stats (non-breaking addition)
    try:
        durations = [r.get("total_duration") or 0 for r in aggregated.get("runs", [])]
        dmin = min(durations) if durations else None
        dmax = max(durations) if durations else None
        davg = (sum(durations) / len(durations)) if durations else None
        def _count_task(task):
            ok = fail = skipped = error = 0
            for r in aggregated.get("runs", []):
                s = (r.get("summary") or {}).get(task)
                if s == 0:
                    ok += 1
                elif s == "skipped":
                    skipped += 1
                elif s == "error":
                    error += 1
                elif s is None:
                    # treat missing as skipped
                    skipped += 1
                else:
                    # non-zero returncode treated as fail
                    try:
                        if int(s) != 0:
                            fail += 1
                        else:
                            ok += 1
                    except Exception:
                        error += 1
            return {"ok": ok, "fail": fail, "skipped": skipped, "error": error}
        aggregated["stats"] = {
            "iterations": len(aggregated.get("runs", [])),
            "tasks": {
                "pytest": _count_task("pytest"),
                "backtest": _count_task("backtest"),
                "frontend_tests": _count_task("frontend_tests"),
            },
            "duration_sec": {"min": dmin, "max": dmax, "avg": davg},
        }
    except Exception:
        pass
    # write aggregated
    with open(OUTDIR / "aggregated.json", "w", encoding="utf-8") as fh:
        json.dump(aggregated, fh, indent=2)
    print("Completed")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--start-at', type=int, default=1)
    p.add_argument(
        '--resume',
        action='store_true',
        help='skip iterations that already have results',
    )
    p.add_argument(
        '--zip-after',
        action='store_true',
        help='Create a zip archive of artifacts/stress after completion',
    )
    args = p.parse_args()
    main(args.count, start_at=args.start_at, resume=args.resume)
    if getattr(args, 'zip_after', False):
        try:
            # Lazy import to avoid hard dependency unless needed
            from scripts.stress.upload_artifacts import zip_artifacts, rotate_zip_archives  # type: ignore
            from scripts.stress.retention import prune_iteration_jsons  # type: ignore
        except Exception:
            # Fallback to relative import when executed as module
            import importlib.util
            up = ROOT / 'scripts' / 'stress' / 'upload_artifacts.py'
            spec = importlib.util.spec_from_file_location('stress.uploader', str(up))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            zip_artifacts = getattr(mod, 'zip_artifacts')
            rotate_zip_archives = getattr(mod, 'rotate_zip_archives')
            rp = ROOT / 'scripts' / 'stress' / 'retention.py'
            spec2 = importlib.util.spec_from_file_location('stress.retention', str(rp))
            mod2 = importlib.util.module_from_spec(spec2)
            assert spec2.loader is not None
            spec2.loader.exec_module(mod2)
            prune_iteration_jsons = getattr(mod2, 'prune_iteration_jsons')
        z = zip_artifacts(OUTDIR)
        print('Zip-after created archive:', z)
        keep = int(os.environ.get('STRESS_KEEP_ZIPS', '5') or '0')
        if keep > 0:
            rotate_zip_archives(OUTDIR.parent, keep=keep)
        keep_iters = int(os.environ.get('STRESS_KEEP_ITERS', '0') or '0')
        alert_threshold = int(os.environ.get('STRESS_PRUNE_ALERT_THRESHOLD', '0') or '0')
        if keep_iters > 0:
            removed = prune_iteration_jsons(OUTDIR, keep=keep_iters)
            print(f"Pruned iteration JSONs; removed={removed}, keep={keep_iters}")
            if alert_threshold > 0 and removed > alert_threshold:
                msg = (
                    f"Iteration retention pruned {removed} files which exceeds "
                    f"STRESS_PRUNE_ALERT_THRESHOLD={alert_threshold}. Investigate artifact retention configuration."
                )
                if os.environ.get('GITHUB_ACTIONS') == 'true':
                    print(f"::warning ::{msg}")
                else:
                    print(f"WARNING: {msg}")
                webhook = os.environ.get('STRESS_PRUNE_ALERT_WEBHOOK')
                if webhook:
                    try:
                        import json as _json
                        from urllib import request as _request
                        payload = _json.dumps({"text": msg}).encode('utf-8')
                        req = _request.Request(webhook, data=payload, headers={'Content-Type': 'application/json'})
                        with _request.urlopen(req, timeout=10) as resp:
                            print(f"Retention alert webhook status: {resp.status}")
                    except Exception as exc:
                        print(f"WARNING: failed to send retention alert webhook: {exc}")
