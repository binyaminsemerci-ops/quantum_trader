"""Rebuild artifacts/stress/aggregated.json from per-iteration iter_*.json files.
This uses the same summarization logic as harness.main to avoid re-running tests.
"""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "artifacts" / "stress"


def _summarize_result(rr):
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


def main():
    runs = []
    files = sorted(OUTDIR.glob("iter_*.json"))
    for p in files:
        try:
            with p.open("r", encoding="utf-8") as fh:
                j = json.load(fh)
        except Exception:
            continue
        summary = {}
        for r in j.get("results", []):
            summary[r.get("name")] = _summarize_result(r.get("result"))
        runs.append({
            "iteration": j.get("iteration"),
            "summary": summary,
            "total_duration": j.get("total_duration"),
        })

    # compute stats compatible with harness
    try:
        durations = [r.get("total_duration") or 0 for r in runs]
        dmin = min(durations) if durations else None
        dmax = max(durations) if durations else None
        davg = (sum(durations) / len(durations)) if durations else None
        def _count_task(task):
            ok = fail = skipped = error = 0
            for r in runs:
                s = (r.get("summary") or {}).get(task)
                if s == 0:
                    ok += 1
                elif s == "skipped":
                    skipped += 1
                elif s == "error":
                    error += 1
                elif s is None:
                    skipped += 1
                else:
                    try:
                        if int(s) != 0:
                            fail += 1
                        else:
                            ok += 1
                    except Exception:
                        error += 1
            return {"ok": ok, "fail": fail, "skipped": skipped, "error": error}
        stats = {
            "iterations": len(runs),
            "tasks": {
                "pytest": _count_task("pytest"),
                "backtest": _count_task("backtest"),
                "frontend_tests": _count_task("frontend_tests"),
            },
            "duration_sec": {"min": dmin, "max": dmax, "avg": davg},
        }
    except Exception:
        stats = None

    aggregated = {
        "runs": runs,
        "git_hash": None,
        "started_at": None,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        **({"stats": stats} if stats is not None else {}),
    }
    out = OUTDIR / "aggregated.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(aggregated, fh, indent=2)
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
