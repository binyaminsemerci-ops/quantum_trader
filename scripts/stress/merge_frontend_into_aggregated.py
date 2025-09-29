"""Merge frontend_aggregated.json into artifacts/stress/aggregated.json.
Adds a top-level key 'frontend_runs' containing the frontend runs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "artifacts" / "stress"

agg_p = OUTDIR / "aggregated.json"
fe_p = OUTDIR / "frontend_aggregated.json"

if not agg_p.exists():
    print("aggregated.json not found; please run harness or rebuild_aggregated.py first")
    raise SystemExit(1)
if not fe_p.exists():
    print("frontend_aggregated.json not found; run frontend repeats first")
    raise SystemExit(1)

agg = json.loads(agg_p.read_text())
fe = json.loads(fe_p.read_text())

fe_runs = fe.get("runs", [])
# compute a small frontend summary
total = len(fe_runs)
ok = sum(1 for r in fe_runs if r.get("summary") == 0)
errors = sum(1 for r in fe_runs if r.get("summary") == "error")
skipped = sum(1 for r in fe_runs if r.get("summary") == "skipped")
avg_duration = None
durations = [r.get("duration") for r in fe_runs if isinstance(r.get("duration"), (int, float))]
if durations:
    avg_duration = sum(durations) / len(durations)

# remove old top-level keys if present
if "frontend_runs" in agg:
    del agg["frontend_runs"]
if "frontend_aggregated_at" in agg:
    del agg["frontend_aggregated_at"]

agg.setdefault("frontend", {})
agg["frontend"]["frontend_runs"] = fe_runs
agg["frontend"]["frontend_aggregated_at"] = fe.get("finished_at") or fe.get("started_at")
agg["frontend"]["frontend_summary"] = {
    "total": total,
    "ok": ok,
    "errors": errors,
    "skipped": skipped,
    "avg_duration": avg_duration,
}

agg_p.write_text(json.dumps(agg, indent=2), encoding="utf-8")
print(f"Merged {fe_p} into {agg_p}")
