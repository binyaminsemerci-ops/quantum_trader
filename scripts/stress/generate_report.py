"""Generate a simple HTML report from artifacts/stress/aggregated.json

Creates artifacts/stress/report.html with a summary table and a tiny sparkline
of durations.
"""
from pathlib import Path
import os
import json

# Allow overriding output directory for testing / custom runs
ROOT = Path(__file__).resolve().parents[2]
_outdir_override = os.environ.get("STRESS_REPORT_OUTDIR")
if _outdir_override:
    outdir_path = Path(_outdir_override)
    if not outdir_path.is_absolute():
        OUTDIR = (ROOT / outdir_path).resolve()
    else:
        OUTDIR = outdir_path
else:
    OUTDIR = ROOT / "artifacts" / "stress"

AGG = OUTDIR / "aggregated.json"
OUT = OUTDIR / "report.html"

if not AGG.exists():
    print("No aggregated.json found; run harness first")
    raise SystemExit(1)

with AGG.open("r", encoding="utf-8") as fh:
    agg = json.load(fh)

runs = agg.get("runs", [])

# Prefer precomputed stats if present (added by harness/rebuild), fallback otherwise
stats = agg.get("stats") or {}
durations = [r.get("total_duration") or 0 for r in runs]
fail_py = None
fail_bt = None
if stats:
    t = stats.get("tasks", {})
    fail_py = (t.get("pytest") or {}).get("fail")
    fail_bt = (t.get("backtest") or {}).get("fail")
if fail_py is None:
    fail_py = sum(1 for r in runs if (r.get("summary", {}).get("pytest") not in (0, "skipped")))
if fail_bt is None:
    fail_bt = sum(1 for r in runs if (r.get("summary", {}).get("backtest") not in (0, "skipped")))

html = []
html.append("<html>")
html.append("<head><meta charset=\"utf-8\"><title>Stress report</title></head>")
html.append("<body>")
html.append("<h1>Stress harness report</h1>")
html.append(
    "<p>runs: %d &nbsp; pytest failures: %s &nbsp; "
    "backtest failures: %s</p>" % (len(runs), str(fail_py), str(fail_bt))
)

# stats table if available
def render_sparkline(values, width=600, height=40, stroke="#1f77b4"):
    if not values:
        return ""
    if len(values) == 1:
        x_coords = [0]
    else:
        step = width / (len(values) - 1)
        x_coords = [int(i * step) for i in range(len(values))]
    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v
    if span == 0:
        ys = [height // 2] * len(values)
    else:
        ys = [int(height - ((v - min_v) / span) * height) for v in values]
    pts = [f"{x},{y}" for x, y in zip(x_coords, ys)]
    baseline = int(height - ((0 - min_v) / span) * height) if span != 0 else height // 2
    svg = (
        f"<svg width=\"{width}\" height=\"{height}\" role=\"img\" aria-label=\"sparkline\">"
        f"<polyline fill=\"none\" stroke=\"{stroke}\" stroke-width=2 points=\"{' '.join(pts)}\"/>"
    )
    if min_v < 0 < max_v:
        svg += f"<line x1=\"0\" y1=\"{baseline}\" x2=\"{width}\" y2=\"{baseline}\" stroke=\"#ccc\" stroke-dasharray=\"4 4\"/>"
    svg += "</svg>"
    return svg


if stats:
    html.append("<h2>Summary</h2>")
    html.append("<table border=1 cellpadding=4>")
    html.append("<tr><th>task</th><th>ok</th><th>fail</th><th>skipped</th><th>error</th></tr>")
    for task, st in (stats.get("tasks") or {}).items():
        html.append(
            f"<tr><td>{task}</td><td>{st.get('ok')}</td><td>{st.get('fail')}</td><td>{st.get('skipped')}</td><td>{st.get('error')}</td></tr>"
        )
    html.append("</table>")

    # percentages
    html.append("<h3>Summary (percentages)</h3>")
    html.append("<table border=1 cellpadding=4>")
    html.append("<tr><th>task</th><th>ok %</th><th>fail %</th><th>skipped %</th><th>error %</th></tr>")
    iters = max(1, int((stats.get("iterations") or len(runs) or 1)))
    task_order = []
    for task, st in (stats.get("tasks") or {}).items():
        ok = (st.get('ok') or 0) * 100.0 / iters
        fail = (st.get('fail') or 0) * 100.0 / iters
        skipped = (st.get('skipped') or 0) * 100.0 / iters
        error = (st.get('error') or 0) * 100.0 / iters
        html.append(
            f"<tr><td>{task}</td><td>{ok:.1f}%</td><td>{fail:.1f}%</td><td>{skipped:.1f}%</td><td>{error:.1f}%</td></tr>"
        )
        task_order.append(task)
    html.append("</table>")

    # Pass-rate badges
    html.append("<h3>Pass rate</h3>")
    badge_css = (
        "display:inline-block;padding:4px 8px;margin:2px;border-radius:6px;"
        "background:#1f77b433;color:#0b3d91;font-weight:600;font-family:monospace;"
    )
    for task, st in (stats.get("tasks") or {}).items():
        rate = 0.0
        if iters:
            rate = (st.get('ok') or 0) * 100.0 / iters
        html.append(f"<span style=\"{badge_css}\">{task}: {rate:.1f}%</span>")

    # Duration histogram (10 bins)
    if durations and len(durations) > 1:
        min_d = min(durations)
        max_d = max(durations)
        span = max(max_d - min_d, 1e-9)
        bins = [0] * 10
        for d in durations:
            idx = int((d - min_d) / span * 9)
            idx = min(max(idx, 0), 9)
            bins[idx] += 1
        max_bin = max(bins) or 1
        html.append("<h3>Duration histogram</h3>")
        html.append("<div style=\"display:flex;align-items:flex-end;gap:4px;height:120px;\">")
        for i, count in enumerate(bins):
            height = int((count / max_bin) * 110) if count else 4
            html.append(
                f"<div style='background:#1f77b4;width:28px;height:{height}px;text-align:center;color:#fff;font-size:12px;' "
                f"title='bin {i+1}: {count}'>{count}</div>"
            )
        html.append("</div>")

    # sparkline per task (success=2, skipped=1, fail=0, error=-1)
    html.append("<h3>Trend (success=2, skipped=1, fail=0, error=-1)</h3>")
    html.append("<table border=1 cellpadding=4>")
    html.append("<tr><th>task</th><th>sparkline</th></tr>")
    color_map = {
        "pytest": "#2ca02c",
        "backtest": "#ff7f0e",
        "frontend_tests": "#1f77b4",
    }
    for task in task_order:
        vals = []
        for r in runs:
            s = (r.get("summary") or {}).get(task)
            if s == 0:
                vals.append(2)
            elif s == "skipped" or s is None:
                vals.append(1)
            elif s == "error":
                vals.append(-1)
            else:
                try:
                    vals.append(2 if int(s) == 0 else 0)
                except Exception:
                    vals.append(0)
        spark = render_sparkline(vals, stroke=color_map.get(task, "#1f77b4"))
        html.append(f"<tr><td>{task}</td><td>{spark}</td></tr>")
    html.append("</table>")

    # CI links if running in GitHub Actions
    srv = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if srv and repo and run_id:
        run_url = f"{srv}/{repo}/actions/runs/{run_id}"
        html.append("<h3>CI</h3>")
        html.append(f"<p>GitHub Actions run: <a href=\"{run_url}\">{run_url}</a></p>")
        html.append("<p>Artifacts: stress-artifacts, stress-report (see run page)</p>")

# simple sparkline (SVG polyline)
if durations:
    maxd = max(durations)
    pts = []
    W = 600
    H = 60
    for i, d in enumerate(durations):
        x = int(i * (W / max(1, len(durations) - 1)))
        y = int(H - (d / maxd * H) if maxd > 0 else H)
        pts.append(f"{x},{y}")
    poly = " ".join(pts)
    svg_line = (
        "<svg width=\"%d\" height=\"%d\">" % (W, H)
        + "<polyline fill=\"none\" stroke=\"#1f77b4\" stroke-width=2 points=\""
        + poly
        + "\"/></svg>"
    )
    html.append(svg_line)

# table of last 20 runs
html.append("<h2>Last runs</h2>")
html.append("<table border=1 cellpadding=4>")
html.append("<tr><th>iter</th><th>pytest</th><th>backtest</th><th>frontend</th><th>duration</th></tr>")
for r in runs[-20:]:
    i = r.get("iteration")
    s = r.get("summary", {})
    html.append(f"<tr><td>{i}</td><td>{s.get('pytest')}</td><td>{s.get('backtest')}</td><td>{s.get('frontend_tests')}</td><td>{r.get('total_duration')}</td></tr>")
html.append("</table>")
html.append("</body></html>")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as fh:
    fh.write('\n'.join(html))

print("Wrote", OUT)
