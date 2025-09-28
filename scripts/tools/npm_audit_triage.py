#!/usr/bin/env python3
"""
Summarize npm audit JSON and produce a concise triage report.

Inputs:
  --input <path>        Path to npm audit JSON (production recommended)
  --md-out <path>       Where to write Markdown report
  --json-out <path>     Where to write machine-readable summary JSON
  --fail-on <level>     Optional: 'high' or 'critical' to set non-zero exit when >=1 found

Output files include counts by severity and top advisories per severity.
The script never requires network and can run in CI or locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SEVERITY_ORDER = ["critical", "high", "moderate", "low", "info"]


def load_audit(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize(audit: dict) -> dict:
    # npm v7+ format under auditReportVersion with vulnerabilities dict
    vulns = audit.get("vulnerabilities") or {}
    counts = {s: 0 for s in SEVERITY_ORDER}
    items = []
    for name, v in vulns.items():
        sev = v.get("severity") or "info"
        if sev not in counts:
            sev = "info"
        counts[sev] += 1
        via = v.get("via") or []
        top_via = None
        # via can be strings or objects
        for ent in via:
            if isinstance(ent, dict):
                top_via = ent
                break
        items.append(
            {
                "module": name,
                "severity": sev,
                "range": v.get("range"),
                "fixAvailable": v.get("fixAvailable"),
                "via_title": (top_via or {}).get("title"),
                "via_url": (top_via or {}).get("url"),
            }
        )
    total = sum(counts.values())
    items_sorted = sorted(
        items,
        key=lambda x: (SEVERITY_ORDER.index(x["severity"]) if x["severity"] in SEVERITY_ORDER else 99, x["module"]),
    )
    return {
        "total": total,
        "counts": counts,
        "items": items_sorted,
    }


def write_md(summary: dict, out_path: Path) -> None:
    lines = []
    lines.append("# npm audit triage (production)")
    lines.append("")
    lines.append("## Counts by severity")
    for sev in SEVERITY_ORDER:
        lines.append(f"- {sev.capitalize()}: {summary['counts'].get(sev, 0)}")
    lines.append("")
    if summary["items"]:
        lines.append("## Notable vulnerabilities")
        shown = 0
        for item in summary["items"]:
            if shown >= 20:
                lines.append("- ... (truncated) ...")
                break
            sev = item["severity"].capitalize()
            name = item["module"]
            rng = item.get("range") or "(unspecified)"
            fix = item.get("fixAvailable")
            fix_s = "fix available" if fix else "no fix yet"
            title = item.get("via_title") or ""
            url = item.get("via_url") or ""
            tail = f" — {title} {url}".strip()
            lines.append(f"- [{sev}] {name}@{rng} — {fix_s}{(' — ' + tail) if tail else ''}")
            shown += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--fail-on", choices=["high", "critical"], default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    md_out = Path(args.md_out)
    json_out = Path(args.json_out)
    audit = load_audit(inp)
    summary = summarize(audit)
    write_md(summary, md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Emit GitHub Actions outputs when possible
    counts = summary.get("counts", {})
    high = int(counts.get("high", 0))
    critical = int(counts.get("critical", 0))
    print(f"TRIAGE_HIGH={high}")
    print(f"TRIAGE_CRITICAL={critical}")
    rc = 0
    if args.fail_on == "critical" and critical > 0:
        rc = 1
    elif args.fail_on == "high" and (critical > 0 or high > 0):
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

