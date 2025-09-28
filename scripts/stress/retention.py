#!/usr/bin/env python3
"""Retention helper for stress artifacts.

Prunes old per-iteration JSON files to keep workspace tidy.

By default it targets both harness iteration files (iter_*.json)
and legacy frontend iterations (frontend_iter_*.json).

Usage:
  python scripts/stress/retention.py --keep 200
  python scripts/stress/retention.py --keep 50 --outdir artifacts/stress
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List
import os


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = ROOT / "artifacts" / "stress"


def prune_iteration_jsons(outdir: Path, keep: int, include_frontend: bool = True) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    patterns: List[str] = ["iter_*.json"]
    if include_frontend:
        patterns.append("frontend_iter_*.json")

    files: List[Path] = []
    for pat in patterns:
        files.extend(outdir.glob(pat))

    if keep <= 0:
        return 0

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for f in files[keep:]:
        try:
            f.unlink()
            removed += 1
        except Exception:
            # ignore failures silently to be safe in CI
            pass
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, required=True, help="Keep only the latest N iteration JSON files")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--no-frontend", action="store_true", help="Do not include legacy frontend_iter_*.json in pruning")
    ap.add_argument("--warn-over", type=int, default=0, help="Emit warning if more than this many files are removed")
    args = ap.parse_args()

    removed = prune_iteration_jsons(Path(args.outdir), args.keep, include_frontend=not args.no_frontend)
    print(f"Removed {removed} old iteration JSON files")
    if args.warn_over and removed > args.warn_over:
        msg = (
            f"Retention removed {removed} files which exceeds warn threshold ({args.warn_over})."
        )
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning ::{msg}")
        else:
            print(f"WARNING: {msg}")
        webhook = os.environ.get("STRESS_PRUNE_ALERT_WEBHOOK")
        if webhook:
            try:
                import json as _json
                from urllib import request as _request
                payload = _json.dumps({"text": msg}).encode("utf-8")
                req = _request.Request(webhook, data=payload, headers={"Content-Type": "application/json"})
                with _request.urlopen(req, timeout=10) as resp:
                    print(f"Retention alert webhook status: {resp.status}")
            except Exception as exc:
                print(f"WARNING: failed to send retention alert webhook: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
