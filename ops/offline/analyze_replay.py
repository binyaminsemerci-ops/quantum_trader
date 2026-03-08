#!/usr/bin/env python3
"""analyze_replay: generate evaluation report from exit.replay JSONL files.

PATCH-9 — replay consumer / evaluation engine.

Usage examples
--------------
    # Analyse everything in logs/replay/
    python ops/offline/analyze_replay.py

    # Specify a custom replay directory
    python ops/offline/analyze_replay.py --replay-dir /data/replay

    # Write JSON report to a file
    python ops/offline/analyze_replay.py --out report.json

    # Also generate DPO dataset
    python ops/offline/analyze_replay.py --dpo-out logs/dpo/dpo_dataset.jsonl

    # Print summary only (no file output)
    python ops/offline/analyze_replay.py --summary-only

Safety
------
Read-only with respect to the replay files.  Only writes to --out and
--dpo-out paths if provided.  Does not connect to Redis or touch any
runtime service.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Allow `python ops/offline/analyze_replay.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.offline.replay_metrics import (  # noqa: E402
    ReplayMetrics,
    aggregate,
    load_jsonl_dir,
)
from ops.offline.dpo_export import build_dpo_dataset, write_dpo_jsonl  # noqa: E402

_DEFAULT_REPLAY_DIR = Path("logs/replay")
_DEFAULT_DPO_DIR    = Path("logs/dpo")


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_reward(rs) -> str:
    """One-line reward stats summary."""
    if rs is None or rs.count == 0:
        return "n/a"
    mean = f"{rs.mean:.4f}" if rs.mean is not None else "n/a"
    return (
        f"mean={mean}  win={rs.win_rate:.1f}%  loss={rs.loss_rate:.1f}%  "
        f"n={rs.count}"
    )


def _bar(label: str, count: int, total: int, width: int = 30) -> str:
    """ASCII progress bar for a count/total fraction."""
    filled = int(width * count / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100.0 * count / total if total else 0.0
    return f"  {label:<25} [{bar}] {count:>5} ({pct:.1f}%)"


# ── Console report ─────────────────────────────────────────────────────────────

def print_report(m: ReplayMetrics, dpo_records: int = 0) -> None:
    sep = "─" * 70

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         PATCH-9  EXIT AGENT REPLAY EVALUATION REPORT            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Generated  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Records    : {m.total_records}  (with reward: {m.records_with_reward})")
    print()

    # ── Overall reward ──────────────────────────────────────────────────────
    print(sep)
    print("  OVERALL REWARD")
    print(f"  {_fmt_reward(m.overall_reward)}")
    print()

    # ── Reward by action ────────────────────────────────────────────────────
    print(sep)
    print("  REWARD BY LIVE ACTION")
    for action, rs in sorted(m.reward_by_action.items()):
        print(f"  {action:<25} {_fmt_reward(rs)}")
    print()

    # ── Reward by side ──────────────────────────────────────────────────────
    print(sep)
    print("  REWARD BY SIDE")
    for side, rs in sorted(m.reward_by_side.items()):
        print(f"  {side:<25} {_fmt_reward(rs)}")
    print()

    # ── Reward by symbol ────────────────────────────────────────────────────
    if m.reward_by_symbol:
        print(sep)
        print("  REWARD BY SYMBOL")
        for sym, rs in sorted(m.reward_by_symbol.items(), key=lambda x: (x[1].mean or 0)):
            print(f"  {sym:<25} {_fmt_reward(rs)}")
        print()

    # ── Divergence ──────────────────────────────────────────────────────────
    print(sep)
    print("  DIVERGENCE  (Qwen3 vs formula)")
    print(f"  Diverged records   : {m.divergence_count}/{m.total_records} "
          f"({m.divergence_rate_pct:.1f}%)")
    print(f"  Qwen3 available    : {m.qwen3_available}")
    if m.qwen3_available:
        print(f"  Qwen3 agrees formula  : {m.qwen3_agrees_formula} "
              f"/ {m.qwen3_available}")
        print(f"  Qwen3 overrides formula: {m.qwen3_overrode_formula} "
              f"/ {m.qwen3_available}")
        print(f"  Qwen3 matches live    : {m.qwen3_agrees_live} "
              f"/ {m.qwen3_available}")
    print()

    # ── Regret distribution ─────────────────────────────────────────────────
    total = m.regret.total
    print(sep)
    print("  REGRET DISTRIBUTION")
    print(_bar("none",               m.regret.none,           total))
    print(_bar("late_hold",          m.regret.late_hold,      total))
    print(_bar("premature_close",    m.regret.premature_close,total))
    print(_bar("divergence_regret",  m.regret.divergence_regret, total))
    print(f"  Actionable rate    : {m.regret.actionable_rate:.1f}%")
    print(f"  Weighted severity  : {m.regret.weighted_severity:.4f}")
    print()

    # ── Preferred action ────────────────────────────────────────────────────
    print(sep)
    print("  PREFERRED ACTION FREQUENCY")
    for action, cnt in sorted(m.preferred_action_freq.items(),
                               key=lambda x: -x[1]):
        print(_bar(action, cnt, m.total_records))
    print(f"  Preferred == live  : {m.preferred_matches_live}/{m.total_records}")
    print(f"  Preferred != live  : {m.preferred_differs}/{m.total_records}  "
          f"(missed signals)")
    print()

    # ── Hold-duration bands ─────────────────────────────────────────────────
    print(sep)
    print("  HOLD DURATION BANDS")
    for band in ("<5m", "5m-1h", "1h-4h", ">4h", "unknown"):
        cnt = m.hold_duration_bands.get(band, 0)
        print(_bar(band, cnt, m.total_records))
    print()

    # ── DPO ─────────────────────────────────────────────────────────────────
    if dpo_records > 0:
        print(sep)
        print(f"  DPO DATASET        : {dpo_records} chosen/rejected pairs exported")
        print()

    # ── Prompt / policy signals ─────────────────────────────────────────────
    print(sep)
    print("  ACTION SIGNALS")
    for sig in m.signals:
        print(f"  ▶  {sig}")
    print()
    print(sep)


# ── JSON report ────────────────────────────────────────────────────────────────

def _metrics_to_dict(m: ReplayMetrics) -> dict:
    """Serialise ReplayMetrics to a JSON-compatible dict."""
    from dataclasses import fields as dc_fields
    d: dict = {}
    d["total_records"] = m.total_records
    d["records_with_reward"] = m.records_with_reward
    d["divergence_count"] = m.divergence_count
    d["divergence_rate_pct"] = m.divergence_rate_pct
    d["qwen3_available"] = m.qwen3_available
    d["qwen3_agrees_formula"] = m.qwen3_agrees_formula
    d["qwen3_overrode_formula"] = m.qwen3_overrode_formula
    d["qwen3_agrees_live"] = m.qwen3_agrees_live
    d["preferred_matches_live"] = m.preferred_matches_live
    d["preferred_differs"] = m.preferred_differs

    def _rs_dict(rs):
        if rs is None:
            return None
        return asdict(rs) | {"win_rate": rs.win_rate, "loss_rate": rs.loss_rate}

    d["overall_reward"] = _rs_dict(m.overall_reward)
    d["reward_by_action"] = {k: _rs_dict(v) for k, v in m.reward_by_action.items()}
    d["reward_by_symbol"] = {k: _rs_dict(v) for k, v in m.reward_by_symbol.items()}
    d["reward_by_side"]   = {k: _rs_dict(v) for k, v in m.reward_by_side.items()}
    d["regret"] = m.regret.to_dict()
    d["preferred_action_freq"] = m.preferred_action_freq
    d["hold_duration_bands"] = m.hold_duration_bands
    d["signals"] = m.signals
    d["generated_utc"] = datetime.now(timezone.utc).isoformat()
    return d


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_replay",
        description="Evaluate exit agent replay data and generate learning signals.",
    )
    p.add_argument(
        "--replay-dir",
        default=str(_DEFAULT_REPLAY_DIR),
        help="Directory containing replay_*.jsonl files (default: %(default)s).",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write JSON report to this path (default: print only).",
    )
    p.add_argument(
        "--dpo-out",
        default=None,
        metavar="PATH",
        help=(
            "Write DPO chosen/rejected JSONL dataset to this path "
            "(default: no DPO export)."
        ),
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="Print a short summary instead of the full report.",
    )
    p.add_argument(
        "--min-reward-gap",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help=(
            "Minimum |reward_chosen - reward_rejected| to include a DPO pair "
            "(default: %(default)s)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        print(f"ERROR: replay directory not found: {replay_dir}", file=sys.stderr)
        return 1

    print(f"Loading replay records from {replay_dir} …", file=sys.stderr)
    records, skipped = load_jsonl_dir(replay_dir)
    if skipped:
        print(f"  Skipped {skipped} malformed lines.", file=sys.stderr)

    if not records:
        print("No records found.", file=sys.stderr)
        return 1

    print(f"Loaded {len(records)} records.", file=sys.stderr)

    m = aggregate(records)

    dpo_count = 0
    if args.dpo_out:
        dpo_path = Path(args.dpo_out)
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        pairs = build_dpo_dataset(records, min_reward_gap=args.min_reward_gap)
        write_dpo_jsonl(dpo_path, pairs)
        dpo_count = len(pairs)
        print(f"DPO dataset written: {dpo_count} pairs → {dpo_path}", file=sys.stderr)

    if args.summary_only:
        r = m.overall_reward
        win = f"{r.win_rate:.1f}%" if r else "n/a"
        mean = f"{r.mean:.4f}" if r and r.mean is not None else "n/a"
        print(
            f"Records={m.total_records}  mean_reward={mean}  win_rate={win}  "
            f"divergence={m.divergence_rate_pct:.1f}%  "
            f"actionable_regret={m.regret.actionable_rate:.1f}%"
        )
    else:
        print_report(m, dpo_records=dpo_count)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = _metrics_to_dict(m)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report written → {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
