#!/usr/bin/env python3
"""export_replay: export quantum:stream:exit.replay to JSONL files.

PATCH-8D — offline replay export.

Usage examples
--------------
    # Export the 1000 most recent records
    python ops/offline/export_replay.py --count 1000

    # Export a full date range (wall-clock time, inclusive)
    python ops/offline/export_replay.py --start 2026-03-01 --end 2026-03-09

    # Export from a specific Redis stream ID to end of stream
    python ops/offline/export_replay.py --start-id 1741200000000-0

    # Dry-run: print JSONL to stdout, write no files
    python ops/offline/export_replay.py --count 50 --dry-run

Output
------
``logs/replay/replay_YYYY-MM-DD.jsonl`` — one JSON object per line, grouped by
UTC day derived from ``record_time_epoch``.  If that field is absent the
record falls back to today's date file.

Safety
------
This script is strictly read-only with respect to Redis.  It calls only
XRANGE / XREVRANGE and never mutates any key, stream, or hash.  It has no
effect whatsoever on the running exit_management_agent service.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis.asyncio as aioredis

# Allow `python ops/offline/export_replay.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ops.offline.replay_schema import ReplayRecord  # noqa: E402

_log = logging.getLogger("ops.offline.export_replay")

_REPLAY_STREAM: str = "quantum:stream:exit.replay"
_DEFAULT_OUT_DIR: Path = Path("logs/replay")
_XRANGE_BATCH: int = 200  # entries per XRANGE/XREVRANGE page


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="export_replay",
        description="Export quantum:stream:exit.replay to JSONL.",
    )
    p.add_argument(
        "--stream",
        default=_REPLAY_STREAM,
        help="Redis stream key (default: %(default)s)",
    )
    p.add_argument(
        "--redis-url",
        default=None,
        help=(
            "Redis connection URL, e.g. redis://localhost:6379.  "
            "Falls back to the REDIS_URL environment variable, then to "
            "redis://localhost:6379."
        ),
    )
    # Mutually exclusive selection modes
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="Export the N most recent records (output is chronological).",
    )
    mode.add_argument(
        "--start",
        metavar="YYYY-MM-DD",
        help="Export from this date (inclusive).  Combine with --end for a range.",
    )
    p.add_argument(
        "--end",
        metavar="YYYY-MM-DD",
        help="End date inclusive (only valid together with --start).",
    )
    p.add_argument(
        "--start-id",
        default="-",
        metavar="ID",
        help="Start from this Redis stream ID when using range mode (default: beginning).",
    )
    p.add_argument(
        "--end-id",
        default="+",
        metavar="ID",
        help="End at this Redis stream ID when using range mode (default: end of stream).",
    )
    p.add_argument(
        "--out-dir",
        default=str(_DEFAULT_OUT_DIR),
        help="Output directory for JSONL files (default: %(default)s).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSONL records to stdout; do not write any files.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


# ── Date / stream-ID helpers ─────────────────────────────────────────────────────

def _date_to_epoch_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to milliseconds since epoch (start of day, UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_stream_id(ms: int) -> str:
    """Format an epoch-millisecond value as a Redis stream ID lower bound."""
    return f"{ms}-0"


def _parse_stream_id(raw) -> tuple[int, int]:
    """Decode a Redis stream ID (bytes or str) to (timestamp_ms, sequence) ints."""
    sid = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    parts = sid.split("-", 1)
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


# ── Redis fetch ──────────────────────────────────────────────────────────────────

async def _fetch_by_count(r, stream: str, count: int) -> list[tuple]:
    """Return the *count* most-recent entries in ascending (chronological) order."""
    entries = await r.xrevrange(stream, "+", "-", count=count)
    return list(reversed(entries))


async def _fetch_by_range(r, stream: str, start_id: str, end_id: str) -> list[tuple]:
    """
    Paginate XRANGE from *start_id* to *end_id*, returning all matched entries.
    Handles streams larger than _XRANGE_BATCH without loading everything in one
    round-trip.
    """
    entries: list[tuple] = []
    cursor = start_id
    while True:
        batch = await r.xrange(stream, cursor, end_id, count=_XRANGE_BATCH)
        if not batch:
            break
        entries.extend(batch)
        if len(batch) < _XRANGE_BATCH:
            break
        ts, seq = _parse_stream_id(batch[-1][0])
        cursor = f"{ts}-{seq + 1}"
    return entries


# ── File I/O ─────────────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, records: list[ReplayRecord]) -> None:
    """
    Append *records* to a JSONL file at *path* (creates file if it does not
    exist).  One JSON object per line, UTF-8 encoded.
    """
    with open(path, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json_line())
            fh.write("\n")


# ── Main export procedure ────────────────────────────────────────────────────────

async def export(
    *,
    redis_url: str,
    stream: str,
    start_id: str,
    end_id: str,
    count: int | None,
    out_dir: Path,
    dry_run: bool,
) -> int:
    """
    Read entries from *stream* and write to JSONL files in *out_dir*.

    Returns the total number of records exported.

    This coroutine never publishes to any Redis key.
    """
    r = aioredis.from_url(redis_url, decode_responses=False)
    try:
        if count is not None:
            raw_entries = await _fetch_by_count(r, stream, count)
        else:
            raw_entries = await _fetch_by_range(r, stream, start_id, end_id)
    finally:
        await r.aclose()

    _log.info("Retrieved %d entries from %s", len(raw_entries), stream)

    if not raw_entries:
        _log.warning("No records found in the specified range.")
        return 0

    records = [
        ReplayRecord.from_redis_entry(entry_id, fields)
        for entry_id, fields in raw_entries
    ]

    if dry_run:
        for rec in records:
            print(rec.to_json_line())
        return len(records)

    # Group by UTC day derived from record_time_epoch (fall back to today)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    by_date: dict[str, list[ReplayRecord]] = {}
    for rec in records:
        if rec.record_time_epoch:
            dt = datetime.fromtimestamp(rec.record_time_epoch, tz=timezone.utc)
            day = dt.strftime("%Y-%m-%d")
        else:
            day = today
        by_date.setdefault(day, []).append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for day, day_records in sorted(by_date.items()):
        out_path = out_dir / f"replay_{day}.jsonl"
        _write_jsonl(out_path, day_records)
        total += len(day_records)
        _log.info("Wrote %d records → %s", len(day_records), out_path)

    return total


# ── Entry point ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Resolve Redis URL: CLI arg > REDIS_URL env var > localhost default
    redis_url = (
        args.redis_url
        or os.environ.get("REDIS_URL", "redis://localhost:6379")
    )

    # Translate --start / --end date strings into Redis stream IDs
    start_id = args.start_id
    end_id = args.end_id

    if args.start:
        start_id = _ms_to_stream_id(_date_to_epoch_ms(args.start))
        if args.end:
            # Cover the full end-day (midnight-to-midnight, UTC)
            end_id = _ms_to_stream_id(
                _date_to_epoch_ms(args.end) + 86_400_000 - 1
            )
    elif args.end:
        parser.error("--end requires --start")

    exported = asyncio.run(
        export(
            redis_url=redis_url,
            stream=args.stream,
            start_id=start_id,
            end_id=end_id,
            count=args.count,
            out_dir=Path(args.out_dir),
            dry_run=args.dry_run,
        )
    )

    print(f"Exported {exported} records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
