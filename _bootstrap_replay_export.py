"""
Bootstrap: extract quantum:stream:exit.audit from VPS Redis and emit as
ReplayRecord-compatible JSONL for PATCH-9 analysis.

Run on VPS:
    python3 /tmp/_bootstrap_replay_export.py

Output: one JSON line per audit record, fields mapped to ReplayRecord schema.

Proxy notes (pre-PATCH-8C data):
- qwen3_action = "null"  (not deployed yet)
- diverged = live_action != formula_action
- reward = None  (no outcome data)
- regret_label = "divergence_regret" if diverged else "none"
- regret_score = None
- preferred_action = formula_action (formula is algorithmic baseline)
- hold_duration_sec = None (not tracked in audit stream)
"""
import json
import sys

try:
    import redis
except ImportError:
    sys.stderr.write("redis-py not available\n")
    sys.exit(1)

r = redis.Redis()

STREAM = "quantum:stream:exit.audit"
COUNT = 10000


def dec(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v) if v is not None else ""


def opt_float(v: str):
    if not v or v in ("null", "None", "nan", ""):
        return None
    try:
        return float(v)
    except Exception:
        return None


def opt_int(v: str):
    f = opt_float(v)
    return int(f) if f is not None else None


# Ordered list of action strengths for divergence direction
_ACTION_RANK = {
    "FULL_CLOSE":        5,
    "TIME_STOP_EXIT":    4,
    "PARTIAL_CLOSE_25":  3,
    "TIGHTEN_TRAIL":     2,
    "MOVE_TO_BREAKEVEN": 1,
    "HOLD":              0,
    "UNKNOWN":          -1,
}


def action_rank(a: str) -> int:
    return _ACTION_RANK.get(a, -1)


def main():
    # Print stream metadata as comment lines (load_jsonl will skip these as bad JSON)
    total = r.xlen(STREAM)
    sys.stderr.write(f"Stream: {STREAM}\n")
    sys.stderr.write(f"Total entries in stream: {total}\n")
    sys.stderr.write(f"Extracting up to {COUNT} most recent entries...\n")

    entries = r.xrevrange(STREAM, count=COUNT)
    if not entries:
        sys.stderr.write("No entries found.\n")
        return

    # Reverse to chronological order
    entries = list(reversed(entries))

    # Date range info
    first_sid = dec(entries[0][0])
    last_sid  = dec(entries[-1][0])

    first_ts = opt_int(first_sid.split("-")[0])
    last_ts  = opt_int(last_sid.split("-")[0])
    if first_ts:
        first_ts = first_ts // 1000  # ms → sec
    if last_ts:
        last_ts = last_ts // 1000

    sys.stderr.write(
        f"Date range: epoch {first_ts} → {last_ts}  "
        f"({len(entries)} records)\n"
    )

    emitted = 0
    for sid, fields in entries:
        sid_s = dec(sid)
        f = {dec(k): dec(v) for k, v in fields.items()}

        live    = f.get("action", "UNKNOWN").strip()
        formula = f.get("formula_action", "").strip()

        # Normalise formula: treat empty/null as same as live (no override)
        if not formula or formula in ("null", "None", "UNKNOWN"):
            formula = live

        diverged = live != formula

        # preferred_action: formula is the algorithmic baseline
        preferred = formula

        rec = {
            "stream_id":          sid_s,
            "decision_id":        f.get("loop_id", sid_s),
            "symbol":             f.get("symbol", "UNKNOWN"),
            "record_time_epoch":  opt_int(f.get("ts", "")),
            "patch":              f.get("patch", "PATCH-7A"),
            "source":             f.get("source", "exit_management_agent"),
            "live_action":        live,
            "formula_action":     formula,
            "qwen3_action":       "null",
            "diverged":           str(diverged).lower(),
            "exit_score":         f.get("exit_score", ""),
            "entry_price":        f.get("entry_price", ""),
            "side":               f.get("side", "UNKNOWN"),
            "quantity":           f.get("quantity", ""),
            "hold_duration_sec":  "",
            "close_price":        f.get("mark_price", ""),
            "closed_by":          "exit_management_agent",
            "outcome_action":     live,
            "reward":             "",
            "regret_label":       "divergence_regret" if diverged else "none",
            "regret_score":       "",
            "preferred_action":   preferred,
        }
        print(json.dumps(rec))
        emitted += 1

    sys.stderr.write(f"Emitted: {emitted} records\n")


if __name__ == "__main__":
    main()
