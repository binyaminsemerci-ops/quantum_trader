#!/usr/bin/env python3
"""
_postfix_eval.py — Post-fix replay evaluation.

Reads all replay records generated after the model switch to llama-3.3-70b-versatile
(boundary: Redis stream ID 1773017591000-0, approx 00:53:11 UTC March 9, 2026),
exports them to JSONL, and prints a full evaluation report.

Run on VPS:
    /home/qt/quantum_trader_venv/bin/python /tmp/_postfix_eval.py
"""
from __future__ import annotations

import json
import os
import collections
import statistics
from pathlib import Path

import redis

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REPLAY_STREAM = "quantum:stream:exit.replay"

# Boundary: first record after service restart with llama-3.3-70b-versatile
# Service restarted at 2026-03-09 00:53:11 UTC → ms epoch 1773017591000
POSTFIX_BOUNDARY_ID = "1773017591000-0"

EXPORT_PATH = "/home/qt/quantum_trader/logs/replay/replay_postfix.jsonl"

# ── Load records ──────────────────────────────────────────────────────────────
def load_records(r: redis.Redis) -> list[dict]:
    """Read all post-fix records from the replay stream."""
    raw = r.xrange(REPLAY_STREAM, POSTFIX_BOUNDARY_ID, "+", count=10000)
    records = []
    for stream_id, fields in raw:
        rec = {k.decode(): v.decode() for k, v in fields.items()}
        rec["_stream_id"] = stream_id.decode()
        records.append(rec)
    return records


def flt(rec: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(rec.get(key, default))
    except (ValueError, TypeError):
        return default


def strf(rec: dict, key: str, default: str = "") -> str:
    return rec.get(key, default)


# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"error": "no records"}

    symbols       = collections.Counter()
    sides         = collections.Counter()
    formula_acts  = collections.Counter()
    qwen3_acts    = collections.Counter()
    live_acts     = collections.Counter()
    regret_labels = collections.Counter()
    pref_acts     = collections.Counter()

    rewards_all:          list[float] = []
    rewards_by_action:    dict[str, list[float]] = collections.defaultdict(list)
    rewards_by_symbol:    dict[str, list[float]] = collections.defaultdict(list)
    rewards_by_side:      dict[str, list[float]] = collections.defaultdict(list)

    divergence_count  = 0   # formula_action != qwen3_action
    fallback_count    = 0   # no real qwen3 call (qwen3_action == formula_action AND diverged==false by fallback)
    # NOTE: replay records don't carry qwen3_fallback flag; we detect "real" calls
    # by checking whether qwen3_action is set AND differs from formula_action (divergence)
    # or is the same. We use the audit stream cross-reference approach below.
    # Good-faith metric: count records where diverged==true
    diverged_true     = 0
    no_qwen3_response = 0   # qwen3_action missing or empty

    for rec in records:
        sym    = strf(rec, "symbol")
        side   = strf(rec, "side")
        fa     = strf(rec, "formula_action")
        qa     = strf(rec, "qwen3_action")
        la     = strf(rec, "live_action")
        div    = strf(rec, "diverged")
        reward = flt(rec, "reward")
        regret = strf(rec, "regret_label", "none")
        pref   = strf(rec, "preferred_action")

        symbols[sym]      += 1
        sides[side]       += 1
        formula_acts[fa]  += 1
        live_acts[la]     += 1
        regret_labels[regret] += 1
        pref_acts[pref]   += 1

        if qa:
            qwen3_acts[qa] += 1
        else:
            no_qwen3_response += 1

        if div.lower() == "true":
            diverged_true += 1

        if qa and fa and qa != fa:
            divergence_count += 1

        rewards_all.append(reward)
        rewards_by_action[la].append(reward)
        rewards_by_symbol[sym].append(reward)
        rewards_by_side[side].append(reward)

    def summarize(lst: list[float]) -> dict:
        if not lst:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(lst),
            "mean":  round(statistics.mean(lst), 4),
            "min":   round(min(lst), 4),
            "max":   round(max(lst), 4),
        }

    return {
        "total_records": n,
        "boundary_stream_id": POSTFIX_BOUNDARY_ID,
        "diverged_true_count": diverged_true,
        "diverged_true_rate_pct": round(100 * diverged_true / n, 2),
        "qwen3_action_mismatch_count": divergence_count,
        "qwen3_action_mismatch_rate_pct": round(100 * divergence_count / n, 2),
        "no_qwen3_response_count": no_qwen3_response,
        "no_qwen3_rate_pct": round(100 * no_qwen3_response / n, 2),
        "reward_overall": summarize(rewards_all),
        "by_symbol": {
            sym: summarize(r_list) for sym, r_list in sorted(rewards_by_symbol.items())
        },
        "by_side": {
            side: summarize(r_list) for side, r_list in sorted(rewards_by_side.items())
        },
        "by_formula_action": {
            act: {"count": cnt, "reward": summarize(rewards_by_action.get(act, []))}
            for act, cnt in formula_acts.most_common()
        },
        "by_qwen3_action": dict(qwen3_acts.most_common()),
        "by_live_action": dict(live_acts.most_common()),
        "regret_label_distribution": dict(regret_labels.most_common()),
        "preferred_action_distribution": dict(pref_acts.most_common()),
        "symbol_counts": dict(symbols.most_common()),
        "side_counts": dict(sides.most_common()),
    }


# ── DPO pair count ────────────────────────────────────────────────────────────
def count_dpo_pairs(records: list[dict]) -> int:
    """A DPO pair requires diverged=true + a clearly better preferred_action."""
    count = 0
    for rec in records:
        div  = strf(rec, "diverged").lower()
        pa   = strf(rec, "preferred_action")
        la   = strf(rec, "live_action")
        if div == "true" and pa and la and pa != la:
            count += 1
    return count


# ── Export JSONL ──────────────────────────────────────────────────────────────
def export_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(stats: dict, dpo_pairs: int, n_records: int) -> None:
    print()
    print("=" * 70)
    print("POST-FIX REPLAY EVALUATION REPORT")
    print(f"Boundary stream ID : {stats['boundary_stream_id']}")
    print(f"                   ≈ 2026-03-09 00:53:11 UTC  (llama-3.3-70b start)")
    print("=" * 70)

    print(f"\n{'Total records':40s}: {stats['total_records']}")

    ov = stats["reward_overall"]
    print(f"{'Average reward (all)':40s}: {ov['mean']:+.4f}  "
          f"[min {ov['min']:+.4f}, max {ov['max']:+.4f}]")

    print(f"\n{'Diverged (diverged=true)':40s}: "
          f"{stats['diverged_true_count']} / {stats['total_records']}  "
          f"({stats['diverged_true_rate_pct']:.1f}%)")
    print(f"{'Qwen3 action ≠ formula action':40s}: "
          f"{stats['qwen3_action_mismatch_count']} / {stats['total_records']}  "
          f"({stats['qwen3_action_mismatch_rate_pct']:.1f}%)")
    print(f"{'No Qwen3 response (empty field)':40s}: "
          f"{stats['no_qwen3_response_count']} / {stats['total_records']}  "
          f"({stats['no_qwen3_rate_pct']:.1f}%)")
    print(f"{'DPO-eligible pairs':40s}: {dpo_pairs}")

    print("\n--- Formula Action Distribution ---")
    for act, info in stats["by_formula_action"].items():
        r = info["reward"]
        print(f"  {act:25s}: {info['count']:4d}  avg_reward={r['mean']:+.4f}  "
              f"[{r['min']:+.4f}, {r['max']:+.4f}]")

    print("\n--- Qwen3 Action Distribution ---")
    for act, cnt in stats["by_qwen3_action"].items():
        print(f"  {act:25s}: {cnt:4d}")

    print("\n--- Live Action Distribution ---")
    for act, cnt in stats["by_live_action"].items():
        print(f"  {act:25s}: {cnt:4d}")

    print("\n--- Reward by Symbol ---")
    for sym, r in stats["by_symbol"].items():
        print(f"  {sym:12s}: n={r['count']:4d}  mean={r['mean']:+.4f}  "
              f"min={r['min']:+.4f}  max={r['max']:+.4f}")

    print("\n--- Reward by Side ---")
    for side, r in stats["by_side"].items():
        print(f"  {side:6s}: n={r['count']:4d}  mean={r['mean']:+.4f}  "
              f"min={r['min']:+.4f}  max={r['max']:+.4f}")

    print("\n--- Regret Label Distribution ---")
    total = sum(stats["regret_label_distribution"].values())
    for label, cnt in stats["regret_label_distribution"].items():
        pct = 100 * cnt / total if total else 0
        print(f"  {label:20s}: {cnt:4d}  ({pct:.1f}%)")

    print("\n--- Preferred Action Distribution ---")
    for act, cnt in stats["preferred_action_distribution"].items():
        print(f"  {act:25s}: {cnt:4d}")

    print("\n--- Symbol Counts ---")
    for sym, cnt in stats["symbol_counts"].items():
        print(f"  {sym:12s}: {cnt}")

    print("\n--- Sufficiency Assessment ---")
    n = stats["total_records"]
    div_rate = stats["diverged_true_rate_pct"]
    non_hold = sum(
        v["count"] for k, v in stats["by_formula_action"].items() if k != "HOLD"
    )

    prompt_ok  = n >= 150 and div_rate > 0
    dpo_ok     = dpo_pairs >= 20
    more_ok    = not dpo_ok or div_rate < 5.0

    print(f"  Prompt tuning  : {'✓ YES' if prompt_ok else '✗ NO'}  "
          f"(need ≥150 records + divergences; have {n}, div_rate={div_rate:.1f}%)")
    print(f"  DPO export     : {'✓ YES' if dpo_ok else '✗ NO'}  "
          f"(need ≥20 pairs; have {dpo_pairs})")
    print(f"  More scenarios : {'✓ YES, collect more' if more_ok else '✗ ENOUGH DATA'}")
    print(f"  Non-HOLD records (formula): {non_hold}")

    print("\n" + "=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.ping()

    print(f"Loading records from {REPLAY_STREAM} since {POSTFIX_BOUNDARY_ID} ...")
    records = load_records(r)
    print(f"Loaded {len(records)} post-fix records.")

    # Cross-check: how many of those have qwen3_action != formula_action?
    export_jsonl(records, EXPORT_PATH)
    print(f"Exported to {EXPORT_PATH}")

    stats = analyze(records)
    dpo_pairs = count_dpo_pairs(records)

    # Write JSON report alongside JSONL
    report_path = EXPORT_PATH.replace(".jsonl", "_report.json")
    with open(report_path, "w") as f:
        json.dump({**stats, "dpo_eligible_pairs": dpo_pairs}, f, indent=2)
    print(f"Report JSON → {report_path}")

    print_report(stats, dpo_pairs, len(records))


if __name__ == "__main__":
    main()
