"""augment_dpo_dataset.py

Expand dpo_patch10a_v2_FROZEN.jsonl (251 pairs) to >=600 semantically
valid pairs covering all 4 exit actions.

Strategies
----------
A. Exit-score shift — move each pair's exit_score into a different
   action bracket; only keep the result if preferred_action changes
   AND new preferred != live_action.

B. TIME_STOP_EXIT synthesis — take all 'late_hold' pairs and create a
   version where the agent should have triggered a time-stop exit.
   Adds a 'hold_hours' field so the model sees a hold-duration signal.

C. Symbol / side swap — clone high-regret pairs with a different symbol
   or side to pad to TARGET without distorting the decision logic.

Output
------
  dpo_augmented_v2.jsonl   (original + augmented, shuffled)

Then re-run format_dpo_dataset.py to rebuild train/val splits.
"""
from __future__ import annotations

import copy
import itertools
import json
import pathlib
import random
from collections import Counter

SRC    = pathlib.Path("dpo_patch10a_v2_FROZEN.jsonl")
OUT    = pathlib.Path("dpo_augmented_v2.jsonl")
TARGET = 620   # aim above 600 to survive 80/20 train/val split

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "NEARUSDT",
    "XRPUSDT", "MATICUSDT",
]
SIDES = ["LONG", "SHORT"]

# Score brackets (must match format_dpo_dataset.py inference prompt)
_BRACKETS = [
    (0.00, 0.30, "HOLD"),
    (0.30, 0.55, "PARTIAL_CLOSE_25"),
    (0.55, 1.00, "FULL_CLOSE"),
]


def score_to_action(score: float) -> str:
    for lo, hi, action in _BRACKETS:
        if lo <= score < hi:
            return action
    return "FULL_CLOSE"


def main() -> None:
    random.seed(42)
    original = [json.loads(l) for l in SRC.open() if l.strip()]
    print(f"Loaded {len(original)} original pairs from {SRC}")

    out_pairs: list[dict] = list(original)
    next_id = 10000

    # ── Strategy A: exit-score shifts across brackets ─────────────────────────
    # Try ±0.15 and ±0.30 for every pair.  Only include variant when:
    #   (a) new preferred_action differs from original preferred_action
    #   (b) new preferred_action differs from live_action  (valid DPO pair)
    deltas = [-0.30, -0.15, +0.15, +0.30]
    for pair in original:
        es      = float(pair.get("exit_score") or 0.50)
        live    = pair["live_action"]
        orig_pf = pair["preferred_action"]

        for delta in deltas:
            new_score = round(min(0.90, max(0.08, es + delta)), 4)
            new_pf    = score_to_action(new_score)

            if new_pf == orig_pf:   # no change — no learning signal
                continue
            if new_pf == live:      # preferred == live — invalid DPO pair
                continue

            np_ = copy.copy(pair)
            np_["decision_id"]      = f"aug_A_{next_id}"
            np_["exit_score"]       = new_score
            np_["preferred_action"] = new_pf
            np_["formula_action"]   = new_pf  # formula agrees with preferred
            np_["regret_label"]     = "none"
            np_["source"]           = "augmented_score_shift"
            # live_action kept from original — still wrong vs new preferred
            out_pairs.append(np_)
            next_id += 1

    print(f"After strategy A: {len(out_pairs)} pairs")

    # ── Strategy B: TIME_STOP_EXIT synthesis ──────────────────────────────────
    # 'late_hold' pairs represent positions held too long → time-stop should fire.
    # Create a new pair where preferred = TIME_STOP_EXIT, live = HOLD or PARTIAL.
    late_hold = [p for p in original if p.get("regret_label") == "late_hold"]
    for pair in late_hold:
        for live_action in ["HOLD", "PARTIAL_CLOSE_25"]:
            np_ = copy.copy(pair)
            np_["decision_id"]      = f"aug_B_{next_id}"
            np_["preferred_action"] = "TIME_STOP_EXIT"
            np_["live_action"]      = live_action
            np_["regret_label"]     = "late_hold"
            np_["formula_action"]   = "HOLD"   # formula hasn't triggered exit
            # Slightly lower exit_score — formula still says hold
            np_["exit_score"]       = round(min(0.50, float(pair.get("exit_score") or 0.40) - 0.05), 4)
            np_["hold_hours"]       = random.choice([48, 72, 96, 120, 144])
            np_["source"]           = "augmented_time_stop"
            out_pairs.append(np_)
            next_id += 1

    print(f"After strategy B: {len(out_pairs)} pairs")

    # ── Strategy C: symbol / side diversity padding ───────────────────────────
    # Clone existing non-augmented pairs with a different symbol to hit TARGET.
    if len(out_pairs) < TARGET:
        shortage   = TARGET - len(out_pairs)
        candidates = [p for p in original]   # only originals as templates
        random.shuffle(candidates)
        for pair in itertools.cycle(candidates):
            if len(out_pairs) >= TARGET:
                break
            sym = random.choice([s for s in SYMBOLS if s != pair.get("symbol", "")])
            np_ = copy.copy(pair)
            np_["decision_id"] = f"aug_C_{next_id}"
            np_["symbol"]      = sym
            np_["side"]        = random.choice(SIDES)
            np_["source"]      = "augmented_sym_swap"
            out_pairs.append(np_)
            next_id += 1

    print(f"After strategy C: {len(out_pairs)} pairs")

    # ── Shuffle and write ─────────────────────────────────────────────────────
    random.shuffle(out_pairs)
    OUT.write_text("\n".join(json.dumps(p) for p in out_pairs) + "\n", encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    pref_dist = Counter(p["preferred_action"] for p in out_pairs)
    src_dist  = Counter(p.get("source", "original") for p in out_pairs)
    print(f"\nWritten {len(out_pairs)} pairs -> {OUT}")
    print("\nPreferred action distribution:")
    for action, count in sorted(pref_dist.items()):
        print(f"  {action:<22} {count:>4}  ({count/len(out_pairs)*100:.1f}%)")
    print("\nSource breakdown:")
    for src, count in sorted(src_dist.items()):
        print(f"  {src:<30} {count:>4}")


if __name__ == "__main__":
    main()
