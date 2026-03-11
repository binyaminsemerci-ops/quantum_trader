"""format_dpo_dataset.py — Convert dpo_patch10a_v2_FROZEN.jsonl to
Hugging Face DPO format (prompt / chosen / rejected) compatible with
TRL DPOTrainer.

Run locally or on the GPU cloud before training:
    python format_dpo_dataset.py

Output
------
  dpo_train.jsonl   (~200 rows, 80%)
  dpo_val.jsonl     (~51 rows, 20%)

Each row:
    {"prompt": "<|system|>...<|user|>...", "chosen": "<|assistant|>...", "rejected": "<|assistant|>..."}
"""
from __future__ import annotations

import json
import pathlib
import random

# ── Paths ──────────────────────────────────────────────────────────────────────
# Use the augmented dataset produced by augment_dpo_dataset.py.
# Falls back to the frozen source if the augmented file is absent.
_AUGMENTED = pathlib.Path("dpo_augmented_v2.jsonl")
SRC      = _AUGMENTED if _AUGMENTED.exists() else pathlib.Path("dpo_patch10a_v2_FROZEN.jsonl")
OUT_DIR  = pathlib.Path("dpo_formatted")
TRAIN_F  = OUT_DIR / "dpo_train.jsonl"
VAL_F    = OUT_DIR / "dpo_val.jsonl"

OUT_DIR.mkdir(exist_ok=True)

# ── System prompt (exact copy from qwen3_layer.py) ────────────────────────────
SYSTEM_PROMPT = (
    "You are a constrained trading risk advisor operating on Binance testnet futures.\n"
    "You receive a JSON object describing one open position and the formula engine's\n"
    "exit recommendation. Your only job is to select an exit action.\n\n"
    "Allowed actions (you MUST use exactly one of these strings):\n"
    "  HOLD\n"
    "  PARTIAL_CLOSE_25\n"
    "  FULL_CLOSE\n"
    "  TIME_STOP_EXIT\n\n"
    "Rules:\n"
    "- Emergency stops are already handled upstream. You will never see urgency=EMERGENCY.\n"
    "- TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN are not available to you.\n"
    "- If uncertain, prefer HOLD (defer to the formula recommendation).\n"
    "- formula_suggestion is advisory. You may agree or override it.\n\n"
    "You MUST respond with ONLY a JSON object — no markdown, no explanation, no extra text:\n"
    '{"action": "<one of the 4 actions>", "confidence": <0.0-1.0>, "reason": "<max 120 chars>"}'
)

# ── Canned reasons for chosen/rejected responses ──────────────────────────────
_CHOSEN_REASONS = {
    ("premature_close",  "HOLD"):             "Position closed too early — insufficient hold time, wait for stronger signal",
    ("late_hold",        "FULL_CLOSE"):        "Exit score strong and hold overdue — close fully now to capture remaining profit",
    ("late_hold",        "PARTIAL_CLOSE_25"):  "Moderate exit signal with long hold — partial close to reduce exposure",
    ("late_hold",        "TIME_STOP_EXIT"):    "Position held beyond time limit — trigger time-stop to free capital",
    ("divergence_regret","FULL_CLOSE"):        "Formula recommended full close; override reverses the divergence loss",
    ("divergence_regret","PARTIAL_CLOSE_25"):  "Formula recommended partial close; reversing Qwen3 override to follow formula",
    ("none",             "PARTIAL_CLOSE_25"):  "Reward negative and exit signal moderate — partial exit de-risks position",
    ("none",             "FULL_CLOSE"):        "Reward strongly negative — exit fully to protect capital",
    ("none",             "HOLD"):              "No strong signal; hold and continue monitoring",
    ("none",             "TIME_STOP_EXIT"):    "Max hold time reached and formula neutral — exit by time stop",
}
_REJECTED_REASONS = {
    "HOLD":             "Formula says hold; deferring to formula recommendation",
    "PARTIAL_CLOSE_25": "Moderate exit score; partial close to lock some profit",
    "FULL_CLOSE":       "High exit score; closing full position",
    "TIME_STOP_EXIT":   "Max hold time reached; exiting by time stop",
}

_ACTION_CONF = {
    "HOLD": 0.65,
    "PARTIAL_CLOSE_25": 0.72,
    "FULL_CLOSE": 0.80,
    "TIME_STOP_EXIT": 0.60,
}


def _chosen_reason(regret: str, preferred: str) -> str:
    return _CHOSEN_REASONS.get((regret, preferred),
           f"{regret}: {preferred} is the better action given the outcome")


def _rejected_reason(action: str) -> str:
    return _REJECTED_REASONS.get(action, f"agent selected {action}")


def _build_user_payload(pair: dict) -> str:
    """Build a realistic inference-like JSON user message from the DPO pair.

    We reconstruct the fields the model actually sees at inference time from
    the proxy signals available in the replay record.  Missing fields are
    omitted rather than fabricated.
    """
    es = float(pair.get("exit_score") or 0.5)
    fa = pair.get("formula_action") or "HOLD"
    payload = {
        "symbol":       pair["symbol"],
        "side":         pair.get("side", "LONG"),
        "exit_score":   round(es, 4),
        "formula_suggestion": {
            "action":     fa,
            "confidence": round(min(1.0, es + 0.1), 4),
            "reason":     "formula engine output",
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def build_example(pair: dict) -> dict:
    preferred = pair["preferred_action"]
    live      = pair["live_action"]
    regret    = pair.get("regret_label", "none")

    preferred_conf = _ACTION_CONF.get(preferred, 0.70)
    live_conf      = _ACTION_CONF.get(live, 0.65)

    # Llama instruct chat template (no BOS — TRL adds it)
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{_build_user_payload(pair)}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )

    chosen = (
        f'{{"action":"{preferred}",'
        f'"confidence":{preferred_conf:.2f},'
        f'"reason":"{_chosen_reason(regret, preferred)}"}}'
    )
    rejected = (
        f'{{"action":"{live}",'
        f'"confidence":{live_conf:.2f},'
        f'"reason":"{_rejected_reason(live)}"}}'
    )

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main():
    pairs = [json.loads(l) for l in SRC.open() if l.strip()]
    print(f"Loaded {len(pairs)} DPO pairs from {SRC}")

    random.seed(42)
    random.shuffle(pairs)

    n_val   = max(40, int(len(pairs) * 0.20))
    n_train = len(pairs) - n_val
    train_p = pairs[:n_train]
    val_p   = pairs[n_train:]

    with TRAIN_F.open("w") as f:
        for p in train_p:
            f.write(json.dumps(build_example(p)) + "\n")
    with VAL_F.open("w") as f:
        for p in val_p:
            f.write(json.dumps(build_example(p)) + "\n")

    print(f"  Train: {n_train} examples → {TRAIN_F}")
    print(f"  Val:   {n_val}   examples → {VAL_F}")
    print()

    # Spot-check first example
    ex = build_example(pairs[0])
    print("── Sample pair (pair[0]) ──────────────────────────")
    print("PROMPT (last 300 chars):", ex["prompt"][-300:])
    print("CHOSEN :", ex["chosen"])
    print("REJECTED:", ex["rejected"])


if __name__ == "__main__":
    main()
