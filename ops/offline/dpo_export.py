"""dpo_export: build chosen/rejected DPO pairs from exit replay records.

PATCH-9 — replay consumer / evaluation engine.

DPO (Direct Preference Optimization) training requires a dataset of
(prompt, chosen_response, rejected_response) triples.  For the exit agent:

    prompt           — market context at decision time (symbol, side, exit_score,
                       hold_duration, regime signals)
    chosen_response  — the action the agent *should* have taken
                       (preferred_action from RewardEngine)
    rejected_response — the action the agent *did* take (live_action)

A pair is only emitted when ``preferred_action != live_action`` AND the
reward gap is large enough to be meaningful (configurable via
``min_reward_gap``).  Pairs where they overlap add noise and are discarded.

The output JSONL format is compatible with HuggingFace ``trl`` DPO trainer:

    {
      "prompt":   "<|system|>...<|user|>...",
      "chosen":   "<|assistant|>FULL_CLOSE — ...",
      "rejected": "<|assistant|>HOLD — ..."
    }

This module has no runtime service dependencies — safe to use offline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from ops.offline.replay_schema import ReplayRecord

_DEFAULT_MIN_REWARD_GAP = 0.2

# ── Prompt template ────────────────────────────────────────────────────────────
# Keep prompt concise — the full reasoning is encoded in the structured fields.

_SYSTEM_PROMPT = (
    "You are an AI exit agent for a crypto futures trading system.  "
    "Given the market snapshot, decide whether to close the position now, "
    "take a partial close, tighten the trailing stop, or hold."
)

_ACTION_DESCRIPTIONS: dict[str, str] = {
    "FULL_CLOSE":         "Close the entire position immediately.",
    "PARTIAL_CLOSE_25":   "Close 25% of the position to lock in partial profit.",
    "HOLD":               "Keep the position open and continue monitoring.",
    "TIGHTEN_TRAIL":      "Tighten the trailing stop to protect gains.",
    "MOVE_TO_BREAKEVEN":  "Move stop loss to breakeven to remove downside risk.",
    "TIME_STOP_EXIT":     "Exit because maximum hold time has been reached.",
    "UNKNOWN":            "Action is unclassified.",
}


def _action_desc(action: str) -> str:
    return _ACTION_DESCRIPTIONS.get(action, action)


# ── DPO record ─────────────────────────────────────────────────────────────────

@dataclass
class DpoPair:
    """A single DPO training example derived from one ReplayRecord."""
    decision_id: str
    symbol: str
    prompt: str
    chosen: str
    rejected: str
    # Metadata kept for inspection / filtering — not part of training format
    chosen_action: str
    rejected_action: str
    reward: Optional[float]
    regret_label: str
    reward_gap: float     # abs(reward_chosen - reward_rejected), proxy estimate

    def to_training_dict(self) -> dict:
        """Return only the fields needed by a DPO trainer."""
        return {
            "prompt":   self.prompt,
            "chosen":   self.chosen,
            "rejected": self.rejected,
        }

    def to_full_dict(self) -> dict:
        """Return the full record including metadata."""
        return {
            "decision_id":      self.decision_id,
            "symbol":           self.symbol,
            "prompt":           self.prompt,
            "chosen":           self.chosen,
            "rejected":         self.rejected,
            "chosen_action":    self.chosen_action,
            "rejected_action":  self.rejected_action,
            "reward":           self.reward,
            "regret_label":     self.regret_label,
            "reward_gap":       round(self.reward_gap, 4),
            "source":           "PATCH-9-dpo-export",
        }


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(rec: ReplayRecord) -> str:
    """
    Build a chat-style prompt string from a ReplayRecord.

    Uses a simple <|system|>/<|user|> template.  Numeric fields that are None
    are rendered as "unknown".
    """
    def _fmt(v, suffix="", decimals=4):
        if v is None:
            return "unknown"
        if isinstance(v, float):
            return f"{v:.{decimals}f}{suffix}"
        return f"{v}{suffix}"

    lines = [
        f"<|system|>{_SYSTEM_PROMPT}",
        "<|user|>",
        f"Symbol          : {rec.symbol}",
        f"Side            : {rec.side}",
        f"Exit score      : {_fmt(rec.exit_score)}",
        f"Entry price     : {_fmt(rec.entry_price, suffix=' USDT', decimals=2)}",
        f"Hold duration   : {_fmt(rec.hold_duration_sec, suffix=' sec')}",
        f"Formula action  : {rec.formula_action}",
        f"Qwen3 action    : {rec.qwen3_action}",
        f"Diverged        : {str(rec.diverged).lower()}",
        "What is the correct exit action?",
    ]
    return "\n".join(lines)


def _build_response(action: str) -> str:
    return f"<|assistant|>{action} — {_action_desc(action)}"


# ── Reward gap estimation ──────────────────────────────────────────────────────

def _estimate_reward_gap(rec: ReplayRecord) -> float:
    """
    Proxy reward gap between preferred_action and live_action.

    If ``reward`` is available, use it as the live_action reward and estimate
    the preferred_action reward from regret_score.  The gap is a lower-bound
    estimate; real gap needs actual counterfactual reward (not available in v1).
    """
    if rec.reward is None:
        # No reward info — use regret_score as proxy gap
        return rec.regret_score or 0.0
    # If the preferred action differs, the gap is abs(reward) + regret proxy
    regret_proxy = rec.regret_score or 0.0
    return abs(rec.reward) + regret_proxy * 0.5


# ── Builder ────────────────────────────────────────────────────────────────────

def build_dpo_dataset(
    records: Sequence[ReplayRecord],
    *,
    min_reward_gap: float = _DEFAULT_MIN_REWARD_GAP,
) -> list[DpoPair]:
    """
    Build a list of DpoPair objects from replay records.

    Eligibility criteria (all must be true):
    1. ``preferred_action != live_action`` — agent made a sub-optimal decision
    2. Estimated reward gap >= ``min_reward_gap``
    3. ``live_action`` and ``preferred_action`` are both non-UNKNOWN

    Parameters
    ----------
    records:
        Sequence of ReplayRecord objects (from load_jsonl / load_jsonl_dir).
    min_reward_gap:
        Minimum estimated reward gap to include a pair.  Lower values include
        more pairs but introduce noisier training signal.
    """
    pairs: list[DpoPair] = []

    for rec in records:
        chosen    = rec.preferred_action
        rejected  = rec.live_action

        # Filter: must be a genuine disagreement
        if chosen == rejected:
            continue
        if chosen in ("UNKNOWN", "", None) or rejected in ("UNKNOWN", "", None):
            continue

        gap = _estimate_reward_gap(rec)
        if gap < min_reward_gap:
            continue

        prompt   = _build_prompt(rec)
        pairs.append(
            DpoPair(
                decision_id=rec.decision_id,
                symbol=rec.symbol,
                prompt=prompt,
                chosen=_build_response(chosen),
                rejected=_build_response(rejected),
                chosen_action=chosen,
                rejected_action=rejected,
                reward=rec.reward,
                regret_label=rec.regret_label,
                reward_gap=gap,
            )
        )

    return pairs


# ── I/O ────────────────────────────────────────────────────────────────────────

def write_dpo_jsonl(path, pairs: list[DpoPair], *, full: bool = True) -> None:
    """
    Write DPO pairs to a JSONL file.

    Parameters
    ----------
    path:
        Output file path (created or appended).
    pairs:
        List of DpoPair objects.
    full:
        If True (default), write full dict including metadata.
        If False, write training dict only (prompt/chosen/rejected).
    """
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            record = pair.to_full_dict() if full else pair.to_training_dict()
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")
