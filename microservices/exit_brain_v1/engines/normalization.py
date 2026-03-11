"""
Normalization — Pure math for probability normalization.

No Redis. No IO. No side effects.
Input: raw model outputs. Output: normalized probabilities.
"""

from __future__ import annotations

import math
from typing import List, Tuple

# Tolerance for probability sum checks
PROBA_SUM_TOLERANCE = 0.02


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def renormalize_probabilities(probs: List[float]) -> List[float]:
    """
    Renormalize a list of probabilities to sum to 1.0.

    If all values are zero or negative, returns uniform distribution.

    Args:
        probs: Raw probability values (may not sum to 1).

    Returns:
        List of probabilities summing to 1.0.
    """
    clamped = [max(0.0, p) for p in probs]
    total = sum(clamped)
    if total <= 0:
        n = len(clamped)
        return [1.0 / n] * n if n > 0 else []
    return [p / total for p in clamped]


def softmax(logits: List[float]) -> List[float]:
    """
    Numerically stable softmax.

    Args:
        logits: Raw logit values from model output.

    Returns:
        Probability distribution summing to 1.0.
    """
    if not logits:
        return []
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def reconstruct_probabilities_from_action(
    action: str,
    confidence: float,
    hold_share: float = 0.6,
) -> Tuple[float, float, float]:
    """
    Reconstruct [sell, hold, buy] probabilities from action + confidence.

    This is the v1 MVP fallback when models only expose action + confidence
    instead of full probability vectors.

    Args:
        action: "SELL", "HOLD", or "BUY".
        confidence: Model confidence in the predicted action [0, 1].
        hold_share: Fraction of residual assigned to HOLD vs opposite action.

    Returns:
        Tuple of (sell_probability, hold_probability, buy_probability).
    """
    conf = clamp(confidence, 0.01, 0.99)
    residual = 1.0 - conf

    if action == "SELL":
        sell_p = conf
        hold_p = residual * hold_share
        buy_p = residual * (1.0 - hold_share)
    elif action == "BUY":
        buy_p = conf
        hold_p = residual * hold_share
        sell_p = residual * (1.0 - hold_share)
    else:  # HOLD
        hold_p = conf
        sell_p = residual * 0.5
        buy_p = residual * 0.5

    return (sell_p, hold_p, buy_p)


def derive_directional_probabilities(
    sell_p: float,
    hold_p: float,
    buy_p: float,
    side: str,
) -> Tuple[float, float]:
    """
    Derive continuation and reversal probability from class probas + side.

    For LONG: continuation = hold + buy, reversal = sell
    For SHORT: continuation = hold + sell, reversal = buy

    Args:
        sell_p: Probability of SELL class.
        hold_p: Probability of HOLD class.
        buy_p: Probability of BUY class.
        side: "LONG" or "SHORT".

    Returns:
        Tuple of (continuation_probability, reversal_probability).
    """
    if side == "LONG":
        continuation = hold_p + buy_p
        reversal = sell_p
    else:  # SHORT
        continuation = hold_p + sell_p
        reversal = buy_p
    return (clamp(continuation), clamp(reversal))
