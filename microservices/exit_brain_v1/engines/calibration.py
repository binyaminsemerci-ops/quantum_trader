"""
Calibration — Pure math for confidence calibration.

No Redis. No IO. No side effects.

v1: Identity calibration (pass-through).
v2+: Platt scaling, temperature scaling, isotonic regression.
"""

from __future__ import annotations

import math
from typing import List

from .normalization import clamp


def identity_calibrate(probabilities: List[float]) -> List[float]:
    """
    v1 identity calibration — returns input unchanged.

    Exists as a named function so downstream code has a consistent
    interface when real calibration is added later.

    Args:
        probabilities: [sell, hold, buy] probabilities.

    Returns:
        Same probabilities, unchanged.
    """
    return list(probabilities)


def temperature_scale(logits: List[float], temperature: float = 1.0) -> List[float]:
    """
    Temperature scaling for PyTorch model logits.

    T > 1.0: softer distribution (less confident)
    T < 1.0: sharper distribution (more confident)
    T = 1.0: standard softmax (identity)

    v2 stub: temperature must be learned per model from validation data.

    Args:
        logits: Raw logit values.
        temperature: Learned temperature parameter (default 1.0 = no-op).

    Returns:
        Calibrated probability distribution.
    """
    if temperature <= 0:
        temperature = 1.0
    scaled = [l / temperature for l in logits]
    max_s = max(scaled) if scaled else 0.0
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    if total <= 0:
        n = len(scaled)
        return [1.0 / n] * n if n > 0 else []
    return [e / total for e in exps]


def platt_scale(probability: float, a: float = 1.0, b: float = 0.0) -> float:
    """
    Platt scaling (logistic calibration) for a single probability.

    calibrated = 1 / (1 + exp(a * log(p / (1-p)) + b))

    v2 stub: a, b must be fitted per model from validation data.

    Args:
        probability: Raw model probability [0, 1].
        a: Slope parameter (default 1.0 = identity).
        b: Intercept parameter (default 0.0 = no shift).

    Returns:
        Calibrated probability.
    """
    p = clamp(probability, 1e-7, 1.0 - 1e-7)
    logit = math.log(p / (1.0 - p))
    calibrated = 1.0 / (1.0 + math.exp(-(a * logit + b)))
    return clamp(calibrated)
