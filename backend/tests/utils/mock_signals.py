"""Mock signal generator for tests and demo pages.

This module lives under tests/ so it is not part of production imports. Use
this generator in tests and local demos only.
"""

from typing import List, Dict, Literal
import datetime
import random


def generate_mock_signals(
    count: int, profile: Literal["left", "right", "mixed"]
) -> List[Dict]:
    """Generate deterministic mock signals for tests/demos.

    Note: uses a deterministic `random.Random(42)` generator for reproducible
    output. This is intentionally non-cryptographic and only used for testing.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
    now = datetime.datetime.now(datetime.timezone.utc)
    signals: List[Dict] = []
    # Deterministic generator for tests/demos; not security-sensitive.
    rnd = random.Random(42)  # nosec B311 - deterministic mock generator

    for i in range(count):
        seconds_ago = (count - i) * 15
        ts = now - datetime.timedelta(seconds=seconds_ago)
        symbol = symbols[i % len(symbols)]

        base = i / max(1, count - 1)
        if profile == "left":
            score = round(max(0.0, 0.05 + (1 - base) * 0.95 * rnd.random()), 3)
        elif profile == "right":
            score = round(min(1.0, 0.05 + base * 0.95 * rnd.random()), 3)
        else:
            score = round(0.3 + (rnd.random() * 0.4), 3)

        side = "buy" if score >= 0.5 else "sell"
        confidence = round(min(1.0, max(0.0, 0.2 + rnd.random() * 0.8)), 3)

        signals.append(
            {
                "id": f"sig-{i}",
                "timestamp": ts,
                "symbol": symbol,
                "side": side,
                "score": score,
                "confidence": confidence,
                "details": {
                    "source": "simulator",
                    "note": f"mock signal #{i} ({profile})",
                },
            }
        )

    return signals
