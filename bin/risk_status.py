#!/usr/bin/env python3
"""Minimal risk status snapshot (read-only)."""

import json
import os
import time
from typing import Optional

import redis

from microservices.risk_policy_enforcer import create_enforcer, SystemState


def _age_seconds(redis_client: redis.Redis, key: str) -> Optional[float]:
    val = redis_client.get(key)
    if not val:
        return None
    try:
        return time.time() - float(val)
    except Exception:
        return None


def _counter(redis_client: redis.Redis, key: str) -> int:
    val = redis_client.get(key)
    if not val:
        return 0
    try:
        return int(val)
    except Exception:
        return 0


def main():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    symbol = os.getenv("RISK_STATUS_SYMBOL", "BTCUSDT")

    client = redis.from_url(redis_url)
    enforcer = create_enforcer(redis_url)
    metrics = enforcer.compute_system_state(symbol=symbol)

    kill_switch = client.get("quantum:global:kill_switch")
    kill_switch_reason = client.get("quantum:global:kill_switch:reason")

    rl_feedback_age = _age_seconds(client, "quantum:svc:rl_feedback_v2:heartbeat")
    rl_trainer_age = _age_seconds(client, "quantum:svc:rl_trainer:heartbeat")

    snapshot = {
        "state": metrics.system_state.value,
        "reason": metrics.failure_reason,
        "uptime_s": int(metrics.uptime_seconds or 0),
        "startup_grace_remaining": int(metrics.startup_grace_remaining or 0),
        "kill_switch": bool(kill_switch),
        "kill_switch_reason": kill_switch_reason.decode() if kill_switch_reason else None,
        "heartbeat": {
            "rl_feedback": "OK" if rl_feedback_age is not None else "MISSING",
            "rl_trainer": "OK" if rl_trainer_age is not None else "MISSING",
            "rl_feedback_age": rl_feedback_age,
            "rl_trainer_age": rl_trainer_age,
        },
        "capital_state": {
            "daily_pnl": metrics.daily_pnl,
            "drawdown_pct": metrics.rolling_drawdown_pct,
            "loss_streak": metrics.consecutive_losses,
        },
        "market_state": {
            "volatility": metrics.realized_volatility,
            "liquidity_ok": metrics.spread_bps is None or metrics.spread_bps <= 10.0,
            "symbol_allowed": metrics.symbol_in_whitelist,
        },
        "counters": {
            "trades_allowed": _counter(client, "quantum:metrics:trades_allowed"),
            "trades_blocked": _counter(client, "quantum:metrics:trades_blocked"),
            "paused_events": _counter(client, "quantum:metrics:paused_events"),
            "kill_switch_events": _counter(client, "quantum:metrics:kill_switch_events"),
        },
    }

    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
