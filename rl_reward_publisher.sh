#!/usr/bin/env bash
set -euo pipefail

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - <<'PY'
import os, json, random, time
import redis

host = os.getenv("REDIS_HOST", "127.0.0.1")
r = redis.Redis(host=host, port=6379)
now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
base = {
    "BTCUSDT": 0.42,
    "ETHUSDT": 0.35,
    "BNBUSDT": -0.05,
    "SOLUSDT": 0.18,
}
for sym, base_reward in base.items():
    jitter = random.uniform(-0.08, 0.08)
    reward = round(base_reward + jitter, 4)
    unreal_usd = round(random.uniform(-50, 150), 2)
    payload = {
        "symbol": sym,
        "unrealized_pnl": unreal_usd,
        "unrealized_pct": reward,
        "realized_pnl": 0.0,
        "realized_pct": 0.0,
        "total_pnl": unreal_usd,
        "realized_trades": random.randint(0, 5),
        "position_size": 1.0,
        "side": "long" if reward >= 0 else "short",
        "entry_price": 0,
        "notional": 0,
        "leverage": 3,
        "timestamp": now,
        "source": "live_synth",
        "reward": reward,
    }
    r.setex(f"quantum:rl:reward:{sym}", 600, json.dumps(payload))
PY
