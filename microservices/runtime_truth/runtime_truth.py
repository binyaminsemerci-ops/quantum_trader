#!/usr/bin/env python3
"""
Quantum Runtime Truth Engine
=============================
LAG 1 – FULL OBSERVABILITY

Publishes verified system state to quantum:health:truth every INTERVAL seconds.
This is NOT logging. This is runtime truth.

Stream: quantum:health:truth
Hash:   quantum:health:truth:latest  (for instant reads)

Fields published:
  ts                       ISO timestamp
  version                  Schema version
  services_ok              Comma-separated list of healthy critical services
  services_dead            Comma-separated list of dead/failed critical services
  exit_monitor_alive       true/false
  governor_alive           true/false
  harvest_active           true/false  (v1 or v2)
  learning_frozen          true/false
  system_mode              FREEZE / LOCKDOWN / LIVE / UNKNOWN
  apply_plan_lag           Consumer lag on apply.plan (intent_executor group)
  apply_result_lag         Consumer lag on apply.result (harvest_brain group)
  exchange_norm_ai_lag     Consumer lag on exchange.normalized (ai-engine group)
  market_tick_lag          Consumer lag on market.tick (ai-engine group)
  redis_position_count     Number of live quantum:position:* keys
  orphan_position_count    Positions with no recent price update (stale > 120s)
  position_mismatch        true/false (Redis vs reconcile engine disagreement)
  alarms                   JSON list of active alarm strings
  overall_health           GREEN / YELLOW / RED
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INTERVAL = float(os.getenv("TRUTH_INTERVAL_SEC", "10"))
STREAM_OUT = "quantum:health:truth"
HASH_OUT = "quantum:health:truth:latest"
STREAM_MAXLEN = 500  # Keep last 500 snapshots (~83 minutes at 10s interval)
VERSION = 2

# Critical services and their importance tier
CRITICAL_SERVICES = {
    # TIER 1 – Must never be down during trading
    "quantum-exit-monitor": "T1",
    "quantum-governor": "T1",
    "quantum-intent-executor": "T1",
    "quantum-execution": "T1",
    "quantum-emergency-exit-worker": "T1",
    # TIER 2 – Important but degraded-mode possible
    "quantum-harvest-brain": "T2",
    "quantum-harvest-v2": "T2",
    "quantum-risk-brake": "T2",
    "quantum-reconcile-engine": "T2",
    "quantum-reconcile-hardened": "T2",
    # TIER 3 – Infrastructure
    "quantum-ai-engine": "T3",
    "quantum-price-feed": "T3",
    "quantum-portfolio-governance": "T3",
}

# Stream → (redis_key, consumer_group_name, lag_alarm_threshold)
STREAM_MONITORS: List[Tuple[str, str, int]] = [
    ("quantum:stream:apply.plan",    "intent_executor",                        500),
    ("quantum:stream:apply.result",  "harvest_brain:execution",               1000),
    ("quantum:stream:exchange.raw",  "quantum:group:ai-engine:exchange.raw",  50000),
]

# Position staleness threshold (seconds before flagging as orphan)
ORPHAN_THRESHOLD_SEC = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | runtime_truth | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("runtime_truth")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def systemctl_is_active(service: str) -> str:
    """Return 'active', 'inactive', 'failed', or 'unknown'."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_stream_lag(r: redis.Redis, stream: str, group: str) -> Optional[int]:
    """Return consumer group lag for a stream, or None on error."""
    try:
        groups = r.xinfo_groups(stream)
        for g in groups:
            name = g.get("name", "")
            if isinstance(name, bytes):
                name = name.decode()
            if name == group:
                lag = g.get("lag", g.get("pel-count", 0))
                return int(lag) if lag is not None else 0
        return None  # Group not found
    except Exception:
        return None


def get_position_stats(r: redis.Redis) -> Tuple[int, int]:
    """
    Return (live_position_count, orphan_count).
    Orphan = position key exists but last_update_ts is stale > ORPHAN_THRESHOLD_SEC
    """
    now = time.time()
    live = 0
    orphan = 0
    try:
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match="quantum:position:*", count=100)
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                if ":ledger:" in key or ":snapshot:" in key:
                    continue
                pos = r.hgetall(key)
                qty = float(pos.get(b"quantity", pos.get("quantity", 0)) or 0)
                if qty == 0:
                    continue
                live += 1
                last_ts = float(pos.get(b"last_update_ts", pos.get("last_update_ts", 0)) or 0)
                if last_ts > 0 and (now - last_ts) > ORPHAN_THRESHOLD_SEC:
                    orphan += 1
            if cursor == 0:
                break
    except Exception as e:
        logger.warning(f"get_position_stats error: {e}")
    return live, orphan


def check_position_mismatch(r: redis.Redis) -> bool:
    """
    Detect disagreement: reconcile engine publishes quantum:reconcile:mismatch
    flag when it detects Redis ≠ Binance. Read that flag.
    """
    try:
        val = r.get("quantum:reconcile:mismatch")
        if val is None:
            return False
        if isinstance(val, bytes):
            val = val.decode()
        return val.strip() in ("1", "true", "True")
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main snapshot builder
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshot(r: redis.Redis) -> Dict[str, Any]:
    alarms: List[str] = []
    ts = datetime.now(timezone.utc).isoformat()

    # ── 1. Service health ──────────────────────────────────────────────────
    services_ok = []
    services_dead = []
    for svc, tier in CRITICAL_SERVICES.items():
        status = systemctl_is_active(svc)
        if status == "active":
            services_ok.append(svc)
        else:
            services_dead.append(f"{svc}({status})")
            if tier == "T1":
                alarms.append(f"T1_SERVICE_DOWN:{svc}:{status}")
            elif tier == "T2":
                alarms.append(f"T2_SERVICE_DOWN:{svc}:{status}")

    exit_monitor_alive = "quantum-exit-monitor" in services_ok
    governor_alive = "quantum-governor" in services_ok
    harvest_active = (
        "quantum-harvest-v2" in services_ok or
        "quantum-harvest-brain" in services_ok
    )

    if not exit_monitor_alive:
        alarms.append("CRITICAL:exit_monitor_down")
    if not governor_alive:
        alarms.append("CRITICAL:governor_down")

    # ── 2. Redis stream lags ───────────────────────────────────────────────
    stream_lags: Dict[str, int] = {}
    lag_keys = ["apply_plan_lag", "apply_result_lag", "exchange_raw_ai_lag"]
    for (stream, group, threshold), lag_key in zip(STREAM_MONITORS, lag_keys):
        lag = get_stream_lag(r, stream, group)
        stream_lags[lag_key] = lag if lag is not None else -1
        if lag is not None and lag > threshold:
            alarms.append(f"HIGH_LAG:{stream}:{group}:{lag}")

    # ── 3. Position health ─────────────────────────────────────────────────
    position_count, orphan_count = get_position_stats(r)
    position_mismatch = check_position_mismatch(r)

    if orphan_count > 0:
        alarms.append(f"ORPHAN_POSITIONS:{orphan_count}")
    if position_mismatch:
        alarms.append("POSITION_MISMATCH:redis_vs_binance")

    # ── 4. System flags ────────────────────────────────────────────────────
    system_mode = "UNKNOWN"
    try:
        mode = r.get("quantum:system:mode")
        if mode:
            system_mode = mode.decode() if isinstance(mode, bytes) else mode
    except Exception:
        pass

    learning_frozen = False
    try:
        frozen = r.get("quantum:learning:frozen")
        learning_frozen = frozen is not None and frozen in (b"1", "1")
    except Exception:
        pass

    if system_mode == "LOCKDOWN":
        alarms.append("SYSTEM_LOCKDOWN_ACTIVE")

    # ── 5. Overall health ──────────────────────────────────────────────────
    critical_alarms = [a for a in alarms if a.startswith("CRITICAL") or a.startswith("T1")]
    warning_alarms  = [a for a in alarms if a.startswith("T2") or a.startswith("HIGH_LAG") or a.startswith("ORPHAN")]

    if critical_alarms:
        overall_health = "RED"
    elif warning_alarms:
        overall_health = "YELLOW"
    else:
        overall_health = "GREEN"

    snapshot = {
        "ts": ts,
        "version": str(VERSION),
        "services_ok": ",".join(services_ok),
        "services_dead": ",".join(services_dead),
        "exit_monitor_alive": str(exit_monitor_alive).lower(),
        "governor_alive": str(governor_alive).lower(),
        "harvest_active": str(harvest_active).lower(),
        "learning_frozen": str(learning_frozen).lower(),
        "system_mode": system_mode,
        "apply_plan_lag": str(stream_lags.get("apply_plan_lag", -1)),
        "apply_result_lag": str(stream_lags.get("apply_result_lag", -1)),
        "exchange_raw_ai_lag": str(stream_lags.get("exchange_raw_ai_lag", -1)),
        "redis_position_count": str(position_count),
        "orphan_position_count": str(orphan_count),
        "position_mismatch": str(position_mismatch).lower(),
        "alarms": json.dumps(alarms),
        "overall_health": overall_health,
    }

    return snapshot


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    logger.info("=" * 60)
    logger.info("Quantum Runtime Truth Engine v2 starting")
    logger.info(f"  Interval  : {INTERVAL}s")
    logger.info(f"  Stream out: {STREAM_OUT}")
    logger.info(f"  Hash out  : {HASH_OUT}")
    logger.info("=" * 60)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    try:
        r.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

    consecutive_errors = 0

    while True:
        cycle_start = time.time()
        try:
            snapshot = build_snapshot(r)

            # Publish to stream (trimmed)
            r.xadd(STREAM_OUT, snapshot, maxlen=STREAM_MAXLEN, approximate=True)

            # Also write to hash for instant O(1) reads
            r.hset(HASH_OUT, mapping=snapshot)
            r.expire(HASH_OUT, int(INTERVAL * 5))  # TTL = 5 intervals

            health = snapshot["overall_health"]
            dead = snapshot["services_dead"]
            alarms = json.loads(snapshot["alarms"])
            pos_count = snapshot["redis_position_count"]
            mode = snapshot["system_mode"]

            if health == "GREEN":
                logger.info(
                    f"[{health}] positions={pos_count} mode={mode} "
                    f"plan_lag={snapshot['apply_plan_lag']} "
                    f"ai_lag={snapshot['exchange_raw_ai_lag']}"
                )
            else:
                logger.warning(
                    f"[{health}] positions={pos_count} mode={mode} "
                    f"dead=[{dead}] alarms={alarms}"
                )

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Snapshot error #{consecutive_errors}: {e}", exc_info=True)
            if consecutive_errors >= 5:
                logger.critical("5 consecutive errors — exiting for systemd restart")
                raise

        elapsed = time.time() - cycle_start
        sleep_time = max(0.1, INTERVAL - elapsed)
        await asyncio.sleep(sleep_time)


if __name__ == "__main__":
    asyncio.run(main())
