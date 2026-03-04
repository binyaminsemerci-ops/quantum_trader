#!/usr/bin/env python3
"""
Quantum Runtime Truth Engine
=============================
LAG 1 – FULL OBSERVABILITY

Publishes verified system state to quantum:health:truth every INTERVAL seconds.
This is NOT logging. This is runtime truth.

Stream: quantum:health:truth
Hash:   quantum:health:truth:latest  (for instant reads)
Key:    quantum:health:os_health     (simple GREEN/YELLOW/RED string, TTL=5x interval)

Fields published:
  ts                       ISO timestamp
  version                  Schema version (3)
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
  freshness_market_klines  Age of last market.klines message in seconds (-1=empty)
  freshness_ai_signal      Age of last ai.signal_generated message in seconds
  freshness_trade_intent   Age of last trade.intent message in seconds
  freshness_exec_result    Age of last trade.execution.res message in seconds
  exit_heartbeat_age_sec   Seconds since last quantum:exit:heartbeat key update
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
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import redis

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INTERVAL = float(os.getenv("TRUTH_INTERVAL_SEC", "10"))
STREAM_OUT = "quantum:health:truth"
HASH_OUT = "quantum:health:truth:latest"
STREAM_MAXLEN = 500  # Keep last 500 snapshots (~83 minutes at 10s interval)
VERSION = 3

# Webhook alerting (set ALERT_WEBHOOK_URL in /etc/quantum/alert.env or env)
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC", "300"))  # 5 minutes between same alarm

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

# Stream → (consumer_group_name, lag_alarm_threshold)
STREAM_MONITORS: List[Tuple[str, str, int]] = [
    ("apply.plan",            "intent_executor",         500),
    ("apply.result",          "harvest_brain:execution", 1000),
    ("exchange.normalized",   "ai-engine",               50000),
    ("market.tick",           "ai-engine",               50000),
]

# Stream freshness checks: stream_suffix → (max_age_sec, alarm_level)
# Uses XREVRANGE COUNT 1 on quantum:stream:{suffix} to get last message timestamp.
# alarm_level: "CRITICAL" triggers RED + webhook, "WARN" triggers YELLOW only.
FRESHNESS_CHECKS: Dict[str, Tuple[float, str]] = {
    "market.klines":       (90.0,   "CRITICAL"),  # 1m candles update ~every 60s; 90s allows 1 miss
    "ai.signal_generated": (60.0,   "WARN"),      # AI generates signals ~30s
    "trade.intent":        (120.0,  "WARN"),       # event-driven, allow 2min quiet
    "trade.execution.res": (300.0,  "CRITICAL"),  # if stale >5min while positions exist → broken
}

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


def get_stream_age_secs(r: redis.Redis, stream_suffix: str) -> float:
    """
    Return seconds since the last message on quantum:stream:{stream_suffix}.
    Returns -1.0 if the stream is empty or missing.
    Redis stream IDs are <ms_timestamp>-<seq>; we parse the ms part.
    """
    try:
        entries = r.xrevrange(f"quantum:stream:{stream_suffix}", count=1)
        if not entries:
            return -1.0
        entry_id, _ = entries[0]
        if isinstance(entry_id, bytes):
            entry_id = entry_id.decode()
        ms = int(entry_id.split("-")[0])
        age = time.time() - (ms / 1000.0)
        return round(age, 1)
    except Exception as e:
        logger.debug(f"get_stream_age_secs({stream_suffix}) error: {e}")
        return -1.0


def get_exit_heartbeat_age(r: redis.Redis) -> float:
    """
    Return seconds since quantum:exit:heartbeat was last written.
    Returns -1.0 if key missing.
    """
    try:
        # Some exit monitors write a Unix timestamp as value
        val = r.get("quantum:exit:heartbeat")
        if val is None:
            return -1.0
        ts = float(val.decode() if isinstance(val, bytes) else val)
        age = time.time() - ts
        return round(age, 1)
    except Exception:
        return -1.0


def send_webhook_alert(url: str, title: str, message: str, color: int = 0xFF0000) -> None:
    """
    Fire-and-forget webhook POST (Discord embeds format).
    Runs in a daemon thread so it never blocks the main snapshot loop.
    Also supports plain Slack/generic webhooks (falls back to {text: ...}).
    """
    if not url or not _REQUESTS_AVAILABLE:
        return

    def _post():
        try:
            payload = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": color,
                    "footer": {"text": f"quantum-runtime-truth | {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"},
                }]
            }
            resp = _requests.post(url, json=payload, timeout=5)
            if resp.status_code not in (200, 204):
                # Try generic Slack format as fallback
                _requests.post(url, json={"text": f"*{title}*\n{message}"}, timeout=5)
        except Exception as ex:
            logger.warning(f"Webhook send failed: {ex}")

    t = threading.Thread(target=_post, daemon=True)
    t.start()


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
    for stream, group, threshold in STREAM_MONITORS:
        lag = get_stream_lag(r, stream, group)
        key = stream.replace(".", "_") + "_lag"
        stream_lags[key] = lag if lag is not None else -1
        if lag is not None and lag > threshold:
            alarms.append(f"HIGH_LAG:{stream}:{group}:{lag}")

    # ── 2b. Stream freshness (last-message age) ───────────────────────────
    freshness: Dict[str, float] = {}
    for suffix, (max_age, alarm_level) in FRESHNESS_CHECKS.items():
        age = get_stream_age_secs(r, suffix)
        freshness[suffix] = age
        # Skip alarm for event-driven streams (trade.intent / execution.res)
        # when there are no active positions — quiet is expected
        if age == -1.0:
            continue  # empty stream — service not yet running, skip
        # For execution.result: only alarm if there are live positions
        if suffix == "trade.execution.res" and alarm_level == "CRITICAL":
            # Defer until position_count is known — checked in step 3b below
            continue
        if age > max_age:
            alarm_tag = f"STALE_STREAM:{suffix}:{age:.0f}s>{max_age:.0f}s"
            if alarm_level == "CRITICAL":
                alarms.append(f"CRITICAL:{alarm_tag}")
            else:
                alarms.append(alarm_tag)

    # ── 2c. Exit heartbeat age ────────────────────────────────────────────
    exit_heartbeat_age = get_exit_heartbeat_age(r)
    if exit_heartbeat_age > 60:
        alarms.append(f"STALE_EXIT_HEARTBEAT:{exit_heartbeat_age:.0f}s")

    # ── 3. Position health ─────────────────────────────────────────────────
    position_count, orphan_count = get_position_stats(r)
    position_mismatch = check_position_mismatch(r)

    if orphan_count > 0:
        alarms.append(f"ORPHAN_POSITIONS:{orphan_count}")
    if position_mismatch:
        alarms.append("POSITION_MISMATCH:redis_vs_binance")

    # ── 3b. Execution result freshness — only matters with live positions ──
    exec_age = freshness.get("trade.execution.res", -1.0)
    exec_max_age, _ = FRESHNESS_CHECKS["trade.execution.res"]
    if exec_age > exec_max_age and position_count > 0:
        alarms.append(f"CRITICAL:STALE_STREAM:trade.execution.res:{exec_age:.0f}s>{exec_max_age:.0f}s")

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
        "exchange_norm_ai_lag": str(stream_lags.get("exchange_normalized_lag", -1)),
        "market_tick_lag": str(stream_lags.get("market_tick_lag", -1)),
        "freshness_market_klines": str(freshness.get("market.klines", -1)),
        "freshness_ai_signal": str(freshness.get("ai.signal_generated", -1)),
        "freshness_trade_intent": str(freshness.get("trade.intent", -1)),
        "freshness_exec_result": str(freshness.get("trade.execution.res", -1)),
        "exit_heartbeat_age_sec": str(exit_heartbeat_age),
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
    logger.info(f"Quantum Runtime Truth Engine v{VERSION} starting")
    logger.info(f"  Interval    : {INTERVAL}s")
    logger.info(f"  Stream out  : {STREAM_OUT}")
    logger.info(f"  Hash out    : {HASH_OUT}")
    logger.info(f"  Webhook     : {'SET' if ALERT_WEBHOOK_URL else 'NOT SET (configure ALERT_WEBHOOK_URL)'}")
    logger.info(f"  Alert cooldown: {ALERT_COOLDOWN_SEC}s")
    logger.info("=" * 60)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    try:
        r.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise

    consecutive_errors = 0
    # Cooldown tracker: alarm_key → last_sent_unix_ts
    last_alert_ts: Dict[str, float] = {}

    while True:
        cycle_start = time.time()
        try:
            snapshot = build_snapshot(r)

            # Publish to stream (trimmed)
            r.xadd(STREAM_OUT, snapshot, maxlen=STREAM_MAXLEN, approximate=True)

            # Write to hash for instant O(1) reads (TTL = 5 intervals)
            r.hset(HASH_OUT, mapping=snapshot)
            r.expire(HASH_OUT, int(INTERVAL * 5))

            # Write simple OS_HEALTH string for fast external reads
            r.set("quantum:health:os_health", snapshot["overall_health"], ex=int(INTERVAL * 5))

            health = snapshot["overall_health"]
            dead = snapshot["services_dead"]
            alarms = json.loads(snapshot["alarms"])
            pos_count = snapshot["redis_position_count"]
            mode = snapshot["system_mode"]
            f_klines = snapshot["freshness_market_klines"]
            f_exec   = snapshot["freshness_exec_result"]

            if health == "GREEN":
                logger.info(
                    f"[{health}] positions={pos_count} mode={mode} "
                    f"plan_lag={snapshot['apply_plan_lag']} "
                    f"klines={f_klines}s exec_res={f_exec}s"
                )
            else:
                logger.warning(
                    f"[{health}] positions={pos_count} mode={mode} "
                    f"dead=[{dead}] klines={f_klines}s exec_res={f_exec}s "
                    f"alarms={alarms}"
                )

            # ── Webhook alerts (with per-alarm cooldown) ──────────────────
            if ALERT_WEBHOOK_URL:
                now = time.time()
                critical = [a for a in alarms if a.startswith("CRITICAL") or a.startswith("T1_SERVICE")]
                for alarm in critical:
                    last = last_alert_ts.get(alarm, 0)
                    if now - last >= ALERT_COOLDOWN_SEC:
                        last_alert_ts[alarm] = now
                        send_webhook_alert(
                            ALERT_WEBHOOK_URL,
                            title=f"🚨 Quantum Runtime CRITICAL — {alarm.split(':')[1] if ':' in alarm else alarm}",
                            message=(
                                f"**Alarm**: `{alarm}`\n"
                                f"**Health**: {health}\n"
                                f"**Positions**: {pos_count} | Mode: `{mode}`\n"
                                f"**Dead services**: {dead or 'none'}\n"
                                f"**klines age**: {f_klines}s | exec_res age: {f_exec}s"
                            ),
                            color=0xFF0000,
                        )
                        logger.warning(f"WEBHOOK_SENT alarm={alarm}")

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
