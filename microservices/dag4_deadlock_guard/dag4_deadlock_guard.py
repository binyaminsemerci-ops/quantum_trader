#!/usr/bin/env python3
"""
dag4_deadlock_guard.py — Redis Stream Deadlock Guard

Problem: Zombie PEL (Pending Entry List) entries accumulate when a consumer
dies mid-processing. These messages are "delivered but never ACK'd" and block
stream progress for that consumer group forever.

Solution: XAUTOCLAIM — every CHECK_INTERVAL_SEC, claim all messages idle
longer than IDLE_THRESHOLD_MS and ACK them (they are zombies by definition).

Streams monitored (hardened with actual VPS group names):
  quantum:stream:apply.plan
    → apply_layer_entry    (PEL was 1)
    → intent_executor      (PEL was 3)
  quantum:stream:apply.result
    → exit_intelligence    (PEL was 10)
    → p35_decision_intel   (PEL was 10)
  quantum:stream:exchange.raw
    → quantum:group:ai-engine:exchange.raw  (PEL was 449)

Safety:
  - Only reclaims messages idle > IDLE_THRESHOLD_MS (default 300s = 5 min)
  - Does NOT touch lag-only groups (no PEL = no deadlock)
  - Publishes stats to quantum:dag4:deadlock_guard:latest (TTL 2min)
  - Disable: redis-cli SET quantum:dag4:deadlock_guard:disabled 1
"""

import os
import sys
import time
import logging
import signal
import redis

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s dag4 %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dag4")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT           = int(os.getenv("REDIS_PORT", "6379"))
CHECK_INTERVAL_SEC   = int(os.getenv("DAG4_INTERVAL_SEC", "60"))
IDLE_THRESHOLD_MS    = int(os.getenv("DAG4_IDLE_MS", "300000"))   # 5 minutes
CLAIM_BATCH_SIZE     = int(os.getenv("DAG4_BATCH", "100"))
CLAIM_CONSUMER       = "dag4_deadlock_guard"
STATE_KEY            = "quantum:dag4:deadlock_guard:latest"
STATE_TTL            = 120
DISABLE_KEY          = "quantum:dag4:deadlock_guard:disabled"

# (stream_key, group_name, description)
WATCH_GROUPS = [
    ("quantum:stream:apply.plan",   "apply_layer_entry",                      "apply.plan/entry"),
    ("quantum:stream:apply.plan",   "intent_executor",                        "apply.plan/intent"),
    ("quantum:stream:apply.result", "exit_intelligence",                      "apply.result/exit_intel"),
    ("quantum:stream:apply.result", "p35_decision_intel",                     "apply.result/p35"),
    ("quantum:stream:exchange.raw", "quantum:group:ai-engine:exchange.raw",   "exchange.raw/ai-engine"),
]

# ── Graceful shutdown ─────────────────────────────────────────────────────
_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    logger.info("Signal %s — shutting down", sig)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Core logic ────────────────────────────────────────────────────────────

def get_pel_count(r: redis.Redis, stream: str, group: str) -> int:
    """Return current PEL count for a stream/group (0 if stream/group missing)."""
    try:
        groups = r.xinfo_groups(stream)
        for g in groups:
            name = g.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode()
            if name == group:
                return g.get("pending", 0)
    except redis.ResponseError:
        pass
    return 0


def reclaim_zombies(r: redis.Redis, stream: str, group: str, label: str) -> int:
    """
    XAUTOCLAIM all messages idle > IDLE_THRESHOLD_MS, then XACK them.
    Returns number of entries freed.
    """
    freed = 0
    start = "0-0"

    while True:
        try:
            # XAUTOCLAIM returns (next_id, [(id, fields), ...], [deleted_ids])
            result = r.xautoclaim(
                stream,
                group,
                CLAIM_CONSUMER,
                IDLE_THRESHOLD_MS,
                start,
                count=CLAIM_BATCH_SIZE,
            )
        except redis.ResponseError as e:
            logger.error("[DAG4] XAUTOCLAIM failed %s/%s: %s", label, group, e)
            break

        next_id  = result[0]
        messages = result[1]

        if messages:
            ids = [msg[0] for msg in messages]
            r.xack(stream, group, *ids)
            freed += len(ids)
            logger.warning(
                "[DAG4] RECLAIMED+ACKED %d zombies from %s | idle>%ds",
                len(ids), label, IDLE_THRESHOLD_MS // 1000,
            )

        # next_id == b'0-0' means we've wrapped around — done
        if not next_id or next_id in (b"0-0", "0-0"):
            break
        start = next_id if not isinstance(next_id, bytes) else next_id.decode()

    return freed


def run_guard_tick(r: redis.Redis) -> dict:
    """Run one guard tick: scan all watched groups, reclaim zombies."""
    total_freed  = 0
    group_stats  = {}

    for stream, group, label in WATCH_GROUPS:
        pel_before = get_pel_count(r, stream, group)

        if pel_before == 0:
            group_stats[label] = {"pel_before": 0, "freed": 0}
            logger.debug("[DAG4] %s PEL=0 — skip", label)
            continue

        logger.info("[DAG4] %s PEL=%d — scanning for zombies (idle>%ds)",
                    label, pel_before, IDLE_THRESHOLD_MS // 1000)
        freed = reclaim_zombies(r, stream, group, label)

        pel_after = get_pel_count(r, stream, group)
        group_stats[label] = {
            "pel_before": pel_before,
            "pel_after":  pel_after,
            "freed":      freed,
        }
        total_freed += freed

        if freed > 0:
            logger.warning(
                "[DAG4] %s freed=%d pel_before=%d pel_after=%d",
                label, freed, pel_before, pel_after,
            )
        else:
            logger.info(
                "[DAG4] %s PEL=%d but 0 entries idle>%ds yet (still being processed)",
                label, pel_before, IDLE_THRESHOLD_MS // 1000,
            )

    return {
        "total_freed":    str(total_freed),
        "groups_checked": str(len(WATCH_GROUPS)),
        **{f"{label}:pel": str(s.get("pel_before", 0)) for label, s in group_stats.items()},
        **{f"{label}:freed": str(s.get("freed", 0)) for label, s in group_stats.items()},
        "idle_threshold_sec": str(IDLE_THRESHOLD_MS // 1000),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("[DAG4] Redis Deadlock Guard starting — idle_threshold=%ds interval=%ds",
                IDLE_THRESHOLD_MS // 1000, CHECK_INTERVAL_SEC)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    # Connection check
    try:
        r.ping()
        logger.info("[DAG4] Redis connection OK")
    except redis.ConnectionError as e:
        logger.error("[DAG4] Redis connection FAILED: %s", e)
        sys.exit(1)

    # Startup report
    logger.info("[DAG4] Watching %d stream/group pairs:", len(WATCH_GROUPS))
    for stream, group, label in WATCH_GROUPS:
        pel = get_pel_count(r, stream, group)
        logger.info("  %s  PEL=%d", label, pel)

    consecutive_errors = 0

    while _RUNNING:
        tick_start = time.monotonic()

        try:
            disabled = r.get(DISABLE_KEY)
            if disabled:
                logger.warning("[DAG4] DISABLED via %s — skipping tick", DISABLE_KEY)
            else:
                stats = run_guard_tick(r)
                r.hset(STATE_KEY, mapping=stats)
                r.expire(STATE_KEY, STATE_TTL)

                total = int(stats.get("total_freed", 0))
                logger.info("[DAG4_TICK] freed=%d groups=%s",
                            total, stats["groups_checked"])
            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            logger.error("[DAG4] Exception (consecutive=%d): %s", consecutive_errors, e)
            if consecutive_errors >= 5:
                logger.critical("[DAG4] 5 consecutive errors — sleeping 120s")
                time.sleep(120)
                consecutive_errors = 0

        elapsed   = time.monotonic() - tick_start
        sleep_for = max(0.0, CHECK_INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    logger.info("[DAG4] Redis Deadlock Guard stopped")


if __name__ == "__main__":
    main()
