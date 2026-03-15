#!/usr/bin/env python3
"""
Position Invariant Enforcer — Hardening Layer
Phases 1, 5, 6

Runs every 60 seconds.
- Fetches Binance TESTNET positions (source of truth)
- Fetches Redis authoritative position hashes
- Auto-corrects drift
- Emits quantum:stream:reconcile.alert on mismatch
- Emits quantum:stream:system.alert for critical events
- Exposes metrics in Redis (position_authoritative_count, etc.)
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.request
from typing import Dict, List, Set

import redis

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [reconcile_engine] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reconcile_engine")

# ── Config ─────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB       = int(os.getenv("REDIS_DB", "0"))

API_KEY        = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET     = os.getenv("BINANCE_TESTNET_API_SECRET", "")
TESTNET_BASE   = "https://testnet.binancefuture.com"

INTERVAL_SEC   = int(os.getenv("RECONCILE_INTERVAL_SEC", "60"))
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", "10"))

STREAM_RECONCILE_ALERT = "quantum:stream:reconcile.alert"
STREAM_SYSTEM_ALERT    = "quantum:stream:system.alert"
STREAM_SLOT_ALERT      = "quantum:stream:slot.alert"

METRICS_KEY            = "quantum:metrics:reconcile_engine"
STREAM_MAXLEN          = 1000


# ── Exchange query ─────────────────────────────────────────

def fetch_exchange_positions() -> Dict[str, Dict]:
    """
    Query Binance TESTNET positionRisk.
    Returns {symbol: {positionAmt, entryPrice, ...}} for abs(positionAmt) > 0.
    Raises on any error — callers must handle.
    """
    if not API_KEY or not API_SECRET:
        raise RuntimeError("BINANCE_TESTNET_API_KEY/SECRET not configured")

    ts = int(time.time() * 1000)
    qs = f"timestamp={ts}"
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{TESTNET_BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
    req = urllib.request.Request(url, headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = json.loads(resp.read())

    return {
        p["symbol"]: p
        for p in raw
        if abs(float(p.get("positionAmt", 0))) > 0
    }


# ── Redis helpers ──────────────────────────────────────────

def fetch_redis_positions(r: redis.Redis) -> Dict[str, Dict]:
    """
    Scan quantum:state:positions:* (canonical position keys).
    Returns {symbol: hash_data} for abs(quantity) > 0.
    """
    result = {}
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="quantum:state:positions:*", count=200)
        for k in keys:
            data = r.hgetall(k)
            try:
                qty = float(data.get("quantity", data.get("position_amt", 0)))
            except (ValueError, TypeError):
                qty = 0.0
            if qty != 0:
                symbol = k.replace("quantum:state:positions:", "")
                result[symbol] = data
        if cursor == 0:
            break
    return result


def get_recent_snapshot_symbols(r: redis.Redis, count: int = 3) -> Set[str]:
    """
    Return symbols seen in last `count` position.snapshot stream events.
    Used as exchange-lag buffer in Phase 4.
    """
    symbols = set()
    try:
        events = r.xrevrange("quantum:stream:position.snapshot", count=count)
        for _, data in events:
            sym = data.get("symbol", "")
            if sym:
                symbols.add(sym)
    except Exception:
        pass
    return symbols


# ── Alert emitters ─────────────────────────────────────────

def emit_alert(
    r: redis.Redis,
    alert_type: str,
    severity: str,
    details: Dict,
    stream: str = STREAM_SYSTEM_ALERT,
) -> None:
    """
    Emit structured alert to quantum:stream:system.alert.

    alert_type: POSITION_DRIFT | SLOT_OVERFLOW | WRITE_GUARD_BLOCK
    severity:   INFO | WARN | CRITICAL
    """
    payload = {
        "type": alert_type,
        "severity": severity,
        "ts": str(int(time.time())),
        **{k: str(v) for k, v in details.items()},
    }
    try:
        r.xadd(stream, payload, maxlen=STREAM_MAXLEN)
    except Exception as e:
        logger.error(f"[ALERT_EMIT] Failed to emit {alert_type}: {e}")


def emit_reconcile_alert(r: redis.Redis, symbol: str, action: str, details: Dict) -> None:
    payload = {
        "symbol": symbol,
        "action": action,
        "ts": str(int(time.time())),
        **{k: str(v) for k, v in details.items()},
    }
    try:
        r.xadd(STREAM_RECONCILE_ALERT, payload, maxlen=STREAM_MAXLEN)
    except Exception as e:
        logger.error(f"[RECONCILE_ALERT] emit failed: {e}")


# ── Metrics ────────────────────────────────────────────────

def publish_metrics(
    r: redis.Redis,
    authoritative_count: int,
    exchange_count: int,
    slot_delta: int,
    mismatch_count: int,
    last_run_ts: int,
) -> None:
    """
    Phase 6: Write metrics to Redis hash for external consumption.
    Key: quantum:metrics:reconcile_engine
    """
    try:
        r.hset(
            METRICS_KEY,
            mapping={
                "position_authoritative_count": str(authoritative_count),
                "position_exchange_count":      str(exchange_count),
                "slot_delta":                   str(slot_delta),
                "reconcile_last_run_ts":        str(last_run_ts),
                "reconcile_mismatch_count":     str(mismatch_count),
                "reconcile_engine_version":     "1.0.0",
            },
        )
        r.expire(METRICS_KEY, 300)  # 5min TTL — stale if engine dies
    except Exception as e:
        logger.warning(f"[METRICS] publish failed: {e}")


# ── Core reconcile logic ────────────────────────────────────

def reconcile_once(r: redis.Redis, cumulative_mismatch_count: int) -> int:
    """
    Single reconcile pass.
    Returns updated cumulative_mismatch_count.
    """
    run_ts = int(time.time())
    logger.info("─" * 60)
    logger.info("[RECONCILE] Starting reconcile pass")

    # ── 1. Exchange truth ──────────────────────────────────
    try:
        exchange_positions = fetch_exchange_positions()
    except Exception as e:
        logger.error(f"[RECONCILE] Exchange query failed: {e} — aborting pass")
        publish_metrics(r, -1, -1, 0, cumulative_mismatch_count, run_ts)
        return cumulative_mismatch_count

    exchange_symbols = set(exchange_positions.keys())
    logger.info(f"[RECONCILE] Exchange: {sorted(exchange_symbols)}")

    # ── 2. Redis authoritative ─────────────────────────────
    redis_positions = fetch_redis_positions(r)
    redis_symbols = set(redis_positions.keys())
    logger.info(f"[RECONCILE] Redis: {sorted(redis_symbols)}")

    # ── 3. Diff ────────────────────────────────────────────
    missing_in_redis   = exchange_symbols - redis_symbols   # on exchange, not in Redis
    ghost_in_redis     = redis_symbols - exchange_symbols   # in Redis, not on exchange
    match_ok           = (exchange_symbols == redis_symbols)

    logger.info(f"[RECONCILE] MATCH_OK={match_ok}")
    if missing_in_redis:
        logger.error(f"[RECONCILE] MISSING_IN_REDIS={sorted(missing_in_redis)}")
    if ghost_in_redis:
        logger.error(f"[RECONCILE] GHOST_IN_REDIS={sorted(ghost_in_redis)}")

    # ── 4. Slot overflow invariant ─────────────────────────
    authoritative_count = len(redis_symbols)
    exchange_count      = len(exchange_symbols)
    slot_delta          = authoritative_count - exchange_count

    if authoritative_count > MAX_POSITIONS:
        logger.critical(
            f"[SLOT_OVERFLOW] authoritative_count={authoritative_count} "
            f"> max_positions={MAX_POSITIONS} — slots exhausted"
        )
        emit_alert(
            r,
            alert_type="SLOT_OVERFLOW",
            severity="CRITICAL",
            details={
                "authoritative_count": authoritative_count,
                "max_positions": MAX_POSITIONS,
                "slot_delta": slot_delta,
            },
            stream=STREAM_SLOT_ALERT,
        )
        emit_alert(
            r,
            alert_type="SLOT_OVERFLOW",
            severity="CRITICAL",
            details={
                "authoritative_count": authoritative_count,
                "max_positions": MAX_POSITIONS,
            },
        )

    # ── 5. Auto-correct mismatches ─────────────────────────
    mismatch_count = len(missing_in_redis) + len(ghost_in_redis)

    if mismatch_count > 0:
        cumulative_mismatch_count += mismatch_count

        # Write missing Redis hashes from exchange
        for symbol in missing_in_redis:
            ep = exchange_positions[symbol]
            try:
                amt = float(ep.get("positionAmt", 0))
                key = f"quantum:state:positions:{symbol}"
                r.hset(
                    key,
                    mapping={
                        "symbol":         symbol,
                        "position_amt":   str(amt),
                        "side":           "LONG" if amt > 0 else "SHORT",
                        "quantity":       str(abs(amt)),
                        "entry_price":    str(ep.get("entryPrice", "0")),
                        "unrealized_pnl": str(ep.get("unRealizedProfit", "0")),
                        "leverage":       str(ep.get("leverage", "1")),
                        "source":         "reconcile_engine_autocorrect",
                        "ts_epoch":       str(run_ts),
                    },
                )
                logger.info(f"[RECONCILE] AUTOCORRECT_ADD {symbol} (missing in Redis)")
                emit_reconcile_alert(r, symbol, "ADD", {"reason": "missing_in_redis"})
                emit_alert(
                    r,
                    "POSITION_DRIFT",
                    "WARN",
                    {"symbol": symbol, "direction": "ADD", "reason": "missing_in_redis"},
                )
            except Exception as e:
                logger.error(f"[RECONCILE] AUTOCORRECT_ADD failed for {symbol}: {e}")

        # Remove ghost Redis hashes — double-check with snapshot buffer
        snapshot_symbols = get_recent_snapshot_symbols(r, count=3)
        for symbol in ghost_in_redis:
            if symbol in snapshot_symbols:
                logger.warning(
                    f"[RECONCILE] SKIP DELETE {symbol}: found in last 3 snapshots "
                    f"(exchange lag guard active)"
                )
                continue
            try:
                canonical_key = f"quantum:state:positions:{symbol}"
                legacy_key = f"quantum:position:{symbol}"
                r.delete(canonical_key)
                r.delete(legacy_key)
                logger.info(f"[RECONCILE] AUTOCORRECT_DEL {symbol} (ghost in Redis)")
                emit_reconcile_alert(r, symbol, "DELETE", {"reason": "ghost_in_redis"})
                emit_alert(
                    r,
                    "POSITION_DRIFT",
                    "WARN",
                    {"symbol": symbol, "direction": "DELETE", "reason": "ghost_in_redis"},
                )
            except Exception as e:
                logger.error(f"[RECONCILE] AUTOCORRECT_DEL failed for {symbol}: {e}")

    else:
        logger.info("[RECONCILE] State consistent — no action required")

    # ── 6. Publish metrics ─────────────────────────────────
    # Re-scan after corrections
    post_redis = fetch_redis_positions(r)
    publish_metrics(
        r,
        authoritative_count=len(post_redis),
        exchange_count=exchange_count,
        slot_delta=len(post_redis) - exchange_count,
        mismatch_count=cumulative_mismatch_count,
        last_run_ts=run_ts,
    )

    logger.info(
        f"[RECONCILE] Done — exchange={exchange_count} "
        f"redis_after={len(post_redis)} "
        f"mismatches_this_pass={mismatch_count} "
        f"cumulative={cumulative_mismatch_count}"
    )
    return cumulative_mismatch_count


# ── Main loop ──────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Position Invariant Enforcer — reconcile_engine v1.0.0")
    logger.info(f"Interval: {INTERVAL_SEC}s | MAX_POSITIONS: {MAX_POSITIONS}")
    logger.info(f"Streams: {STREAM_RECONCILE_ALERT} | {STREAM_SYSTEM_ALERT}")
    logger.info("=" * 60)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # Verify Redis
    try:
        r.ping()
        logger.info(f"[INIT] Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.critical(f"[INIT] Redis connection failed: {e}")
        sys.exit(1)

    # Verify creds
    if not API_KEY or not API_SECRET:
        logger.critical("[INIT] BINANCE_TESTNET_API_KEY/SECRET not set")
        sys.exit(1)

    cumulative_mismatch_count = 0
    cycle = 0

    while True:
        cycle += 1
        try:
            logger.info(f"[RECONCILE] Cycle #{cycle}")
            cumulative_mismatch_count = reconcile_once(r, cumulative_mismatch_count)
        except Exception as e:
            logger.error(f"[RECONCILE] Unhandled error in cycle #{cycle}: {e}", exc_info=True)

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
