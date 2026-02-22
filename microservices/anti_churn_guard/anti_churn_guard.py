#!/usr/bin/env python3
"""
Anti-Churn Guard — Churning Prevention System
================================================
Root cause of February 2026 drawdown: 86.5% of closed trades were churn alerts
(verified by Layer 5: 868/1003 trades = round-trips within same day).

This service reads Layer 5 churn data and enforces cooldowns:
  - If a symbol has >= CHURN_THRESHOLD round-trips today → blacklist for COOLDOWN_HOURS
  - Blacklist key: quantum:churn:blacklist:<SYM>  (TTL = cooldown expires automatically)
  - Execution layer MUST check this key before opening a new position

Also enforces:
  - DAILY_LOSS_LIMIT: if a symbol loses > N% today → blacklist for rest of day
  - PATTERN_BLOCK: if symbol has lost 3 consecutive trades → 4h cooldown
  - GLOBAL_HEAT: if *any* symbol hits global daily loss limit → freeze all new entries (not closes)

Operator reads:
  redis-cli keys "quantum:churn:blacklist:*"
  redis-cli hgetall quantum:churn:guard:status
  redis-cli ttl quantum:churn:blacklist:BTCUSDT

Operator override (emergency):
  redis-cli del quantum:churn:blacklist:BTCUSDT   # manually unblock
  redis-cli SET quantum:churn:guard:disabled 1    # disable entire guard (DANGEROUS)
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Set

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s churn_guard %(message)s",
)
log = logging.getLogger("churn_guard")

# ── Config ───────────────────────────────────────────────────────────────
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_DB          = 0
EVAL_INTERVAL     = int(os.getenv("EVAL_INTERVAL",      "15"))   # seconds
CHURN_THRESHOLD   = int(os.getenv("CHURN_THRESHOLD",     "4"))   # round-trips/day → blacklist (was 6, tightened)
COOLDOWN_HOURS    = float(os.getenv("COOLDOWN_HOURS",   "4.0"))  # hours after churn detection
DAILY_LOSS_LIMIT  = float(os.getenv("DAILY_LOSS_LIMIT", "2.0"))  # % equity loss per symbol per day
CONSEC_LOSS_LIMIT = int(os.getenv("CONSEC_LOSS_LIMIT",  "3"))    # consecutive losses → 4h cooldown
GLOBAL_DAILY_LOSS = float(os.getenv("GLOBAL_DAILY_LOSS","5.0"))  # % total equity → freeze new entries

STREAM_TRADES     = "quantum:stream:trade.closed"
CG_NAME           = "churn_guard"
KEY_STATUS        = "quantum:churn:guard:status"
KEY_GUARD_DISABLE = "quantum:churn:guard:disabled"
KEY_GLOBAL_FREEZE = "quantum:churn:global_freeze"
KEY_EQUITY        = "quantum:equity:current"

BLACKLIST_PREFIX  = "quantum:churn:blacklist:"
COOLDOWN_TTL      = int(COOLDOWN_HOURS * 3600)


# ── In-memory state ───────────────────────────────────────────────────────
_today          = ""
_daily_trades:  Dict[str, List[dict]]   = defaultdict(list)   # symbol → [trade]
_consec_losses: Dict[str, int]          = defaultdict(int)    # symbol → consecutive loss count
_blacklisted:   Set[str]                = set()
_stats = {
    "trades_scanned":  0,
    "blacklists_added": 0,
    "blacklists_active": 0,
    "global_freeze": False,
}


# ── Daily Reset ───────────────────────────────────────────────────────────
def _check_day_reset():
    global _today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _today:
        _today = today
        _daily_trades.clear()
        _consec_losses.clear()
        _blacklisted.clear()
        log.info(f"[GUARD] Day reset → {today} — all churn counters cleared")


# ── Blacklist Helper ──────────────────────────────────────────────────────
async def blacklist_symbol(r: aioredis.Redis, symbol: str, reason: str,
                            ttl_seconds: int = COOLDOWN_TTL):
    key = f"{BLACKLIST_PREFIX}{symbol}"
    ts  = int(time.time())
    expires_at = ts + ttl_seconds
    await r.hset(key, mapping={
        "symbol":     symbol,
        "reason":     reason,
        "ts":         str(ts),
        "expires_at": str(expires_at),
        "ttl_hours":  str(round(ttl_seconds / 3600, 1)),
    })
    await r.expire(key, ttl_seconds)
    _blacklisted.add(symbol)
    _stats["blacklists_added"] += 1
    log.warning(
        f"[GUARD] 🚫 BLACKLIST {symbol} for {ttl_seconds//3600:.1f}h "
        f"| reason: {reason}"
    )

    # Also push to alert stream
    await r.lpush("quantum:churn:alerts", json.dumps({
        "ts": ts, "symbol": symbol, "reason": reason,
        "expires_at": expires_at,
    }))
    await r.ltrim("quantum:churn:alerts", 0, 199)


async def unblacklist_expired(r: aioredis.Redis):
    """Sync in-memory set with Redis (TTL-based expiry)."""
    expired = set()
    for sym in list(_blacklisted):
        exists = await r.exists(f"{BLACKLIST_PREFIX}{sym}")
        if not exists:
            expired.add(sym)
    for sym in expired:
        _blacklisted.discard(sym)
        log.info(f"[GUARD] ✅ Cooldown expired for {sym} — unblocked")


# ── Trade Event Processor ─────────────────────────────────────────────────
async def process_trade(r: aioredis.Redis, msg_id: str, fields: dict,
                         equity: float):
    _check_day_reset()
    _stats["trades_scanned"] += 1

    sym      = fields.get("symbol", fields.get("Symbol", "UNKNOWN"))
    pnl_pct  = float(fields.get("pnl_pct", 0.0))
    pnl_usdt = float(fields.get("pnl", fields.get("realizedPnl", 0.0)))

    trade_rec = {
        "msg_id":   msg_id,
        "ts":       int(time.time()),
        "pnl_pct":  pnl_pct,
        "pnl_usdt": pnl_usdt,
    }
    _daily_trades[sym].append(trade_rec)

    n_trades_today   = len(_daily_trades[sym])
    # Round-trips = (n_trades / 2) since each position = 1 open + 1 close
    round_trips_today = n_trades_today // 2

    # Track consecutive losses
    if pnl_pct < 0:
        _consec_losses[sym] += 1
    else:
        _consec_losses[sym] = 0

    # Already blacklisted?
    already = await r.exists(f"{BLACKLIST_PREFIX}{sym}")
    if already:
        return

    # Disabled guard?
    if await r.exists(KEY_GUARD_DISABLE):
        return

    # ── C1: Churn threshold ──────────────────────────────────────────────
    if round_trips_today >= CHURN_THRESHOLD:
        await blacklist_symbol(
            r, sym, f"churn:{round_trips_today}_round_trips_today",
            COOLDOWN_TTL,
        )
        return

    # ── C2: Daily loss limit ─────────────────────────────────────────────
    daily_pnl_usdt = sum(t["pnl_usdt"] for t in _daily_trades[sym])
    if equity > 0:
        daily_loss_pct = abs(min(0, daily_pnl_usdt)) / equity * 100
        if daily_loss_pct >= DAILY_LOSS_LIMIT:
            # Block for rest of today (until midnight UTC)
            now   = datetime.now(timezone.utc)
            secs_until_midnight = int(
                (86400 - (now.hour * 3600 + now.minute * 60 + now.second))
            )
            await blacklist_symbol(
                r, sym,
                f"daily_loss:{daily_loss_pct:.2f}%_of_equity",
                max(secs_until_midnight, 3600),
            )
            return

    # ── C3: Consecutive losses ───────────────────────────────────────────
    if _consec_losses[sym] >= CONSEC_LOSS_LIMIT:
        await blacklist_symbol(
            r, sym,
            f"consec_losses:{_consec_losses[sym]}_in_a_row",
            4 * 3600,  # 4 hour cooldown
        )
        _consec_losses[sym] = 0  # reset after blacklist
        return


# ── Global Freeze Check ───────────────────────────────────────────────────
async def check_global_freeze(r: aioredis.Redis, equity: float):
    """If overall daily loss > GLOBAL_DAILY_LOSS% → freeze new entries globally."""
    try:
        # Sum daily P&L across all symbols
        total_daily_loss = 0.0
        for sym_trades in _daily_trades.values():
            sym_pnl = sum(t["pnl_usdt"] for t in sym_trades)
            if sym_pnl < 0:
                total_daily_loss += abs(sym_pnl)

        loss_pct = total_daily_loss / equity * 100 if equity > 0 else 0.0
        should_freeze = loss_pct >= GLOBAL_DAILY_LOSS

        if should_freeze and not _stats["global_freeze"]:
            await r.set(KEY_GLOBAL_FREEZE, f"daily_loss:{loss_pct:.2f}pct")
            _stats["global_freeze"] = True
            log.warning(
                f"[GUARD] 🔴 GLOBAL FREEZE activated — "
                f"total daily loss={loss_pct:.2f}% >= {GLOBAL_DAILY_LOSS}%"
            )
        elif not should_freeze and _stats["global_freeze"]:
            await r.delete(KEY_GLOBAL_FREEZE)
            _stats["global_freeze"] = False
            log.info("[GUARD] ✅ Global freeze lifted")

    except Exception as e:
        log.error(f"Global freeze check error: {e}")


# ── Stream Consumer ───────────────────────────────────────────────────────
async def consume_trades(r: aioredis.Redis):
    try:
        await r.xgroup_create(STREAM_TRADES, CG_NAME, id="$", mkstream=True)
        log.info(f"[GUARD] Consumer group {CG_NAME} on {STREAM_TRADES} — from NOW (not history)")
    except Exception:
        log.info(f"[GUARD] Consumer group {CG_NAME} already exists")

    consumer_name = f"churn_guard_{os.getpid()}"
    log.info("[GUARD] Listening for new trade.closed events...")

    while True:
        try:
            msgs = await r.xreadgroup(
                CG_NAME, consumer_name, {STREAM_TRADES: ">"}, count=20, block=5000
            )
            if not msgs:
                continue

            # Get equity for loss calculations
            eq_data = await r.hgetall(KEY_EQUITY)
            equity  = float(eq_data.get("equity", 3645.0)) if eq_data else 3645.0

            for msg_id, fields in msgs[0][1]:
                await process_trade(r, msg_id, fields, equity)
                await r.xack(STREAM_TRADES, CG_NAME, msg_id)

            await check_global_freeze(r, equity)

        except Exception as e:
            log.error(f"Stream loop error: {e}", exc_info=True)
            await asyncio.sleep(2)


# ── Status Publisher ──────────────────────────────────────────────────────
async def publish_status(r: aioredis.Redis):
    while True:
        try:
            await unblacklist_expired(r)

            # Count active blacklists in Redis
            active = []
            for sym in list(_blacklisted):
                ttl = await r.ttl(f"{BLACKLIST_PREFIX}{sym}")
                if ttl > 0:
                    active.append(f"{sym}({ttl//60}m)")

            # Symbols with high churn today (not yet blacklisted)
            _check_day_reset()
            approaching = []
            for sym, trades in _daily_trades.items():
                rt = len(trades) // 2
                if rt >= CHURN_THRESHOLD - 1 and sym not in _blacklisted:
                    approaching.append(f"{sym}:{rt}rt")

            status = {
                "ts":                  str(int(time.time())),
                "ts_human":            datetime.now(timezone.utc).isoformat(),
                "trades_scanned":      str(_stats["trades_scanned"]),
                "blacklists_added":    str(_stats["blacklists_added"]),
                "blacklisted_now":     ",".join([s.split("(")[0] for s in active]) or "none",
                "blacklisted_count":   str(len(active)),
                "approaching_churn":   ",".join(approaching) or "none",
                "global_freeze":       "ACTIVE" if _stats["global_freeze"] else "OFF",
                "churn_threshold":     str(CHURN_THRESHOLD),
                "cooldown_hours":      str(COOLDOWN_HOURS),
                "daily_loss_limit_pct":str(DAILY_LOSS_LIMIT),
                "consec_loss_limit":   str(CONSEC_LOSS_LIMIT),
            }
            await r.hset(KEY_STATUS, mapping=status)

            n_bl = len(active)
            log.info(
                f"[GUARD] active_blacklists={n_bl} "
                f"scanned={_stats['trades_scanned']} "
                f"blocked={_stats['blacklists_added']} "
                f"global_freeze={_stats['global_freeze']}"
                + (f" | {', '.join(active)}" if active else "")
            )

        except Exception as e:
            log.warning(f"Status publish error: {e}")
        await asyncio.sleep(EVAL_INTERVAL)


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    log.info("[GUARD] Anti-Churn Guard starting")
    log.info(f"[GUARD] thresholds: churn={CHURN_THRESHOLD}rt/day cooldown={COOLDOWN_HOURS}h "
             f"daily_loss={DAILY_LOSS_LIMIT}% consec_loss={CONSEC_LOSS_LIMIT} "
             f"global_freeze={GLOBAL_DAILY_LOSS}%")
    log.info("[GUARD] Execution layer must check quantum:churn:blacklist:<SYM> before opening positions")
    log.info("[GUARD] Layer 5 finding: 86.5% churn was root cause of Feb 2026 drawdown")

    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
    await r.ping()
    log.info("[GUARD] Redis OK")
    log.info("[GUARD] Operator commands:")
    log.info("  redis-cli del quantum:churn:blacklist:BTCUSDT  # manually unblock")
    log.info("  redis-cli SET quantum:churn:guard:disabled 1   # disable guard (DANGEROUS)")
    log.info("  redis-cli keys 'quantum:churn:blacklist:*'      # see all blacklists")

    asyncio.create_task(publish_status(r))
    await consume_trades(r)


if __name__ == "__main__":
    asyncio.run(main())
