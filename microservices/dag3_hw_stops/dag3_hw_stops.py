#!/usr/bin/env python3
"""
dag3_hw_stops.py — Hardware TP/SL Guardian

Places and maintains exchange-native STOP_MARKET orders on Binance testnet
for every open position tracked in Redis.

Safety model:
  - R_STOP = 0.7 (matches harvest_v2 HIGH_VOL regime stop)
  - R_TARGET = 4.2 (matches harvest_v2 HIGH_VOL regime target)
  - STOP_MARKET order = reduce_only, closes full position
  - Every CHECK_INTERVAL_SEC: scan positions, verify orders, replace if missing
  - On position closed (key deleted): cancel orphan orders
  - Publishes guardian state to quantum:dag3:hw_stops:latest (hash, TTL=2min)

Rollback:
  redis-cli SET quantum:dag3:hw_stops:disabled 1  → stops placing new orders
"""

import os
import sys
import time
import hmac
import hashlib
import logging
import signal
import json
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

# ── Path fix ──────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/qt/quantum_trader")
sys.path.insert(0, "/opt/quantum")

import redis as redis_lib

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s dag3 %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dag3")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST          = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT          = int(os.getenv("REDIS_PORT", "6379"))
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET  = os.getenv("BINANCE_API_SECRET", "")
BINANCE_BASE_URL    = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
CHECK_INTERVAL_SEC  = int(os.getenv("DAG3_INTERVAL_SEC", "30"))
R_STOP              = float(os.getenv("DAG3_R_STOP", "0.7"))   # stop at -0.7R
R_TARGET            = float(os.getenv("DAG3_R_TARGET", "4.2"))  # TP at +4.2R
POSITION_KEY_PREFIX = "quantum:position:"
STATE_KEY           = "quantum:dag3:hw_stops:latest"
STATE_TTL           = 120  # seconds
DISABLE_KEY         = "quantum:dag3:hw_stops:disabled"

# ── Precision map (futures contract specs) ────────────────────────────────
# price_precision: decimal places for price
# qty_precision:   decimal places for quantity
SYMBOL_PRECISION = {
    "BTCUSDT":  {"price": 1, "qty": 3},
    "ETHUSDT":  {"price": 2, "qty": 3},
    "LINKUSDT": {"price": 3, "qty": 0},
    "UNIUSDT":  {"price": 3, "qty": 0},
    "LTCUSDT":  {"price": 2, "qty": 3},
    "SOLUSDT":  {"price": 2, "qty": 0},
    "NEARUSDT": {"price": 4, "qty": 0},
    "BNBUSDT":  {"price": 2, "qty": 2},
    "DOGEUSDT": {"price": 5, "qty": 0},
    "AVAXUSDT": {"price": 2, "qty": 1},
}
DEFAULT_PRECISION = {"price": 4, "qty": 2}

# ── Graceful shutdown ─────────────────────────────────────────────────────
_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    logger.info("Signal %s — shutting down", sig)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Binance REST helpers ──────────────────────────────────────────────────

def _sign(params: dict, secret: str) -> str:
    query = urllib.parse.urlencode(params)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _binance_request(method: str, path: str, params: dict, signed: bool = True) -> dict:
    """Execute a signed Binance Futures REST request."""
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = _sign(params, BINANCE_API_SECRET)

    url = BINANCE_BASE_URL + path
    data = urllib.parse.urlencode(params).encode()

    req = urllib.request.Request(
        url if method == "GET" else url,
        data=data if method != "GET" else None,
        method=method,
    )
    req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
    if method == "GET":
        url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, method="GET")
        req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        logger.error("Binance HTTP %s %s → %s: %s", method, path, e.code, body)
        return {"error": body, "code": e.code}
    except Exception as e:
        logger.error("Binance request failed: %s", e)
        return {"error": str(e)}


def get_open_orders(symbol: str) -> list:
    return _binance_request("GET", "/fapi/v1/openOrders", {"symbol": symbol})


def cancel_order(symbol: str, order_id: int) -> dict:
    return _binance_request("DELETE", "/fapi/v1/order", {
        "symbol": symbol,
        "orderId": order_id,
    })


def place_stop_market(symbol: str, side: str, stop_price: float, close_qty: float) -> dict:
    """Place a STOP_MARKET reduce-only order."""
    prec = SYMBOL_PRECISION.get(symbol, DEFAULT_PRECISION)
    stop_str = f"{stop_price:.{prec['price']}f}"
    qty_str  = f"{close_qty:.{prec['qty']}f}"
    return _binance_request("POST", "/fapi/v1/order", {
        "symbol":           symbol,
        "side":             side,
        "type":             "STOP_MARKET",
        "stopPrice":        stop_str,
        "closePosition":    "true",
        "workingType":      "MARK_PRICE",
    })


def place_take_profit_market(symbol: str, side: str, tp_price: float) -> dict:
    """Place a TAKE_PROFIT_MARKET reduce-only order."""
    prec = SYMBOL_PRECISION.get(symbol, DEFAULT_PRECISION)
    tp_str = f"{tp_price:.{prec['price']}f}"
    return _binance_request("POST", "/fapi/v1/order", {
        "symbol":           symbol,
        "side":             side,
        "type":             "TAKE_PROFIT_MARKET",
        "stopPrice":        tp_str,
        "closePosition":    "true",
        "workingType":      "MARK_PRICE",
    })


# ── Position parsing ──────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_risk_usdt: float
    stop_price: float = 0.0
    tp_price: float = 0.0

    def close_side(self) -> str:
        return "SELL" if self.side.upper() in ("LONG", "BUY") else "BUY"


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_positions(r: redis_lib.Redis) -> list[Position]:
    keys = r.keys(f"{POSITION_KEY_PREFIX}*")
    positions = []
    for key in keys:
        raw = r.hgetall(key)
        if not raw:
            continue
        symbol        = raw.get(b"symbol", b"").decode()
        side          = raw.get(b"side", b"LONG").decode().upper()
        quantity      = _safe_float(raw.get(b"quantity"))
        entry_price   = _safe_float(raw.get(b"entry_price"))
        risk_usdt     = _safe_float(raw.get(b"entry_risk_usdt"))

        if not symbol or quantity <= 0 or entry_price <= 0 or risk_usdt <= 0:
            logger.warning("SKIP invalid position key=%s", key)
            continue

        stop_dist  = (R_STOP * risk_usdt) / quantity
        tp_dist    = (R_TARGET * risk_usdt) / quantity
        if side.upper() in ("LONG", "BUY"):
            stop_price = entry_price - stop_dist
            tp_price   = entry_price + tp_dist
        else:
            stop_price = entry_price + stop_dist
            tp_price   = entry_price - tp_dist

        positions.append(Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_risk_usdt=risk_usdt,
            stop_price=max(stop_price, 0.0001),
            tp_price=max(tp_price, 0.0001),
        ))
    return positions


# ── Guardian logic ────────────────────────────────────────────────────────

def ensure_hardware_stops(r: redis_lib.Redis) -> dict:
    """
    For each open position:
      1. Check if a STOP_MARKET and TAKE_PROFIT_MARKET order already exist
      2. If not (or if price drifted >5%), cancel stale and place fresh orders

    Returns a summary dict for the state hash.
    """
    positions = load_positions(r)
    if not positions:
        logger.info("No open positions — nothing to guard")
        return {"positions_guarded": "0", "status": "IDLE"}

    guarded = []
    errors  = []

    for pos in positions:
        logger.info(
            "GUARD %s side=%s qty=%s entry=%.4f stop=%.4f tp=%.4f",
            pos.symbol, pos.side, pos.quantity,
            pos.entry_price, pos.stop_price, pos.tp_price,
        )

        open_orders = get_open_orders(pos.symbol)
        if isinstance(open_orders, dict) and "error" in open_orders:
            logger.error("GET open_orders failed for %s: %s", pos.symbol, open_orders)
            errors.append(pos.symbol)
            continue

        existing_stop = None
        existing_tp   = None
        stale_ids     = []

        for order in open_orders:
            otype = order.get("type", "")
            if otype == "STOP_MARKET":
                existing_price = float(order.get("stopPrice", 0))
                drift_pct = abs(existing_price - pos.stop_price) / pos.stop_price * 100
                if drift_pct < 5.0:
                    existing_stop = order
                    logger.info("  STOP_MARKET OK id=%s price=%s drift=%.2f%%",
                                order["orderId"], order["stopPrice"], drift_pct)
                else:
                    logger.warning("  STOP_MARKET STALE drift=%.2f%% — replacing", drift_pct)
                    stale_ids.append(order["orderId"])
            elif otype == "TAKE_PROFIT_MARKET":
                existing_price = float(order.get("stopPrice", 0))
                drift_pct = abs(existing_price - pos.tp_price) / pos.tp_price * 100
                if drift_pct < 5.0:
                    existing_tp = order
                    logger.info("  TAKE_PROFIT_MARKET OK id=%s price=%s drift=%.2f%%",
                                order["orderId"], order["stopPrice"], drift_pct)
                else:
                    logger.warning("  TP STALE drift=%.2f%% — replacing", drift_pct)
                    stale_ids.append(order["orderId"])

        # Cancel stale
        for oid in stale_ids:
            res = cancel_order(pos.symbol, oid)
            logger.info("  CANCELLED stale order %s → %s", oid, res.get("status", "?"))

        # Place stop if missing
        if not existing_stop:
            result = place_stop_market(
                pos.symbol, pos.close_side(), pos.stop_price, pos.quantity
            )
            if "orderId" in result:
                logger.warning(
                    "[DAG3] STOP_MARKET PLACED %s stop=%.4f orderId=%s",
                    pos.symbol, pos.stop_price, result["orderId"]
                )
                r.hset(f"quantum:dag3:order:{pos.symbol}:stop", mapping={
                    "orderId": str(result["orderId"]),
                    "stopPrice": f"{pos.stop_price:.6f}",
                    "placed_ts": str(int(time.time())),
                })
            else:
                logger.error("[DAG3] STOP PLACE FAILED %s: %s", pos.symbol, result)
                errors.append(f"{pos.symbol}:stop")

        # Place TP if missing
        if not existing_tp:
            result = place_take_profit_market(
                pos.symbol, pos.close_side(), pos.tp_price
            )
            if "orderId" in result:
                logger.info(
                    "[DAG3] TP_MARKET PLACED %s tp=%.4f orderId=%s",
                    pos.symbol, pos.tp_price, result["orderId"]
                )
                r.hset(f"quantum:dag3:order:{pos.symbol}:tp", mapping={
                    "orderId": str(result["orderId"]),
                    "tpPrice": f"{pos.tp_price:.6f}",
                    "placed_ts": str(int(time.time())),
                })
            else:
                logger.warning("[DAG3] TP PLACE FAILED %s: %s (non-critical)", pos.symbol, result)

        guarded.append(pos.symbol)

    return {
        "positions_guarded": str(len(guarded)),
        "guarded":            ",".join(guarded),
        "errors":             ",".join(errors) if errors else "",
        "status":             "ERROR" if errors else "OK",
        "r_stop":             str(R_STOP),
        "r_target":           str(R_TARGET),
        "ts":                 str(int(time.time())),
    }


# ── Cancel orphan orders for closed positions ─────────────────────────────

def cancel_orphan_orders(r: redis_lib.Redis, active_symbols: set):
    """Cancel any DAG3 orders for symbols that no longer have open positions."""
    order_keys = r.keys("quantum:dag3:order:*")
    for key in order_keys:
        key_str = key.decode()
        # format: quantum:dag3:order:BTCUSDT:stop
        parts = key_str.split(":")
        if len(parts) < 5:
            continue
        symbol = parts[3]
        if symbol not in active_symbols:
            stored = r.hgetall(key)
            if stored:
                oid = stored.get(b"orderId", b"").decode()
                if oid:
                    logger.info("[DAG3] Cancel orphan order %s (position closed)", key_str)
                    cancel_order(symbol, int(oid))
            r.delete(key)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("[DAG3] Hardware TP/SL Guardian starting — base=%s R_stop=%.2f R_tp=%.2f",
                BINANCE_BASE_URL, R_STOP, R_TARGET)

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.error("[DAG3] BINANCE_API_KEY / BINANCE_API_SECRET not set — aborting")
        sys.exit(1)

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    # Startup: run immediately
    consecutive_errors = 0

    while _RUNNING:
        tick_start = time.monotonic()

        try:
            disabled = r.get(DISABLE_KEY)
            if disabled:
                logger.warning("[DAG3] DISABLED via %s — skipping tick", DISABLE_KEY)
            else:
                summary = ensure_hardware_stops(r)
                active  = set(summary.get("guarded", "").split(",")) - {""}
                cancel_orphan_orders(r, active)

                # Publish state
                summary["check_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                r.hset(STATE_KEY, mapping=summary)
                r.expire(STATE_KEY, STATE_TTL)

                status = summary.get("status", "?")
                errors = summary.get("errors", "")
                logger.info("[DAG3_TICK] guarded=%s status=%s errors=%s",
                            summary.get("positions_guarded", "0"), status, errors or "(none)")
                consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            logger.error("[DAG3] Exception (consecutive=%d): %s", consecutive_errors, e)
            if consecutive_errors >= 5:
                logger.critical("[DAG3] Too many consecutive errors — sleeping 120s")
                time.sleep(120)
                consecutive_errors = 0

        elapsed   = time.monotonic() - tick_start
        sleep_for = max(0.0, CHECK_INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    logger.info("[DAG3] Hardware TP/SL Guardian stopped")


if __name__ == "__main__":
    main()
