"""
hardware_stop_manager.py
========================
Safety Floor: Binance-native STOP_MARKET + TAKE_PROFIT_MARKET for every position.

Sits beneath ExitBrain / software exits — does NOT replace them.

TOGGLE:  HARDWARE_STOPS_ENABLED=0  → entire module is bypassed.
TESTNET: Inherits base URL from the `client` object already initialised in executor_service.py.

Redis keys used:
  quantum:runtime:position_mode          Hash  {dualSidePosition, ts}  TTL 300 s
  quantum:position:<symbol>:hardware_stops  Hash  {sl_order_id, tp_order_id,
                                                    mode, ts, entry_price, qty}

Supports:
  - USDT-M futures (fapi)
  - USDC-M futures (if using same client / fapi — override USDC_FAPI_BASE env if needed)
  - CROSS margin (no ISOLATED flag overrides from this module)
  - One-way mode  (positionSide omitted)
  - Hedge mode    (positionSide=LONG|SHORT auto-detected)
  - Testnet / Mainnet (follows existing client.FUTURES_URL)

Author: quantum_trader auto-executor layer
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional

logger = logging.getLogger("hardware_stop_manager")

# ──────────────────────────────────────────────────────────────────────────────
# Feature toggle — set HARDWARE_STOPS_ENABLED=0 to fully disable without
# touching any other code.
# ──────────────────────────────────────────────────────────────────────────────
_ENABLED: bool = os.getenv("HARDWARE_STOPS_ENABLED", "1").strip() not in ("0", "false", "no", "off")

# Cache TTL for position-mode query
_POSITION_MODE_TTL_SEC: int = 300  # 5 minutes

# Redis keys
_REDIS_MODE_KEY: str = "quantum:runtime:position_mode"
_REDIS_STOPS_KEY: str = "quantum:position:{symbol}:hardware_stops"

# Fallback SL/TP percentages when neither Risk Kernel nor ATR is available
_DEFAULT_SL_PCT: float = 0.015   # 1.5%
_DEFAULT_TP_MULT: float = 2.0    # 2 × SL distance (2R)

# How close (pct of entry) existing order can be before it is considered "correct"
_PRICE_TOLERANCE_FRACTION: float = 0.001   # 0.1 %

# Max retries for transient Binance errors
_MAX_PLACE_RETRIES: int = 2


# ──────────────────────────────────────────────────────────────────────────────
# Low-level Binance call helper (mirrors safe_futures_call in executor_service)
# to avoid importing that module (circular-import risk).
# ──────────────────────────────────────────────────────────────────────────────
_UNSIGNED_METHODS = frozenset({
    "futures_time", "futures_exchange_info", "futures_ticker",
    "futures_orderbook", "futures_klines", "futures_trades",
})


def _call(client, func_name: str, **kwargs):
    """Call a python-binance futures method, auto-adding recvWindow."""
    if client is None:
        raise RuntimeError("Binance client is not initialised")
    if func_name not in _UNSIGNED_METHODS:
        kwargs.setdefault("recvWindow", 10_000)
    return getattr(client, func_name)(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# A) POSITION MODE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def get_position_mode(client, redis_client) -> bool:
    """
    Return True if Binance account is in hedge (dual-side) mode.

    Result is cached in Redis key `quantum:runtime:position_mode` for 5 min.
    Safe to call on every order — cache avoids API hammering.
    """
    try:
        cached = redis_client.hgetall(_REDIS_MODE_KEY)
        if cached:
            ts = float(cached.get("ts", 0))
            if time.time() - ts < _POSITION_MODE_TTL_SEC:
                result = cached.get("dualSidePosition", "false").lower() == "true"
                logger.debug("position_mode cache hit: dual=%s", result)
                return result

        # Cache miss or stale — query exchange
        response = _call(client, "futures_get_position_mode")
        # python-binance returns {"dualSidePosition": bool}
        dual = bool(response.get("dualSidePosition", False))

        redis_client.hset(_REDIS_MODE_KEY, mapping={
            "dualSidePosition": str(dual).lower(),
            "ts": str(time.time()),
        })
        redis_client.expire(_REDIS_MODE_KEY, _POSITION_MODE_TTL_SEC + 60)

        logger.info("position_mode fetched from exchange: dual=%s", dual)
        return dual

    except Exception as exc:
        logger.warning("get_position_mode failed (%s) — defaulting to one-way", exc)
        return False  # Safest default: omit positionSide


# ──────────────────────────────────────────────────────────────────────────────
# B/C) PRICE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _tick_round(price: float, tick_size: float, direction: str) -> float:
    """Round price to nearest tick, using ceiling for SL-short/TP-long etc."""
    if tick_size <= 0:
        return round(price, 8)
    tick = Decimal(str(tick_size))
    p = Decimal(str(price))
    if direction == "up":
        return float((p / tick).to_integral_value(rounding=ROUND_UP) * tick)
    else:
        return float((p / tick).to_integral_value(rounding=ROUND_DOWN) * tick)


def _compute_prices(
    side: str,          # "BUY" (long) or "SELL" (short)
    entry_price: float,
    tick_size: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    atr: Optional[float] = None,
    sl_atr_mult: float = 1.5,
    tp_r_mult: float = 2.0,
) -> tuple[float, float]:
    """
    Resolve final SL and TP prices.

    Priority:
      1. Explicit sl_price / tp_price from Risk Kernel
      2. ATR-based fallback  (sl = 1.5×ATR; tp = 2R)
      3. Hard-coded percentage fallback (_DEFAULT_SL_PCT)

    Returns (stop_loss_price, take_profit_price) validated and tick-rounded.
    Raises ValueError if prices fail directional sanity checks.
    """
    is_long = side.upper() == "BUY"

    # ── Resolve SL ──────────────────────────────────────────
    if sl_price and sl_price > 0:
        resolved_sl = sl_price
    elif atr and atr > 0:
        dist = sl_atr_mult * atr
        resolved_sl = entry_price - dist if is_long else entry_price + dist
    else:
        dist = entry_price * _DEFAULT_SL_PCT
        resolved_sl = entry_price - dist if is_long else entry_price + dist

    # ── Resolve TP ──────────────────────────────────────────
    sl_dist = abs(entry_price - resolved_sl)
    if tp_price and tp_price > 0:
        resolved_tp = tp_price
    elif atr and atr > 0:
        dist = sl_atr_mult * tp_r_mult * atr
        resolved_tp = entry_price + dist if is_long else entry_price - dist
    else:
        resolved_tp = (entry_price + sl_dist * tp_r_mult
                       if is_long else entry_price - sl_dist * tp_r_mult)

    # ── Tick-round (conservative direction) ─────────────────
    if is_long:
        # SL must be BELOW entry  → round down
        final_sl = _tick_round(resolved_sl, tick_size, "down")
        # TP must be ABOVE entry  → round up
        final_tp = _tick_round(resolved_tp, tick_size, "up")
    else:
        # SL must be ABOVE entry  → round up
        final_sl = _tick_round(resolved_sl, tick_size, "up")
        # TP must be BELOW entry  → round down
        final_tp = _tick_round(resolved_tp, tick_size, "down")

    # ── Validate ────────────────────────────────────────────
    if final_sl <= 0 or final_tp <= 0:
        raise ValueError(f"Price <= 0: SL={final_sl} TP={final_tp}")

    if is_long:
        if final_sl >= entry_price:
            raise ValueError(f"LONG SL {final_sl} must be < entry {entry_price}")
        if final_tp <= entry_price:
            raise ValueError(f"LONG TP {final_tp} must be > entry {entry_price}")
    else:
        if final_sl <= entry_price:
            raise ValueError(f"SHORT SL {final_sl} must be > entry {entry_price}")
        if final_tp >= entry_price:
            raise ValueError(f"SHORT TP {final_tp} must be < entry {entry_price}")

    return final_sl, final_tp


# ──────────────────────────────────────────────────────────────────────────────
# D) IDEMPOTENCY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _stops_redis_key(symbol: str) -> str:
    return _REDIS_STOPS_KEY.format(symbol=symbol.upper())


def _load_cached_stops(redis_client, symbol: str) -> dict:
    return redis_client.hgetall(_stops_redis_key(symbol)) or {}


def _save_cached_stops(redis_client, symbol: str, sl_order_id, tp_order_id,
                       mode: str, entry_price: float, qty: float):
    redis_client.hset(_stops_redis_key(symbol), mapping={
        "sl_order_id": str(sl_order_id),
        "tp_order_id": str(tp_order_id),
        "mode": mode,
        "ts": str(time.time()),
        "entry_price": str(entry_price),
        "qty": str(qty),
    })
    redis_client.expire(_stops_redis_key(symbol), 86_400)  # 24 h


def _delete_cached_stops(redis_client, symbol: str):
    redis_client.delete(_stops_redis_key(symbol))


def _get_existing_hardware_orders(client, symbol: str) -> tuple[dict, dict]:
    """
    Query live open orders and return (existing_sl, existing_tp) as dicts.
    Only considers reduceOnly STOP_MARKET / TAKE_PROFIT_MARKET.
    Returns empty dict if not found.
    """
    sl_order: dict = {}
    tp_order: dict = {}
    try:
        orders = _call(client, "futures_get_open_orders", symbol=symbol.upper())
        for o in orders:
            if not o.get("reduceOnly", False) and not o.get("closePosition", False):
                continue
            otype = o.get("type", "")
            if otype == "STOP_MARKET" and not sl_order:
                sl_order = o
            elif otype == "TAKE_PROFIT_MARKET" and not tp_order:
                tp_order = o
    except Exception as exc:
        logger.warning("_get_existing_hardware_orders failed for %s: %s", symbol, exc)
    return sl_order, tp_order


def _orders_match(existing_price: float, target_price: float,
                  entry_price: float) -> bool:
    if entry_price <= 0:
        return False
    return abs(existing_price - target_price) / entry_price < _PRICE_TOLERANCE_FRACTION


# ──────────────────────────────────────────────────────────────────────────────
# B) MAIN PUBLIC FUNCTION: place_hardware_stops
# ──────────────────────────────────────────────────────────────────────────────

def place_hardware_stops(
    client,
    redis_client,
    *,
    symbol: str,
    side: str,               # "BUY" or "SELL"  (direction of open position)
    entry_price: float,
    qty: float,
    sl_price: Optional[float] = None,   # from Risk Kernel preferred
    tp_price: Optional[float] = None,   # from Risk Kernel preferred
    atr: Optional[float] = None,        # fallback ATR value
    tick_size: Optional[float] = None,  # will be fetched if None
    is_paper: bool = False,
) -> dict:
    """
    Place STOP_MARKET + TAKE_PROFIT_MARKET orders on Binance after a fill.

    Returns dict with keys: placed (bool), sl_order_id, tp_order_id, skipped_reason.
    Never raises — errors are logged and the main trade is NOT failed.
    """
    result = {"placed": False, "sl_order_id": None, "tp_order_id": None,
              "skipped_reason": None}

    # ── Feature toggle ───────────────────────────────────────
    if not _ENABLED:
        result["skipped_reason"] = "HARDWARE_STOPS_ENABLED=0"
        logger.info("[%s] hardware stops DISABLED via env", symbol)
        return result

    if is_paper:
        result["skipped_reason"] = "paper_trading"
        return result

    if client is None:
        result["skipped_reason"] = "no_binance_client"
        logger.warning("[%s] skipping hardware stops: no client", symbol)
        return result

    try:
        # ── A) Detect position mode ──────────────────────────
        hedge_mode = get_position_mode(client, redis_client)
        mode_str = "hedge" if hedge_mode else "oneway"

        # ── Fetch tick size if not supplied ─────────────────
        if tick_size is None or tick_size <= 0:
            tick_size = _fetch_tick_size(client, symbol)

        # ── C) Compute validated SL + TP prices ─────────────
        final_sl, final_tp = _compute_prices(
            side=side,
            entry_price=entry_price,
            tick_size=tick_size,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr,
        )

        logger.info(
            "[%s] hardware_stops target: side=%s entry=%.6f SL=%.6f TP=%.6f "
            "mode=%s atr=%s",
            symbol, side, entry_price, final_sl, final_tp, mode_str, atr,
        )

        # ── D) Idempotency check ─────────────────────────────
        existing_sl_order, existing_tp_order = _get_existing_hardware_orders(
            client, symbol
        )
        cached = _load_cached_stops(redis_client, symbol)

        sl_needs_place = True
        tp_needs_place = True

        if existing_sl_order:
            ex_sl_price = float(existing_sl_order.get("stopPrice", 0))
            if _orders_match(ex_sl_price, final_sl, entry_price):
                logger.info("[%s] SL already correct @ %.6f — skipping", symbol, ex_sl_price)
                result["sl_order_id"] = str(existing_sl_order["orderId"])
                sl_needs_place = False
            else:
                # Cancel stale SL
                _safe_cancel(client, symbol, existing_sl_order["orderId"], "SL")

        if existing_tp_order:
            ex_tp_price = float(existing_tp_order.get("stopPrice", 0))
            if _orders_match(ex_tp_price, final_tp, entry_price):
                logger.info("[%s] TP already correct @ %.6f — skipping", symbol, ex_tp_price)
                result["tp_order_id"] = str(existing_tp_order["orderId"])
                tp_needs_place = False
            else:
                _safe_cancel(client, symbol, existing_tp_order["orderId"], "TP")

        if not sl_needs_place and not tp_needs_place:
            result["placed"] = True
            result["skipped_reason"] = "already_correct"
            return result

        # ── B) Build common order params ─────────────────────
        is_long = side.upper() == "BUY"
        # For a long position: close side = SELL; for short: close side = BUY
        close_side = "SELL" if is_long else "BUY"
        position_side_param = {}
        if hedge_mode:
            position_side_param = {"positionSide": "LONG" if is_long else "SHORT"}

        # ── Place SL ─────────────────────────────────────────
        if sl_needs_place:
            sl_id = _place_order_with_retry(
                client, symbol=symbol, side=close_side,
                order_type="STOP_MARKET", stop_price=final_sl,
                position_side_param=position_side_param, label="SL",
            )
            if sl_id:
                result["sl_order_id"] = sl_id

        # ── Place TP ─────────────────────────────────────────
        if tp_needs_place:
            tp_id = _place_order_with_retry(
                client, symbol=symbol, side=close_side,
                order_type="TAKE_PROFIT_MARKET", stop_price=final_tp,
                position_side_param=position_side_param, label="TP",
            )
            if tp_id:
                result["tp_order_id"] = tp_id

        # ── Save to Redis ────────────────────────────────────
        if result["sl_order_id"] or result["tp_order_id"]:
            _save_cached_stops(
                redis_client, symbol,
                sl_order_id=result["sl_order_id"] or "FAILED",
                tp_order_id=result["tp_order_id"] or "FAILED",
                mode=mode_str,
                entry_price=entry_price,
                qty=qty,
            )
            result["placed"] = True
            logger.info(
                "[%s] hardware_stops placed: SL=%s TP=%s mode=%s",
                symbol, result["sl_order_id"], result["tp_order_id"], mode_str,
            )

        return result

    except Exception as exc:
        logger.error("[%s] place_hardware_stops UNEXPECTED ERROR: %s", symbol, exc, exc_info=True)
        result["skipped_reason"] = f"exception:{exc}"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# E) LIFECYCLE: cancel on full close
# ──────────────────────────────────────────────────────────────────────────────

def cancel_hardware_stops(client, redis_client, *, symbol: str, is_paper: bool = False):
    """
    Cancel open SL/TP hardware stops when position is fully closed.
    Safe to call even if orders are already gone.
    """
    if not _ENABLED or is_paper or client is None:
        return

    try:
        sl_order, tp_order = _get_existing_hardware_orders(client, symbol)
        if sl_order:
            _safe_cancel(client, symbol, sl_order["orderId"], "SL")
        if tp_order:
            _safe_cancel(client, symbol, tp_order["orderId"], "TP")
        _delete_cached_stops(redis_client, symbol)
        logger.info("[%s] hardware_stops cancelled and cache cleared", symbol)
    except Exception as exc:
        logger.warning("[%s] cancel_hardware_stops error: %s", symbol, exc)


# ──────────────────────────────────────────────────────────────────────────────
# E) LIFECYCLE: adjust on partial fill
# ──────────────────────────────────────────────────────────────────────────────

def sync_hardware_stops_on_partial(
    client,
    redis_client,
    *,
    symbol: str,
    side: str,
    entry_price: float,
    remaining_qty: float,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    atr: Optional[float] = None,
    tick_size: Optional[float] = None,
    is_paper: bool = False,
):
    """
    When a position is partially closed, cancel existing hardware stops and
    re-place correct ones for the remaining quantity.

    If `closePosition=True` semantics are used (which we do), the exchange
    closes *all* of the position regardless of qty — so this call simply
    re-places after cancelling to keep Redis and exchange in sync.
    """
    if not _ENABLED or is_paper or client is None:
        return

    if remaining_qty <= 0:
        cancel_hardware_stops(client, redis_client, symbol=symbol)
        return

    # Cancel existing stops
    sl_order, tp_order = _get_existing_hardware_orders(client, symbol)
    if sl_order:
        _safe_cancel(client, symbol, sl_order["orderId"], "SL-partial")
    if tp_order:
        _safe_cancel(client, symbol, tp_order["orderId"], "TP-partial")

    # Re-place for remaining qty
    place_hardware_stops(
        client, redis_client,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        qty=remaining_qty,
        sl_price=sl_price,
        tp_price=tp_price,
        atr=atr,
        tick_size=tick_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_tick_size(client, symbol: str) -> float:
    """Fetch tickSize from exchange info. Fallback: 0.01."""
    try:
        info = _call(client, "futures_exchange_info")
        for s in info.get("symbols", []):
            if s["symbol"] == symbol.upper():
                for f in s.get("filters", []):
                    if f["filterType"] == "PRICE_FILTER":
                        return float(f["tickSize"])
    except Exception as exc:
        logger.warning("[%s] _fetch_tick_size error: %s — using 0.01", symbol, exc)
    return 0.01


def _place_order_with_retry(
    client,
    symbol: str,
    side: str,
    order_type: str,
    stop_price: float,
    position_side_param: dict,
    label: str,
) -> Optional[str]:
    """
    Place a reduceOnly stop/tp order. Returns orderId string or None on failure.
    Uses closePosition=True so Binance closes the entire position size — making
    this qty-agnostic (handles partial fills automatically on the exchange side).
    """
    params = dict(
        symbol=symbol.upper(),
        side=side,
        type=order_type,
        stopPrice=f"{stop_price:.10f}".rstrip("0").rstrip("."),
        closePosition=True,
        workingType="MARK_PRICE",
        priceProtect=True,
        **position_side_param,
    )

    last_exc = None
    for attempt in range(1, _MAX_PLACE_RETRIES + 1):
        try:
            resp = _call(client, "futures_create_order", **params)
            order_id = str(resp["orderId"])
            logger.info(
                "[%s] %s placed: orderId=%s stopPrice=%s attempt=%d",
                symbol, label, order_id, stop_price, attempt,
            )
            return order_id
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            # -2021 = ReduceOnly order rejected (position already gone)
            # -1111 = precision error  (should not happen after tick rounding)
            if "-2021" in err_str:
                logger.warning("[%s] %s rejected -2021 (no position): %s", symbol, label, exc)
                return None
            logger.warning(
                "[%s] %s place attempt %d/%d failed: %s",
                symbol, label, attempt, _MAX_PLACE_RETRIES, exc,
            )
            time.sleep(0.5)

    logger.error("[%s] %s all retries exhausted: %s", symbol, label, last_exc)
    return None


def _safe_cancel(client, symbol: str, order_id, label: str):
    """Cancel an order, swallowing errors."""
    try:
        _call(client, "futures_cancel_order",
              symbol=symbol.upper(), orderId=int(order_id))
        logger.info("[%s] cancelled %s orderId=%s", symbol, label, order_id)
    except Exception as exc:
        # -2011 = unknown order (already filled/cancelled — OK)
        if "-2011" not in str(exc):
            logger.warning("[%s] cancel %s orderId=%s failed: %s", symbol, label, order_id, exc)
