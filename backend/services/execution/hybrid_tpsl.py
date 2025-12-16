from __future__ import annotations

import asyncio
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional, Tuple

from .safe_order_executor import SafeOrderExecutor

logger = logging.getLogger(__name__)

_CONF_HIGH = 0.80
_CONF_LOW = 0.20
_MIN_CALLBACK = 0.1  # percent
_MAX_CALLBACK = 5.0  # percent


def _decimals_from_step(step: float) -> int:
    step_str = f"{step:.10f}".rstrip("0")
    if "." not in step_str:
        return 0
    return len(step_str.split(".")[-1])


def _round_price(price: float, tick_size: float, *, favor: str) -> float:
    """Round price to tick size without increasing risk."""
    if tick_size <= 0:
        return round(price, 8)
    precision = _decimals_from_step(tick_size)
    quant = Decimal(str(tick_size))
    rounding = ROUND_DOWN if favor == "down" else ROUND_UP
    return float(Decimal(str(price)).quantize(quant, rounding=rounding).quantize(Decimal(f"1.{'0'*precision}")))


def _round_qty(qty: float, step_size: float) -> float:
    if step_size <= 0:
        return qty
    quant = Decimal(str(step_size))
    rounded = Decimal(str(qty)).quantize(quant, rounding=ROUND_DOWN)
    return float(rounded)


def _opposite_side(side: str) -> str:
    return "SELL" if str(side).upper() in {"BUY", "LONG"} else "BUY"


def _calc_prices(entry_price: float, pct: float, side: str, *, target: bool) -> float:
    is_long = str(side).upper() in {"BUY", "LONG"}
    if target:
        return entry_price * (1 + pct) if is_long else entry_price * (1 - pct)
    return entry_price * (1 - pct) if is_long else entry_price * (1 + pct)


def calculate_hybrid_levels(
    *,
    entry_price: float,
    side: str,
    risk_sl_percent: float,
    base_tp_percent: float,
    ai_tp_percent: Optional[float],
    ai_trail_percent: Optional[float],
    confidence: float,
) -> Dict[str, Any]:
    """Blend baseline risk controls with AI overlays while never widening risk."""
    conf = max(0.0, min(float(confidence or 0.0), 1.0))
    baseline_sl = max(0.0, float(risk_sl_percent or 0.0))
    base_tp = max(0.0, float(base_tp_percent or 0.0))
    ai_tp = max(0.0, float(ai_tp_percent or 0.0))
    ai_trail = max(0.0, float(ai_trail_percent or 0.0))

    final_sl = baseline_sl
    final_tp = base_tp
    trail_cb = 0.0
    mode = "base"

    if conf < _CONF_LOW:
        logger.info("[HYBRID-TPSL] Using base TP because confidence=%.2f", conf)
        return {
            "final_sl_percent": final_sl,
            "final_tp_percent": final_tp,
            "trail_callback_percent": trail_cb,
            "mode": mode,
        }

    if conf >= _CONF_HIGH:
        if ai_tp > base_tp:
            logger.info(
                "[HYBRID-TPSL] Extended TP from %.2f%% -> %.2f%% (confidence=%.2f)",
                base_tp * 100,
                ai_tp * 100,
                conf,
            )
            final_tp = ai_tp
            mode = "ai-extended"
        if ai_trail > 0:
            trail_cb = ai_trail
            mode = "hybrid" if mode != "ai-extended" else "hybrid"
            logger.info("[HYBRID-TPSL] Trailing activated: %.3f%% callback", trail_cb * 100)
        # High confidence = tighten SL by 10% without widening risk
        tightened = max(0.0, baseline_sl * 0.9)
        if tightened < baseline_sl:
            logger.info(
                "[HYBRID-TPSL] SL tightened from -%.3f%% -> -%.3f%%",
                baseline_sl * 100,
                tightened * 100,
            )
            final_sl = tightened
    else:
        # Mid confidence: keep conservative TP, optional softer trailing
        if ai_trail > 0:
            trail_cb = ai_trail * 0.5
            mode = "trailing"
            logger.info(
                "[HYBRID-TPSL] Soft trailing enabled: %.3f%% (confidence=%.2f)",
                trail_cb * 100,
                conf,
            )

    return {
        "final_sl_percent": final_sl,
        "final_tp_percent": final_tp,
        "trail_callback_percent": trail_cb,
        "mode": mode,
    }


async def _fetch_filters(client: Any, symbol: str) -> Tuple[float, float]:
    """Return (tick_size, step_size) best-effort."""
    tick_size = 0.0
    step_size = 0.0

    # DO NOT call _configure_symbol here - it would reset leverage to default!
    # Symbol is already configured when the market order was placed.
    # Just fetch the cached filters directly.

    sym_filters = {}
    if hasattr(client, "_symbol_filters"):
        sym_filters = getattr(client, "_symbol_filters", {}).get(symbol, {})

    if not sym_filters and hasattr(client, "get_exchange_info"):
        try:
            info = await client.get_exchange_info() if asyncio.iscoroutinefunction(client.get_exchange_info) else client.get_exchange_info()
            for entry in info.get("symbols", []):
                if entry.get("symbol") == symbol:
                    sym_filters = {f["filterType"]: f for f in entry.get("filters", [])}
                    break
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not fetch exchange info for %s: %s", symbol, exc)

    price_filter = sym_filters.get("PRICE_FILTER") if isinstance(sym_filters, dict) else None
    lot_filter = sym_filters.get("LOT_SIZE") if isinstance(sym_filters, dict) else None

    try:
        tick_size = float(price_filter.get("tickSize", 0.0)) if price_filter else 0.0
    except Exception:
        tick_size = 0.0
    try:
        step_size = float(lot_filter.get("stepSize", 0.0)) if lot_filter else 0.0
    except Exception:
        step_size = 0.0

    return tick_size, step_size


async def place_hybrid_orders(
    *,
    client: Any,
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    risk_sl_percent: float,
    base_tp_percent: float,
    ai_tp_percent: Optional[float],
    ai_trail_percent: Optional[float],
    confidence: float,
    policy_store: Optional[Any] = None,
) -> bool:
    """
    Cancel stale TP/SL and place baseline SL + TP with AI overlay and trailing.
    Returns True if baseline orders were placed successfully.
    
    D7 Enhancement: Uses SafeOrderExecutor for robust retry logic.
    """
    if client is None:
        logger.warning("[HYBRID-TPSL] No client provided; skipping hybrid placement")
        return False
    if not hasattr(client, "_signed_request"):
        logger.warning("[HYBRID-TPSL] Client lacks _signed_request; skipping hybrid placement")
        return False

    if entry_price <= 0 or qty <= 0:
        logger.error("[HYBRID-TPSL] Invalid entry data: price=%.8f qty=%.8f", entry_price, qty)
        return False

    # Initialize SafeOrderExecutor (D7)
    safe_executor = SafeOrderExecutor(
        policy_store=policy_store,
        safety_guard=None,  # Safety guard already applied at entry order level
        logger_instance=logger
    )

    side_norm = str(side).upper()
    is_long = side_norm in {"BUY", "LONG"}
    exit_side = _opposite_side(side_norm)

    levels = calculate_hybrid_levels(
        entry_price=entry_price,
        side=side_norm,
        risk_sl_percent=risk_sl_percent,
        base_tp_percent=base_tp_percent,
        ai_tp_percent=ai_tp_percent,
        ai_trail_percent=ai_trail_percent,
        confidence=confidence,
    )

    tick_size, step_size = await _fetch_filters(client, symbol)

    # Round qty using adapter helper if available
    try:
        rounded_qty = float(client._round_quantity(symbol, qty)) if hasattr(client, "_round_quantity") else qty
    except Exception:
        rounded_qty = _round_qty(qty, step_size)

    if rounded_qty <= 0:
        logger.error("[HYBRID-TPSL] Rounded quantity invalid (%.8f)", rounded_qty)
        return False

    final_sl_pct = levels["final_sl_percent"] or risk_sl_percent or 0.0005
    final_tp_pct = levels["final_tp_percent"] or base_tp_percent
    trail_cb_pct = levels.get("trail_callback_percent") or 0.0

    sl_price = _calc_prices(entry_price, final_sl_pct, side_norm, target=False)
    base_tp_price = _calc_prices(entry_price, base_tp_percent, side_norm, target=True)
    ai_tp_price = _calc_prices(entry_price, final_tp_pct, side_norm, target=True)

    # Ensure SL/TP are not overlapping with entry
    if is_long:
        if sl_price >= entry_price:
            sl_price = entry_price * 0.999
        if base_tp_price <= entry_price:
            base_tp_price = entry_price * 1.001
    else:
        if sl_price <= entry_price:
            sl_price = entry_price * 1.001
        if base_tp_price >= entry_price:
            base_tp_price = entry_price * 0.999

    sl_price = _round_price(sl_price, tick_size, favor="up" if is_long else "down")
    base_tp_price = _round_price(base_tp_price, tick_size, favor="down" if is_long else "up")
    ai_tp_price = _round_price(ai_tp_price, tick_size, favor="down" if is_long else "up")

    # Cancel existing protection first
    try:
        await client._signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})
    except Exception as exc:
        logger.warning("[HYBRID-TPSL] Could not cancel existing orders for %s: %s", symbol, exc)

    # D7: Submit helper replaced with SafeOrderExecutor
    async def _submit_with_safety(payload: Dict[str, Any], label: str, order_type: str) -> Optional[Any]:
        """Wrapper to use SafeOrderExecutor for order submission."""
        # Create async wrapper for _signed_request
        async def submit_func(**params):
            return await client._signed_request("POST", "/fapi/v1/order", params)
        
        result = await safe_executor.place_order_with_safety(
            submit_func=submit_func,
            order_params=payload,
            symbol=symbol,
            side=side_norm,
            sl_price=payload.get('stopPrice') if order_type == "sl" else None,
            tp_price=payload.get('stopPrice') if order_type == "tp" else None,
            client=client,
            order_type=order_type
        )
        
        if result.success:
            logger.info(
                f"[HYBRID-TPSL] {label} order placed: order_id={result.order_id}, "
                f"attempts={result.attempts}"
            )
            return {"orderId": result.order_id}  # Return dict for compatibility
        else:
            logger.error(
                f"[HYBRID-TPSL] {label} order failed after {result.attempts} attempts: "
                f"{result.error_message}"
            )
            return None

    async def _place_static_fallback() -> bool:
        logger.warning("[HYBRID-TPSL] Fallback to static TP/SL applied")
        try:
            await client._signed_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})
        except Exception:
            pass
        static_sl = {
            "symbol": symbol,
            "side": exit_side,
            "type": "STOP_MARKET",
            "stopPrice": f"{sl_price:.{_decimals_from_step(tick_size)}f}" if tick_size else sl_price,
            "closePosition": True,
            "workingType": "MARK_PRICE",
            "positionSide": "LONG" if is_long else "SHORT",
        }
        static_tp = {
            "symbol": symbol,
            "side": exit_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{base_tp_price:.{_decimals_from_step(tick_size)}f}" if tick_size else base_tp_price,
            "closePosition": True,
            "workingType": "MARK_PRICE",
            "positionSide": "LONG" if is_long else "SHORT",
        }
        sl_ok = await _submit_with_safety(static_sl, "STATIC_SL", "sl")
        tp_ok = await _submit_with_safety(static_tp, "STATIC_TP", "tp")
        return bool(sl_ok and tp_ok)

    # 1) Stop loss (non-negotiable)
    sl_payload = {
        "symbol": symbol,
        "side": exit_side,
        "type": "STOP_MARKET",
        "stopPrice": f"{sl_price:.{_decimals_from_step(tick_size)}f}" if tick_size else sl_price,
        "closePosition": True,
        "workingType": "MARK_PRICE",
        "positionSide": "LONG" if is_long else "SHORT",
    }
    sl_result = await _submit_with_safety(sl_payload, "STOP_LOSS", "sl")
    if not sl_result:
        await _place_static_fallback()
        return False

    # Determine position split
    ai_active = confidence >= _CONF_HIGH and ai_tp_percent and final_tp_pct > base_tp_percent
    base_qty = rounded_qty
    ai_qty = 0.0
    if ai_active:
        base_qty = _round_qty(rounded_qty * 0.5, step_size) or rounded_qty
        ai_qty = max(0.0, rounded_qty - base_qty)
        if ai_qty < step_size:
            ai_qty = 0.0

    # 2) Base TP (always)
    base_tp_payload = {
        "symbol": symbol,
        "side": exit_side,
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": f"{base_tp_price:.{_decimals_from_step(tick_size)}f}" if tick_size else base_tp_price,
        "workingType": "MARK_PRICE",
        "quantity": f"{base_qty:.{_decimals_from_step(step_size) if step_size else 6}f}",
        "positionSide": "LONG" if is_long else "SHORT",
    }
    base_result = await _submit_with_safety(base_tp_payload, "BASE_TP", "tp")
    if not base_result:
        await _place_static_fallback()
        return False

    # 3) AI TP (optional extension)
    if ai_active and ai_qty > 0:
        ai_tp_payload = {
            "symbol": symbol,
            "side": exit_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{ai_tp_price:.{_decimals_from_step(tick_size)}f}" if tick_size else ai_tp_price,
            "workingType": "MARK_PRICE",
            "quantity": f"{ai_qty:.{_decimals_from_step(step_size) if step_size else 6}f}",
            "positionSide": "LONG" if is_long else "SHORT",
        }
        await _submit_with_safety(ai_tp_payload, "AI_TP", "tp")

    # 4) Trailing stop (protect profits)
    if trail_cb_pct > 0:
        cb_rate = min(max(trail_cb_pct * 100, _MIN_CALLBACK), _MAX_CALLBACK)
        activation_pct = max(trail_cb_pct * 2, 0.002)
        activation_price = _calc_prices(entry_price, activation_pct, side_norm, target=True)
        activation_price = _round_price(
            activation_price,
            tick_size,
            favor="down" if is_long else "up",
        )
        trail_payload = {
            "symbol": symbol,
            "side": exit_side,
            "type": "TRAILING_STOP_MARKET",
            "quantity": f"{rounded_qty:.{_decimals_from_step(step_size) if step_size else 6}f}",
            "callbackRate": f"{cb_rate:.2f}",
            "activationPrice": f"{activation_price:.{_decimals_from_step(tick_size)}f}" if tick_size else activation_price,
            "positionSide": "LONG" if is_long else "SHORT",
        }
        await _submit_with_safety(trail_payload, "TRAILING_STOP", "trailing")

    return True
