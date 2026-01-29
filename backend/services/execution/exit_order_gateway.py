"""
Exit Order Gateway
==================

Central gateway for ALL exit-related orders (TP/SL/trailing).

Phase 1 Goals (Current):
- Observability: Log all exit orders with module ownership
- Soft Guards: Warn about ownership conflicts in EXIT_BRAIN_V3 mode
- No Hard Blocking: Allow all orders through for backward compatibility

Future Phases:
- Enforce single MUSCLE (only Exit Brain Executor can place orders)
- Hard block legacy modules in EXIT_BRAIN_V3 mode
- Coordinate with Exit Brain plan lifecycle
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from backend.config.exit_mode import (
    get_exit_mode,
    is_exit_brain_mode,
    is_legacy_exit_mode,
    is_exit_brain_live_fully_enabled,
    EXIT_MODE_EXIT_BRAIN_V3,
    EXIT_MODE_LEGACY
)

logger = logging.getLogger(__name__)

# Module ownership tracking
# In EXIT_BRAIN_V3 mode, ideally ONLY exit_brain_executor should place orders
LEGACY_MODULES = [
    "position_monitor",
    "hybrid_tpsl", 
    "trailing_stop_manager",
    "dynamic_trailing_rearm",
    "event_driven_executor"
]

EXPECTED_EXIT_BRAIN_MODULES = [
    "exit_brain_executor",  # Future executor
    "exit_brain_v3"
]

# Order kind tracking for analytics
VALID_ORDER_KINDS = [
    "tp", "sl", "trailing", "breakeven", "partial_tp", "other_exit",
    "hard_sl",  # Exit Brain V3 risk floor SL
    "loss_guard_emergency",  # Exit Brain V3 max loss guard
    "emergency_exit",  # Exit Brain V3 emergency full exit
    "partial_close"  # Exit Brain V3 partial close
]


class ExitOrderMetrics:
    """
    Simple metrics tracker for exit order flows.
    
    Helps identify BRAIN/MUSCLE conflicts.
    """
    
    def __init__(self):
        self.orders_by_module: Dict[str, int] = {}
        self.orders_by_kind: Dict[str, int] = {}
        self.ownership_conflicts: int = 0
        self.blocked_legacy_orders: int = 0
        self.total_orders: int = 0
        
    def record_order(
        self, 
        module_name: str, 
        order_kind: str,
        is_conflict: bool = False,
        is_blocked: bool = False
    ):
        """Record order placement for metrics."""
        self.total_orders += 1
        self.orders_by_module[module_name] = self.orders_by_module.get(module_name, 0) + 1
        self.orders_by_kind[order_kind] = self.orders_by_kind.get(order_kind, 0) + 1
        
        if is_conflict:
            self.ownership_conflicts += 1
        
        if is_blocked:
            self.blocked_legacy_orders += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_orders": self.total_orders,
            "orders_by_module": self.orders_by_module,
            "orders_by_kind": self.orders_by_kind,
            "ownership_conflicts": self.ownership_conflicts,
            "blocked_legacy_orders": self.blocked_legacy_orders
        }


# Global metrics instance
_exit_metrics = ExitOrderMetrics()


def get_exit_order_metrics() -> ExitOrderMetrics:
    """Get global exit order metrics."""
    return _exit_metrics


async def submit_exit_order(
    module_name: str,
    symbol: str,
    order_params: Dict[str, Any],
    order_kind: str,
    client: Any,  # Binance client or adapter
    explanation: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Central gateway for all exit-related orders.
    
    Phase 1 Implementation:
    - Logs all exit orders with ownership tracking
    - Soft warnings for ownership conflicts in EXIT_BRAIN_V3 mode
    - No hard blocking (allows all orders through)
    
    Args:
        module_name: Name of calling module (e.g. "position_monitor", "hybrid_tpsl")
        symbol: Trading symbol
        order_params: Order parameters dict for futures_create_order
        order_kind: Type of exit order ("tp", "sl", "trailing", etc.)
        client: Binance client or adapter with futures_create_order method
        explanation: Optional human-readable explanation of why order is being placed
        
    Returns:
        Order result dict from exchange, or None if order failed
        
    Raises:
        Exception: If order submission fails (propagates from client)
    """
    exit_mode = get_exit_mode()
    order_type = order_params.get("type", "UNKNOWN")
    
    # Validate order_kind
    if order_kind not in VALID_ORDER_KINDS:
        logger.warning(
            f"[EXIT_GATEWAY] Unknown order_kind='{order_kind}'. "
            f"Valid kinds: {VALID_ORDER_KINDS}"
        )
    
    # Build context for logging
    context = {
        "module": module_name,
        "symbol": symbol,
        "order_kind": order_kind,
        "order_type": order_type,
        "exit_mode": exit_mode,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if explanation:
        context["explanation"] = explanation
    
    # Check for ownership conflicts and hard blocking (LIVE mode)
    is_conflict = False
    is_blocked = False
    live_mode_active = is_exit_brain_live_fully_enabled()
    
    if exit_mode == EXIT_MODE_EXIT_BRAIN_V3:
        # In Exit Brain mode, legacy modules should NOT place orders
        if module_name in LEGACY_MODULES:
            is_conflict = True
            
            # HARD BLOCK in LIVE mode
            if live_mode_active:
                is_blocked = True
                logger.error(
                    f"[EXIT_GUARD] 🛑 BLOCKED: Legacy module '{module_name}' attempted to place "
                    f"{order_kind} ({order_type}) for {symbol} in EXIT_BRAIN_V3 LIVE mode. "
                    f"Executor is the single MUSCLE. Order NOT sent to exchange. "
                    f"Context: {context}"
                )
                
                # Record metrics and return None (order blocked)
                _exit_metrics.record_order(module_name, order_kind, is_conflict=True, is_blocked=True)
                return None
            else:
                # SOFT WARNING in SHADOW mode
                logger.warning(
                    f"[EXIT_GUARD] 🚨 OWNERSHIP CONFLICT: Legacy module '{module_name}' "
                    f"placing {order_kind} ({order_type}) for {symbol} in EXIT_BRAIN_V3 mode. "
                    f"This indicates BRAIN+MUSCLE conflict. Exit Brain Executor should own exit orders. "
                    f"Context: {context}"
                )
        elif module_name not in EXPECTED_EXIT_BRAIN_MODULES:
            # Unknown module - warn but allow
            logger.warning(
                f"[EXIT_GUARD] ℹ️  Unknown module '{module_name}' placing exit order in EXIT_BRAIN_V3 mode. "
                f"Context: {context}"
            )
        else:
            # Expected Exit Brain module - allow and log success
            mode_str = "LIVE" if live_mode_active else "SHADOW"
            logger.info(
                f"[EXIT_GUARD] ✅ Exit Brain module '{module_name}' placing {order_kind} for {symbol} "
                f"(mode={mode_str}). Context: {context}"
            )
    else:
        # Legacy mode - all modules allowed
        logger.debug(
            f"[EXIT_GATEWAY] {module_name} placing {order_kind} ({order_type}) for {symbol} "
            f"in LEGACY mode. Context: {context}"
        )
    
    # Record metrics
    _exit_metrics.record_order(module_name, order_kind, is_conflict, is_blocked)
    
    # Actually submit order to exchange
    try:
        # Check if client has futures_create_order method
        if not hasattr(client, 'futures_create_order'):
            logger.error(
                f"[EXIT_GATEWAY] Client {type(client).__name__} has no futures_create_order method. "
                f"Cannot submit order for {symbol}."
            )
            return None
        
        # [PRECISION LAYER] Apply centralized quantization to all order parameters
        from backend.domains.exits.exit_brain_v3.precision import (
            quantize_price, quantize_stop_price, quantize_quantity
        )
        
        # Quantize price (for limit orders)
        if 'price' in order_params:
            original_price = float(order_params['price'])
            quantized_price = quantize_price(symbol, original_price, client)
            order_params['price'] = str(quantized_price)
            logger.debug(
                f"[EXIT_GATEWAY] {symbol} price: {original_price:.8f} -> {quantized_price:.8f}"
            )
        
        # Quantize stopPrice (for stop orders)
        if 'stopPrice' in order_params:
            original_stop = float(order_params['stopPrice'])
            quantized_stop = quantize_stop_price(symbol, original_stop, client)
            order_params['stopPrice'] = str(quantized_stop)
            logger.debug(
                f"[EXIT_GATEWAY] {symbol} stopPrice: {original_stop:.8f} -> {quantized_stop:.8f}"
            )
        
        # Quantize quantity (for all orders)
        if 'quantity' in order_params:
            original_qty = float(order_params['quantity'])
            quantized_qty = quantize_quantity(symbol, original_qty, client)
            if quantized_qty == 0.0:
                logger.error(
                    f"[EXIT_GATEWAY] {symbol}: Quantity {original_qty} too small after quantization. "
                    f"Order NOT submitted."
                )
                return None
            order_params['quantity'] = str(quantized_qty)
            logger.debug(
                f"[EXIT_GATEWAY] {symbol} quantity: {original_qty:.8f} -> {quantized_qty:.8f}"
            )
        
        # [REDUCE-ONLY FIX] Always set reduceOnly=true for exit orders (futures)
        # This allows small exit orders below minNotional threshold
        if order_kind in ['tp', 'sl', 'trailing', 'hard_sl', 'partial_tp', 'breakeven', 'partial_close', 'emergency_exit']:
            # Only add reduceOnly if closePosition is not already set
            if 'closePosition' not in order_params or not order_params.get('closePosition'):
                order_params['reduceOnly'] = True
                logger.debug(
                    f"[EXIT_GATEWAY] {symbol}: Added reduceOnly=true for {order_kind} order"
                )
        
        # [MIN_NOTIONAL GUARD] Check if order meets minimum notional value
        min_notional = 5.0  # Binance futures minimum
        if 'quantity' in order_params:
            try:
                # Get current mark price to calculate notional
                ticker = client.futures_symbol_ticker(symbol=symbol)
                mark_price = float(ticker.get('price', 0))
                quantity = float(order_params['quantity'])
                notional = mark_price * quantity
                
                if notional < min_notional:
                    # If reduceOnly is set, we can proceed (Binance allows small reduce-only orders)
                    if order_params.get('reduceOnly') or order_params.get('closePosition'):
                        logger.warning(
                            f"[EXIT_GATEWAY] {symbol}: Notional ${notional:.2f} < ${min_notional:.2f} "
                            f"but reduceOnly/closePosition is set, allowing order."
                        )
                    else:
                        # Try to increase quantity to meet minimum
                        min_qty_needed = min_notional / mark_price
                        from backend.domains.exits.exit_brain_v3.precision import quantize_quantity
                        adjusted_qty = quantize_quantity(symbol, min_qty_needed, client)
                        
                        if adjusted_qty > 0:
                            logger.warning(
                                f"[EXIT_GATEWAY] {symbol}: Notional ${notional:.2f} < ${min_notional:.2f}. "
                                f"Increasing quantity {quantity} -> {adjusted_qty} to meet minimum."
                            )
                            order_params['quantity'] = str(adjusted_qty)
                        else:
                            logger.error(
                                f"[EXIT_GATEWAY] {symbol}: Notional ${notional:.2f} < ${min_notional:.2f} "
                                f"and cannot increase quantity. Skipping order to avoid -4164 error."
                            )
                            return None
            except Exception as notional_err:
                logger.warning(
                    f"[EXIT_GATEWAY] {symbol}: Could not verify minNotional: {notional_err}. "
                    f"Proceeding with order."
                )
        
        # Log order submission attempt (include positionSide if present for hedge mode visibility)
        position_side_str = f", positionSide={order_params.get('positionSide', 'N/A')}" if 'positionSide' in order_params else ""
        reduce_only_str = f", reduceOnly={order_params.get('reduceOnly', False)}"
        logger.info(
            f"[EXIT_GATEWAY] 📤 Submitting {order_kind} order: "
            f"module={module_name}, symbol={symbol}, type={order_type}{position_side_str}{reduce_only_str}, "
            f"params_keys={list(order_params.keys())}"
        )
        
        # ========================================================================
        # POLICY ENFORCEMENT: BLOCK ALL CONDITIONAL ORDERS
        # ========================================================================
        # Quantum Trader architecture: ALL exits must use internal intents,
        # executed as MARKET orders only. Conditional orders (STOP_MARKET,
        # TAKE_PROFIT_MARKET, etc.) bypass internal decision pipeline.
        #
        # Enforcement: Hard fail-closed at gateway choke point.
        # Owner: Exit Brain v3.5 (sole exit decision authority).
        # ========================================================================
        BLOCKED_CONDITIONAL_TYPES = [
            'STOP', 'STOP_MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT',
            'TAKE_PROFIT', 'TAKE_PROFIT_MARKET', 'TAKE_PROFIT_LIMIT',
            'TRAILING_STOP_MARKET'
        ]
        
        if order_type in BLOCKED_CONDITIONAL_TYPES:
            logger.error(
                f"[EXIT_GATEWAY] 🚨 POLICY VIOLATION: Conditional order blocked. "
                f"type={order_type}, module={module_name}, symbol={symbol}, "
                f"params={order_params}"
            )
            raise ValueError(
                f"Conditional orders not allowed (type={order_type}). "
                f"Use internal intents with MARKET execution only. "
                f"See Exit Brain v3.5 architecture for proper exit flow."
            )
        
        # Forward to actual exchange client (MARKET orders only)
        result = client.futures_create_order(**order_params)
        
        # Log success
        order_id = result.get('orderId', 'UNKNOWN') if result else 'NONE'
        logger.info(
            f"[EXIT_GATEWAY] ✅ Order placed successfully: "
            f"module={module_name}, symbol={symbol}, order_id={order_id}, kind={order_kind}"
        )
        
        return result
        
    except Exception as e:
        logger.error(
            f"[EXIT_GATEWAY] ❌ Order submission failed: "
            f"module={module_name}, symbol={symbol}, kind={order_kind}, error={e}",
            exc_info=True
        )
        raise  # Propagate exception to caller for their error handling


def log_exit_metrics_summary():
    """
    Log summary of exit order metrics.
    
    Useful for understanding which modules are acting as MUSCLE.
    """
    from backend.config.exit_mode import is_exit_brain_live_fully_enabled
    
    metrics = _exit_metrics.get_summary()
    live_mode = is_exit_brain_live_fully_enabled()
    mode_str = f"{get_exit_mode()} {'(LIVE)' if live_mode else ''}"
    
    logger.info(
        f"[EXIT_METRICS] Summary: "
        f"total_orders={metrics['total_orders']}, "
        f"conflicts={metrics['ownership_conflicts']}, "
        f"blocked={metrics['blocked_legacy_orders']}, "
        f"mode={mode_str}"
    )
    
    if metrics['orders_by_module']:
        logger.info(
            f"[EXIT_METRICS] Orders by module: {metrics['orders_by_module']}"
        )
    
    if metrics['orders_by_kind']:
        logger.info(
            f"[EXIT_METRICS] Orders by kind: {metrics['orders_by_kind']}"
        )
    
    if metrics['blocked_legacy_orders'] > 0:
        logger.warning(
            f"[EXIT_METRICS] 🛑 Blocked {metrics['blocked_legacy_orders']} legacy module orders "
            f"in EXIT_BRAIN_V3 LIVE mode (expected behavior)."
        )
    
    if metrics['ownership_conflicts'] > 0:
        logger.warning(
            f"[EXIT_METRICS] ⚠️  Detected {metrics['ownership_conflicts']} ownership conflicts. "
            f"Legacy modules attempting exit orders in EXIT_BRAIN_V3 mode."
        )
