"""
Safe Order Executor - Robust retry logic for order placement

SPRINT 1 - D7: Slippage + Retry Logic
Wraps Binance order submission with:
- Exponential backoff retry
- -2021 error handling (order would immediately trigger)
- Network error handling
- PolicyStore integration for retry limits
"""
import logging
import asyncio
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# [PHASE 1] Exit Order Gateway for observability
try:
    from backend.services.execution.exit_order_gateway import submit_exit_order
    EXIT_GATEWAY_AVAILABLE = True
except ImportError:
    EXIT_GATEWAY_AVAILABLE = False
    logger.warning("[EXIT_GATEWAY] Not available in SafeOrderExecutor - will place orders directly")


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: Optional[str] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    attempts: int = 0
    duration_sec: float = 0.0


class SafeOrderExecutor:
    """
    Robust order executor with retry logic and error handling.
    
    Features:
    - Exponential backoff retry (0.5s, 1s, 2s, 4s, ...)
    - -2021 error handling with price adjustment
    - Network error handling
    - PolicyStore integration for dynamic config
    - Detailed logging for debugging
    
    Args:
        policy_store: PolicyStore instance for dynamic config
        safety_guard: ExecutionSafetyGuard for price adjustment
        logger: Optional logger instance
    
    Example:
        executor = SafeOrderExecutor(policy_store, safety_guard)
        
        result = await executor.place_order_with_safety(
            submit_func=client.futures_create_order,
            order_params={"symbol": "BTCUSDT", "side": "BUY", ...},
            symbol="BTCUSDT",
            side="buy",
            sl_price=49000.0,
            tp_price=52000.0
        )
        
        if result.success:
            logger.info(f"Order placed: {result.order_id}")
        else:
            logger.error(f"Order failed: {result.error_message}")
    """
    
    def __init__(self, policy_store=None, safety_guard=None, logger_instance=None):
        self.policy_store = policy_store
        self.safety_guard = safety_guard
        self.logger = logger_instance or logger
        
        # Default retry config (can be overridden by PolicyStore)
        self.default_max_retries = 5
        self.default_backoff_base_sec = 0.5
        self.default_max_backoff_sec = 10.0
        
        # Error codes to retry
        self.retryable_errors = {
            -2021,  # Order would immediately trigger
            -1001,  # Internal error
            -1003,  # Too many requests (though D6 handles this)
            -1015,  # Too many orders (though D6 handles this)
            -1021,  # Timestamp error
            -2013,  # Order does not exist (can retry)
            -2015,  # Invalid API key (transient if rate limited)
        }
        
        # -2021 specific handling
        self.sl_adjustment_buffer_pct = 0.001  # 0.1% buffer for SL adjustment
        
        self.logger.info(
            "[SAFE-EXECUTOR] Initialized: max_retries=5, "
            "backoff_base=0.5s, sl_buffer=0.1%"
        )
    
    def _get_policy_value(self, key: str, default: Any) -> Any:
        """Get value from PolicyStore or use default."""
        if self.policy_store and hasattr(self.policy_store, 'get_policy'):
            try:
                value = self.policy_store.get_policy(key)
                if value is not None:
                    return value
            except Exception as e:
                self.logger.debug(f"[SAFE-EXECUTOR] Could not read policy {key}: {e}")
        return default
    
    async def place_order_with_safety(
        self,
        submit_func: Callable,
        order_params: Dict[str, Any],
        symbol: str,
        side: str,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        client: Optional[Any] = None,
        order_type: str = "entry"
    ) -> OrderResult:
        """
        Place order with retry logic and error handling.
        
        Args:
            submit_func: Async function to submit order (e.g., client.futures_create_order)
            order_params: Parameters to pass to submit_func
            symbol: Trading symbol
            side: "buy" or "sell"
            sl_price: Stop loss price (for adjustment if needed)
            tp_price: Take profit price (for adjustment if needed)
            client: Binance client (for fetching current price if needed)
            order_type: "entry", "sl", "tp", "trailing" (for logging)
        
        Returns:
            OrderResult with success status and details
        """
        max_retries = self._get_policy_value("execution.max_order_retries", self.default_max_retries)
        backoff_base = self._get_policy_value("execution.retry_backoff_base_sec", self.default_backoff_base_sec)
        max_backoff = self._get_policy_value("execution.max_retry_backoff_sec", self.default_max_backoff_sec)
        
        start_time = time.time()
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            attempt += 1
            
            try:
                self.logger.debug(
                    f"[SAFE-EXECUTOR] Attempt {attempt}/{max_retries} for {symbol} "
                    f"{order_type} order: {order_params.get('side', 'N/A')}"
                )
                
                # [PHASE 1] Route exit orders through gateway for observability
                # Only applies to exit orders (tp/sl/trailing), not entry orders
                is_exit_order = order_type in ["sl", "tp", "trailing", "partial_tp", "breakeven"]
                
                if is_exit_order and EXIT_GATEWAY_AVAILABLE and client:
                    # Determine order_kind for gateway
                    order_kind_map = {
                        "sl": "sl",
                        "tp": "tp",
                        "trailing": "trailing",
                        "partial_tp": "partial_tp",
                        "breakeven": "breakeven"
                    }
                    order_kind = order_kind_map.get(order_type, "other_exit")
                    
                    # Route through gateway (gateway logs & forwards to submit_func)
                    # Note: We pass client but gateway will call submit_func internally
                    response = await submit_exit_order(
                        module_name="safe_order_executor",
                        symbol=symbol,
                        order_params=order_params,
                        order_kind=order_kind,
                        client=client,
                        explanation=f"SafeOrderExecutor {order_type} attempt {attempt}/{max_retries}"
                    )
                else:
                    # Not an exit order OR gateway not available - use direct submission
                    if asyncio.iscoroutinefunction(submit_func):
                        response = await submit_func(**order_params)
                    else:
                        response = submit_func(**order_params)
                
                # Extract order ID
                order_id = None
                if isinstance(response, dict):
                    order_id = response.get('orderId') or response.get('order_id')
                elif hasattr(response, 'orderId'):
                    order_id = response.orderId
                elif hasattr(response, 'order_id'):
                    order_id = response.order_id
                
                duration = time.time() - start_time
                
                self.logger.info(
                    f"[SAFE-EXECUTOR] ✅ {symbol} {order_type} order placed: "
                    f"order_id={order_id}, attempts={attempt}, duration={duration:.2f}s"
                )
                
                return OrderResult(
                    success=True,
                    order_id=str(order_id) if order_id else None,
                    attempts=attempt,
                    duration_sec=duration
                )
            
            except Exception as e:
                last_error = e
                error_code = getattr(e, 'code', None)
                error_message = str(e)
                
                self.logger.warning(
                    f"[SAFE-EXECUTOR] ❌ Attempt {attempt}/{max_retries} failed for {symbol} "
                    f"{order_type} order: code={error_code}, msg={error_message}"
                )
                
                # Check if error is retryable
                if error_code not in self.retryable_errors:
                    self.logger.error(
                        f"[SAFE-EXECUTOR] Non-retryable error {error_code} for {symbol}, "
                        f"aborting after {attempt} attempts"
                    )
                    break
                
                # Handle -2021 specifically (order would immediately trigger)
                if error_code == -2021 and order_type == "sl" and sl_price is not None:
                    adjusted_sl = await self._adjust_sl_for_2021(
                        symbol, side, sl_price, client
                    )
                    if adjusted_sl and adjusted_sl != sl_price:
                        self.logger.info(
                            f"[SAFE-EXECUTOR] Adjusting SL for {symbol} from "
                            f"${sl_price:.2f} to ${adjusted_sl:.2f}"
                        )
                        order_params['stopPrice'] = adjusted_sl
                        sl_price = adjusted_sl  # Update for next retry
                    else:
                        self.logger.warning(
                            f"[SAFE-EXECUTOR] Could not adjust SL for {symbol}, "
                            f"will retry with same price"
                        )
                
                # Calculate backoff
                if attempt < max_retries:
                    backoff = min(backoff_base * (2 ** (attempt - 1)), max_backoff)
                    self.logger.debug(
                        f"[SAFE-EXECUTOR] Waiting {backoff:.2f}s before retry {attempt + 1}"
                    )
                    await asyncio.sleep(backoff)
        
        # All retries failed
        duration = time.time() - start_time
        error_code = getattr(last_error, 'code', None)
        error_message = str(last_error) if last_error else "Unknown error"
        
        self.logger.error(
            f"[SAFE-EXECUTOR] ❌ All {attempt} attempts failed for {symbol} {order_type} order: "
            f"code={error_code}, msg={error_message}, duration={duration:.2f}s"
        )
        
        return OrderResult(
            success=False,
            error_code=error_code,
            error_message=error_message,
            attempts=attempt,
            duration_sec=duration
        )
    
    async def _adjust_sl_for_2021(
        self,
        symbol: str,
        side: str,
        current_sl: float,
        client: Any
    ) -> Optional[float]:
        """
        Adjust SL price to avoid -2021 error.
        
        -2021 happens when SL would immediately trigger (e.g., LONG with SL above current price).
        We need to fetch current price and adjust SL with buffer.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            current_sl: Current SL price that triggered error
            client: Binance client to fetch current price
        
        Returns:
            Adjusted SL price or None if adjustment failed
        """
        if not client:
            self.logger.warning(f"[SAFE-EXECUTOR] Cannot adjust SL for {symbol}: no client provided")
            return None
        
        try:
            # Fetch current mark price
            if hasattr(client, '_signed_request'):
                ticker = await client._signed_request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
                current_price = float(ticker.get('markPrice', 0))
            elif hasattr(client, 'futures_mark_price'):
                ticker = client.futures_mark_price(symbol=symbol)
                current_price = float(ticker.get('markPrice', 0))
            else:
                self.logger.warning(f"[SAFE-EXECUTOR] Client does not support mark price fetch")
                return None
            
            if current_price <= 0:
                self.logger.warning(f"[SAFE-EXECUTOR] Invalid current price {current_price} for {symbol}")
                return None
            
            # Get buffer from policy
            buffer_pct = self._get_policy_value(
                "execution.sl_adjustment_buffer_pct",
                self.sl_adjustment_buffer_pct
            )
            
            # Adjust based on side
            is_long = side.lower() in ["buy", "long"]
            
            if is_long:
                # LONG: SL must be BELOW current price
                buffer = current_price * buffer_pct
                adjusted_sl = current_price - buffer
                
                # Ensure adjusted SL is actually lower than current SL (don't make it worse)
                if adjusted_sl >= current_sl:
                    adjusted_sl = current_sl * 0.999  # Move 0.1% lower than original
                
                self.logger.info(
                    f"[SAFE-EXECUTOR] {symbol} LONG: current_price=${current_price:.2f}, "
                    f"adjusted SL from ${current_sl:.2f} to ${adjusted_sl:.2f}"
                )
            else:
                # SHORT: SL must be ABOVE current price
                buffer = current_price * buffer_pct
                adjusted_sl = current_price + buffer
                
                # Ensure adjusted SL is actually higher than current SL (don't make it worse)
                if adjusted_sl <= current_sl:
                    adjusted_sl = current_sl * 1.001  # Move 0.1% higher than original
                
                self.logger.info(
                    f"[SAFE-EXECUTOR] {symbol} SHORT: current_price=${current_price:.2f}, "
                    f"adjusted SL from ${current_sl:.2f} to ${adjusted_sl:.2f}"
                )
            
            return adjusted_sl
        
        except Exception as e:
            self.logger.error(f"[SAFE-EXECUTOR] Failed to adjust SL for {symbol}: {e}")
            return None
    
    async def place_order_batch_with_safety(
        self,
        orders: list,
        submit_func: Callable,
        symbol: str,
        client: Optional[Any] = None
    ) -> list:
        """
        Place multiple orders with safety (e.g., SL + TP together).
        
        Args:
            orders: List of order_params dicts
            submit_func: Async function to submit orders
            symbol: Trading symbol
            client: Binance client
        
        Returns:
            List of OrderResult objects
        """
        results = []
        
        for idx, order_params in enumerate(orders):
            order_type = order_params.get('type', 'unknown')
            side = order_params.get('side', 'N/A')
            
            result = await self.place_order_with_safety(
                submit_func=submit_func,
                order_params=order_params,
                symbol=symbol,
                side=side,
                sl_price=order_params.get('stopPrice'),
                tp_price=order_params.get('price') if order_type == 'TAKE_PROFIT_MARKET' else None,
                client=client,
                order_type=order_type
            )
            
            results.append(result)
            
            # Log batch progress
            self.logger.debug(
                f"[SAFE-EXECUTOR] Batch order {idx + 1}/{len(orders)} for {symbol}: "
                f"{'✅' if result.success else '❌'}"
            )
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            f"[SAFE-EXECUTOR] Batch complete for {symbol}: "
            f"{success_count}/{len(results)} orders placed"
        )
        
        return results
