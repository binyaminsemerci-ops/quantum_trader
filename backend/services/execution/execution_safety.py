"""
Execution Safety Guard - Pre-execution validation and adjustment

SPRINT 1 - D7: Slippage + Retry Logic
Validates orders before execution to prevent:
- Excessive slippage
- "Order would immediately trigger" errors
- Invalid SL/TP placement
"""
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of order validation."""
    is_valid: bool
    reason: str
    adjusted_entry: Optional[float] = None
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None


class ExecutionSafetyGuard:
    """
    Pre-execution validation and adjustment for orders.
    
    Features:
    - Slippage validation (planned vs current market price)
    - SL/TP logic validation (correct side of entry)
    - Price adjustment within policy limits
    - PolicyStore integration for dynamic thresholds
    
    Args:
        policy_store: PolicyStore instance for dynamic config
        logger: Optional logger instance
    
    Example:
        guard = ExecutionSafetyGuard(policy_store)
        
        result = await guard.validate_and_adjust_order(
            symbol="BTCUSDT",
            side="buy",
            planned_entry_price=50000.0,
            current_market_price=50100.0,
            sl_price=49000.0,
            tp_price=52000.0
        )
        
        if not result.is_valid:
            logger.error(f"Order rejected: {result.reason}")
        elif result.adjusted_sl or result.adjusted_tp:
            logger.warning(f"Order adjusted: {result.reason}")
    """
    
    def __init__(self, policy_store=None, logger_instance=None):
        self.policy_store = policy_store
        self.logger = logger_instance or logger
        
        # Default thresholds (can be overridden by PolicyStore)
        self.default_max_slippage_pct = 0.005  # 0.5%
        self.default_max_sl_distance_pct = 0.10  # 10%
        self.default_min_sl_buffer_pct = 0.001  # 0.1%
        self.default_max_tp_distance_pct = 0.50  # 50%
        
        self.logger.info(
            "[SAFETY-GUARD] Initialized: max_slippage=0.5%, "
            "sl_buffer=0.1%, max_sl_distance=10%"
        )
    
    def _get_policy_value(self, key: str, default: float) -> float:
        """Get value from PolicyStore or use default."""
        if self.policy_store and hasattr(self.policy_store, 'get_policy'):
            try:
                value = self.policy_store.get_policy(key)
                if value is not None:
                    return float(value)
            except Exception as e:
                self.logger.debug(f"[SAFETY-GUARD] Could not read policy {key}: {e}")
        return default
    
    async def validate_and_adjust_order(
        self,
        symbol: str,
        side: str,
        planned_entry_price: float,
        current_market_price: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        leverage: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate order parameters and adjust if needed.
        
        Checks:
        1. Slippage between planned and current price
        2. SL placement (must be on correct side of entry)
        3. TP placement (must be on correct side of entry)
        4. SL/TP distances (within policy limits)
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            planned_entry_price: Price order was planned at
            current_market_price: Current market price
            sl_price: Stop loss price (optional)
            tp_price: Take profit price (optional)
            leverage: Leverage (optional, affects risk thresholds)
        
        Returns:
            ValidationResult with is_valid flag and adjusted prices
        """
        side_normalized = side.lower()
        is_long = side_normalized in ["buy", "long"]
        
        # Get policy thresholds
        max_slippage_pct = self._get_policy_value(
            "execution.max_slippage_pct",
            self.default_max_slippage_pct
        )
        max_sl_distance_pct = self._get_policy_value(
            "execution.max_sl_distance_pct",
            self.default_max_sl_distance_pct
        )
        min_sl_buffer_pct = self._get_policy_value(
            "execution.min_sl_buffer_pct",
            self.default_min_sl_buffer_pct
        )
        max_tp_distance_pct = self._get_policy_value(
            "execution.max_tp_distance_pct",
            self.default_max_tp_distance_pct
        )
        
        # Adjust thresholds for high leverage (tighter limits)
        if leverage and leverage > 10:
            max_slippage_pct *= 0.5  # Stricter for high leverage
            self.logger.debug(
                f"[SAFETY-GUARD] High leverage {leverage}x: "
                f"adjusted max_slippage to {max_slippage_pct*100:.2f}%"
            )
        
        # 1. Check slippage
        slippage = abs(planned_entry_price - current_market_price) / current_market_price
        if slippage > max_slippage_pct:
            self.logger.warning(
                f"[SAFETY-GUARD] ❌ Excessive slippage for {symbol}: "
                f"{slippage*100:.3f}% > {max_slippage_pct*100:.3f}% "
                f"(planned=${planned_entry_price:.2f}, current=${current_market_price:.2f})"
            )
            return ValidationResult(
                is_valid=False,
                reason=f"Slippage {slippage*100:.3f}% exceeds limit {max_slippage_pct*100:.3f}%"
            )
        
        # Use current market price as entry
        adjusted_entry = current_market_price
        adjusted_sl = sl_price
        adjusted_tp = tp_price
        adjustments_made = []
        
        # 2. Validate and adjust SL
        if sl_price is not None:
            if is_long:
                # LONG: SL must be below entry
                if sl_price >= current_market_price:
                    # Too high - adjust down with buffer
                    buffer = current_market_price * min_sl_buffer_pct
                    adjusted_sl = current_market_price - buffer
                    adjustments_made.append(
                        f"SL adjusted from ${sl_price:.2f} to ${adjusted_sl:.2f} "
                        f"(was above entry)"
                    )
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} LONG: SL ${sl_price:.2f} >= entry ${current_market_price:.2f}, "
                        f"adjusted to ${adjusted_sl:.2f}"
                    )
                
                # Check SL distance
                sl_distance_pct = (current_market_price - adjusted_sl) / current_market_price
                if sl_distance_pct > max_sl_distance_pct:
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} LONG: SL too far "
                        f"({sl_distance_pct*100:.1f}% > {max_sl_distance_pct*100:.1f}%)"
                    )
                    return ValidationResult(
                        is_valid=False,
                        reason=f"SL distance {sl_distance_pct*100:.1f}% exceeds limit {max_sl_distance_pct*100:.1f}%"
                    )
            else:
                # SHORT: SL must be above entry
                if sl_price <= current_market_price:
                    # Too low - adjust up with buffer
                    buffer = current_market_price * min_sl_buffer_pct
                    adjusted_sl = current_market_price + buffer
                    adjustments_made.append(
                        f"SL adjusted from ${sl_price:.2f} to ${adjusted_sl:.2f} "
                        f"(was below entry)"
                    )
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} SHORT: SL ${sl_price:.2f} <= entry ${current_market_price:.2f}, "
                        f"adjusted to ${adjusted_sl:.2f}"
                    )
                
                # Check SL distance
                sl_distance_pct = (adjusted_sl - current_market_price) / current_market_price
                if sl_distance_pct > max_sl_distance_pct:
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} SHORT: SL too far "
                        f"({sl_distance_pct*100:.1f}% > {max_sl_distance_pct*100:.1f}%)"
                    )
                    return ValidationResult(
                        is_valid=False,
                        reason=f"SL distance {sl_distance_pct*100:.1f}% exceeds limit {max_sl_distance_pct*100:.1f}%"
                    )
        
        # 3. Validate and adjust TP
        if tp_price is not None:
            if is_long:
                # LONG: TP must be above entry
                if tp_price <= current_market_price:
                    # Too low - adjust up with buffer
                    buffer = current_market_price * min_sl_buffer_pct
                    adjusted_tp = current_market_price + buffer
                    adjustments_made.append(
                        f"TP adjusted from ${tp_price:.2f} to ${adjusted_tp:.2f} "
                        f"(was below entry)"
                    )
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} LONG: TP ${tp_price:.2f} <= entry ${current_market_price:.2f}, "
                        f"adjusted to ${adjusted_tp:.2f}"
                    )
                
                # Check TP distance
                tp_distance_pct = (adjusted_tp - current_market_price) / current_market_price
                if tp_distance_pct > max_tp_distance_pct:
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} LONG: TP very far "
                        f"({tp_distance_pct*100:.1f}% > {max_tp_distance_pct*100:.1f}%), "
                        f"allowing but logging"
                    )
            else:
                # SHORT: TP must be below entry
                if tp_price >= current_market_price:
                    # Too high - adjust down with buffer
                    buffer = current_market_price * min_sl_buffer_pct
                    adjusted_tp = current_market_price - buffer
                    adjustments_made.append(
                        f"TP adjusted from ${tp_price:.2f} to ${adjusted_tp:.2f} "
                        f"(was above entry)"
                    )
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} SHORT: TP ${tp_price:.2f} >= entry ${current_market_price:.2f}, "
                        f"adjusted to ${adjusted_tp:.2f}"
                    )
                
                # Check TP distance
                tp_distance_pct = (current_market_price - adjusted_tp) / current_market_price
                if tp_distance_pct > max_tp_distance_pct:
                    self.logger.warning(
                        f"[SAFETY-GUARD] {symbol} SHORT: TP very far "
                        f"({tp_distance_pct*100:.1f}% > {max_tp_distance_pct*100:.1f}%), "
                        f"allowing but logging"
                    )
        
        # Success
        if adjustments_made:
            reason = "Order adjusted: " + "; ".join(adjustments_made)
            self.logger.info(f"[SAFETY-GUARD] ✅ {symbol}: {reason}")
        else:
            reason = "Order validated successfully"
            self.logger.debug(f"[SAFETY-GUARD] ✅ {symbol}: No adjustments needed")
        
        return ValidationResult(
            is_valid=True,
            reason=reason,
            adjusted_entry=adjusted_entry if adjusted_entry != planned_entry_price else None,
            adjusted_sl=adjusted_sl if adjusted_sl != sl_price else None,
            adjusted_tp=adjusted_tp if adjusted_tp != tp_price else None
        )
    
    async def get_current_market_price(self, symbol: str, client=None) -> Optional[float]:
        """
        Get current market price for symbol.
        
        Args:
            symbol: Trading symbol
            client: Binance client (optional, for fetching live price)
        
        Returns:
            Current market price or None if unavailable
        """
        if client and hasattr(client, '_signed_request'):
            try:
                # Fetch mark price (most accurate for futures)
                ticker = await client._signed_request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
                if ticker and 'price' in ticker:
                    return float(ticker['price'])
            except Exception as e:
                self.logger.warning(f"[SAFETY-GUARD] Could not fetch market price for {symbol}: {e}")
        
        return None
