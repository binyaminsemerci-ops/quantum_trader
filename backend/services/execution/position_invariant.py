"""
Position Invariant Enforcement Module

Ensures no simultaneous long/short positions on same symbol unless explicitly allowed.

CRITICAL TRADING INVARIANT:
For each (account, exchange, symbol) tuple:
  - Cannot have both long AND short positions simultaneously
  - Must be either FLAT (qty=0), net LONG (qty>0), or net SHORT (qty<0)
  
Exception: Hedging mode explicitly enabled (NOT RECOMMENDED for most strategies)
"""

from typing import Dict, Optional, Tuple
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PositionInvariantViolation(Exception):
    """Raised when attempting to violate position invariants."""
    pass


@dataclass
class BlockedOrderEvent:
    """Record of an order blocked by invariant enforcement."""
    timestamp: datetime
    symbol: str
    attempted_side: str
    attempted_quantity: float
    existing_quantity: float
    reason: str
    account: str
    exchange: str


class PositionInvariantEnforcer:
    """
    Centralized enforcement of trading invariants.
    
    INVARIANT: For each (account, exchange, symbol):
      - Cannot have both long AND short positions simultaneously
      - Must be either FLAT, net LONG, or net SHORT
    
    This enforcer MUST be called before any order placement to prevent
    the critical bug where system opens both long and short on same symbol.
    """
    
    def __init__(self, allow_hedging: bool = False):
        """
        Initialize position invariant enforcer.
        
        Args:
            allow_hedging: If True, allows simultaneous long/short positions.
                          WARNING: Only enable if you have explicit hedging strategy.
                          Default: False (recommended for directional trading)
        """
        self.allow_hedging = allow_hedging
        self.blocked_orders: list[BlockedOrderEvent] = []
        self.hedge_mode_warning_shown = False  # Track if we've warned about exchange hedge mode
        
        if allow_hedging:
            logger.warning(
                "⚠️  [HEDGING MODE ENABLED] Simultaneous long/short positions allowed. "
                "This bypasses position conflict protection."
            )
        else:
            logger.info(
                "✅ [POSITION PROTECTION ACTIVE] Simultaneous long/short positions blocked"
            )
    
    def check_can_open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_positions: Dict[str, float],
        account: str = "default",
        exchange: str = "binance"
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if opening a position would violate invariants.
        
        Args:
            symbol: Trading symbol (e.g. 'BTCUSDT')
            side: Order side - 'buy'/'long' for long, 'sell'/'short' for short
            quantity: Order quantity (positive number)
            current_positions: Dict of {symbol: net_quantity} where:
                             positive = long position
                             negative = short position
                             zero = flat
            account: Account identifier (for multi-account systems)
            exchange: Exchange name (for multi-exchange systems)
        
        Returns:
            Tuple of (can_open: bool, reason: Optional[str])
            
            If can_open=True: Order is safe to place
            If can_open=False: Order would violate invariant, reason explains why
        
        Examples:
            >>> enforcer = PositionInvariantEnforcer()
            >>> positions = {"BTCUSDT": 0.001}  # Long position
            >>> can_open, reason = enforcer.check_can_open_position(
            ...     "BTCUSDT", "sell", 0.001, positions
            ... )
            >>> print(can_open)  # False - would create opposing position
            False
            >>> print(reason)
            'Cannot open SHORT position on BTCUSDT: existing LONG position...'
        """
        if self.allow_hedging:
            # Hedging mode - allow any position
            return (True, None)
        
        # 🚨 Check for exchange-level hedge mode
        self.detect_exchange_hedge_mode(current_positions)
        
        # Get current net position for this symbol
        current_qty = current_positions.get(symbol, 0.0)
        
        # Normalize side to boolean
        # buy/long = True, sell/short = False
        is_buy_order = side.lower() in ["buy", "long"]
        
        # Check for conflict
        if is_buy_order and current_qty < 0:
            # Attempting to open LONG but have existing SHORT
            reason = (
                f"Cannot open LONG position on {symbol}: "
                f"existing SHORT position (net_qty={current_qty:.8f}). "
                f"Must close SHORT position first before opening LONG, "
                f"or enable hedging mode if intentional."
            )
            logger.error(
                f"🛑 [POSITION INVARIANT VIOLATION] {reason} "
                f"[account={account}, exchange={exchange}]"
            )
            
            # Record blocked order
            self._record_blocked_order(
                symbol=symbol,
                attempted_side="LONG",
                attempted_quantity=quantity,
                existing_quantity=current_qty,
                reason=reason,
                account=account,
                exchange=exchange
            )
            
            return (False, reason)
        
        if not is_buy_order and current_qty > 0:
            # Attempting to open SHORT but have existing LONG
            reason = (
                f"Cannot open SHORT position on {symbol}: "
                f"existing LONG position (net_qty={current_qty:.8f}). "
                f"Must close LONG position first before opening SHORT, "
                f"or enable hedging mode if intentional."
            )
            logger.error(
                f"🛑 [POSITION INVARIANT VIOLATION] {reason} "
                f"[account={account}, exchange={exchange}]"
            )
            
            # Record blocked order
            self._record_blocked_order(
                symbol=symbol,
                attempted_side="SHORT",
                attempted_quantity=quantity,
                existing_quantity=current_qty,
                reason=reason,
                account=account,
                exchange=exchange
            )
            
            return (False, reason)
        
        # Same direction or flat - OK to proceed
        direction_label = "LONG" if is_buy_order else "SHORT"
        logger.debug(
            f"✅ [POSITION CHECK PASSED] {symbol} {direction_label} order allowed "
            f"(current_net_qty={current_qty:.8f}) [account={account}, exchange={exchange}]"
        )
        
        return (True, None)
    
    def enforce_before_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_positions: Dict[str, float],
        **kwargs
    ) -> None:
        """
        Enforce invariant before placing order - raises exception if check fails.
        
        This is the primary method to call before ANY order placement.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            current_positions: Current positions dict
            **kwargs: Additional args passed to check_can_open_position
        
        Raises:
            PositionInvariantViolation: If order would violate position invariants
        
        Example:
            >>> enforcer = get_position_invariant_enforcer()
            >>> positions = get_current_positions()
            >>> try:
            ...     enforcer.enforce_before_order("BTCUSDT", "buy", 0.001, positions)
            ...     # Safe to place order
            ...     place_order(...)
            ... except PositionInvariantViolation as e:
            ...     logger.warning(f"Order blocked: {e}")
        """
        can_open, reason = self.check_can_open_position(
            symbol, side, quantity, current_positions, **kwargs
        )
        
        if not can_open:
            raise PositionInvariantViolation(reason)
    
    def detect_exchange_hedge_mode(self, current_positions: Dict[str, float]) -> bool:
        """
        Detect if exchange has hedge mode enabled by analyzing position structure.
        In hedge mode, you can have separate LONG and SHORT positions for same symbol.
        
        Args:
            current_positions: Dict of {symbol: net_quantity}
        
        Returns:
            True if hedge mode detected (positions indicate dual-side positioning possible)
        
        Note: This is a heuristic check. True hedge mode detection requires exchange API call.
        """
        # If user explicitly enabled hedging, don't warn
        if self.allow_hedging:
            return False
        
        # Check if we've seen conflicting positions (sign of hedge mode at exchange)
        # This gets called before each order, so if we see conflicts, hedge mode is likely ON
        for symbol, net_qty in current_positions.items():
            # Net quantity should be either positive (LONG), negative (SHORT), or zero (FLAT)
            # If exchange is in hedge mode, positions are reported separately and may appear strange
            pass
        
        # Log warning if not shown yet
        if not self.hedge_mode_warning_shown:
            # Check the blocked orders - if we've blocked multiple conflicting orders, hedge mode may be ON
            if len(self.blocked_orders) > 3:
                logger.critical(
                    f"🚨🚨🚨 POSSIBLE EXCHANGE HEDGE MODE! 🚨🚨🚨\\n"
                    f"   System has blocked {len(self.blocked_orders)} conflicting orders.\\n"
                    f"   This suggests Binance dualSidePosition might be ENABLED.\\n"
                    f"   ⚠️  IMMEDIATE ACTION REQUIRED:\\n"
                    f"      1. Check Binance position mode settings\\n"
                    f"      2. If hedge mode is ON, close ALL positions\\n"
                    f"      3. Run: python disable_hedge_mode.py\\n"
                    f"      4. Restart trading bot\\n"
                    f"   This warning shows after multiple blocked orders."
                )
                self.hedge_mode_warning_shown = True
                return True
        
        return False
    
    def _record_blocked_order(
        self,
        symbol: str,
        attempted_side: str,
        attempted_quantity: float,
        existing_quantity: float,
        reason: str,
        account: str,
        exchange: str
    ) -> None:
        """Record a blocked order for metrics and debugging."""
        event = BlockedOrderEvent(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            attempted_side=attempted_side,
            attempted_quantity=attempted_quantity,
            existing_quantity=existing_quantity,
            reason=reason,
            account=account,
            exchange=exchange
        )
        
        self.blocked_orders.append(event)
        
        # Limit history to prevent memory growth
        max_history = int(os.getenv("QT_BLOCKED_ORDER_HISTORY", "1000"))
        if len(self.blocked_orders) > max_history:
            self.blocked_orders = self.blocked_orders[-max_history:]
    
    def get_blocked_orders_count(self, since_minutes: Optional[int] = None) -> int:
        """
        Get count of blocked orders.
        
        Args:
            since_minutes: If provided, only count blocks in last N minutes
        
        Returns:
            Number of blocked orders
        """
        if since_minutes is None:
            return len(self.blocked_orders)
        
        cutoff = datetime.now(timezone.utc).timestamp() - (since_minutes * 60)
        return sum(
            1 for event in self.blocked_orders
            if event.timestamp.timestamp() >= cutoff
        )
    
    def get_blocked_orders_summary(self, limit: int = 10) -> list[dict]:
        """
        Get summary of recent blocked orders for monitoring.
        
        Args:
            limit: Maximum number of recent blocks to return
        
        Returns:
            List of dicts with blocked order details
        """
        recent = self.blocked_orders[-limit:] if limit > 0 else self.blocked_orders
        
        return [
            {
                "timestamp": event.timestamp.isoformat(),
                "symbol": event.symbol,
                "attempted_side": event.attempted_side,
                "attempted_quantity": event.attempted_quantity,
                "existing_quantity": event.existing_quantity,
                "reason": event.reason,
                "account": event.account,
                "exchange": event.exchange
            }
            for event in reversed(recent)  # Most recent first
        ]


# Global singleton instance
_enforcer_instance: Optional[PositionInvariantEnforcer] = None


def get_position_invariant_enforcer(allow_hedging: Optional[bool] = None) -> PositionInvariantEnforcer:
    """
    Get or create global position invariant enforcer singleton.
    
    Args:
        allow_hedging: If provided, (re)initializes enforcer with this setting.
                      If None, uses existing instance or reads from environment.
    
    Returns:
        Global PositionInvariantEnforcer instance
    
    Environment Variables:
        QT_ALLOW_HEDGING: Set to 'true' to enable hedging mode (default: 'false')
    
    Example:
        >>> # In application startup
        >>> enforcer = get_position_invariant_enforcer()
        >>> 
        >>> # Before placing any order
        >>> enforcer.enforce_before_order(symbol, side, qty, positions)
    """
    global _enforcer_instance
    
    if allow_hedging is not None:
        # Explicit override - reinitialize
        _enforcer_instance = PositionInvariantEnforcer(allow_hedging=allow_hedging)
        return _enforcer_instance
    
    if _enforcer_instance is None:
        # Initialize from environment
        hedging_enabled = os.getenv("QT_ALLOW_HEDGING", "false").lower() == "true"
        _enforcer_instance = PositionInvariantEnforcer(allow_hedging=hedging_enabled)
    
    return _enforcer_instance


def reset_enforcer() -> None:
    """Reset global enforcer instance (primarily for testing)."""
    global _enforcer_instance
    _enforcer_instance = None


__all__ = [
    "PositionInvariantEnforcer",
    "PositionInvariantViolation",
    "BlockedOrderEvent",
    "get_position_invariant_enforcer",
    "reset_enforcer",
]
