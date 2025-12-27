"""
Exit Brain v3 - Core Types for Dynamic Executor

Phase 2A types for AI-driven exit decision pipeline.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal, Tuple, Set
from enum import Enum


@dataclass
class PositionContext:
    """
    Complete context for an open position.
    
    Used by ExitBrainAdapter to make AI-driven exit decisions.
    """
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    current_price: float
    size: float  # Absolute position size
    unrealized_pnl: float  # Unrealized PnL in % terms
    
    # Optional enrichment
    leverage: Optional[float] = None
    account: Optional[str] = None
    exchange: Optional[str] = "binance"
    
    # AI/Strategy context
    regime: Optional[str] = None  # trend, range, high_vol, etc.
    risk_state: Optional[str] = None  # normal, drawdown, high_risk, etc.
    strategy_id: Optional[str] = None
    
    # Metadata
    opened_at: Optional[str] = None
    duration_hours: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def unrealized_pnl_abs(self) -> float:
        """Absolute unrealized PnL in quote currency."""
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size
    
    @property
    def is_profitable(self) -> bool:
        """Is position in profit?"""
        return self.unrealized_pnl > 0
    
    @property
    def is_long(self) -> bool:
        """Is this a long position?"""
        return self.side == "long"


class ExitDecisionType(str, Enum):
    """
    Type of exit decision made by Exit Brain.
    
    Phase 2A: All logged only (shadow mode).
    Phase 2B: Will translate to actual orders via exit_order_gateway.
    """
    NO_CHANGE = "no_change"  # AI says keep current exits
    PARTIAL_CLOSE = "partial_close"  # Take partial profit
    MOVE_SL = "move_sl"  # Tighten or adjust stop loss
    UPDATE_TP_LIMITS = "update_tp_limits"  # Adjust take profit levels
    FULL_EXIT_NOW = "full_exit_now"  # Exit entire position immediately
    
    def __str__(self) -> str:
        return self.value


@dataclass
class ExitDecision:
    """
    AI-driven exit decision for a position.
    
    Phase 2A: Logged only (what AI would do).
    Phase 2B: Translated to orders via exit_order_gateway.
    """
    decision_type: ExitDecisionType
    symbol: str
    
    # Partial close parameters
    fraction_to_close: Optional[float] = None  # 0.0 to 1.0
    
    # Stop loss parameters
    new_sl_price: Optional[float] = None
    sl_reason: Optional[str] = None  # "breakeven", "trailing", "tighter", etc.
    
    # Take profit parameters
    new_tp_levels: Optional[List[float]] = None  # List of TP price levels
    tp_fractions: Optional[List[float]] = None  # Corresponding fractions per level
    
    # Exit reasoning
    reason: Optional[str] = None  # Human-readable explanation
    confidence: Optional[float] = None  # 0.0 to 1.0 if available
    
    # AI/RL metadata
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # Context snapshot
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    
    def __post_init__(self):
        """Validate decision parameters."""
        if self.decision_type == ExitDecisionType.PARTIAL_CLOSE:
            if self.fraction_to_close is None or not (0.0 < self.fraction_to_close <= 1.0):
                raise ValueError(
                    f"PARTIAL_CLOSE requires fraction_to_close in (0.0, 1.0], "
                    f"got {self.fraction_to_close}"
                )
        
        if self.decision_type == ExitDecisionType.MOVE_SL:
            if self.new_sl_price is None:
                raise ValueError("MOVE_SL requires new_sl_price")
        
        if self.decision_type == ExitDecisionType.UPDATE_TP_LIMITS:
            if not self.new_tp_levels:
                raise ValueError("UPDATE_TP_LIMITS requires new_tp_levels")
    
    def summary(self) -> str:
        """One-line summary for logging."""
        parts = [f"type={self.decision_type}"]
        
        if self.fraction_to_close is not None:
            parts.append(f"close_frac={self.fraction_to_close:.2%}")
        
        if self.new_sl_price is not None:
            parts.append(f"new_sl=${self.new_sl_price:.4f}")
            if self.sl_reason:
                parts.append(f"sl_reason={self.sl_reason}")
        
        if self.new_tp_levels:
            parts.append(f"tp_levels={len(self.new_tp_levels)}")
        
        if self.reason:
            parts.append(f"reason='{self.reason}'")
        
        if self.confidence is not None:
            parts.append(f"conf={self.confidence:.2f}")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON logging."""
        return {
            "decision_type": str(self.decision_type),
            "symbol": self.symbol,
            "fraction_to_close": self.fraction_to_close,
            "new_sl_price": self.new_sl_price,
            "sl_reason": self.sl_reason,
            "new_tp_levels": self.new_tp_levels,
            "tp_fractions": self.tp_fractions,
            "reason": self.reason,
            "confidence": self.confidence,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "meta": self.meta
        }


@dataclass
class ShadowExecutionLog:
    """
    Log entry for what AI would have done (Phase 2A shadow mode).
    
    Allows comparison of AI decisions vs actual legacy exits.
    """
    timestamp: str
    symbol: str
    side: str
    
    # Position state
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float
    
    # AI decision
    decision: ExitDecision
    
    # Context
    regime: Optional[str] = None
    risk_state: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON logging."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "size": self.size,
            "unrealized_pnl": self.unrealized_pnl,
            "decision": self.decision.to_dict(),
            "regime": self.regime,
            "risk_state": self.risk_state
        }


@dataclass
class PositionExitState:
    """
    Internal state for active monitoring of a position's exit levels.
    
    This is the core of Exit Brain V3's HYBRID exit system:
    - Planner sets SL/TP levels via decisions
    - Executor stores levels here (not on exchange)
    - Monitoring loop checks price vs levels every cycle
    - When level hit → execute MARKET reduce-only order
    
    HYBRID STOP-LOSS MODEL:
    - active_sl: Internal/soft SL (AI-driven, dynamic, optimizes exits)
    - hard_sl_price: Binance STOP_MARKET order (static max-loss floor, survives crashes)
    
    Rules:
    - LONG: SL triggers when price <= active_sl, TP when price >= tp_price
    - SHORT: SL triggers when price >= active_sl, TP when price <= tp_price
    - size_pct on TP legs applies to REMAINING position size
    - triggered_legs prevents duplicate execution
    
    DYNAMIC PARTIAL TP & RATCHETING:
    - initial_size: Original position size at creation (for fraction calculations)
    - remaining_size: Updated after each partial TP execution
    - tp_hits_count: Number of TPs hit (triggers SL ratcheting rules)
    - max_unrealized_profit_pct: High-water mark for trailing logic
    - loss_guard_triggered: Prevents duplicate loss guard execution
    - entry_price: Entry price for risk calculations
    """
    symbol: str
    side: Literal["LONG", "SHORT"]  # Uppercase for Binance positionSide
    position_size: float  # Current absolute size (updated as TPs fill)
    
    # Active levels (set by AI decisions, monitored by executor)
    active_sl: Optional[float] = None  # Current SL price (internal/soft)
    tp_levels: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size_pct), ...]
    
    # Execution tracking
    triggered_legs: Set[int] = field(default_factory=set)  # Indices of executed TP legs
    last_price: Optional[float] = None  # Last observed price
    last_updated: Optional[str] = None  # ISO timestamp of last update
    
    # Hard stop-loss on Binance (safety net)
    hard_sl_price: Optional[float] = None  # Binance STOP_MARKET price
    hard_sl_order_id: Optional[str] = None  # Binance order ID for cancellation
    
    # Dynamic partial TP tracking (NEW)
    entry_price: Optional[float] = None  # Entry price for position
    initial_size: Optional[float] = None  # Original position size at state creation
    remaining_size: Optional[float] = None  # Current remaining size after partial closes
    tp_hits_count: int = 0  # Number of TPs hit (for ratcheting logic)
    max_unrealized_profit_pct: float = 0.0  # High-water mark for trailing/analytics
    loss_guard_triggered: bool = False  # Prevents duplicate loss guard fires
    
    # CHALLENGE_100 mode tracking
    tp1_taken: bool = False  # Has TP1 (30% @ +1R) been taken
    opened_at_ts: Optional[float] = None  # Position open timestamp (for time stop)
    highest_favorable_price: Optional[float] = None  # For trailing (LONG: highest seen, SHORT: lowest seen)
    challenge_mode_active: bool = False  # Is this position using CHALLENGE_100 logic
    
    def __post_init__(self):
        """Validate and normalize state."""
        if self.position_size < 0:
            raise ValueError(f"position_size must be >= 0, got {self.position_size}")
        
        # Ensure side is uppercase
        if self.side not in ("LONG", "SHORT"):
            # Try to convert from lowercase
            if self.side.upper() in ("LONG", "SHORT"):
                object.__setattr__(self, 'side', self.side.upper())
            else:
                raise ValueError(f"side must be 'LONG' or 'SHORT', got {self.side}")
        
        # Initialize dynamic TP tracking fields if not set
        if self.initial_size is None:
            object.__setattr__(self, 'initial_size', self.position_size)
        if self.remaining_size is None:
            object.__setattr__(self, 'remaining_size', self.position_size)
    
    def get_remaining_size(self) -> float:
        """
        Calculate remaining position size after triggered TPs.
        
        Uses remaining_size field if available (updated by executor),
        otherwise calculates from initial_size and triggered_legs.
        """
        # Use explicitly tracked remaining_size if available
        if self.remaining_size is not None:
            return self.remaining_size
        
        # Fallback: calculate from triggered legs
        if not self.tp_levels or not self.triggered_legs:
            return self.position_size
        
        # Calculate how much has been closed by triggered TPs
        closed_fraction = sum(
            size_pct for i, (_, size_pct) in enumerate(self.tp_levels)
            if i in self.triggered_legs
        )
        
        initial = self.initial_size if self.initial_size is not None else self.position_size
        return initial * (1.0 - closed_fraction)
    
    def should_trigger_sl(self, current_price: float) -> bool:
        """Check if SL should trigger at current price."""
        if self.active_sl is None:
            return False
        
        if self.side == "LONG":
            return current_price <= self.active_sl
        else:  # SHORT
            return current_price >= self.active_sl
    
    def get_triggerable_tp_legs(self, current_price: float) -> List[Tuple[int, float, float]]:
        """
        Get TP legs that should trigger at current price.
        
        Returns:
            List of (leg_index, tp_price, size_pct) for legs that should execute
        """
        triggerable = []
        
        for i, (tp_price, size_pct) in enumerate(self.tp_levels):
            # Skip already triggered
            if i in self.triggered_legs:
                continue
            
            # Check if price hit this TP
            should_trigger = False
            if self.side == "LONG":
                should_trigger = current_price >= tp_price
            else:  # SHORT
                should_trigger = current_price <= tp_price
            
            if should_trigger:
                triggerable.append((i, tp_price, size_pct))
        
        return triggerable
