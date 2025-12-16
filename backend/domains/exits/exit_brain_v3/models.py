"""
Exit Brain v3 Data Models - DTOs for exit plans and context.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum


class ExitKind(str, Enum):
    """Type of exit leg"""
    TP = "TP"              # Take profit
    SL = "SL"              # Stop loss
    TRAIL = "TRAIL"        # Trailing stop
    EMERGENCY = "EMERGENCY"  # Emergency exit (ESS / Risk v3)
    PARTIAL = "PARTIAL"    # Partial exit at specific level


@dataclass
class ExitContext:
    """
    Complete context for exit decision making.
    Aggregates all relevant information about position, market, and risk state.
    """
    # Position basics
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    size: float  # Position size
    leverage: float = 1.0
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0  # As percentage of margin
    unrealized_pnl_usd: float = 0.0
    position_age_seconds: int = 0
    
    # Market regime
    volatility: float = 0.0  # ATR/price or similar
    trend_strength: float = 0.0  # -1 to 1
    market_regime: str = "NORMAL"  # NORMAL, VOLATILE, TRENDING, RANGE_BOUND
    
    # AI/RL hints (optional)
    rl_tp_hint: Optional[float] = None  # Suggested TP % from RL
    rl_sl_hint: Optional[float] = None  # Suggested SL % from RL
    rl_confidence: Optional[float] = None  # RL decision confidence
    signal_confidence: Optional[float] = None  # Original entry signal confidence
    
    # Risk context
    risk_mode: str = "NORMAL"  # NORMAL, CONSERVATIVE, CRITICAL, ESS_ACTIVE
    max_loss_pct: float = 0.025  # Max acceptable loss (2.5% default)
    target_rr_ratio: float = 2.0  # Target risk-reward ratio
    
    # Trailing hints
    trail_hint: Optional[float] = None  # Suggested trail callback %
    trail_enabled: bool = True
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitLeg:
    """
    Single exit component (one TP level, SL, or trailing rule).
    Multiple legs form complete exit strategy.
    """
    kind: ExitKind
    size_pct: float  # 0.0-1.0, portion of position to exit
    trigger_price: Optional[float] = None  # Exact trigger price (None = market-based)
    trigger_pct: Optional[float] = None  # Trigger as % from entry (+ for TP, - for SL)
    trail_callback: Optional[float] = None  # For TRAIL: callback %
    priority: int = 0  # Execution priority (0=highest)
    condition: str = "IMMEDIATE"  # IMMEDIATE, STAGED, CONDITIONAL
    reason: str = ""  # Human-readable rationale
    r_multiple: Optional[float] = None  # R multiple this leg represents (for TP profiles)
    profile_leg_index: Optional[int] = None  # Index in TPProfile.tp_legs (for tracking)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate leg configuration"""
        if not (0.0 <= self.size_pct <= 1.0):
            raise ValueError(f"size_pct must be 0.0-1.0, got {self.size_pct}")
        
        if self.kind == ExitKind.TRAIL and self.trail_callback is None:
            raise ValueError("TRAIL legs require trail_callback")


@dataclass
class ExitPlan:
    """
    Complete exit strategy for a position.
    Contains all exit legs (TP, SL, trailing) coordinated by Exit Brain.
    """
    symbol: str
    legs: List[ExitLeg]
    strategy_id: str  # Identifier for this strategy template
    source: str  # "EXIT_BRAIN_V3", "RL_V3", "DYNAMIC_TPSL", etc.
    reason: str  # Why this plan was chosen
    confidence: float = 0.0  # Overall confidence in plan (0.0-1.0)
    
    # Profile tracking (for TP profile system)
    profile_name: Optional[str] = None  # Name of TPProfile used
    market_regime: Optional[str] = None  # Regime profile was selected for
    
    # Aggregate metrics
    total_tp_pct: float = 0.0  # Sum of all TP sizes
    total_sl_pct: float = 0.0  # Sum of all SL sizes
    has_trailing: bool = False
    has_emergency: bool = False
    
    # Lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate aggregate metrics"""
        self.total_tp_pct = sum(leg.size_pct for leg in self.legs if leg.kind == ExitKind.TP)
        self.total_sl_pct = sum(leg.size_pct for leg in self.legs if leg.kind == ExitKind.SL)
        self.has_trailing = any(leg.kind == ExitKind.TRAIL for leg in self.legs)
        self.has_emergency = any(leg.kind == ExitKind.EMERGENCY for leg in self.legs)
    
    def get_legs_by_kind(self, kind: ExitKind) -> List[ExitLeg]:
        """Filter legs by type"""
        return [leg for leg in self.legs if leg.kind == kind]
    
    def get_primary_tp(self) -> Optional[ExitLeg]:
        """Get highest priority TP leg"""
        tp_legs = self.get_legs_by_kind(ExitKind.TP)
        return min(tp_legs, key=lambda x: x.priority) if tp_legs else None
    
    def get_primary_sl(self) -> Optional[ExitLeg]:
        """Get highest priority SL leg"""
        sl_legs = self.get_legs_by_kind(ExitKind.SL)
        return min(sl_legs, key=lambda x: x.priority) if sl_legs else None
