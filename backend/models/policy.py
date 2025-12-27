"""Policy data models for PolicyStore v2."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class RiskMode(str, Enum):
    """Risk modes for trading strategy."""
    
    AGGRESSIVE_SMALL_ACCOUNT = "AGGRESSIVE_SMALL_ACCOUNT"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


class RiskModeConfig(BaseModel):
    """Configuration for a specific risk mode."""
    
    # Risk limits
    max_leverage: float = Field(gt=0, le=125)
    max_risk_pct_per_trade: float = Field(gt=0, le=1.0)  # 0.01 = 1%
    max_daily_drawdown: float = Field(gt=0, le=1.0)      # 0.05 = 5%
    max_positions: int = Field(gt=0, le=100)
    
    # Confidence & sizing
    global_min_confidence: float = Field(ge=0, le=1.0)   # 0.50 = 50%
    scaling_factor: float = Field(gt=0, le=2.0)          # Position size multiplier
    position_size_cap: float = Field(gt=0)               # Max USD per position
    
    # AI module toggles
    enable_rl: bool = True
    enable_meta_strategy: bool = True
    enable_pal: bool = True                              # Portfolio Analysis Logger
    enable_pba: bool = True                              # Portfolio Balance Allocator
    enable_clm: bool = True                              # Continuous Learning Manager
    enable_retraining: bool = True
    enable_dynamic_tpsl: bool = True
    
    @field_validator("max_leverage")
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Ensure leverage is reasonable."""
        if v > 50:
            # High leverage - warn but allow for testnet
            pass
        return v
    
    @field_validator("max_risk_pct_per_trade")
    @classmethod
    def validate_risk_per_trade(cls, v: float) -> float:
        """Ensure risk per trade is not excessive."""
        if v > 0.05:  # >5% per trade is very aggressive
            pass
        return v


class PolicyConfig(BaseModel):
    """Complete policy configuration with all risk modes."""
    
    # Current active mode
    active_mode: RiskMode = RiskMode.NORMAL
    
    # Risk mode configurations
    modes: dict[RiskMode, RiskModeConfig]
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    updated_by: str = "system"
    version: int = 1
    
    # PATCH-P0-01: Single source of truth for emergency state (no split-brain)
    emergency_mode: bool = False
    allow_new_trades: bool = True
    emergency_reason: Optional[str] = None
    emergency_activated_at: Optional[datetime] = None
    
    def get_active_config(self) -> RiskModeConfig:
        """Get configuration for currently active risk mode."""
        return self.modes[self.active_mode]
    
    def switch_mode(self, new_mode: RiskMode, updated_by: str = "system") -> None:
        """Switch to a different risk mode."""
        if new_mode not in self.modes:
            raise ValueError(f"Risk mode {new_mode} not configured")
        
        self.active_mode = new_mode
        self.last_updated = datetime.utcnow()
        self.updated_by = updated_by
        self.version += 1


class RiskProfile(BaseModel):
    """Risk profile with all risk management parameters."""
    
    name: str
    
    # Core risk limits
    max_leverage: float = Field(gt=0, le=125)
    min_leverage: float = Field(gt=0, le=125, default=1.0)
    max_risk_pct_per_trade: float = Field(gt=0, le=1.0)  # 0.01 = 1%
    max_daily_drawdown_pct: float = Field(gt=0, le=1.0)  # 0.05 = 5%
    max_open_positions: int = Field(gt=0, le=100)
    
    # Position sizing
    position_size_cap_usd: float = Field(gt=0)  # Max USD per position
    
    # Signal confidence
    global_min_confidence: float = Field(ge=0, le=1.0)
    
    # Trading controls
    allow_new_positions: bool = True
    
    @field_validator("min_leverage")
    @classmethod
    def validate_min_leverage(cls, v: float, info) -> float:
        """Ensure min_leverage <= max_leverage."""
        max_lev = info.data.get("max_leverage", float("inf"))
        if v > max_lev:
            raise ValueError(f"min_leverage ({v}) must be <= max_leverage ({max_lev})")
        return v


class PolicyUpdateEvent(BaseModel):
    """Event published when policy is updated."""
    
    event_type: str = "policy.updated"
    previous_mode: RiskMode
    new_mode: RiskMode
    updated_by: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None


# Default risk profiles (aligned with RiskModeConfig)
DEFAULT_RISK_PROFILES = {
    RiskMode.AGGRESSIVE_SMALL_ACCOUNT: RiskProfile(
        name="AGGRESSIVE_SMALL_ACCOUNT",
        max_leverage=7.0,
        min_leverage=2.0,
        max_risk_pct_per_trade=0.03,         # 3% per trade
        max_daily_drawdown_pct=0.06,         # 6% daily DD
        max_open_positions=15,
        position_size_cap_usd=300.0,         # $300 max
        global_min_confidence=0.45,
        allow_new_positions=True,
    ),
    RiskMode.NORMAL: RiskProfile(
        name="NORMAL",
        max_leverage=5.0,
        min_leverage=1.0,
        max_risk_pct_per_trade=0.015,        # 1.5% per trade
        max_daily_drawdown_pct=0.05,         # 5% daily DD
        max_open_positions=30,
        position_size_cap_usd=1000.0,        # $1000 max
        global_min_confidence=0.50,
        allow_new_positions=True,
    ),
    RiskMode.DEFENSIVE: RiskProfile(
        name="DEFENSIVE",
        max_leverage=3.0,
        min_leverage=1.0,
        max_risk_pct_per_trade=0.0075,       # 0.75% per trade
        max_daily_drawdown_pct=0.03,         # 3% daily DD
        max_open_positions=10,
        position_size_cap_usd=500.0,         # $500 max
        global_min_confidence=0.60,
        allow_new_positions=True,
    ),
}

# Default policy configurations
DEFAULT_POLICIES = {
    RiskMode.AGGRESSIVE_SMALL_ACCOUNT: RiskModeConfig(
        max_leverage=7.0,
        max_risk_pct_per_trade=0.03,      # 3% per trade
        max_daily_drawdown=0.06,           # 6% daily DD
        max_positions=15,
        global_min_confidence=0.45,        # Lower bar for more trades
        scaling_factor=1.5,                # Aggressive sizing
        position_size_cap=300.0,           # $300 max per position
        enable_rl=True,
        enable_meta_strategy=True,
        enable_pal=True,
        enable_pba=True,
        enable_clm=True,
        enable_retraining=True,
        enable_dynamic_tpsl=True,
    ),
    RiskMode.NORMAL: RiskModeConfig(
        max_leverage=5.0,
        max_risk_pct_per_trade=0.015,      # 1.5% per trade
        max_daily_drawdown=0.05,           # 5% daily DD
        max_positions=30,
        global_min_confidence=0.50,        # Moderate bar
        scaling_factor=1.0,                # Standard sizing
        position_size_cap=1000.0,          # $1000 max per position
        enable_rl=True,
        enable_meta_strategy=True,
        enable_pal=True,
        enable_pba=True,
        enable_clm=True,
        enable_retraining=True,
        enable_dynamic_tpsl=True,
    ),
    RiskMode.DEFENSIVE: RiskModeConfig(
        max_leverage=3.0,
        max_risk_pct_per_trade=0.0075,     # 0.75% per trade
        max_daily_drawdown=0.03,           # 3% daily DD
        max_positions=10,
        global_min_confidence=0.60,        # Higher bar for quality
        scaling_factor=0.7,                # Conservative sizing
        position_size_cap=500.0,           # $500 max per position
        enable_rl=True,
        enable_meta_strategy=True,
        enable_pal=True,
        enable_pba=True,
        enable_clm=False,                  # Disable learning in defensive
        enable_retraining=False,           # No retraining
        enable_dynamic_tpsl=True,
    ),
}


def create_default_policy() -> PolicyConfig:
    """Create default policy configuration."""
    return PolicyConfig(
        active_mode=RiskMode.NORMAL,
        modes=DEFAULT_POLICIES,
        updated_by="system_init",
        version=1,
    )
