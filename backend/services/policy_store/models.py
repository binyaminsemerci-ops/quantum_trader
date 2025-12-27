"""
Policy data models for Quantum Trader.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskMode(str, Enum):
    """Risk mode for trading operations."""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


@dataclass
class GlobalPolicy:
    """
    Global trading policy configuration.
    
    This represents the system-wide trading parameters that affect
    all strategies and risk management components.
    """
    # Risk settings
    risk_mode: RiskMode = RiskMode.NORMAL
    global_min_confidence: float = 0.65
    max_risk_per_trade: float = 0.02
    max_positions: int = 10
    max_daily_trades: int = 10
    max_drawdown_pct: float = 10.0
    
    # Strategy control
    allowed_strategies: list[str] = field(default_factory=list)
    blocked_strategies: list[str] = field(default_factory=list)
    
    # Symbol control
    allowed_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_loss_threshold: float = -5.0  # %
    
    # Metadata
    updated_at: datetime = field(default_factory=datetime.utcnow)
    updated_by: str = "system"
    version: int = 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "risk_mode": self.risk_mode.value,
            "global_min_confidence": self.global_min_confidence,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_positions": self.max_positions,
            "max_daily_trades": self.max_daily_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "allowed_strategies": self.allowed_strategies,
            "blocked_strategies": self.blocked_strategies,
            "allowed_symbols": self.allowed_symbols,
            "blocked_symbols": self.blocked_symbols,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "circuit_breaker_loss_threshold": self.circuit_breaker_loss_threshold,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GlobalPolicy":
        """Create from dictionary."""
        return cls(
            risk_mode=RiskMode(data["risk_mode"]),
            global_min_confidence=data["global_min_confidence"],
            max_risk_per_trade=data["max_risk_per_trade"],
            max_positions=data["max_positions"],
            max_daily_trades=data["max_daily_trades"],
            max_drawdown_pct=data["max_drawdown_pct"],
            allowed_strategies=data.get("allowed_strategies", []),
            blocked_strategies=data.get("blocked_strategies", []),
            allowed_symbols=data.get("allowed_symbols", []),
            blocked_symbols=data.get("blocked_symbols", []),
            circuit_breaker_enabled=data.get("circuit_breaker_enabled", True),
            circuit_breaker_loss_threshold=data.get("circuit_breaker_loss_threshold", -5.0),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            updated_by=data.get("updated_by", "system"),
            version=data.get("version", 1),
        )
