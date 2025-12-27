"""CEO Policy - Decision rules and thresholds for AI CEO.

This module defines the operating modes, thresholds, and decision
mappings that the AI CEO uses to manage global trading state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """Global operating modes for the trading system."""
    
    EXPANSION = "EXPANSION"              # Normal/slightly aggressive
    GROWTH = "GROWTH"                    # Increased aggressivity
    DEFENSIVE = "DEFENSIVE"              # Lower risk, reduced exposure
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"  # Minimal trading, protect capital
    BLACK_SWAN = "BLACK_SWAN"           # Emergency mode, close all positions


class MarketRegime(str, Enum):
    """Market regime classifications."""
    
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


class SystemHealth(str, Enum):
    """System health states."""
    
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    OFFLINE = "OFFLINE"


@dataclass
class CEOThresholds:
    """Thresholds for CEO decision-making."""
    
    # Drawdown thresholds
    max_daily_drawdown_expansion: float = 0.03      # 3%
    max_daily_drawdown_growth: float = 0.05         # 5%
    max_daily_drawdown_defensive: float = 0.015     # 1.5%
    critical_drawdown_threshold: float = 0.08       # 8% → BLACK_SWAN
    
    # Risk score thresholds (0-100 scale from AI-RO)
    risk_score_expansion: float = 50.0
    risk_score_growth: float = 70.0
    risk_score_defensive: float = 30.0
    risk_score_critical: float = 85.0              # → BLACK_SWAN
    
    # Strategy performance thresholds (win rate)
    min_win_rate_expansion: float = 0.52
    min_win_rate_growth: float = 0.55
    min_win_rate_defensive: float = 0.48
    
    # Market regime confidence thresholds
    min_regime_confidence: float = 0.70
    
    # Position count limits per mode
    max_positions_expansion: int = 10
    max_positions_growth: int = 15
    max_positions_defensive: int = 5
    max_positions_capital_preservation: int = 2
    
    # System health requirements
    require_healthy_for_growth: bool = True
    allow_degraded_in_defensive: bool = True


class ModeTransitionRule(BaseModel):
    """Rule for transitioning between operating modes."""
    
    from_mode: OperatingMode
    to_mode: OperatingMode
    
    # Conditions (all must be true)
    max_drawdown: Optional[float] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    min_win_rate: Optional[float] = None
    required_regime: Optional[MarketRegime] = None
    required_health: Optional[SystemHealth] = None
    
    # Cooldown between transitions (seconds)
    cooldown: int = 300  # 5 minutes


class CEOPolicy:
    """
    Policy engine for AI CEO decision-making.
    
    Responsibilities:
    - Define operating modes and their characteristics
    - Map system state to recommended operating mode
    - Validate mode transitions
    - Provide mode-specific configuration adjustments
    """
    
    def __init__(self, thresholds: Optional[CEOThresholds] = None):
        """Initialize CEO policy with thresholds."""
        self.thresholds = thresholds or CEOThresholds()
        self._transition_rules = self._build_transition_rules()
        
        logger.info(
            f"CEOPolicy initialized with thresholds: "
            f"critical_drawdown={self.thresholds.critical_drawdown_threshold}, "
            f"risk_score_critical={self.thresholds.risk_score_critical}"
        )
    
    def _build_transition_rules(self) -> list[ModeTransitionRule]:
        """Build transition rules between modes."""
        return [
            # Emergency transitions (always checked first)
            ModeTransitionRule(
                from_mode=OperatingMode.EXPANSION,
                to_mode=OperatingMode.BLACK_SWAN,
                max_drawdown=self.thresholds.critical_drawdown_threshold,
                cooldown=0,  # Immediate
            ),
            ModeTransitionRule(
                from_mode=OperatingMode.GROWTH,
                to_mode=OperatingMode.BLACK_SWAN,
                max_risk_score=self.thresholds.risk_score_critical,
                cooldown=0,
            ),
            
            # Growth → Defensive (risk too high)
            ModeTransitionRule(
                from_mode=OperatingMode.GROWTH,
                to_mode=OperatingMode.DEFENSIVE,
                max_risk_score=self.thresholds.risk_score_defensive,
                cooldown=180,  # 3 minutes
            ),
            
            # Expansion → Defensive (drawdown exceeded)
            ModeTransitionRule(
                from_mode=OperatingMode.EXPANSION,
                to_mode=OperatingMode.DEFENSIVE,
                max_drawdown=self.thresholds.max_daily_drawdown_expansion,
                cooldown=180,
            ),
            
            # Defensive → Expansion (recovery)
            ModeTransitionRule(
                from_mode=OperatingMode.DEFENSIVE,
                to_mode=OperatingMode.EXPANSION,
                min_win_rate=self.thresholds.min_win_rate_expansion,
                required_health=SystemHealth.HEALTHY,
                cooldown=600,  # 10 minutes
            ),
            
            # Expansion → Growth (strong performance)
            ModeTransitionRule(
                from_mode=OperatingMode.EXPANSION,
                to_mode=OperatingMode.GROWTH,
                min_win_rate=self.thresholds.min_win_rate_growth,
                min_risk_score=self.thresholds.risk_score_expansion,
                required_health=SystemHealth.HEALTHY,
                cooldown=600,
            ),
        ]
    
    def recommend_mode(
        self,
        current_mode: OperatingMode,
        drawdown: float,
        risk_score: float,
        win_rate: float,
        regime: MarketRegime,
        health: SystemHealth,
        open_positions: int,
    ) -> tuple[OperatingMode, str]:
        """
        Recommend operating mode based on current system state.
        
        Args:
            current_mode: Current operating mode
            drawdown: Current daily drawdown (0-1 scale)
            risk_score: Risk score from AI-RO (0-100 scale)
            win_rate: Recent strategy win rate (0-1 scale)
            regime: Current market regime
            health: System health status
            open_positions: Number of open positions
        
        Returns:
            Tuple of (recommended_mode, reason)
        """
        # Emergency checks first
        if drawdown >= self.thresholds.critical_drawdown_threshold:
            return OperatingMode.BLACK_SWAN, f"Critical drawdown: {drawdown:.2%}"
        
        if risk_score >= self.thresholds.risk_score_critical:
            return OperatingMode.BLACK_SWAN, f"Critical risk score: {risk_score:.1f}"
        
        if health == SystemHealth.OFFLINE:
            return OperatingMode.BLACK_SWAN, "System offline"
        
        if health == SystemHealth.CRITICAL:
            return OperatingMode.CAPITAL_PRESERVATION, "System critical"
        
        # Capital preservation conditions
        if drawdown >= self.thresholds.max_daily_drawdown_growth:
            return OperatingMode.CAPITAL_PRESERVATION, f"High drawdown: {drawdown:.2%}"
        
        # Defensive conditions
        if drawdown >= self.thresholds.max_daily_drawdown_expansion:
            return OperatingMode.DEFENSIVE, f"Elevated drawdown: {drawdown:.2%}"
        
        if risk_score <= self.thresholds.risk_score_defensive:
            return OperatingMode.DEFENSIVE, f"Low risk score: {risk_score:.1f}"
        
        if win_rate < self.thresholds.min_win_rate_defensive and current_mode != OperatingMode.GROWTH:
            return OperatingMode.DEFENSIVE, f"Low win rate: {win_rate:.2%}"
        
        if health == SystemHealth.DEGRADED and not self.thresholds.allow_degraded_in_defensive:
            return OperatingMode.DEFENSIVE, "System degraded"
        
        # Growth conditions (most aggressive)
        if (
            win_rate >= self.thresholds.min_win_rate_growth
            and risk_score >= self.thresholds.risk_score_growth
            and drawdown < self.thresholds.max_daily_drawdown_expansion
            and health == SystemHealth.HEALTHY
        ):
            return OperatingMode.GROWTH, "Strong performance and healthy system"
        
        # Expansion (default good state)
        if (
            win_rate >= self.thresholds.min_win_rate_expansion
            and risk_score >= self.thresholds.risk_score_expansion
            and drawdown < self.thresholds.max_daily_drawdown_expansion
        ):
            return OperatingMode.EXPANSION, "Normal operating conditions"
        
        # Default to current mode if no clear recommendation
        return current_mode, "No change - conditions stable"
    
    def get_mode_config(self, mode: OperatingMode) -> dict[str, any]:
        """
        Get PolicyStore configuration adjustments for a given mode.
        
        Returns dict of config keys to update in PolicyStore.
        """
        configs = {
            OperatingMode.EXPANSION: {
                "max_leverage": 10.0,
                "max_risk_pct_per_trade": 0.02,  # 2%
                "max_daily_drawdown": 0.03,       # 3%
                "max_open_positions": self.thresholds.max_positions_expansion,
                "global_min_confidence": 0.60,
                "scaling_factor": 1.0,
                "enable_rl": True,
                "enable_meta_strategy": True,
                "enable_pal": True,
                "enable_pba": True,
                "enable_dynamic_tpsl": True,
            },
            OperatingMode.GROWTH: {
                "max_leverage": 15.0,
                "max_risk_pct_per_trade": 0.03,  # 3%
                "max_daily_drawdown": 0.05,       # 5%
                "max_open_positions": self.thresholds.max_positions_growth,
                "global_min_confidence": 0.65,
                "scaling_factor": 1.2,
                "enable_rl": True,
                "enable_meta_strategy": True,
                "enable_pal": True,
                "enable_pba": True,
                "enable_dynamic_tpsl": True,
            },
            OperatingMode.DEFENSIVE: {
                "max_leverage": 5.0,
                "max_risk_pct_per_trade": 0.01,  # 1%
                "max_daily_drawdown": 0.015,      # 1.5%
                "max_open_positions": self.thresholds.max_positions_defensive,
                "global_min_confidence": 0.70,
                "scaling_factor": 0.7,
                "enable_rl": True,
                "enable_meta_strategy": True,
                "enable_pal": True,
                "enable_pba": True,
                "enable_dynamic_tpsl": True,
            },
            OperatingMode.CAPITAL_PRESERVATION: {
                "max_leverage": 2.0,
                "max_risk_pct_per_trade": 0.005, # 0.5%
                "max_daily_drawdown": 0.01,       # 1%
                "max_open_positions": self.thresholds.max_positions_capital_preservation,
                "global_min_confidence": 0.80,
                "scaling_factor": 0.5,
                "enable_rl": False,
                "enable_meta_strategy": False,
                "enable_pal": True,
                "enable_pba": False,
                "enable_dynamic_tpsl": True,
            },
            OperatingMode.BLACK_SWAN: {
                "max_leverage": 1.0,
                "max_risk_pct_per_trade": 0.001, # 0.1%
                "max_daily_drawdown": 0.005,      # 0.5%
                "max_open_positions": 0,
                "global_min_confidence": 1.0,
                "scaling_factor": 0.0,
                "enable_rl": False,
                "enable_meta_strategy": False,
                "enable_pal": False,
                "enable_pba": False,
                "enable_dynamic_tpsl": False,
            },
        }
        
        return configs.get(mode, configs[OperatingMode.EXPANSION])
    
    def validate_transition(
        self,
        from_mode: OperatingMode,
        to_mode: OperatingMode,
        last_transition_time: float,
        current_time: float,
    ) -> tuple[bool, str]:
        """
        Validate if mode transition is allowed.
        
        Args:
            from_mode: Current mode
            to_mode: Desired mode
            last_transition_time: Unix timestamp of last transition
            current_time: Current Unix timestamp
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Same mode - always allowed
        if from_mode == to_mode:
            return True, "No transition"
        
        # BLACK_SWAN transitions always allowed (emergency)
        if to_mode == OperatingMode.BLACK_SWAN:
            return True, "Emergency transition"
        
        # Check cooldown
        time_since_transition = current_time - last_transition_time
        
        for rule in self._transition_rules:
            if rule.from_mode == from_mode and rule.to_mode == to_mode:
                if time_since_transition < rule.cooldown:
                    remaining = rule.cooldown - time_since_transition
                    return False, f"Cooldown: {remaining:.0f}s remaining"
                return True, "Transition allowed"
        
        # No explicit rule - allow after default cooldown
        default_cooldown = 300  # 5 minutes
        if time_since_transition < default_cooldown:
            remaining = default_cooldown - time_since_transition
            return False, f"Default cooldown: {remaining:.0f}s remaining"
        
        return True, "Transition allowed (no explicit rule)"
