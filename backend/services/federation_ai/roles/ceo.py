"""
AI-CEO (Chief Executive Officer)
=================================

Global governor of the trading system.

Responsibilities:
- Set global trading mode (LIVE/SHADOW/PAUSED/EMERGENCY)
- Choose capital profile (MICRO/LOW/NORMAL/AGGRESSIVE)
- Approve/deny major system changes
- Emergency shutdown authority

Decision Logic:
- Upgrade profile: stable profits, low DD, healthy system
- Downgrade profile: losses, high DD, system degradation
- Pause trading: critical health, ESS WARNING/CRITICAL
- Emergency stop: cascading failures, extreme DD
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    CapitalProfileDecision,
    TradingModeDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    DecisionType,
    DecisionPriority,
    TradingMode,
    CapitalProfile,
)

logger = structlog.get_logger(__name__)


class AICEO(FederationRole):
    """AI Chief Executive Officer - Global governor"""
    
    # Thresholds for capital profile changes
    PROFILE_THRESHOLDS = {
        CapitalProfile.MICRO: {
            "max_risk_per_trade": 0.005,  # 0.5%
            "max_daily_risk": 0.02,        # 2%
            "max_positions": 2,
        },
        CapitalProfile.LOW: {
            "max_risk_per_trade": 0.01,   # 1%
            "max_daily_risk": 0.03,        # 3%
            "max_positions": 3,
        },
        CapitalProfile.NORMAL: {
            "max_risk_per_trade": 0.02,   # 2%
            "max_daily_risk": 0.05,        # 5%
            "max_positions": 5,
        },
        CapitalProfile.AGGRESSIVE: {
            "max_risk_per_trade": 0.05,   # 5%
            "max_daily_risk": 0.10,        # 10%
            "max_positions": 8,
        },
    }
    
    def __init__(self):
        super().__init__("ai-ceo")
        self.current_profile = CapitalProfile.LOW  # Start conservative
        self.current_mode = TradingMode.LIVE
        self.consecutive_profitable_days = 0
        self.consecutive_loss_days = 0
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate portfolio state and adjust capital profile.
        
        Logic:
        - Profitable day + low DD → consider upgrade
        - Loss day + high DD → consider downgrade
        - Max DD exceeded → emergency downgrade
        """
        decisions = []
        
        # Track daily performance
        if snapshot.realized_pnl_today > 0:
            self.consecutive_profitable_days += 1
            self.consecutive_loss_days = 0
        elif snapshot.realized_pnl_today < 0:
            self.consecutive_loss_days += 1
            self.consecutive_profitable_days = 0
        
        # Check if profile change needed
        new_profile = self._evaluate_profile_change(snapshot)
        
        if new_profile and new_profile != self.current_profile:
            self.logger.info(
                "Capital profile change",
                from_profile=self.current_profile,
                to_profile=new_profile,
                drawdown=snapshot.drawdown_pct,
                pnl_today=snapshot.realized_pnl_today,
            )
            
            profile_config = self.PROFILE_THRESHOLDS[new_profile]
            
            decision = FederationDecision(
                decision_type=DecisionType.CAPITAL_PROFILE,
                role_source="ceo",
                priority=DecisionPriority.HIGH,
                reason=self._get_profile_change_reason(snapshot, new_profile),
                payload={
                    "profile": new_profile.value,
                    "max_risk_per_trade_pct": profile_config["max_risk_per_trade"],
                    "max_daily_risk_pct": profile_config["max_daily_risk"],
                    "max_positions": profile_config["max_positions"],
                }
            )
            decisions.append(decision)
            self.current_profile = new_profile
        
        return decisions
    
    async def on_health_update(
        self,
        health: SystemHealthSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate system health and adjust trading mode.
        
        Logic:
        - OPTIMAL/HEALTHY → LIVE
        - DEGRADED → consider SHADOW
        - CRITICAL → PAUSED
        - ESS CRITICAL → EMERGENCY stop
        """
        decisions = []
        
        new_mode = self._evaluate_mode_change(health)
        
        if new_mode and new_mode != self.current_mode:
            self.logger.warning(
                "Trading mode change",
                from_mode=self.current_mode,
                to_mode=new_mode,
                system_status=health.overall_status,
                ess_state=health.ess_state,
            )
            
            decision = FederationDecision(
                decision_type=DecisionType.MODE_CHANGE,
                role_source="ceo",
                priority=DecisionPriority.CRITICAL if new_mode == TradingMode.EMERGENCY else DecisionPriority.HIGH,
                reason=self._get_mode_change_reason(health, new_mode),
                payload={
                    "mode": new_mode.value,
                    "duration_minutes": 30 if new_mode == TradingMode.PAUSED else None,
                }
            )
            decisions.append(decision)
            self.current_mode = new_mode
        
        return decisions
    
    def _evaluate_profile_change(self, snapshot: PortfolioSnapshot) -> CapitalProfile:
        """
        Determine if capital profile should change.
        
        Upgrade conditions (e.g., LOW → NORMAL):
        - 5+ consecutive profitable days
        - DD < 3%
        - Win rate > 60%
        
        Downgrade conditions (e.g., NORMAL → LOW):
        - 3+ consecutive loss days
        - DD > 5%
        - Win rate < 40%
        """
        current_idx = list(CapitalProfile).index(self.current_profile)
        
        # Emergency downgrade to MICRO
        if snapshot.drawdown_pct > 0.08 or snapshot.max_drawdown_pct > 0.10:
            return CapitalProfile.MICRO
        
        # Downgrade conditions
        if (
            self.consecutive_loss_days >= 3
            or snapshot.drawdown_pct > 0.05
            or (snapshot.win_rate_today < 0.40 and snapshot.num_positions > 0)
        ):
            if current_idx > 0:
                return list(CapitalProfile)[current_idx - 1]
        
        # Upgrade conditions
        if (
            self.consecutive_profitable_days >= 5
            and snapshot.drawdown_pct < 0.03
            and snapshot.win_rate_today > 0.60
            and snapshot.sharpe_ratio_7d and snapshot.sharpe_ratio_7d > 1.5
        ):
            if current_idx < len(CapitalProfile) - 1:
                return list(CapitalProfile)[current_idx + 1]
        
        return self.current_profile
    
    def _evaluate_mode_change(self, health: SystemHealthSnapshot) -> TradingMode:
        """Determine if trading mode should change"""
        
        # Emergency stop
        if health.ess_state == "CRITICAL" or health.overall_status == "EMERGENCY":
            return TradingMode.EMERGENCY
        
        # Pause trading
        if health.overall_status == "CRITICAL" or health.ess_state == "WARNING":
            return TradingMode.PAUSED
        
        # Shadow mode for degraded systems
        if health.overall_status == "DEGRADED":
            return TradingMode.SHADOW
        
        # Resume live trading if healthy
        if health.overall_status in ["OPTIMAL", "HEALTHY"] and health.ess_state == "NOMINAL":
            return TradingMode.LIVE
        
        return self.current_mode
    
    def _get_profile_change_reason(self, snapshot: PortfolioSnapshot, new_profile: CapitalProfile) -> str:
        """Generate human-readable reason for profile change"""
        if snapshot.drawdown_pct > 0.08:
            return f"Emergency downgrade: DD {snapshot.drawdown_pct:.1%} exceeds safe limits"
        elif self.consecutive_loss_days >= 3:
            return f"Downgrade: {self.consecutive_loss_days} consecutive loss days"
        elif self.consecutive_profitable_days >= 5:
            return f"Upgrade: {self.consecutive_profitable_days} consecutive profitable days, DD {snapshot.drawdown_pct:.1%}"
        else:
            return f"Profile adjustment based on performance metrics"
    
    def _get_mode_change_reason(self, health: SystemHealthSnapshot, new_mode: TradingMode) -> str:
        """Generate human-readable reason for mode change"""
        if new_mode == TradingMode.EMERGENCY:
            return f"EMERGENCY STOP: ESS {health.ess_state}, System {health.overall_status}"
        elif new_mode == TradingMode.PAUSED:
            return f"Paused: System health {health.overall_status}, ESS {health.ess_state}"
        elif new_mode == TradingMode.SHADOW:
            return f"Shadow mode: System degraded ({health.overall_status})"
        elif new_mode == TradingMode.LIVE:
            return f"Resuming live trading: System healthy ({health.overall_status})"
        else:
            return "Mode change based on system state"
