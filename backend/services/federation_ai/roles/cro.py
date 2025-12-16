"""
AI-CRO (Chief Risk Officer)
============================

Global risk manager and ESS coordinator.

Responsibilities:
- Set global risk limits (leverage, position size, exposure)
- Dynamically adjust ESS thresholds
- Monitor portfolio risk metrics (VaR, DD, exposure)
- Mandate position reductions when risk too high
- Coordinate with AI-CEO on emergency stops

Decision Logic:
- Tighten risk: High DD, ESS escalation, volatile markets
- Loosen risk: Low DD, stable profits, calm markets
- Adjust ESS: Adaptive thresholds based on market conditions
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    RiskAdjustmentDecision,
    ESSPolicyDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    DecisionType,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class AICRO(FederationRole):
    """AI Chief Risk Officer - Global risk manager"""
    
    # Base risk limits (can be adjusted)
    BASE_RISK_LIMITS = {
        "max_leverage": 10.0,
        "max_position_size_usd": 10000.0,
        "max_drawdown_pct": 0.08,      # 8%
        "max_exposure_pct": 0.90,      # 90% of capital
        "stop_loss_multiplier": 1.0,   # 1x = normal, 1.5x = tighter
    }
    
    # ESS thresholds (adaptive)
    BASE_ESS_THRESHOLDS = {
        "caution": 0.03,    # 3% DD
        "warning": 0.05,    # 5% DD
        "critical": 0.08,   # 8% DD
    }
    
    def __init__(self):
        super().__init__("ai-cro")
        self.current_risk_limits = self.BASE_RISK_LIMITS.copy()
        self.current_ess_thresholds = self.BASE_ESS_THRESHOLDS.copy()
        self.last_adjustment_drawdown = 0.0
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate portfolio risk and adjust limits.
        
        Logic:
        - DD increasing → tighten risk
        - DD stable/decreasing → maintain/loosen
        - Exposure > 80% → reduce position sizes
        """
        decisions = []
        
        # Check if risk adjustment needed
        risk_decision = self._evaluate_risk_adjustment(snapshot)
        if risk_decision:
            self.logger.info(
                "Risk adjustment",
                drawdown=snapshot.drawdown_pct,
                exposure_pct=snapshot.total_exposure_usd / snapshot.total_equity if snapshot.total_equity > 0 else 0,
                num_positions=snapshot.num_positions,
            )
            decisions.append(risk_decision)
        
        # Check if ESS thresholds need adjustment
        ess_decision = self._evaluate_ess_adjustment(snapshot)
        if ess_decision:
            self.logger.info(
                "ESS threshold adjustment",
                drawdown=snapshot.drawdown_pct,
                max_dd=snapshot.max_drawdown_pct,
            )
            decisions.append(ess_decision)
        
        return decisions
    
    async def on_health_update(
        self,
        health: SystemHealthSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate system health and tighten risk if degraded.
        """
        decisions = []
        
        # Tighten risk immediately if ESS escalates
        if health.ess_state in ["WARNING", "CRITICAL"]:
            self.logger.warning(
                "ESS escalation detected, tightening risk",
                ess_state=health.ess_state,
            )
            
            decision = FederationDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                role_source="cro",
                priority=DecisionPriority.CRITICAL,
                reason=f"ESS escalation to {health.ess_state}",
                payload={
                    "max_leverage": 5.0,              # Reduce leverage
                    "max_position_size_usd": 3000.0,  # Smaller positions
                    "max_drawdown_pct": 0.05,         # Stricter DD limit
                    "max_exposure_pct": 0.50,         # Reduce exposure
                    "stop_loss_multiplier": 1.5,      # Tighter stops
                }
            )
            decisions.append(decision)
            
            # Update current limits
            self.current_risk_limits.update(decision.payload)
        
        return decisions
    
    def _evaluate_risk_adjustment(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Determine if risk limits need adjustment.
        
        Tighten conditions:
        - DD > 5%
        - Exposure > 80%
        - Win rate < 45%
        
        Loosen conditions:
        - DD < 2%
        - Stable profits
        - Win rate > 65%
        """
        dd_pct = snapshot.drawdown_pct
        exposure_pct = snapshot.total_exposure_usd / snapshot.total_equity if snapshot.total_equity > 0 else 0
        
        # Significant DD increase since last adjustment
        if dd_pct > self.last_adjustment_drawdown + 0.02:  # +2%
            # Tighten risk
            new_limits = {
                "max_leverage": max(5.0, self.current_risk_limits["max_leverage"] * 0.7),
                "max_position_size_usd": self.current_risk_limits["max_position_size_usd"] * 0.7,
                "max_drawdown_pct": 0.06,  # Stricter
                "max_exposure_pct": 0.70,
                "stop_loss_multiplier": 1.3,
            }
            
            self.current_risk_limits.update(new_limits)
            self.last_adjustment_drawdown = dd_pct
            
            return FederationDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                role_source="cro",
                priority=DecisionPriority.HIGH,
                reason=f"Drawdown increased to {dd_pct:.1%}, tightening risk",
                payload=new_limits,
            )
        
        # High exposure
        elif exposure_pct > 0.85:
            new_limits = {
                "max_leverage": self.current_risk_limits["max_leverage"],
                "max_position_size_usd": self.current_risk_limits["max_position_size_usd"] * 0.8,
                "max_drawdown_pct": self.current_risk_limits["max_drawdown_pct"],
                "max_exposure_pct": 0.75,
                "stop_loss_multiplier": self.current_risk_limits["stop_loss_multiplier"],
            }
            
            self.current_risk_limits.update(new_limits)
            
            return FederationDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                role_source="cro",
                priority=DecisionPriority.HIGH,
                reason=f"Exposure {exposure_pct:.1%} too high, reducing position sizes",
                payload=new_limits,
            )
        
        # Can loosen if performance good
        elif (
            dd_pct < 0.02
            and snapshot.win_rate_today > 0.65
            and snapshot.sharpe_ratio_7d and snapshot.sharpe_ratio_7d > 2.0
        ):
            # Gradually restore to base limits
            new_limits = {
                "max_leverage": min(self.BASE_RISK_LIMITS["max_leverage"], self.current_risk_limits["max_leverage"] * 1.2),
                "max_position_size_usd": min(self.BASE_RISK_LIMITS["max_position_size_usd"], self.current_risk_limits["max_position_size_usd"] * 1.2),
                "max_drawdown_pct": self.BASE_RISK_LIMITS["max_drawdown_pct"],
                "max_exposure_pct": self.BASE_RISK_LIMITS["max_exposure_pct"],
                "stop_loss_multiplier": 1.0,
            }
            
            self.current_risk_limits.update(new_limits)
            self.last_adjustment_drawdown = dd_pct
            
            return FederationDecision(
                decision_type=DecisionType.RISK_ADJUSTMENT,
                role_source="cro",
                priority=DecisionPriority.NORMAL,
                reason=f"Performance stable (DD {dd_pct:.1%}, WR {snapshot.win_rate_today:.1%}), restoring limits",
                payload=new_limits,
            )
        
        return None
    
    def _evaluate_ess_adjustment(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Dynamically adjust ESS thresholds based on portfolio state.
        
        Logic:
        - High volatility → tighter thresholds
        - Stable profits → standard thresholds
        - Near max DD → emergency tightening
        """
        dd_pct = snapshot.drawdown_pct
        max_dd_pct = snapshot.max_drawdown_pct
        
        # Emergency tightening if near max DD
        if dd_pct > 0.06 or max_dd_pct > 0.08:
            new_thresholds = {
                "caution_threshold_pct": 0.02,   # 2%
                "warning_threshold_pct": 0.04,   # 4%
                "critical_threshold_pct": 0.06,  # 6%
            }
            
            self.current_ess_thresholds = new_thresholds
            
            return FederationDecision(
                decision_type=DecisionType.ESS_POLICY,
                role_source="cro",
                priority=DecisionPriority.CRITICAL,
                reason=f"Near max DD ({dd_pct:.1%}/{max_dd_pct:.1%}), tightening ESS",
                payload=new_thresholds,
            )
        
        # Restore standard thresholds if stable
        elif dd_pct < 0.02 and self.current_ess_thresholds["critical_threshold_pct"] < 0.08:
            new_thresholds = self.BASE_ESS_THRESHOLDS.copy()
            self.current_ess_thresholds = new_thresholds
            
            return FederationDecision(
                decision_type=DecisionType.ESS_POLICY,
                role_source="cro",
                priority=DecisionPriority.NORMAL,
                reason=f"Drawdown normalized ({dd_pct:.1%}), restoring standard ESS thresholds",
                payload=new_thresholds,
            )
        
        return None
