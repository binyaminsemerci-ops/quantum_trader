"""
AI-Supervisor (Chief Guardian)
===============================

Emergency override and trading freeze authority.

Responsibilities:
- Override dangerous decisions from other roles
- Emergency freeze trading (immediate halt)
- Monitor decision conflicts
- Highest priority in Federation hierarchy
- Protect against catastrophic losses

Decision Logic:
- Immediate freeze: ESS CRITICAL, extreme DD, system EMERGENCY
- Override: Decisions that violate safety constraints
- Watchdog: Monitor all Federation decisions
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    OverrideDecision,
    FreezeDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    DecisionType,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class AISupervisor(FederationRole):
    """AI Supervisor - Emergency guardian"""
    
    # Safety thresholds (stricter than other roles)
    EMERGENCY_THRESHOLDS = {
        "max_drawdown_pct": 0.10,       # 10% DD = freeze
        "max_loss_per_day_pct": 0.05,   # 5% daily loss = freeze
        "max_consecutive_losses": 5,    # 5 losses in a row = freeze
        "min_capital_usd": 1000.0,      # Below $1000 = freeze
    }
    
    def __init__(self):
        super().__init__("ai-supervisor")
        self.freeze_active = False
        self.overridden_decisions = []
        self.consecutive_losses = 0
        self.freeze_history = []
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Monitor portfolio for emergency conditions.
        
        Logic:
        - DD > 10% → immediate freeze
        - Daily loss > 5% → immediate freeze
        - Consecutive losses → escalating freeze
        - Capital < $1000 → freeze until manual review
        """
        decisions = []
        
        # Check if freeze needed
        freeze_decision = self._evaluate_freeze_condition(snapshot)
        if freeze_decision:
            self.logger.critical(
                "EMERGENCY FREEZE TRIGGERED",
                reason=freeze_decision.payload["reason"],
                severity=freeze_decision.payload["severity"],
            )
            self.freeze_active = True
            self.freeze_history.append(freeze_decision)
            decisions.append(freeze_decision)
        
        # Check if freeze can be lifted
        elif self.freeze_active:
            unfreeze_decision = self._evaluate_unfreeze_condition(snapshot)
            if unfreeze_decision:
                self.logger.info(
                    "Freeze lifted",
                    reason=unfreeze_decision.reason,
                )
                self.freeze_active = False
                decisions.append(unfreeze_decision)
        
        return decisions
    
    async def on_health_update(
        self,
        health: SystemHealthSnapshot
    ) -> List[FederationDecision]:
        """
        Monitor system health for emergency conditions.
        
        Logic:
        - ESS CRITICAL → freeze
        - System EMERGENCY → freeze
        """
        decisions = []
        
        # Immediate freeze on critical system state
        if health.ess_state == "CRITICAL" or health.system_status == "EMERGENCY":
            freeze_decision = FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.CRITICAL,
                reason=f"System critical state: ESS {health.ess_state}, System {health.system_status}",
                payload={
                    "duration_minutes": 60,  # 1 hour
                    "affected_subsystems": ["trading", "execution", "signals"],
                    "severity": "CRITICAL",
                    "requires_manual_review": True,
                    "reason": "Critical system health detected",
                },
            )
            
            self.logger.critical(
                "CRITICAL FREEZE - System health",
                ess_state=health.ess_state,
                system_status=health.system_status,
            )
            
            self.freeze_active = True
            self.freeze_history.append(freeze_decision)
            decisions.append(freeze_decision)
        
        return decisions
    
    def _evaluate_freeze_condition(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Evaluate if trading should be frozen.
        
        Triggers:
        1. DD > 10%
        2. Daily loss > 5%
        3. 5+ consecutive losses
        4. Capital < $1000
        """
        dd_pct = snapshot.drawdown_pct
        daily_loss_pct = abs(snapshot.realized_pnl_today / snapshot.total_equity) if snapshot.total_equity > 0 else 0
        equity = snapshot.total_equity
        
        # Track consecutive losses
        if snapshot.realized_pnl_today < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Condition 1: Extreme drawdown
        if dd_pct > self.EMERGENCY_THRESHOLDS["max_drawdown_pct"]:
            return FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.CRITICAL,
                reason=f"Extreme drawdown {dd_pct:.1%} exceeds limit {self.EMERGENCY_THRESHOLDS['max_drawdown_pct']:.1%}",
                payload={
                    "duration_minutes": 120,  # 2 hours
                    "affected_subsystems": ["trading", "execution", "signals"],
                    "severity": "CRITICAL",
                    "requires_manual_review": True,
                    "reason": f"Drawdown {dd_pct:.1%} > {self.EMERGENCY_THRESHOLDS['max_drawdown_pct']:.1%}",
                },
            )
        
        # Condition 2: Excessive daily loss
        elif snapshot.realized_pnl_today < 0 and daily_loss_pct > self.EMERGENCY_THRESHOLDS["max_loss_per_day_pct"]:
            return FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.CRITICAL,
                reason=f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.EMERGENCY_THRESHOLDS['max_loss_per_day_pct']:.1%}",
                payload={
                    "duration_minutes": 60,
                    "affected_subsystems": ["trading", "execution"],
                    "severity": "HIGH",
                    "requires_manual_review": True,
                    "reason": f"Daily loss {snapshot.realized_pnl_today:.2f} USD ({daily_loss_pct:.1%})",
                },
            )
        
        # Condition 3: Consecutive losses
        elif self.consecutive_losses >= self.EMERGENCY_THRESHOLDS["max_consecutive_losses"]:
            return FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.HIGH,
                reason=f"{self.consecutive_losses} consecutive losses detected",
                payload={
                    "duration_minutes": 30,
                    "affected_subsystems": ["trading"],
                    "severity": "MEDIUM",
                    "requires_manual_review": False,
                    "reason": f"Consecutive losses: {self.consecutive_losses}",
                },
            )
        
        # Condition 4: Insufficient capital
        elif equity < self.EMERGENCY_THRESHOLDS["min_capital_usd"]:
            return FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.CRITICAL,
                reason=f"Capital {equity:.2f} USD below minimum {self.EMERGENCY_THRESHOLDS['min_capital_usd']:.2f}",
                payload={
                    "duration_minutes": 0,  # Indefinite
                    "affected_subsystems": ["trading", "execution", "signals", "risk"],
                    "severity": "CRITICAL",
                    "requires_manual_review": True,
                    "reason": f"Insufficient capital: {equity:.2f} USD",
                },
            )
        
        return None
    
    def _evaluate_unfreeze_condition(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Evaluate if freeze can be lifted.
        
        Requirements:
        - DD back below 5%
        - No recent losses
        - Capital stable
        """
        dd_pct = snapshot.drawdown_pct
        equity = snapshot.total_equity
        
        # Can unfreeze if conditions normalized
        if (
            dd_pct < 0.05
            and equity > self.EMERGENCY_THRESHOLDS["min_capital_usd"] * 1.5
            and snapshot.realized_pnl_today >= 0
        ):
            return FederationDecision(
                decision_type=DecisionType.FREEZE,
                role_source="supervisor",
                priority=DecisionPriority.HIGH,
                reason="Conditions normalized, lifting freeze",
                payload={
                    "duration_minutes": 0,  # Lift freeze
                    "affected_subsystems": [],
                    "severity": "NONE",
                    "requires_manual_review": False,
                    "reason": f"DD {dd_pct:.1%}, Equity {equity:.2f} USD, P&L positive",
                },
            )
        
        return None
    
    def override_decision(
        self,
        decision_id: str,
        reason: str
    ) -> FederationDecision:
        """
        Override a dangerous decision from another role.
        
        Use cases:
        - CEO wants to switch to AGGRESSIVE but DD > 5%
        - CIO wants to expand symbols but system degraded
        - CRO loosens risk but consecutive losses active
        """
        override = FederationDecision(
            decision_type=DecisionType.OVERRIDE,
            role_source="supervisor",
            priority=DecisionPriority.CRITICAL,
            reason=reason,
            payload={
                "overridden_decision_id": decision_id,
                "reason": reason,
                "override_timestamp": None,  # Will be set by base class
            },
        )
        
        self.overridden_decisions.append(decision_id)
        
        self.logger.warning(
            "Decision overridden",
            decision_id=decision_id,
            reason=reason,
        )
        
        return override
