"""
Fund Management Domain - AI CEO v2 (Fund CEO)

Strategic fund management and overall direction for the AI-powered hedge fund.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StrategicDecisionType(Enum):
    """Types of strategic decisions."""
    CAPITAL_ALLOCATION = "capital_allocation"
    STRATEGY_APPROVAL = "strategy_approval"
    FUND_EXPANSION = "fund_expansion"
    RISK_ADJUSTMENT = "risk_adjustment"
    PERFORMANCE_TARGET = "performance_target"


@dataclass
class StrategicDirective:
    """Strategic directive from CEO."""
    directive_id: str
    decision_type: StrategicDecisionType
    description: str
    target_metrics: Dict[str, float]
    issued_at: datetime
    effective_until: Optional[datetime] = None


class AIFundCEO:
    """
    AI CEO v2 (Fund CEO) - Strategic fund management.
    
    Responsibilities:
    - Set fund strategy and objectives
    - Approve capital allocation across strategies
    - Monitor fund performance vs targets
    - Make strategic decisions (new strategies, fund expansion)
    - Interface with external stakeholders (future: LPs, regulators)
    
    Decision Authority: HIGHEST (can override all except CRO veto)
    """
    
    def __init__(
        self,
        fund_name: str,
        policy_store,
        event_bus,
        target_annual_return: float = 0.25,  # 25%
        max_annual_drawdown: float = 0.15,   # -15%
        target_sharpe_ratio: float = 2.0,
    ):
        """
        Initialize AI Fund CEO.
        
        Args:
            fund_name: Name of the fund
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            target_annual_return: Target annual return (default 25%)
            max_annual_drawdown: Max acceptable drawdown (default -15%)
            target_sharpe_ratio: Target Sharpe ratio (default 2.0)
        """
        self.fund_name = fund_name
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Performance targets
        self.target_annual_return = target_annual_return
        self.max_annual_drawdown = max_annual_drawdown
        self.target_sharpe_ratio = target_sharpe_ratio
        
        # Strategic state
        self.active_directives: List[StrategicDirective] = []
        self.approved_strategies: Dict[str, Dict] = {}
        self.capital_allocations: Dict[str, float] = {}
        
        # Subscribe to key events
        self.event_bus.subscribe("fund.performance.report", self._handle_performance_report)
        self.event_bus.subscribe("fund.risk.assessment.updated", self._handle_risk_update)
        self.event_bus.subscribe("governance.decision.proposed", self._handle_decision_proposal)
        
        logger.info(
            f"[CEO] AI Fund CEO initialized: fund={fund_name}, "
            f"target_return={target_annual_return:.1%}, "
            f"max_drawdown={max_annual_drawdown:.1%}, "
            f"target_sharpe={target_sharpe_ratio:.1f}"
        )
    
    async def approve_capital_allocation(
        self,
        strategy_id: str,
        allocation_pct: float,
        expected_return: float,
        max_drawdown: float,
        reason: str
    ) -> bool:
        """
        Approve capital allocation to a strategy.
        
        Args:
            strategy_id: Strategy identifier
            allocation_pct: Capital allocation percentage (0-1)
            expected_return: Expected annual return
            max_drawdown: Maximum acceptable drawdown
            reason: Allocation rationale
        
        Returns:
            True if approved, False if rejected
        """
        # Validate allocation
        if allocation_pct < 0.05 or allocation_pct > 0.30:
            logger.warning(
                f"[CEO] Capital allocation rejected for {strategy_id}: "
                f"{allocation_pct:.1%} outside allowed range (5%-30%)"
            )
            return False
        
        # Check if total allocation would exceed 100%
        total_allocated = sum(self.capital_allocations.values())
        if total_allocated + allocation_pct > 0.90:  # Reserve 10% cash
            logger.warning(
                f"[CEO] Capital allocation rejected for {strategy_id}: "
                f"total allocation would exceed 90% (currently {total_allocated:.1%})"
            )
            return False
        
        # Approve allocation
        self.capital_allocations[strategy_id] = allocation_pct
        
        logger.info(
            f"[CEO] ✅ Capital allocation APPROVED for {strategy_id}: "
            f"{allocation_pct:.1%} (expected_return={expected_return:.1%}, "
            f"max_drawdown={max_drawdown:.1%})"
        )
        
        # Publish approval event
        await self.event_bus.publish(
            "fund.strategy.allocated",
            {
                "strategy_id": strategy_id,
                "capital_allocation": allocation_pct,
                "expected_return": expected_return,
                "max_drawdown": max_drawdown,
                "allocation_reason": reason,
                "approved_by": "CEO",
                "approved_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return True
    
    async def issue_directive(
        self,
        decision_type: StrategicDecisionType,
        description: str,
        target_metrics: Dict[str, float],
        effective_days: Optional[int] = None
    ) -> str:
        """
        Issue strategic directive to the organization.
        
        Args:
            decision_type: Type of strategic decision
            description: Directive description
            target_metrics: Target metrics to achieve
            effective_days: Days until directive expires (None = permanent)
        
        Returns:
            Directive ID
        """
        import uuid
        
        directive_id = f"CEO-DIR-{uuid.uuid4().hex[:8].upper()}"
        
        effective_until = None
        if effective_days:
            from datetime import timedelta
            effective_until = datetime.now(timezone.utc) + timedelta(days=effective_days)
        
        directive = StrategicDirective(
            directive_id=directive_id,
            decision_type=decision_type,
            description=description,
            target_metrics=target_metrics,
            issued_at=datetime.now(timezone.utc),
            effective_until=effective_until
        )
        
        self.active_directives.append(directive)
        
        logger.warning(
            f"[CEO] 📋 STRATEGIC DIRECTIVE ISSUED: {directive_id}\n"
            f"   Type: {decision_type.value}\n"
            f"   Description: {description}\n"
            f"   Target Metrics: {target_metrics}\n"
            f"   Effective Until: {effective_until.isoformat() if effective_until else 'Permanent'}"
        )
        
        # Publish directive event
        await self.event_bus.publish(
            "fund.directive.issued",
            {
                "directive_id": directive_id,
                "decision_type": decision_type.value,
                "description": description,
                "target_metrics": target_metrics,
                "issued_at": directive.issued_at.isoformat(),
                "effective_until": effective_until.isoformat() if effective_until else None,
                "issuer": "CEO"
            }
        )
        
        return directive_id
    
    async def _handle_performance_report(self, event_data: dict) -> None:
        """Handle fund performance reports."""
        period = event_data.get("period")
        total_return = event_data.get("total_return", 0.0)
        sharpe_ratio = event_data.get("sharpe_ratio", 0.0)
        max_drawdown = event_data.get("max_drawdown", 0.0)
        
        logger.info(
            f"[CEO] 📊 Performance Report ({period}):\n"
            f"   Return: {total_return:.2%}\n"
            f"   Sharpe: {sharpe_ratio:.2f}\n"
            f"   Max Drawdown: {max_drawdown:.2%}"
        )
        
        # Check if performance below targets
        if period == "monthly":
            monthly_target = self.target_annual_return / 12.0
            if total_return < monthly_target * 0.5:  # 50% of target
                logger.warning(
                    f"[CEO] ⚠️ Performance below target: "
                    f"{total_return:.2%} < {monthly_target * 0.5:.2%}"
                )
                # Issue directive to CIO for strategy review
                await self.issue_directive(
                    StrategicDecisionType.RISK_ADJUSTMENT,
                    f"Monthly performance below target ({total_return:.2%}). "
                    f"CIO to review strategy allocation and risk parameters.",
                    {"min_monthly_return": monthly_target * 0.5},
                    effective_days=30
                )
    
    async def _handle_risk_update(self, event_data: dict) -> None:
        """Handle portfolio risk updates."""
        portfolio_var = event_data.get("portfolio_var", 0.0)
        portfolio_cvar = event_data.get("portfolio_cvar", 0.0)
        
        logger.debug(
            f"[CEO] Risk Update: VaR={portfolio_var:.2%}, CVaR={portfolio_cvar:.2%}"
        )
        
        # Check if risk exceeds acceptable levels
        if portfolio_cvar > self.max_annual_drawdown * 1.5:  # 150% of max drawdown
            logger.critical(
                f"[CEO] 🚨 Portfolio CVaR exceeds acceptable level: "
                f"{portfolio_cvar:.2%} > {self.max_annual_drawdown * 1.5:.2%}"
            )
            # Escalate to CRO for immediate risk reduction
            await self.event_bus.publish(
                "fund.risk.escalation",
                {
                    "reason": "cvar_breach",
                    "current_cvar": portfolio_cvar,
                    "max_acceptable": self.max_annual_drawdown * 1.5,
                    "escalated_by": "CEO",
                    "escalated_to": "CRO",
                    "priority": "CRITICAL"
                }
            )
    
    async def _handle_decision_proposal(self, event_data: dict) -> None:
        """Handle governance decision proposals."""
        decision_id = event_data.get("decision_id")
        proposer = event_data.get("proposer")
        decision_type = event_data.get("decision_type")
        description = event_data.get("description")
        
        logger.info(
            f"[CEO] Decision proposal received: {decision_id}\n"
            f"   Proposer: {proposer}\n"
            f"   Type: {decision_type}\n"
            f"   Description: {description}"
        )
        
        # CEO participates in voting (handled by governance system)
    
    def get_status(self) -> dict:
        """Get CEO status and metrics."""
        return {
            "fund_name": self.fund_name,
            "target_annual_return": self.target_annual_return,
            "max_annual_drawdown": self.max_annual_drawdown,
            "target_sharpe_ratio": self.target_sharpe_ratio,
            "active_directives": len(self.active_directives),
            "approved_strategies": len(self.approved_strategies),
            "total_capital_allocated": sum(self.capital_allocations.values()),
            "capital_allocations": self.capital_allocations
        }
