"""
Fund Management Domain - AI CRO v2 (Chief Risk Officer)

Enterprise risk management with veto power over risky decisions.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskDecisionType(Enum):
    """Types of risk decisions."""
    POSITION_VETO = "position_veto"
    STRATEGY_SUSPENSION = "strategy_suspension"
    LEVERAGE_REDUCTION = "leverage_reduction"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"
    RISK_LIMIT_ADJUSTMENT = "risk_limit_adjustment"


@dataclass
class RiskVeto:
    """Risk veto decision."""
    veto_id: str
    decision_type: RiskDecisionType
    target_entity: str  # strategy_id, position_id, etc.
    reason: str
    risk_metrics: Dict[str, float]
    issued_at: datetime
    override_allowed: bool = False


class AIFundCRO:
    """
    AI CRO v2 (Chief Risk Officer) - Enterprise risk management.
    
    Responsibilities:
    - Monitor portfolio risk in real-time
    - Veto risky trades/strategies
    - Enforce risk limits across all strategies
    - Manage portfolio-level risk (VaR, CVaR, correlations)
    - Emergency liquidation authority
    
    Decision Authority: VETO POWER (can override CEO, cannot be overridden)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        max_portfolio_var: float = 0.10,     # 10% VaR (95% confidence)
        max_portfolio_cvar: float = 0.15,    # 15% CVaR
        max_leverage: float = 30.0,          # 30x max leverage
        max_single_position: float = 0.15,   # 15% of portfolio
        correlation_threshold: float = 0.70,  # 70% correlation warning
    ):
        """
        Initialize AI Fund CRO.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            max_portfolio_var: Max portfolio VaR (95% confidence)
            max_portfolio_cvar: Max portfolio CVaR
            max_leverage: Maximum allowed leverage
            max_single_position: Max single position size (% of portfolio)
            correlation_threshold: Correlation warning threshold
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Risk limits
        self.max_portfolio_var = max_portfolio_var
        self.max_portfolio_cvar = max_portfolio_cvar
        self.max_leverage = max_leverage
        self.max_single_position = max_single_position
        self.correlation_threshold = correlation_threshold
        
        # Veto tracking
        self.active_vetos: List[RiskVeto] = []
        self.suspended_strategies: Dict[str, datetime] = {}
        
        # Risk state
        self.current_portfolio_var: float = 0.0
        self.current_portfolio_cvar: float = 0.0
        self.current_leverage: float = 0.0
        
        # Subscribe to risk events
        self.event_bus.subscribe("position.opened", self._handle_position_opened)
        self.event_bus.subscribe("fund.risk.assessment.updated", self._handle_risk_update)
        self.event_bus.subscribe("fund.strategy.allocated", self._handle_strategy_allocation)
        self.event_bus.subscribe("fund.risk.escalation", self._handle_risk_escalation)
        
        logger.info(
            f"[CRO] AI Fund CRO initialized:\n"
            f"   Max Portfolio VaR: {max_portfolio_var:.1%}\n"
            f"   Max Portfolio CVaR: {max_portfolio_cvar:.1%}\n"
            f"   Max Leverage: {max_leverage:.0f}x\n"
            f"   Max Single Position: {max_single_position:.1%}"
        )
    
    async def veto_position(
        self,
        position_id: str,
        reason: str,
        risk_metrics: Dict[str, float]
    ) -> str:
        """
        Veto a position opening.
        
        Args:
            position_id: Position identifier
            reason: Veto reason
            risk_metrics: Risk metrics triggering veto
        
        Returns:
            Veto ID
        """
        import uuid
        
        veto_id = f"CRO-VETO-{uuid.uuid4().hex[:8].upper()}"
        
        veto = RiskVeto(
            veto_id=veto_id,
            decision_type=RiskDecisionType.POSITION_VETO,
            target_entity=position_id,
            reason=reason,
            risk_metrics=risk_metrics,
            issued_at=datetime.now(timezone.utc),
            override_allowed=False
        )
        
        self.active_vetos.append(veto)
        
        logger.critical(
            f"[CRO] 🚫 POSITION VETOED: {position_id}\n"
            f"   Veto ID: {veto_id}\n"
            f"   Reason: {reason}\n"
            f"   Risk Metrics: {risk_metrics}"
        )
        
        # Publish veto event
        await self.event_bus.publish(
            "fund.risk.veto.issued",
            {
                "veto_id": veto_id,
                "decision_type": RiskDecisionType.POSITION_VETO.value,
                "target_entity": position_id,
                "reason": reason,
                "risk_metrics": risk_metrics,
                "issued_at": veto.issued_at.isoformat(),
                "issued_by": "CRO",
                "override_allowed": False
            }
        )
        
        return veto_id
    
    async def suspend_strategy(
        self,
        strategy_id: str,
        reason: str,
        suspension_hours: int = 24
    ) -> str:
        """
        Suspend a strategy from trading.
        
        Args:
            strategy_id: Strategy identifier
            reason: Suspension reason
            suspension_hours: Hours to suspend (default 24)
        
        Returns:
            Veto ID
        """
        import uuid
        from datetime import timedelta
        
        veto_id = f"CRO-SUSPEND-{uuid.uuid4().hex[:8].upper()}"
        
        suspension_until = datetime.now(timezone.utc) + timedelta(hours=suspension_hours)
        self.suspended_strategies[strategy_id] = suspension_until
        
        veto = RiskVeto(
            veto_id=veto_id,
            decision_type=RiskDecisionType.STRATEGY_SUSPENSION,
            target_entity=strategy_id,
            reason=reason,
            risk_metrics={"suspension_hours": suspension_hours},
            issued_at=datetime.now(timezone.utc),
            override_allowed=False
        )
        
        self.active_vetos.append(veto)
        
        logger.critical(
            f"[CRO] ⛔ STRATEGY SUSPENDED: {strategy_id}\n"
            f"   Veto ID: {veto_id}\n"
            f"   Reason: {reason}\n"
            f"   Suspension Until: {suspension_until.isoformat()}\n"
            f"   Duration: {suspension_hours}h"
        )
        
        # Publish suspension event
        await self.event_bus.publish(
            "fund.risk.strategy.suspended",
            {
                "veto_id": veto_id,
                "strategy_id": strategy_id,
                "reason": reason,
                "suspended_at": veto.issued_at.isoformat(),
                "suspension_until": suspension_until.isoformat(),
                "suspension_hours": suspension_hours,
                "issued_by": "CRO"
            }
        )
        
        return veto_id
    
    async def enforce_leverage_reduction(
        self,
        current_leverage: float,
        target_leverage: float,
        reason: str
    ) -> str:
        """
        Enforce immediate leverage reduction.
        
        Args:
            current_leverage: Current portfolio leverage
            target_leverage: Target leverage to reduce to
            reason: Reduction reason
        
        Returns:
            Veto ID
        """
        import uuid
        
        veto_id = f"CRO-DELEVER-{uuid.uuid4().hex[:8].upper()}"
        
        veto = RiskVeto(
            veto_id=veto_id,
            decision_type=RiskDecisionType.LEVERAGE_REDUCTION,
            target_entity="PORTFOLIO",
            reason=reason,
            risk_metrics={
                "current_leverage": current_leverage,
                "target_leverage": target_leverage,
                "reduction_pct": (current_leverage - target_leverage) / current_leverage
            },
            issued_at=datetime.now(timezone.utc),
            override_allowed=False
        )
        
        self.active_vetos.append(veto)
        
        logger.critical(
            f"[CRO] 📉 LEVERAGE REDUCTION ENFORCED\n"
            f"   Veto ID: {veto_id}\n"
            f"   Current Leverage: {current_leverage:.1f}x\n"
            f"   Target Leverage: {target_leverage:.1f}x\n"
            f"   Reduction: {veto.risk_metrics['reduction_pct']:.1%}\n"
            f"   Reason: {reason}"
        )
        
        # Publish leverage reduction event
        await self.event_bus.publish(
            "fund.risk.leverage.reduction",
            {
                "veto_id": veto_id,
                "current_leverage": current_leverage,
                "target_leverage": target_leverage,
                "reduction_pct": veto.risk_metrics['reduction_pct'],
                "reason": reason,
                "issued_at": veto.issued_at.isoformat(),
                "issued_by": "CRO",
                "priority": "CRITICAL"
            }
        )
        
        return veto_id
    
    async def _handle_position_opened(self, event_data: dict) -> None:
        """Handle position opening events - check if veto required."""
        position_id = event_data.get("position_id")
        symbol = event_data.get("symbol")
        size_usd = event_data.get("size_usd", 0.0)
        leverage = event_data.get("leverage", 1.0)
        
        # Get current portfolio value
        portfolio_value = await self._get_portfolio_value()
        
        # Check single position size
        position_pct = size_usd / portfolio_value if portfolio_value > 0 else 0
        if position_pct > self.max_single_position:
            await self.veto_position(
                position_id,
                f"Position size {position_pct:.1%} exceeds max {self.max_single_position:.1%}",
                {
                    "position_size_usd": size_usd,
                    "portfolio_value": portfolio_value,
                    "position_pct": position_pct,
                    "max_allowed": self.max_single_position
                }
            )
            return
        
        # Check leverage
        if leverage > self.max_leverage:
            await self.veto_position(
                position_id,
                f"Leverage {leverage:.0f}x exceeds max {self.max_leverage:.0f}x",
                {
                    "leverage": leverage,
                    "max_leverage": self.max_leverage
                }
            )
            return
        
        logger.debug(
            f"[CRO] Position {position_id} ({symbol}) passed risk checks: "
            f"size={position_pct:.1%}, leverage={leverage:.0f}x"
        )
    
    async def _handle_risk_update(self, event_data: dict) -> None:
        """Handle portfolio risk assessment updates."""
        self.current_portfolio_var = event_data.get("portfolio_var", 0.0)
        self.current_portfolio_cvar = event_data.get("portfolio_cvar", 0.0)
        
        logger.debug(
            f"[CRO] Risk Update: VaR={self.current_portfolio_var:.2%}, "
            f"CVaR={self.current_portfolio_cvar:.2%}"
        )
        
        # Check VaR breach
        if self.current_portfolio_var > self.max_portfolio_var:
            logger.warning(
                f"[CRO] ⚠️ Portfolio VaR breach: "
                f"{self.current_portfolio_var:.2%} > {self.max_portfolio_var:.2%}"
            )
            await self.event_bus.publish(
                "fund.risk.var.breach",
                {
                    "current_var": self.current_portfolio_var,
                    "max_var": self.max_portfolio_var,
                    "breach_pct": (self.current_portfolio_var - self.max_portfolio_var) / self.max_portfolio_var,
                    "reported_by": "CRO"
                }
            )
        
        # Check CVaR breach (more severe)
        if self.current_portfolio_cvar > self.max_portfolio_cvar:
            logger.critical(
                f"[CRO] 🚨 Portfolio CVaR BREACH: "
                f"{self.current_portfolio_cvar:.2%} > {self.max_portfolio_cvar:.2%}"
            )
            # Trigger immediate leverage reduction
            current_leverage = await self._get_current_leverage()
            target_leverage = current_leverage * 0.70  # Reduce to 70%
            await self.enforce_leverage_reduction(
                current_leverage,
                target_leverage,
                f"CVaR breach: {self.current_portfolio_cvar:.2%} > {self.max_portfolio_cvar:.2%}"
            )
    
    async def _handle_strategy_allocation(self, event_data: dict) -> None:
        """Handle strategy allocation events - validate risk parameters."""
        strategy_id = event_data.get("strategy_id")
        max_drawdown = event_data.get("max_drawdown", 0.0)
        
        # Check if strategy's max drawdown exceeds portfolio limits
        if max_drawdown > self.max_portfolio_cvar:
            logger.warning(
                f"[CRO] ⚠️ Strategy {strategy_id} max drawdown {max_drawdown:.2%} "
                f"exceeds portfolio CVaR limit {self.max_portfolio_cvar:.2%}"
            )
            # Don't veto, but issue warning to CEO
            await self.event_bus.publish(
                "fund.risk.warning",
                {
                    "warning_type": "strategy_drawdown_high",
                    "strategy_id": strategy_id,
                    "strategy_max_drawdown": max_drawdown,
                    "portfolio_cvar_limit": self.max_portfolio_cvar,
                    "issued_by": "CRO",
                    "addressed_to": "CEO"
                }
            )
    
    async def _handle_risk_escalation(self, event_data: dict) -> None:
        """Handle risk escalations from CEO or other components."""
        reason = event_data.get("reason")
        priority = event_data.get("priority")
        
        logger.critical(
            f"[CRO] 🚨 Risk escalation received: {reason} (priority={priority})"
        )
        
        # Take immediate action based on escalation
        if reason == "cvar_breach":
            current_cvar = event_data.get("current_cvar")
            # Emergency leverage reduction
            current_leverage = await self._get_current_leverage()
            await self.enforce_leverage_reduction(
                current_leverage,
                current_leverage * 0.50,  # Cut leverage in half
                f"Emergency response to CVaR breach: {current_cvar:.2%}"
            )
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value (placeholder)."""
        # TODO: Integrate with actual portfolio tracking
        return 10000.0  # Placeholder
    
    async def _get_current_leverage(self) -> float:
        """Get current portfolio leverage (placeholder)."""
        # TODO: Integrate with actual position tracking
        return 15.0  # Placeholder
    
    def get_status(self) -> dict:
        """Get CRO status and metrics."""
        return {
            "max_portfolio_var": self.max_portfolio_var,
            "max_portfolio_cvar": self.max_portfolio_cvar,
            "max_leverage": self.max_leverage,
            "max_single_position": self.max_single_position,
            "current_portfolio_var": self.current_portfolio_var,
            "current_portfolio_cvar": self.current_portfolio_cvar,
            "current_leverage": self.current_leverage,
            "active_vetos": len(self.active_vetos),
            "suspended_strategies": len(self.suspended_strategies)
        }
