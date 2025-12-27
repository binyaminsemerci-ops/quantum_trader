"""
AI-CFO (Chief Financial Officer)
=================================

Capital allocator and cashflow manager.

Responsibilities:
- Decide profit allocation (lock, reinvest, reserve)
- Manage cash reserves for margin calls
- Monitor capital efficiency
- Coordinate with AI-CEO on capital profile changes
- Track P&L distribution

Decision Logic:
- High profits → lock profits, build reserves
- Stable growth → balanced reinvestment
- Drawdown → preserve capital, minimal reinvestment
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    CashflowDecision,
    PortfolioSnapshot,
    DecisionType,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class AICFO(FederationRole):
    """AI Chief Financial Officer - Capital allocator"""
    
    # Base cashflow policy
    BASE_CASHFLOW_POLICY = {
        "profit_lock_pct": 0.30,        # Lock 30% of profits
        "reinvest_pct": 0.60,           # Reinvest 60%
        "reserve_buffer_pct": 0.10,     # 10% reserve
    }
    
    # Reserve thresholds
    MIN_RESERVE_PCT = 0.05  # 5% of equity
    TARGET_RESERVE_PCT = 0.15  # 15% of equity
    
    def __init__(self):
        super().__init__("ai-cfo")
        self.current_policy = self.BASE_CASHFLOW_POLICY.copy()
        self.total_locked_profits = 0.0
        self.reserve_balance = 0.0
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate cashflow allocation based on portfolio state.
        
        Logic:
        - Profit realized → decide allocation
        - Low reserves → prioritize building reserves
        - High DD → lock more profits
        """
        decisions = []
        
        # Check if cashflow policy needs adjustment
        cashflow_decision = self._evaluate_cashflow_policy(snapshot)
        if cashflow_decision:
            self.logger.info(
                "Cashflow policy adjustment",
                profit_lock=cashflow_decision.payload["profit_lock_pct"],
                reinvest=cashflow_decision.payload["reinvest_pct"],
                reserve=cashflow_decision.payload["reserve_buffer_pct"],
            )
            decisions.append(cashflow_decision)
        
        return decisions
    
    def _evaluate_cashflow_policy(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Adjust profit allocation based on portfolio state.
        
        High profits → lock more
        High DD → lock more, lower reinvestment
        Low reserves → build reserves
        """
        dd_pct = snapshot.drawdown_pct
        pnl_today = snapshot.realized_pnl_today
        equity = snapshot.total_equity
        
        # Calculate current reserve percentage
        reserve_pct = self.reserve_balance / equity if equity > 0 else 0
        
        # Scenario 1: High drawdown - maximize profit locking
        if dd_pct > 0.05:
            new_policy = {
                "profit_lock_pct": 0.60,     # Lock 60%
                "reinvest_pct": 0.25,        # Minimal reinvestment
                "reserve_buffer_pct": 0.15,  # Build reserves
            }
            
            if new_policy != self.current_policy:
                self.current_policy = new_policy
                
                return FederationDecision(
                    decision_type=DecisionType.CASHFLOW,
                    role_source="cfo",
                    priority=DecisionPriority.HIGH,
                    reason=f"High DD ({dd_pct:.1%}), maximizing profit locking",
                    payload=new_policy,
                )
        
        # Scenario 2: Low reserves - prioritize reserve building
        elif reserve_pct < self.MIN_RESERVE_PCT:
            new_policy = {
                "profit_lock_pct": 0.20,     # Minimal locking
                "reinvest_pct": 0.50,        # Moderate reinvestment
                "reserve_buffer_pct": 0.30,  # Aggressive reserve build
            }
            
            if new_policy != self.current_policy:
                self.current_policy = new_policy
                
                return FederationDecision(
                    decision_type=DecisionType.CASHFLOW,
                    role_source="cfo",
                    priority=DecisionPriority.HIGH,
                    reason=f"Low reserves ({reserve_pct:.1%}), building buffer",
                    payload=new_policy,
                )
        
        # Scenario 3: Strong profits - balanced allocation
        elif pnl_today > equity * 0.02 and dd_pct < 0.02:
            new_policy = {
                "profit_lock_pct": 0.30,     # Standard locking
                "reinvest_pct": 0.60,        # Strong reinvestment
                "reserve_buffer_pct": 0.10,  # Maintain reserves
            }
            
            if new_policy != self.current_policy:
                self.current_policy = new_policy
                
                return FederationDecision(
                    decision_type=DecisionType.CASHFLOW,
                    role_source="cfo",
                    priority=DecisionPriority.NORMAL,
                    reason=f"Strong profits ({pnl_today:.2f} USD, {pnl_today/equity:.1%} of equity), balanced allocation",
                    payload=new_policy,
                )
        
        # Scenario 4: Excess reserves - can be more aggressive
        elif reserve_pct > self.TARGET_RESERVE_PCT:
            new_policy = {
                "profit_lock_pct": 0.25,     # Lower locking
                "reinvest_pct": 0.70,        # Aggressive reinvestment
                "reserve_buffer_pct": 0.05,  # Minimal reserves
            }
            
            if new_policy != self.current_policy:
                self.current_policy = new_policy
                
                return FederationDecision(
                    decision_type=DecisionType.CASHFLOW,
                    role_source="cfo",
                    priority=DecisionPriority.NORMAL,
                    reason=f"Strong reserves ({reserve_pct:.1%}), increasing reinvestment",
                    payload=new_policy,
                )
        
        return None
