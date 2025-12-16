"""
Hedge Fund OS v2 - Integration Module

Complete integration of all 8 Hedge Fund OS v2 components.

Components:
1. AI CEO v2 (Fund CEO) - Strategic fund management
2. AI CRO v2 (Chief Risk Officer) - Enterprise risk with veto power
3. AI CIO (Chief Investment Officer) - Portfolio management
4. Compliance OS - Real-time compliance monitoring
5. Federation v3 - Multi-agent coordination
6. Audit OS - Complete audit trail
7. Regulation Engine - Dynamic regulatory compliance
8. Decision Transparency Layer - Explainable AI decisions

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from typing import Optional
from backend.core.policy_store import PolicyStore
from backend.core.event_bus import EventBus
from backend.domains.fund_management.ceo_v2 import AIFundCEO
from backend.domains.fund_management.cro_v2 import AIFundCRO
from backend.domains.fund_management.cio import AIFundCIO
from backend.domains.governance.compliance_os import ComplianceOS
from backend.domains.governance.federation_v3 import FederationV3
from backend.domains.governance.audit_os import AuditOS
from backend.domains.governance.regulation_engine import RegulationEngine
from backend.domains.governance.transparency_layer import DecisionTransparencyLayer

logger = logging.getLogger(__name__)


class HedgeFundOS:
    """
    Hedge Fund OS v2 - Complete institutional-grade hedge fund operating system.
    
    Architecture:
    - Event-driven coordination via EventBus v2
    - Centralized policy management via PolicyStore v2
    - Multi-agent consensus via Federation v3
    - Complete auditability via Audit OS
    - Regulatory compliance via Regulation Engine & Compliance OS
    - Explainable decisions via Transparency Layer
    
    Decision Hierarchy:
    1. CRO (VETO POWER) - Can override all except regulations
    2. CEO (HIGHEST) - Strategic decisions, can override CIO
    3. CIO (MEDIUM) - Portfolio management, subject to CEO approval
    4. Compliance OS (ENFORCER) - Blocks non-compliant trades
    5. Regulation Engine (ENFORCER) - Enforces regulatory rules
    
    Observers (no decision power):
    - Audit OS - Records everything
    - Transparency Layer - Explains everything
    - Federation v3 - Coordinates consensus
    """
    
    def __init__(
        self,
        policy_store: PolicyStore,
        event_bus: EventBus,
        fund_name: str = "Quantum Hedge Fund",
        target_annual_return: float = 0.25,  # 25%
        max_annual_drawdown: float = 0.15,   # -15%
    ):
        """
        Initialize Hedge Fund OS v2.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            fund_name: Name of the hedge fund
            target_annual_return: Target annual return
            max_annual_drawdown: Maximum acceptable drawdown
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        self.fund_name = fund_name
        
        logger.info(
            f"\n{'='*80}\n"
            f"  HEDGE FUND OS v2 - INITIALIZATION\n"
            f"{'='*80}\n"
            f"  Fund: {fund_name}\n"
            f"  Target Return: {target_annual_return:.1%}\n"
            f"  Max Drawdown: {max_annual_drawdown:.1%}\n"
            f"{'='*80}"
        )
        
        # Initialize all 8 components
        self._initialize_components(target_annual_return, max_annual_drawdown)
        
        logger.info(
            f"\n{'='*80}\n"
            f"  ✅ HEDGE FUND OS v2 INITIALIZED\n"
            f"  All 8 components operational\n"
            f"{'='*80}\n"
        )
    
    def _initialize_components(
        self,
        target_annual_return: float,
        max_annual_drawdown: float
    ) -> None:
        """Initialize all 8 Hedge Fund OS v2 components."""
        
        # 1. CEO - Strategic fund management
        logger.info("[1/8] Initializing AI CEO v2 (Fund CEO)...")
        self.ceo = AIFundCEO(
            fund_name=self.fund_name,
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            target_annual_return=target_annual_return,
            max_annual_drawdown=max_annual_drawdown,
            target_sharpe_ratio=2.0
        )
        
        # 2. CRO - Enterprise risk with veto power
        logger.info("[2/8] Initializing AI CRO v2 (Chief Risk Officer)...")
        self.cro = AIFundCRO(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            max_portfolio_var=0.10,
            max_portfolio_cvar=max_annual_drawdown,
            max_leverage=30.0,
            max_single_position=0.15
        )
        
        # 3. CIO - Portfolio management
        logger.info("[3/8] Initializing AI CIO (Chief Investment Officer)...")
        self.cio = AIFundCIO(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            rebalance_threshold=0.10,
            min_diversification=5,
            max_correlation=0.60,
            cash_reserve_pct=0.10
        )
        
        # 4. Compliance OS - Real-time compliance monitoring
        logger.info("[4/8] Initializing Compliance OS...")
        self.compliance = ComplianceOS(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            max_position_size=0.15,
            max_leverage=30.0,
            max_sector_concentration=0.40,
            enable_wash_trading_detection=True
        )
        
        # 5. Federation v3 - Multi-agent coordination
        logger.info("[5/8] Initializing Federation v3...")
        self.federation = FederationV3(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            voting_timeout_minutes=5,
            quorum_percentage=0.67,
            majority_percentage=0.67
        )
        
        # 6. Audit OS - Complete audit trail
        logger.info("[6/8] Initializing Audit OS...")
        self.audit = AuditOS(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            audit_log_path="./data/audit",
            enable_cryptographic_hash=True,
            retention_days=365
        )
        
        # 7. Regulation Engine - Dynamic regulatory compliance
        logger.info("[7/8] Initializing Regulation Engine...")
        self.regulation = RegulationEngine(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            active_jurisdictions=None  # Use default (crypto exchanges)
        )
        
        # 8. Transparency Layer - Explainable AI decisions
        logger.info("[8/8] Initializing Decision Transparency Layer...")
        self.transparency = DecisionTransparencyLayer(
            policy_store=self.policy_store,
            event_bus=self.event_bus,
            min_explainability_score=0.70
        )
    
    async def startup(self) -> None:
        """Perform startup sequence."""
        logger.info("[HedgeFundOS] Starting up...")
        
        # Publish startup event
        await self.event_bus.publish(
            "fund.system.started",
            {
                "fund_name": self.fund_name,
                "components": [
                    "CEO", "CRO", "CIO", "Compliance", 
                    "Federation", "Audit", "Regulation", "Transparency"
                ],
                "version": "v2.0.0"
            }
        )
        
        logger.info("[HedgeFundOS] ✅ Startup complete")
    
    async def shutdown(self) -> None:
        """Perform shutdown sequence."""
        logger.info("[HedgeFundOS] Shutting down...")
        
        # Publish shutdown event
        await self.event_bus.publish(
            "fund.system.stopped",
            {
                "fund_name": self.fund_name
            }
        )
        
        logger.info("[HedgeFundOS] ✅ Shutdown complete")
    
    def get_system_status(self) -> dict:
        """
        Get complete system status.
        
        Returns:
            System status dictionary
        """
        return {
            "fund_name": self.fund_name,
            "version": "v2.0.0",
            "components": {
                "ceo": self.ceo.get_status(),
                "cro": self.cro.get_status(),
                "cio": self.cio.get_status(),
                "compliance": self.compliance.get_status(),
                "federation": self.federation.get_status(),
                "audit": self.audit.get_status(),
                "regulation": self.regulation.get_status(),
                "transparency": self.transparency.get_status()
            }
        }


# Factory function for easy initialization
async def create_hedge_fund_os(
    policy_store: PolicyStore,
    event_bus: EventBus,
    fund_name: str = "Quantum Hedge Fund",
    target_annual_return: float = 0.25,
    max_annual_drawdown: float = 0.15
) -> HedgeFundOS:
    """
    Create and initialize Hedge Fund OS v2.
    
    Args:
        policy_store: PolicyStore v2 instance
        event_bus: EventBus v2 instance
        fund_name: Name of the hedge fund
        target_annual_return: Target annual return (default 25%)
        max_annual_drawdown: Maximum acceptable drawdown (default -15%)
    
    Returns:
        Initialized HedgeFundOS instance
    
    Example:
        ```python
        from backend.core.policy_store import PolicyStore
        from backend.core.event_bus import EventBus
        from backend.domains.hedge_fund_os import create_hedge_fund_os
        
        # Initialize infrastructure
        policy_store = PolicyStore(redis_url="redis://localhost:6379")
        event_bus = EventBus(redis_url="redis://localhost:6379")
        await policy_store.initialize()
        await event_bus.initialize()
        
        # Create Hedge Fund OS
        fund_os = await create_hedge_fund_os(
            policy_store=policy_store,
            event_bus=event_bus,
            fund_name="Quantum Hedge Fund",
            target_annual_return=0.30,  # 30%
            max_annual_drawdown=0.12    # -12%
        )
        
        # Start the system
        await fund_os.startup()
        
        # Get system status
        status = fund_os.get_system_status()
        print(status)
        ```
    """
    hedge_fund_os = HedgeFundOS(
        policy_store=policy_store,
        event_bus=event_bus,
        fund_name=fund_name,
        target_annual_return=target_annual_return,
        max_annual_drawdown=max_annual_drawdown
    )
    
    await hedge_fund_os.startup()
    
    return hedge_fund_os
