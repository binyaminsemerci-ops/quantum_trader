"""
Exit Brain v3 Router - Helper for mapping signals/positions to plans.
"""

import logging
from typing import Dict, Optional

from backend.domains.exits.exit_brain_v3.models import ExitContext, ExitPlan
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.integration import build_context_from_position

logger = logging.getLogger(__name__)


class ExitRouter:
    """
    Routes position/signal updates to Exit Brain and tracks active plans.
    
    🔧 SINGLETON: All services share the same instance and plan cache.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Ensure only one ExitRouter instance exists (singleton pattern)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize router with planner (only runs once)"""
        if ExitRouter._initialized:
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"[EXIT ROUTER] Returning existing singleton instance (id={id(self)})")
            return
        
        self.brain = ExitBrainV3()
        self.active_plans: Dict[str, ExitPlan] = {}
        self.logger = logging.getLogger(__name__)
        ExitRouter._initialized = True
        self.logger.info(f"[EXIT ROUTER] Singleton instance created (id={id(self)}) - all services share same plan cache")
    
    async def get_or_create_plan(
        self,
        position: Dict,
        rl_hints: Optional[Dict] = None,
        risk_context: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        force_rebuild: bool = False
    ) -> ExitPlan:
        """
        Get existing exit plan or create new one for position.
        
        Args:
            position: Binance position dict
            rl_hints: Optional RL v3 hints
            risk_context: Optional Risk v3 state
            market_data: Optional market metrics
            force_rebuild: Force creation of new plan
            
        Returns:
            ExitPlan (cached or newly created)
        """
        symbol = position["symbol"]
        
        # Check cache
        if not force_rebuild and symbol in self.active_plans:
            plan = self.active_plans[symbol]
            self.logger.debug(f"[EXIT ROUTER] Using cached plan for {symbol}")
            return plan
        
        # Build context
        ctx = build_context_from_position(
            position=position,
            rl_hints=rl_hints,
            risk_context=risk_context,
            market_data=market_data
        )
        
        # Create plan
        plan = await self.brain.build_exit_plan(ctx)
        
        # Cache plan
        self.active_plans[symbol] = plan
        self.logger.info(
            f"[EXIT ROUTER] Created new plan for {symbol} on instance id={id(self)}: "
            f"{len(plan.legs)} legs, Strategy={plan.strategy_id}, Cache now has {len(self.active_plans)} plans"
        )
        
        return plan
    
    def invalidate_plan(self, symbol: str):
        """Remove cached plan (forces rebuild on next request)"""
        if symbol in self.active_plans:
            del self.active_plans[symbol]
            self.logger.debug(f"[EXIT ROUTER] Invalidated plan for {symbol}")
    
    def get_active_plan(self, symbol: str) -> Optional[ExitPlan]:
        """Get cached plan without creating new one"""
        plan = self.active_plans.get(symbol)
        self.logger.debug(
            f"[EXIT ROUTER] get_active_plan({symbol}) called on instance id={id(self)}, "
            f"cache has {len(self.active_plans)} plans, returning: {plan is not None}"
        )
        return plan
