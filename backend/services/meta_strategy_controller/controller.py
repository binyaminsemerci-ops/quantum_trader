"""
Meta Strategy Controller - The "Brain" of Quantum Trader AI OS.

Analyzes market conditions and sets global trading policy.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from typing import Protocol as TypingProtocol
from backend.services.eventbus import EventBus, PolicyUpdatedEvent, HealthStatusChangedEvent, Event
from backend.services.policy_store import PolicyStore, RiskMode
from .models import MarketAnalysis, MarketRegime

logger = logging.getLogger(__name__)


class MetricsRepository(TypingProtocol):
    """Interface for accessing system-wide metrics"""
    
    def get_current_drawdown_pct(self) -> float:
        """Get current drawdown as percentage"""
        ...
    
    def get_global_winrate(self, last_trades: int = 200) -> float:
        """Get global winrate from recent trades"""
        ...
    
    def get_equity_curve(self, days: int = 30) -> list[tuple[datetime, float]]:
        """Get equity curve as (timestamp, balance) tuples"""
        ...
    
    def get_global_regime(self) -> str:
        """Get current market regime"""
        ...
    
    def get_volatility_level(self) -> str:
        """Get current volatility regime (LOW/NORMAL/HIGH/EXTREME)"""
        ...
    
    def get_consecutive_losses(self) -> int:
        """Get number of consecutive losing trades"""
        ...
    
    def get_days_since_last_profit(self) -> int:
        """Get days since last profitable trade"""
        ...


class StrategyRepository(TypingProtocol):
    """Interface for strategy repository"""
    
    def get_all_strategies(self) -> list:
        """Get all registered strategies"""
        ...
    
    def get_strategy(self, strategy_id: str):
        """Get a specific strategy by ID"""
        ...


class MetaStrategyController:
    """
    Meta Strategy Controller AI.
    
    The MSC AI is the top-level decision maker that:
    - Analyzes market conditions
    - Determines optimal risk mode
    - Sets global trading parameters
    - Reacts to system health alerts
    - Publishes policy updates
    
    The MSC AI runs continuously and updates policy when conditions change.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        policy_store: PolicyStore,
        update_interval: int = 300,  # 5 minutes
        auto_adjust: bool = True,
    ):
        """
        Initialize the Meta Strategy Controller.
        
        Args:
            event_bus: EventBus for publishing/subscribing
            policy_store: PolicyStore for reading/writing policy
            update_interval: Seconds between policy evaluations
            auto_adjust: Whether to automatically adjust policy
        """
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.update_interval = update_interval
        self.auto_adjust = auto_adjust
        
        self._running = False
        self._last_analysis: Optional[MarketAnalysis] = None
        self._emergency_mode = False
        
        # Subscribe to health alerts
        event_bus.subscribe("health.status_changed", self.on_health_alert)
        
        logger.info(
            f"MSC AI initialized (update_interval={update_interval}s, "
            f"auto_adjust={auto_adjust})"
        )
    
    async def analyze_market_conditions(self) -> MarketAnalysis:
        """
        Analyze current market conditions.
        
        This is where the MSC AI gathers data from various sources:
        - Regime detector
        - Volatility indicators
        - Performance metrics
        - Risk indicators
        
        Returns:
            MarketAnalysis with current conditions
        """
        # TODO: Integrate with actual regime detector and metrics
        # For now, return a placeholder analysis
        
        # In real implementation, this would:
        # 1. Query regime detector for current regime
        # 2. Calculate volatility from recent price data
        # 3. Get recent performance metrics from analytics
        # 4. Calculate correlation and liquidity scores
        # 5. Compute overall risk score
        
        logger.debug("Analyzing market conditions...")
        
        # Placeholder analysis (replace with real implementation)
        analysis = MarketAnalysis(
            regime=MarketRegime.RANGING,
            volatility=0.5,
            trend_strength=0.6,
            correlation=0.4,
            liquidity_score=0.8,
            risk_score=0.3,
            recent_win_rate=0.60,
            recent_sharpe=1.8,
            recent_drawdown=-2.5,
            timestamp=datetime.utcnow(),
        )
        
        self._last_analysis = analysis
        return analysis
    
    def determine_optimal_risk_mode(self, analysis: MarketAnalysis) -> RiskMode:
        """
        Determine the optimal risk mode based on market analysis.
        
        Decision logic:
        - DEFENSIVE: High risk, recent losses, choppy markets
        - AGGRESSIVE: Strong trends, low risk, good recent performance
        - NORMAL: Default/balanced conditions
        
        Args:
            analysis: MarketAnalysis to base decision on
            
        Returns:
            Optimal RiskMode
        """
        # Emergency override
        if self._emergency_mode:
            logger.warning("Emergency mode active - forcing DEFENSIVE")
            return RiskMode.DEFENSIVE
        
        # Check for defensive conditions
        if analysis.is_unfavorable_requires_defensive():
            logger.info("Unfavorable conditions detected - switching to DEFENSIVE")
            return RiskMode.DEFENSIVE
        
        # Check for aggressive conditions
        if analysis.is_favorable_for_aggressive():
            logger.info("Favorable conditions detected - switching to AGGRESSIVE")
            return RiskMode.AGGRESSIVE
        
        # Default to NORMAL
        return RiskMode.NORMAL
    
    async def update_policy(self, reason: str) -> None:
        """
        Analyze market and update policy if needed.
        
        This is the main decision loop:
        1. Analyze market conditions
        2. Determine optimal risk mode
        3. Compare with current policy
        4. Update if changed
        5. Publish PolicyUpdatedEvent
        
        Args:
            reason: Why the policy update was triggered
        """
        try:
            # Get current policy
            current_policy = await self.policy_store.get_policy()
            
            # Analyze market
            analysis = await self.analyze_market_conditions()
            
            # Determine optimal risk mode
            optimal_mode = self.determine_optimal_risk_mode(analysis)
            
            # Check if change is needed
            if optimal_mode == current_policy.risk_mode:
                logger.debug(f"Risk mode unchanged: {optimal_mode.value}")
                return
            
            # Update policy
            logger.info(
                f"Updating risk mode: {current_policy.risk_mode.value} → {optimal_mode.value} "
                f"(reason: {reason})"
            )
            
            await self.policy_store.update_risk_mode(optimal_mode, updated_by="MSC_AI")
            
            # Get updated policy for event
            updated_policy = await self.policy_store.get_policy()
            
            # Publish PolicyUpdatedEvent
            event = PolicyUpdatedEvent.create(
                risk_mode=updated_policy.risk_mode,
                allowed_strategies=updated_policy.allowed_strategies,
                global_min_confidence=updated_policy.global_min_confidence,
                max_risk_per_trade=updated_policy.max_risk_per_trade,
                max_positions=updated_policy.max_positions,
                updated_by="MSC_AI",
            )
            
            await self.event_bus.publish(event)
            
            logger.info(f"Policy updated and broadcasted: {optimal_mode.value}")
            
        except Exception as e:
            logger.error(f"Error updating policy: {e}", exc_info=True)
    
    async def on_health_alert(self, event: Event) -> None:
        """
        React to health status changes.
        
        If health becomes CRITICAL, immediately switch to DEFENSIVE mode.
        
        Args:
            event: HealthStatusChangedEvent
        """
        try:
            new_status = event.payload["new_status"]
            component = event.payload["component"]
            reason = event.payload["reason"]
            
            logger.warning(
                f"Health alert received: {component} → {new_status} ({reason})"
            )
            
            if new_status == "CRITICAL":
                logger.error("CRITICAL health status - activating emergency defensive mode")
                self._emergency_mode = True
                await self.update_policy(f"CRITICAL health alert: {component} - {reason}")
            
            elif new_status == "DEGRADED":
                logger.warning("DEGRADED health status - evaluating policy")
                await self.update_policy(f"Health degraded: {component} - {reason}")
            
            elif new_status == "HEALTHY" and self._emergency_mode:
                logger.info("Health restored - deactivating emergency mode")
                self._emergency_mode = False
                await self.update_policy("Health restored to HEALTHY")
            
        except Exception as e:
            logger.error(f"Error handling health alert: {e}", exc_info=True)
    
    async def run_forever(self) -> None:
        """
        Run the MSC AI in continuous mode.
        
        Periodically evaluates market conditions and updates policy.
        This should be run as a background task.
        """
        self._running = True
        logger.info("MSC AI started, entering continuous evaluation loop")
        
        while self._running:
            try:
                if self.auto_adjust:
                    await self.update_policy("scheduled_evaluation")
                
                # Wait for next interval
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                logger.info("MSC AI loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in MSC AI loop: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(60)  # Wait 1 min before retry
    
    def stop(self) -> None:
        """Stop the MSC AI."""
        logger.info("Stopping MSC AI...")
        self._running = False
    
    def get_last_analysis(self) -> Optional[MarketAnalysis]:
        """Get the most recent market analysis."""
        return self._last_analysis
    
    async def force_defensive_mode(self, reason: str) -> None:
        """
        Force defensive mode (e.g., from manual override or emergency).
        
        Args:
            reason: Why defensive mode is being forced
        """
        logger.warning(f"Forcing DEFENSIVE mode: {reason}")
        self._emergency_mode = True
        await self.update_policy(f"Manual override: {reason}")
    
    async def clear_emergency_mode(self) -> None:
        """Clear emergency mode and resume normal operation."""
        logger.info("Clearing emergency mode")
        self._emergency_mode = False
        await self.update_policy("Emergency mode cleared")
