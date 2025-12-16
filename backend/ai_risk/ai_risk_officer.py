"""AI Risk Officer - Intelligent risk monitoring agent.

The AI Risk Officer continuously monitors portfolio risk at multiple levels
and provides real-time risk assessment, alerts, and recommendations.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import redis.asyncio as redis
from redis.asyncio import Redis

from backend.ai_risk.risk_brain import PortfolioRiskData, RiskAssessment, RiskBrain
from backend.ai_risk.risk_models import RiskModels
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class AI_RiskOfficer:
    """
    AI Risk Officer - Intelligent Risk Monitoring Agent.
    
    Responsibilities:
    - Continuously assess portfolio risk
    - Calculate VaR, ES, tail risk
    - Monitor drawdown and exposure
    - Generate risk alerts
    - Recommend risk limit adjustments
    - Provide risk state to AI CEO
    - Integrate with Risk OS
    
    Integration:
    - Runs in analytics-os-service or risk-os-service
    - Uses EventBus v2 for communication
    - Reads from PolicyStore v2
    - Publishes risk_state_update, risk_alert, risk_ceiling_update
    """
    
    DEFAULT_ASSESSMENT_INTERVAL = 30.0  # seconds
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        policy_store: PolicyStore,
        brain: Optional[RiskBrain] = None,
        assessment_interval: float = DEFAULT_ASSESSMENT_INTERVAL,
    ):
        """
        Initialize AI Risk Officer.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus instance
            policy_store: PolicyStore instance
            brain: Optional RiskBrain instance (for testing)
            assessment_interval: Seconds between assessments
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.brain = brain or RiskBrain()
        self.assessment_interval = assessment_interval
        
        # State tracking
        self._running = False
        self._assessment_task: Optional[asyncio.Task] = None
        
        # Cached data
        self._latest_portfolio_state: Optional[dict] = None
        self._returns_history: np.ndarray = np.array([])
        
        # Instance ID
        self._instance_id = f"ai_ro_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"AI_RiskOfficer initialized: instance_id={self._instance_id}, "
            f"assessment_interval={assessment_interval}s"
        )
    
    async def initialize(self) -> None:
        """Initialize AI Risk Officer."""
        # Verify EventBus
        await self.event_bus.initialize()
        
        # Verify PolicyStore
        policy = await self.policy_store.get_policy()
        logger.info(f"AI_RO verified PolicyStore: mode={policy.active_mode.value}")
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Load initial returns history
        await self._load_returns_history()
        
        logger.info("AI_RiskOfficer initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Portfolio updates
        self.event_bus.subscribe("position_opened", self._handle_position_event)
        self.event_bus.subscribe("position_closed", self._handle_position_event)
        self.event_bus.subscribe("portfolio_state_update", self._handle_portfolio_state_update)
        
        # Trade execution
        self.event_bus.subscribe("trade_executed", self._handle_trade_executed)
        
        # Market data updates
        self.event_bus.subscribe("market_data_update", self._handle_market_data_update)
        
        logger.info("AI_RO subscribed to events")
    
    async def _load_returns_history(self) -> None:
        """Load historical returns from Redis or database."""
        # Try to load from Redis cache
        try:
            cached_returns = await self.redis.get("quantum:risk:returns_history")
            if cached_returns:
                self._returns_history = np.frombuffer(cached_returns, dtype=np.float64)
                logger.info(f"AI_RO loaded {len(self._returns_history)} historical returns")
            else:
                # Initialize with empty array (will populate as trades execute)
                self._returns_history = np.array([])
                logger.info("AI_RO initialized with empty returns history")
        except Exception as e:
            logger.error(f"Failed to load returns history: {e}")
            self._returns_history = np.array([])
    
    async def start(self) -> None:
        """Start AI Risk Officer assessment loop."""
        if self._running:
            logger.warning("AI_RO already running")
            return
        
        self._running = True
        
        # Start EventBus processing
        await self.event_bus.start()
        
        # Start assessment loop
        self._assessment_task = asyncio.create_task(self._assessment_loop())
        
        logger.info("AI_RiskOfficer started")
    
    async def stop(self) -> None:
        """Stop AI Risk Officer gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel assessment task
        if self._assessment_task:
            self._assessment_task.cancel()
            try:
                await self._assessment_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AI_RiskOfficer stopped")
    
    async def _assessment_loop(self) -> None:
        """Main risk assessment loop."""
        logger.info(f"AI_RO assessment loop started (interval={self.assessment_interval}s)")
        
        while self._running:
            try:
                # Check if Risk Officer is enabled
                policy = await self.policy_store.get_policy()
                active_config = policy.get_active_config()
                
                # Check for enable_ai_ro flag (if exists)
                if hasattr(active_config, 'enable_ai_ro') and not active_config.enable_ai_ro:
                    logger.debug("AI_RO disabled in PolicyStore")
                    await asyncio.sleep(self.assessment_interval)
                    continue
                
                # Gather risk data
                risk_data = await self._gather_risk_data()
                
                # Generate trace ID
                trace_id = f"ro_assessment_{uuid.uuid4().hex[:12]}"
                
                # Perform risk assessment
                assessment = self.brain.assess_risk(risk_data)
                
                # Log assessment
                logger.info(
                    f"[{trace_id}] Risk Assessment: score={assessment.risk_score:.1f}, "
                    f"level={assessment.risk_level}, "
                    f"alerts={len(assessment.risk_alerts)}"
                )
                
                # Publish assessment
                await self._publish_assessment(assessment, trace_id)
                
                # Update PolicyStore if needed
                await self._update_risk_limits(assessment, trace_id)
                
            except Exception as e:
                logger.error(f"Error in risk assessment loop: {e}", exc_info=True)
            
            # Wait for next cycle
            await asyncio.sleep(self.assessment_interval)
    
    async def _gather_risk_data(self) -> PortfolioRiskData:
        """Gather current risk data from portfolio state and PolicyStore."""
        # Get portfolio state (cached or query)
        portfolio_state = self._latest_portfolio_state or {}
        
        # Get current policy
        policy = await self.policy_store.get_policy()
        active_config = policy.get_active_config()
        
        # Extract data
        total_exposure = portfolio_state.get("total_exposure", 0.0)
        open_positions = portfolio_state.get("open_positions", 0)
        account_balance = portfolio_state.get("account_balance", 10000.0)
        current_drawdown = portfolio_state.get("current_drawdown", 0.0)
        max_drawdown_today = portfolio_state.get("max_drawdown_today", 0.0)
        total_pnl_today = portfolio_state.get("total_pnl_today", 0.0)
        current_volatility = portfolio_state.get("market_volatility", 0.02)
        regime = portfolio_state.get("market_regime", "UNKNOWN")
        
        # Get risk limits from PolicyStore
        max_leverage = active_config.max_leverage
        max_risk_per_trade = active_config.max_risk_pct_per_trade
        max_daily_drawdown = active_config.max_daily_drawdown
        
        return PortfolioRiskData(
            returns_history=self._returns_history,
            total_exposure=total_exposure,
            open_positions=open_positions,
            account_balance=account_balance,
            current_drawdown=current_drawdown,
            max_drawdown_today=max_drawdown_today,
            total_pnl_today=total_pnl_today,
            current_volatility=current_volatility,
            regime=regime,
            max_leverage=max_leverage,
            max_risk_per_trade=max_risk_per_trade,
            max_daily_drawdown=max_daily_drawdown,
            timestamp=datetime.utcnow(),
        )
    
    async def _publish_assessment(self, assessment: RiskAssessment, trace_id: str) -> None:
        """Publish risk assessment to EventBus."""
        # Publish risk state update
        await self.event_bus.publish(
            "risk_state_update",
            assessment.to_dict(),
            trace_id=trace_id,
        )
        
        # Publish risk alert if needed
        if assessment.risk_alerts:
            await self.event_bus.publish(
                "risk_alert",
                {
                    "level": assessment.risk_level,
                    "alerts": assessment.risk_alerts,
                    "risk_score": assessment.risk_score,
                    "should_reduce_exposure": assessment.should_reduce_exposure,
                    "should_pause_trading": assessment.should_pause_trading,
                    "timestamp": assessment.assessment_timestamp.isoformat(),
                },
                trace_id=trace_id,
            )
    
    async def _update_risk_limits(self, assessment: RiskAssessment, trace_id: str) -> None:
        """
        Update risk limits in PolicyStore if assessment recommends changes.
        
        Only updates if recommendations differ significantly from current settings.
        """
        policy = await self.policy_store.get_policy()
        active_config = policy.get_active_config()
        
        # Check if recommendations differ significantly
        current_leverage = active_config.max_leverage
        rec_leverage = assessment.recommended_max_leverage
        
        leverage_diff = abs(current_leverage - rec_leverage) / current_leverage
        
        # Only update if difference is > 10%
        if leverage_diff > 0.10:
            logger.info(
                f"[{trace_id}] AI_RO recommending leverage adjustment: "
                f"{current_leverage:.1f}x → {rec_leverage:.1f}x"
            )
            
            # Update config
            active_config.max_leverage = rec_leverage
            active_config.max_risk_pct_per_trade = assessment.recommended_max_risk_per_trade
            active_config.max_positions = assessment.recommended_max_positions
            
            # Save to PolicyStore
            await self.policy_store.set_policy(policy, updated_by="ai_ro")
            
            # Publish risk ceiling update event
            await self.event_bus.publish(
                "risk_ceiling_update",
                {
                    "max_leverage": rec_leverage,
                    "max_risk_per_trade": assessment.recommended_max_risk_per_trade,
                    "max_positions": assessment.recommended_max_positions,
                    "reason": f"Risk score: {assessment.risk_score:.1f}",
                    "timestamp": datetime.utcnow().isoformat(),
                },
                trace_id=trace_id,
            )
    
    # Event handlers
    
    async def _handle_position_event(self, event_data: dict) -> None:
        """Handle position opened/closed events."""
        logger.debug(f"AI_RO received position event: {event_data.get('event_type')}")
        # Trigger state refresh
    
    async def _handle_portfolio_state_update(self, event_data: dict) -> None:
        """Handle portfolio state update."""
        self._latest_portfolio_state = event_data.get("payload", {})
        logger.debug("AI_RO updated portfolio state")
    
    async def _handle_trade_executed(self, event_data: dict) -> None:
        """Handle trade executed event and update returns history."""
        payload = event_data.get("payload", {})
        pnl_pct = payload.get("pnl_percentage")
        
        if pnl_pct is not None:
            # Add to returns history
            self._returns_history = np.append(self._returns_history, pnl_pct)
            
            # Keep only recent history (last 500 trades)
            if len(self._returns_history) > 500:
                self._returns_history = self._returns_history[-500:]
            
            # Cache to Redis
            try:
                await self.redis.set(
                    "quantum:risk:returns_history",
                    self._returns_history.tobytes(),
                    ex=86400,  # 24 hour expiry
                )
            except Exception as e:
                logger.error(f"Failed to cache returns history: {e}")
    
    async def _handle_market_data_update(self, event_data: dict) -> None:
        """Handle market data update."""
        logger.debug("AI_RO received market data update")
    
    # Public API
    
    async def get_status(self) -> dict:
        """Get AI Risk Officer status."""
        recent_assessments = self.brain.get_assessment_history(limit=10)
        
        return {
            "instance_id": self._instance_id,
            "running": self._running,
            "assessment_interval": self.assessment_interval,
            "returns_history_length": len(self._returns_history),
            "recent_assessments_count": len(recent_assessments),
            "last_assessment": recent_assessments[-1].to_dict() if recent_assessments else None,
            "portfolio_state_cached": self._latest_portfolio_state is not None,
        }
    
    async def force_assessment(self) -> RiskAssessment:
        """Force immediate risk assessment (for testing/debugging)."""
        logger.info("AI_RO forcing immediate assessment")
        risk_data = await self._gather_risk_data()
        assessment = self.brain.assess_risk(risk_data)
        
        trace_id = f"ro_forced_{uuid.uuid4().hex[:8]}"
        await self._publish_assessment(assessment, trace_id)
        
        return assessment
