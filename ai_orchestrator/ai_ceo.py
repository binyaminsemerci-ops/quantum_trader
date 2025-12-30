"""AI CEO - Meta-Orchestrator for Quantum Trader.

The AI CEO is the highest-level decision-making agent that:
- Aggregates inputs from all system components
- Evaluates global trading state
- Selects operating mode (EXPANSION, GROWTH, DEFENSIVE, etc.)
- Updates PolicyStore with mode-specific configuration
- Publishes decisions via EventBus
- Monitors and responds to system alerts

This agent runs continuously and makes decisions at configurable intervals.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.ai_orchestrator.ceo_brain import CEOBrain, CEODecision, SystemState
from backend.ai_orchestrator.ceo_policy import (
    CEOPolicy,
    MarketRegime,
    OperatingMode,
    SystemHealth,
)
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class AI_CEO:
    """
    AI CEO - Meta-Orchestrator Agent.
    
    Responsibilities:
    - Run decision loop at configurable intervals
    - Subscribe to critical events (risk_alert, system_degraded, etc.)
    - Aggregate state from AI-RO, AI-SO, PBA, PAL, PIL
    - Make global operating mode decisions
    - Update PolicyStore with mode configuration
    - Publish ceo_decision, ceo_mode_switch, ceo_alert events
    - Provide health/status endpoint
    
    Integration:
    - Runs in analytics-os-service or dedicated orchestrator-service
    - Uses EventBus v2 for event-driven communication
    - Uses PolicyStore v2 for configuration management
    """
    
    DEFAULT_DECISION_INTERVAL = 30.0  # seconds
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        policy_store: PolicyStore,
        brain: Optional[CEOBrain] = None,
        decision_interval: float = DEFAULT_DECISION_INTERVAL,
    ):
        """
        Initialize AI CEO.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus instance
            policy_store: PolicyStore instance
            brain: Optional CEOBrain instance (for testing)
            decision_interval: Seconds between decision cycles
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.brain = brain or CEOBrain()
        self.decision_interval = decision_interval
        
        # State tracking
        self._running = False
        self._decision_task: Optional[asyncio.Task] = None
        self._event_listeners: list[asyncio.Task] = []
        
        # Cached state from other agents
        self._latest_risk_state: Optional[dict] = None
        self._latest_strategy_state: Optional[dict] = None
        self._latest_portfolio_state: Optional[dict] = None
        self._latest_system_health: Optional[dict] = None
        
        # Instance ID
        self._instance_id = f"ai_ceo_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"AI_CEO initialized: instance_id={self._instance_id}, "
            f"decision_interval={decision_interval}s"
        )
    
    async def initialize(self) -> None:
        """Initialize AI CEO and verify dependencies."""
        # Verify EventBus
        await self.event_bus.initialize()
        
        # Verify PolicyStore
        policy = await self.policy_store.get_policy()
        logger.info(f"AI_CEO verified PolicyStore: mode={policy.active_mode.value}")
        
        # Subscribe to critical events
        self._subscribe_to_events()
        
        logger.info("AI_CEO initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to critical system events."""
        # Risk alerts from AI-RO
        self.event_bus.subscribe("risk_alert", self._handle_risk_alert)
        self.event_bus.subscribe("risk_state_update", self._handle_risk_state_update)
        
        # Strategy updates from AI-SO
        self.event_bus.subscribe("strategy_state_update", self._handle_strategy_state_update)
        self.event_bus.subscribe("strategy_alert", self._handle_strategy_alert)
        
        # Portfolio updates from PBA/PAL
        self.event_bus.subscribe("position_opened", self._handle_position_event)
        self.event_bus.subscribe("position_closed", self._handle_position_event)
        self.event_bus.subscribe("portfolio_state_update", self._handle_portfolio_state_update)
        
        # System health
        self.event_bus.subscribe("system_health_update", self._handle_system_health_update)
        self.event_bus.subscribe("system_degraded", self._handle_system_degraded)
        
        # Model updates
        self.event_bus.subscribe("model_updated", self._handle_model_updated)
        
        logger.info("AI_CEO subscribed to critical events")
    
    async def start(self) -> None:
        """Start AI CEO decision loop and event processing."""
        if self._running:
            logger.warning("AI_CEO already running")
            return
        
        self._running = True
        
        # Start EventBus processing
        await self.event_bus.start()
        
        # Start decision loop
        self._decision_task = asyncio.create_task(self._decision_loop())
        
        logger.info("AI_CEO started")
    
    async def stop(self) -> None:
        """Stop AI CEO gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel decision task
        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AI_CEO stopped")
    
    async def _decision_loop(self) -> None:
        """
        Main decision loop.
        
        Runs continuously, making decisions at configured intervals.
        """
        logger.info(f"AI_CEO decision loop started (interval={self.decision_interval}s)")
        
        while self._running:
            try:
                # Check if CEO is enabled in PolicyStore
                policy = await self.policy_store.get_policy()
                active_config = policy.get_active_config()
                
                # Check for enable_ai_ceo flag (if exists)
                if hasattr(active_config, 'enable_ai_ceo') and not active_config.enable_ai_ceo:
                    logger.debug("AI_CEO disabled in PolicyStore, skipping decision cycle")
                    await asyncio.sleep(self.decision_interval)
                    continue
                
                # Gather current state
                state = await self._gather_system_state()
                
                # Generate trace ID for this decision
                trace_id = f"ceo_decision_{uuid.uuid4().hex[:12]}"
                
                # Make decision
                decision = self.brain.evaluate(
                    state=state,
                    current_mode=OperatingMode(policy.active_mode.value),
                )
                
                # Log decision
                logger.info(
                    f"[{trace_id}] CEO Decision: mode={decision.operating_mode.value}, "
                    f"changed={decision.mode_changed}, "
                    f"reason={decision.primary_reason}, "
                    f"confidence={decision.decision_confidence:.2f}"
                )
                
                # Execute decision
                await self._execute_decision(decision, trace_id)
                
            except Exception as e:
                logger.error(f"Error in CEO decision loop: {e}", exc_info=True)
            
            # Wait for next cycle
            await asyncio.sleep(self.decision_interval)
    
    async def _gather_system_state(self) -> SystemState:
        """
        Gather current system state from all components.
        
        Combines cached state from events with direct queries when needed.
        """
        # Get risk state from AI-RO (cached or default)
        risk_state = self._latest_risk_state or {}
        risk_score = risk_state.get("risk_score", 50.0)
        var_estimate = risk_state.get("var_estimate", 0.0)
        expected_shortfall = risk_state.get("expected_shortfall", 0.0)
        tail_risk_score = risk_state.get("tail_risk_score", 0.0)
        
        # Get strategy state from AI-SO (cached or default)
        strategy_state = self._latest_strategy_state or {}
        win_rate = strategy_state.get("win_rate", 0.50)
        sharpe_ratio = strategy_state.get("sharpe_ratio", 1.0)
        profit_factor = strategy_state.get("profit_factor", 1.0)
        avg_win_loss_ratio = strategy_state.get("avg_win_loss_ratio", 1.0)
        
        # Get portfolio state from PBA/PAL (cached or query Redis)
        portfolio_state = self._latest_portfolio_state or {}
        current_drawdown = portfolio_state.get("current_drawdown", 0.0)
        max_drawdown_today = portfolio_state.get("max_drawdown_today", 0.0)
        total_pnl_today = portfolio_state.get("total_pnl_today", 0.0)
        open_positions = portfolio_state.get("open_positions", 0)
        total_exposure = portfolio_state.get("total_exposure", 0.0)
        
        # Get market state (from regime detector - cached or default)
        market_regime = portfolio_state.get("market_regime", MarketRegime.UNKNOWN.value)
        regime_confidence = portfolio_state.get("regime_confidence", 0.5)
        market_volatility = portfolio_state.get("market_volatility", 0.02)
        
        # Get system health (cached or query)
        health_state = self._latest_system_health or {}
        system_health = SystemHealth(health_state.get("overall_health", SystemHealth.HEALTHY.value))
        ai_engine_healthy = health_state.get("ai_engine_healthy", True)
        rl_engine_healthy = health_state.get("rl_engine_healthy", True)
        execution_healthy = health_state.get("execution_healthy", True)
        risk_os_healthy = health_state.get("risk_os_healthy", True)
        
        return SystemState(
            risk_score=risk_score,
            var_estimate=var_estimate,
            expected_shortfall=expected_shortfall,
            tail_risk_score=tail_risk_score,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win_loss_ratio=avg_win_loss_ratio,
            current_drawdown=current_drawdown,
            max_drawdown_today=max_drawdown_today,
            total_pnl_today=total_pnl_today,
            open_positions=open_positions,
            total_exposure=total_exposure,
            market_regime=MarketRegime(market_regime),
            regime_confidence=regime_confidence,
            market_volatility=market_volatility,
            system_health=system_health,
            ai_engine_healthy=ai_engine_healthy,
            rl_engine_healthy=rl_engine_healthy,
            execution_healthy=execution_healthy,
            risk_os_healthy=risk_os_healthy,
            timestamp=datetime.utcnow(),
        )
    
    async def _execute_decision(self, decision: CEODecision, trace_id: str) -> None:
        """
        Execute CEO decision.
        
        - Update PolicyStore if mode changed
        - Publish events
        - Send alerts if needed
        """
        # Publish ceo_decision event (always)
        await self.event_bus.publish(
            "ceo_decision",
            decision.to_dict(),
            trace_id=trace_id,
        )
        
        # Update PolicyStore if needed
        if decision.update_policy_store and decision.policy_updates:
            try:
                policy = await self.policy_store.get_policy()
                active_config = policy.get_active_config()
                
                # Update config fields
                for key, value in decision.policy_updates.items():
                    if hasattr(active_config, key):
                        setattr(active_config, key, value)
                
                # Switch mode if changed
                if decision.mode_changed:
                    # Note: This assumes PolicyStore has mode switching capability
                    # or that modes are stored in a way we can update
                    logger.info(
                        f"[{trace_id}] Updating PolicyStore: mode={decision.operating_mode.value}"
                    )
                    # Ideally: await self.policy_store.switch_mode(decision.operating_mode, "ai_ceo")
                    # For now, update the policy directly
                    await self.policy_store.set_policy(policy, updated_by="ai_ceo")
                
            except Exception as e:
                logger.error(f"[{trace_id}] Failed to update PolicyStore: {e}", exc_info=True)
        
        # Publish mode switch event if changed
        if decision.mode_changed:
            await self.event_bus.publish(
                "ceo_mode_switch",
                {
                    "new_mode": decision.operating_mode.value,
                    "reason": decision.primary_reason,
                    "timestamp": decision.decision_timestamp.isoformat(),
                },
                trace_id=trace_id,
            )
        
        # Publish alert if needed
        if decision.alert_level in ["warning", "critical"] and decision.alert_message:
            await self.event_bus.publish(
                "ceo_alert",
                {
                    "level": decision.alert_level,
                    "message": decision.alert_message,
                    "mode": decision.operating_mode.value,
                    "timestamp": decision.decision_timestamp.isoformat(),
                },
                trace_id=trace_id,
            )
        
        # Publish goal report (periodic summary)
        await self.event_bus.publish(
            "ceo_goal_report",
            {
                "operating_mode": decision.operating_mode.value,
                "decision_confidence": decision.decision_confidence,
                "contributing_factors": decision.contributing_factors,
                "timestamp": decision.decision_timestamp.isoformat(),
            },
            trace_id=trace_id,
        )
    
    # Event handlers (update cached state)
    
    async def _handle_risk_alert(self, event_data: dict) -> None:
        """Handle risk alert from AI-RO."""
        logger.warning(f"AI_CEO received risk alert: {event_data}")
        # Could trigger immediate re-evaluation if critical
        alert_level = event_data.get("level")
        if alert_level == "critical":
            logger.info("Critical risk alert - triggering immediate decision cycle")
            # Note: In production, implement immediate decision trigger
    
    async def _handle_risk_state_update(self, event_data: dict) -> None:
        """Handle risk state update from AI-RO."""
        self._latest_risk_state = event_data.get("payload", {})
        logger.debug(f"AI_CEO updated risk state: risk_score={self._latest_risk_state.get('risk_score')}")
    
    async def _handle_strategy_state_update(self, event_data: dict) -> None:
        """Handle strategy state update from AI-SO."""
        self._latest_strategy_state = event_data.get("payload", {})
        logger.debug(f"AI_CEO updated strategy state: win_rate={self._latest_strategy_state.get('win_rate')}")
    
    async def _handle_strategy_alert(self, event_data: dict) -> None:
        """Handle strategy alert from AI-SO."""
        logger.info(f"AI_CEO received strategy alert: {event_data}")
    
    async def _handle_position_event(self, event_data: dict) -> None:
        """Handle position opened/closed events."""
        logger.debug(f"AI_CEO received position event: {event_data.get('event_type')}")
        # Trigger portfolio state refresh
    
    async def _handle_portfolio_state_update(self, event_data: dict) -> None:
        """Handle portfolio state update from PBA/PAL."""
        self._latest_portfolio_state = event_data.get("payload", {})
        logger.debug(
            f"AI_CEO updated portfolio state: "
            f"pnl={self._latest_portfolio_state.get('total_pnl_today')}"
        )
    
    async def _handle_system_health_update(self, event_data: dict) -> None:
        """Handle system health update."""
        self._latest_system_health = event_data.get("payload", {})
        logger.debug(
            f"AI_CEO updated system health: "
            f"status={self._latest_system_health.get('overall_health')}"
        )
    
    async def _handle_system_degraded(self, event_data: dict) -> None:
        """Handle system degraded event."""
        logger.warning(f"AI_CEO received system degraded: {event_data}")
        # Could trigger immediate defensive mode
    
    async def _handle_model_updated(self, event_data: dict) -> None:
        """Handle model updated event."""
        logger.info(f"AI_CEO received model update: {event_data}")
    
    # Public API
    
    async def get_status(self) -> dict:
        """Get AI CEO status and current state."""
        current_mode = self.brain.get_current_mode()
        recent_decisions = self.brain.get_decision_history(limit=10)
        
        return {
            "instance_id": self._instance_id,
            "running": self._running,
            "current_mode": current_mode.value,
            "decision_interval": self.decision_interval,
            "recent_decisions_count": len(recent_decisions),
            "last_decision": recent_decisions[-1].to_dict() if recent_decisions else None,
            "cached_states": {
                "risk_state_available": self._latest_risk_state is not None,
                "strategy_state_available": self._latest_strategy_state is not None,
                "portfolio_state_available": self._latest_portfolio_state is not None,
                "system_health_available": self._latest_system_health is not None,
            },
        }
    
    async def force_decision(self) -> CEODecision:
        """Force an immediate decision cycle (for testing/debugging)."""
        logger.info("AI_CEO forcing immediate decision cycle")
        state = await self._gather_system_state()
        policy = await self.policy_store.get_policy()
        
        decision = self.brain.evaluate(
            state=state,
            current_mode=OperatingMode(policy.active_mode.value),
        )
        
        trace_id = f"ceo_forced_{uuid.uuid4().hex[:8]}"
        await self._execute_decision(decision, trace_id)
        
        return decision
