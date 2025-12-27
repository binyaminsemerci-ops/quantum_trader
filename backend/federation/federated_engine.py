"""Federated Engine - Unified orchestration engine.

This module combines outputs from AI CEO, AI Risk Officer, and
AI Strategy Officer into a single global decision state.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.core.event_bus import EventBus
from backend.federation.integration_layer import IntegrationLayer

logger = logging.getLogger(__name__)


@dataclass
class GlobalState:
    """
    Global decision snapshot combining all AI agent outputs.
    
    This is the "single source of truth" for the current state
    of the entire trading system.
    """
    
    # CEO state
    global_mode: str                     # Operating mode (EXPANSION, GROWTH, etc.)
    mode_decision_confidence: float      # CEO's confidence in mode decision
    mode_reason: str                     # Why this mode was chosen
    
    # Risk state
    effective_risk_limits: dict          # Current risk limits
    risk_score: float                    # Overall risk score (0-100)
    risk_level: str                      # Risk level (low, moderate, high, critical)
    should_reduce_exposure: bool         # Risk officer recommendation
    should_pause_trading: bool           # Risk officer recommendation
    
    # Strategy state
    active_strategies: list[str]         # Primary + fallback strategies
    disabled_strategies: list[str]       # Strategies currently disabled
    meta_strategy_mode: str              # Meta-strategy (trend, mean_reversion, etc.)
    recommended_models: list[str]        # Active ML models
    strategy_confidence: float           # Strategy officer confidence
    
    # Feature flags (derived from mode + risk + strategy)
    disabled_features: list[str]         # Features to disable
    
    # Aggregated alerts
    alerts: list[dict]                   # All alerts from all agents
    
    # Metadata
    snapshot_timestamp: datetime
    data_completeness: float             # 0-1 scale (how much data we have)
    all_agents_healthy: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["snapshot_timestamp"] = self.snapshot_timestamp.isoformat()
        return data


class FederatedEngine:
    """
    Federated Engine - Unified AI Orchestration.
    
    Responsibilities:
    - Subscribe to events from AI CEO, AI-RO, AI-SO
    - Aggregate states via IntegrationLayer
    - Build GlobalState snapshots
    - Publish global_state_update events
    - Provide API for current global state
    - Handle missing/degraded agent data
    
    Integration:
    - Runs in analytics-os-service or orchestrator-service
    - Uses EventBus v2 for communication
    - Provides REST API endpoint for state queries
    """
    
    DEFAULT_UPDATE_INTERVAL = 15.0  # seconds
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        integration_layer: Optional[IntegrationLayer] = None,
        update_interval: float = DEFAULT_UPDATE_INTERVAL,
    ):
        """
        Initialize Federated Engine.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus instance
            integration_layer: Optional IntegrationLayer (for testing)
            update_interval: Seconds between global state updates
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.integration = integration_layer or IntegrationLayer()
        self.update_interval = update_interval
        
        # State tracking
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Latest global state
        self._latest_global_state: Optional[GlobalState] = None
        
        # Instance ID
        self._instance_id = f"fed_engine_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"FederatedEngine initialized: instance_id={self._instance_id}, "
            f"update_interval={update_interval}s"
        )
    
    async def initialize(self) -> None:
        """Initialize Federated Engine."""
        # Verify EventBus
        await self.event_bus.initialize()
        
        # Subscribe to agent events
        self._subscribe_to_events()
        
        logger.info("FederatedEngine initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to events from all AI agents."""
        # CEO events
        self.event_bus.subscribe("ceo_decision", self._handle_ceo_decision)
        self.event_bus.subscribe("ceo_mode_switch", self._handle_ceo_mode_switch)
        self.event_bus.subscribe("ceo_alert", self._handle_ceo_alert)
        
        # Risk Officer events
        self.event_bus.subscribe("risk_state_update", self._handle_risk_state_update)
        self.event_bus.subscribe("risk_alert", self._handle_risk_alert)
        self.event_bus.subscribe("risk_ceiling_update", self._handle_risk_ceiling_update)
        
        # Strategy Officer events
        self.event_bus.subscribe("strategy_state_update", self._handle_strategy_state_update)
        self.event_bus.subscribe("strategy_recommendation", self._handle_strategy_recommendation)
        self.event_bus.subscribe("strategy_alert", self._handle_strategy_alert)
        
        logger.info("FederatedEngine subscribed to agent events")
    
    async def start(self) -> None:
        """Start Federated Engine."""
        if self._running:
            logger.warning("FederatedEngine already running")
            return
        
        self._running = True
        
        # Start EventBus processing
        await self.event_bus.start()
        
        # Start global state update loop
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info("FederatedEngine started")
    
    async def stop(self) -> None:
        """Stop Federated Engine gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("FederatedEngine stopped")
    
    async def _update_loop(self) -> None:
        """Main global state update loop."""
        logger.info(f"FederatedEngine update loop started (interval={self.update_interval}s)")
        
        while self._running:
            try:
                # Build global state snapshot
                global_state = self._build_global_state()
                
                # Store latest
                self._latest_global_state = global_state
                
                # Generate trace ID
                trace_id = f"fed_update_{uuid.uuid4().hex[:12]}"
                
                # Publish global state update
                await self.event_bus.publish(
                    "global_state_update",
                    global_state.to_dict(),
                    trace_id=trace_id,
                )
                
                logger.debug(
                    f"[{trace_id}] Global state updated: mode={global_state.global_mode}, "
                    f"risk={global_state.risk_level}, "
                    f"completeness={global_state.data_completeness:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error in federated update loop: {e}", exc_info=True)
            
            # Wait for next cycle
            await asyncio.sleep(self.update_interval)
    
    def _build_global_state(self) -> GlobalState:
        """
        Build global state snapshot from integrated agent states.
        
        Handles missing/stale data gracefully with defaults.
        """
        # Get states from integration layer
        ceo_state = self.integration.get_ceo_state()
        risk_state = self.integration.get_risk_state()
        strategy_state = self.integration.get_strategy_state()
        
        # Calculate data completeness
        available_count = sum([
            ceo_state is not None,
            risk_state is not None,
            strategy_state is not None,
        ])
        data_completeness = available_count / 3.0
        
        all_agents_healthy = available_count == 3
        
        # Extract CEO state (or defaults)
        global_mode = ceo_state.operating_mode if ceo_state else "EXPANSION"
        mode_confidence = ceo_state.decision_confidence if ceo_state else 0.5
        mode_reason = ceo_state.primary_reason if ceo_state else "Waiting for CEO decision"
        
        # Extract Risk state (or defaults)
        risk_score = risk_state.risk_score if risk_state else 50.0
        risk_level = risk_state.risk_level if risk_state else "moderate"
        should_reduce_exposure = risk_state.should_reduce_exposure if risk_state else False
        should_pause_trading = risk_state.should_pause_trading if risk_state else False
        
        effective_risk_limits = {}
        if risk_state:
            effective_risk_limits = {
                "max_leverage": risk_state.recommended_max_leverage,
                "max_risk_per_trade": risk_state.recommended_max_risk_per_trade,
                "var_estimate": risk_state.var_estimate,
                "expected_shortfall": risk_state.expected_shortfall,
            }
        
        # Extract Strategy state (or defaults)
        primary_strategy = strategy_state.primary_strategy if strategy_state else "ensemble_conservative"
        fallback_strategies = strategy_state.fallback_strategies if strategy_state else []
        active_strategies = [primary_strategy] + fallback_strategies
        
        disabled_strategies = strategy_state.disabled_strategies if strategy_state else []
        meta_strategy_mode = strategy_state.meta_strategy_mode if strategy_state else "ensemble"
        recommended_models = strategy_state.recommended_models if strategy_state else ["xgboost", "lightgbm"]
        strategy_confidence = strategy_state.confidence if strategy_state else 0.5
        
        # Determine disabled features based on mode and risk
        disabled_features = self._determine_disabled_features(global_mode, risk_level)
        
        # Aggregate alerts
        alerts = []
        if ceo_state and ceo_state.alerts:
            alerts.extend([{"source": "ceo", "message": a} for a in ceo_state.alerts])
        if risk_state and risk_state.alerts:
            alerts.extend([{"source": "risk", "message": a} for a in risk_state.alerts])
        if strategy_state and strategy_state.alerts:
            alerts.extend([{"source": "strategy", "message": a} for a in strategy_state.alerts])
        
        return GlobalState(
            global_mode=global_mode,
            mode_decision_confidence=mode_confidence,
            mode_reason=mode_reason,
            effective_risk_limits=effective_risk_limits,
            risk_score=risk_score,
            risk_level=risk_level,
            should_reduce_exposure=should_reduce_exposure,
            should_pause_trading=should_pause_trading,
            active_strategies=active_strategies,
            disabled_strategies=disabled_strategies,
            meta_strategy_mode=meta_strategy_mode,
            recommended_models=recommended_models,
            strategy_confidence=strategy_confidence,
            disabled_features=disabled_features,
            alerts=alerts,
            snapshot_timestamp=datetime.utcnow(),
            data_completeness=data_completeness,
            all_agents_healthy=all_agents_healthy,
        )
    
    def _determine_disabled_features(self, mode: str, risk_level: str) -> list[str]:
        """Determine which features should be disabled based on mode and risk."""
        disabled = []
        
        # Disable features in BLACK_SWAN mode
        if mode == "BLACK_SWAN":
            disabled.extend([
                "new_positions",
                "rl_position_sizing",
                "meta_strategy",
                "retraining",
            ])
        
        # Disable features in CAPITAL_PRESERVATION mode
        elif mode == "CAPITAL_PRESERVATION":
            disabled.extend([
                "rl_position_sizing",
                "meta_strategy",
            ])
        
        # Disable features based on risk level
        if risk_level == "critical":
            disabled.extend([
                "new_positions",
                "leverage_increase",
            ])
        elif risk_level == "high":
            disabled.append("leverage_increase")
        
        return list(set(disabled))  # Remove duplicates
    
    # Event handlers
    
    async def _handle_ceo_decision(self, event_data: dict) -> None:
        """Handle CEO decision event."""
        payload = event_data.get("payload", {})
        
        self.integration.update_ceo_state(
            operating_mode=payload.get("operating_mode", "EXPANSION"),
            mode_changed=payload.get("mode_changed", False),
            decision_confidence=payload.get("decision_confidence", 0.5),
            primary_reason=payload.get("primary_reason", ""),
            alerts=[payload.get("alert_message")] if payload.get("alert_message") else [],
        )
        
        logger.debug("FederatedEngine processed CEO decision")
    
    async def _handle_ceo_mode_switch(self, event_data: dict) -> None:
        """Handle CEO mode switch event."""
        logger.info(f"FederatedEngine received mode switch: {event_data}")
    
    async def _handle_ceo_alert(self, event_data: dict) -> None:
        """Handle CEO alert event."""
        logger.warning(f"FederatedEngine received CEO alert: {event_data}")
    
    async def _handle_risk_state_update(self, event_data: dict) -> None:
        """Handle Risk Officer state update."""
        payload = event_data.get("payload", {})
        
        self.integration.update_risk_state(
            risk_score=payload.get("risk_score", 50.0),
            risk_level=payload.get("risk_level", "moderate"),
            var_estimate=payload.get("var_estimate", 0.0),
            expected_shortfall=payload.get("expected_shortfall", 0.0),
            tail_risk_score=payload.get("tail_risk_score", 0.0),
            recommended_max_leverage=payload.get("recommended_max_leverage", 10.0),
            recommended_max_risk_per_trade=payload.get("recommended_max_risk_per_trade", 0.02),
            should_reduce_exposure=payload.get("should_reduce_exposure", False),
            should_pause_trading=payload.get("should_pause_trading", False),
            alerts=payload.get("risk_alerts", []),
        )
        
        logger.debug("FederatedEngine processed risk state update")
    
    async def _handle_risk_alert(self, event_data: dict) -> None:
        """Handle Risk Officer alert."""
        logger.warning(f"FederatedEngine received risk alert: {event_data}")
    
    async def _handle_risk_ceiling_update(self, event_data: dict) -> None:
        """Handle risk ceiling update."""
        logger.info(f"FederatedEngine received risk ceiling update: {event_data}")
    
    async def _handle_strategy_state_update(self, event_data: dict) -> None:
        """Handle Strategy Officer state update."""
        payload = event_data.get("payload", {})
        
        self.integration.update_strategy_state(
            primary_strategy=payload.get("primary_strategy", "ensemble_conservative"),
            fallback_strategies=payload.get("fallback_strategies", []),
            disabled_strategies=payload.get("disabled_strategies", []),
            meta_strategy_mode=payload.get("meta_strategy_mode", "ensemble"),
            overall_win_rate=payload.get("win_rate", 0.5),
            overall_sharpe=payload.get("sharpe_ratio", 1.0),
            recommended_models=payload.get("recommended_models", []),
            confidence=payload.get("confidence", 0.5),
            alerts=[],
        )
        
        logger.debug("FederatedEngine processed strategy state update")
    
    async def _handle_strategy_recommendation(self, event_data: dict) -> None:
        """Handle Strategy Officer recommendation."""
        logger.info("FederatedEngine received strategy recommendation")
    
    async def _handle_strategy_alert(self, event_data: dict) -> None:
        """Handle Strategy Officer alert."""
        logger.warning(f"FederatedEngine received strategy alert: {event_data}")
    
    # Public API
    
    def get_current_global_state(self) -> Optional[GlobalState]:
        """
        Get current global state snapshot.
        
        This is the primary API for other systems to query the
        federated decision state.
        """
        return self._latest_global_state
    
    async def get_status(self) -> dict:
        """Get Federated Engine status."""
        health = self.integration.get_health_status()
        
        return {
            "instance_id": self._instance_id,
            "running": self._running,
            "update_interval": self.update_interval,
            "has_global_state": self._latest_global_state is not None,
            "integration_health": health,
            "latest_global_state": self._latest_global_state.to_dict() if self._latest_global_state else None,
        }
    
    async def force_update(self) -> GlobalState:
        """Force immediate global state update (for testing/debugging)."""
        logger.info("FederatedEngine forcing immediate update")
        
        global_state = self._build_global_state()
        self._latest_global_state = global_state
        
        trace_id = f"fed_forced_{uuid.uuid4().hex[:8]}"
        await self.event_bus.publish(
            "global_state_update",
            global_state.to_dict(),
            trace_id=trace_id,
        )
        
        return global_state
