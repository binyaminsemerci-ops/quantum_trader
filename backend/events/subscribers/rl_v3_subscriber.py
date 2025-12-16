"""
RL v3 EventBus Subscriber - PPO-based decision engine.

Integrates RL v3 (PPO) with EventBus v2 for event-driven trading.
Listens to market/trading events, generates decisions via RLv3Manager,
and publishes decision events back to EventBus.

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0 (PPO Architecture)
"""

import asyncio
import structlog
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.domains.learning.rl_v3.features_v3 import build_feature_vector
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore

logger = structlog.get_logger(__name__)


class RLv3Subscriber:
    """
    RL v3 EventBus subscriber for event-driven PPO decisions.
    
    Subscribes to market.tick or signal.generated events,
    uses RLv3Manager.predict() to generate decisions,
    publishes rl_v3.decision events to EventBus.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        rl_manager: RLv3Manager,
        logger: Any = None,
        policy_store: Optional[PolicyStore] = None,
        shadow_mode: bool = True
    ):
        """
        Initialize RL v3 subscriber.
        
        Args:
            event_bus: EventBus v2 instance for pub/sub
            rl_manager: RLv3Manager for PPO predictions
            logger: Structured logger (uses default if None)
            policy_store: PolicyStore v2 for risk configs (optional)
            shadow_mode: If True, only observe without executing trades
        """
        self.event_bus = event_bus
        self.rl_manager = rl_manager
        self.logger = logger or structlog.get_logger(__name__)
        self.policy_store = policy_store
        self.shadow_mode = shadow_mode
        
        # Metrics store (singleton)
        self.metrics = RLv3MetricsStore.instance()
        
        # Subscription task references
        self._tasks = []
        self._running = False
        
        self.logger.info(
            "[RL v3 Subscriber] Initialized",
            shadow_mode=shadow_mode,
            model_path=rl_manager.config.model_path
        )
    
    async def start(self):
        """
        Start subscribing to events.
        
        Subscribes to:
        - signal.generated (primary input stream)
        - market.tick (alternative input for real-time)
        """
        if self._running:
            self.logger.warning("[RL v3 Subscriber] Already running")
            return
        
        self._running = True
        
        # Subscribe to signal.generated events (primary)
        self.event_bus.subscribe("signal.generated", self._handle_signal)
        
        # Subscribe to market.tick events (secondary, for real-time)
        self.event_bus.subscribe("market.tick", self._handle_tick)
        
        self.logger.info(
            "[RL v3 Subscriber] Started",
            subscriptions=["signal.generated", "market.tick"]
        )
    
    async def stop(self):
        """Stop subscriber and cancel all tasks."""
        self._running = False
        
        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._tasks.clear()
        
        self.logger.info("[RL v3 Subscriber] Stopped")
    
    async def _handle_signal(self, payload: Dict[str, Any]):
        """
        Handle signal.generated event.
        
        Args:
            payload: Event payload with market data
        """
        try:
            trace_id = payload.get("trace_id", "")
            symbol = payload.get("symbol", "UNKNOWN")
            
            self.logger.debug(
                "[RL v3 Subscriber] Processing signal",
                symbol=symbol,
                trace_id=trace_id
            )
            
            # Build observation dict for RL agent
            obs_dict = self._build_obs_dict(payload)
            
            # Get RL v3 decision
            result = self.rl_manager.predict(obs_dict)
            
            # Publish decision event
            await self._publish_decision(
                symbol=symbol,
                action=result["action"],
                confidence=result["confidence"],
                value=result.get("value", 0.0),
                raw_obs=obs_dict,
                trace_id=trace_id
            )
            
            self.logger.info(
                "[RL v3 Subscriber] Decision published",
                symbol=symbol,
                action=result["action"],
                confidence=result["confidence"],
                trace_id=trace_id
            )
            
        except Exception as e:
            self.logger.error(
                "[RL v3 Subscriber] Error handling signal",
                error=str(e),
                payload=payload,
                exc_info=True
            )
    
    async def _handle_tick(self, payload: Dict[str, Any]):
        """
        Handle market.tick event.
        
        Args:
            payload: Event payload with tick data
        """
        try:
            trace_id = payload.get("trace_id", "")
            symbol = payload.get("symbol", "UNKNOWN")
            
            self.logger.debug(
                "[RL v3 Subscriber] Processing tick",
                symbol=symbol,
                trace_id=trace_id
            )
            
            # Build observation dict from tick data
            obs_dict = self._build_obs_dict_from_tick(payload)
            
            # Get RL v3 decision
            result = self.rl_manager.predict(obs_dict)
            
            # Publish decision event
            await self._publish_decision(
                symbol=symbol,
                action=result["action"],
                confidence=result["confidence"],
                value=result.get("value", 0.0),
                raw_obs=obs_dict,
                trace_id=trace_id
            )
            
            self.logger.debug(
                "[RL v3 Subscriber] Tick decision published",
                symbol=symbol,
                action=result["action"],
                confidence=result["confidence"],
                trace_id=trace_id
            )
            
        except Exception as e:
            self.logger.error(
                "[RL v3 Subscriber] Error handling tick",
                error=str(e),
                payload=payload,
                exc_info=True
            )
    
    def _build_obs_dict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build observation dict from signal.generated payload.
        
        Args:
            payload: Signal event payload
            
        Returns:
            Observation dict compatible with build_feature_vector()
        """
        return {
            "symbol": payload.get("symbol", "UNKNOWN"),
            "price": payload.get("price", 0.0),
            "volume": payload.get("volume", 0.0),
            "volatility": payload.get("volatility", 0.0),
            "prediction": payload.get("prediction", 0),
            "confidence": payload.get("confidence", 0.5),
            "position_size": payload.get("position_size", 0.0),
            "account_balance": payload.get("account_balance", 10000.0),
            "open_positions": payload.get("open_positions", 0),
            "market_features": payload.get("market_features", {}),
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).timestamp())
        }
    
    def _build_obs_dict_from_tick(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build observation dict from market.tick payload.
        
        Args:
            payload: Tick event payload
            
        Returns:
            Observation dict compatible with build_feature_vector()
        """
        return {
            "symbol": payload.get("symbol", "UNKNOWN"),
            "price": payload.get("close", payload.get("price", 0.0)),
            "volume": payload.get("volume", 0.0),
            "volatility": payload.get("volatility", 0.0),
            "prediction": 0,  # No prediction from tick
            "confidence": 0.5,  # Neutral confidence
            "position_size": 0.0,  # Will be filled by executor
            "account_balance": 10000.0,  # Will be filled by executor
            "open_positions": 0,  # Will be filled by executor
            "market_features": {},
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).timestamp())
        }
    
    async def _publish_decision(
        self,
        symbol: str,
        action: int,
        confidence: float,
        value: float,
        raw_obs: Dict[str, Any],
        trace_id: str
    ):
        """
        Publish RL v3 decision event to EventBus.
        
        Args:
            symbol: Trading symbol
            action: Action index (0-5)
            confidence: Confidence score (0.0-1.0)
            value: Value estimate from critic
            raw_obs: Raw observation dict (subset for logging)
            trace_id: Trace ID for distributed tracing
        """
        # Build decision payload
        decision_payload = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "confidence": confidence,
            "value": value,
            "source": "RL_V3_PPO",
            "shadow_mode": self.shadow_mode,
            "trace_id": trace_id,
            "raw_obs": {
                "price": raw_obs.get("price", 0.0),
                "volatility": raw_obs.get("volatility", 0.0),
                "prediction": raw_obs.get("prediction", 0),
                "position_size": raw_obs.get("position_size", 0.0)
            }
        }
        
        # Publish to EventBus
        try:
            await self.event_bus.publish(
                "rl_v3.decision",
                decision_payload,
                trace_id=trace_id
            )
            
            # Record metrics
            self.metrics.record_decision(decision_payload)
            
        except Exception as e:
            self.logger.error(
                "[RL v3 Subscriber] Failed to publish decision",
                error=str(e),
                symbol=symbol,
                trace_id=trace_id,
                exc_info=True
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get subscriber statistics.
        
        Returns:
            Statistics dict with recent decisions and summary
        """
        return {
            "running": self._running,
            "shadow_mode": self.shadow_mode,
            "model_path": self.rl_manager.config.model_path,
            "recent_decisions": self.metrics.get_recent_decisions(limit=50),
            "summary": self.metrics.get_summary()
        }
