"""
RL Subscriber v3 - Event-Driven PPO Integration
================================================

Integrates RL v3 (PPO) with EventFlow system using domain-based architecture.

Handles:
- SIGNAL_GENERATED events → RL v3 decision
- TRADE_EXECUTED events → Experience collection
- POSITION_CLOSED events → Reward calculation

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0 (PPO Architecture)
"""

from typing import Dict, Any, Optional
import structlog
import numpy as np
from pathlib import Path

from backend.core.event_bus import EventBus
from backend.events.event_types import EventType
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config

logger = structlog.get_logger(__name__)


class RLSubscriberV3:
    """
    RL subscriber v3 for event-driven PPO integration.
    
    Subscribes to:
    - SIGNAL_GENERATED: Use RL v3 PPO to generate trading decision
    - MARKET_DATA_UPDATED: Collect market state for prediction
    - POSITION_CLOSED: Calculate rewards and store experience
    
    Publishes:
    - RL_V3_DECISION: PPO-based trading decision with confidence
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[RLv3Config] = None,
        shadow_mode: bool = True
    ):
        """
        Initialize RL Subscriber v3.
        
        Args:
            event_bus: Event bus v2
            config: RL v3 configuration (uses default if None)
            shadow_mode: If True, only logs decisions without executing
        """
        self.event_bus = event_bus
        self.shadow_mode = shadow_mode
        
        # Initialize RL v3 manager
        self.manager = RLv3Manager(config)
        
        # Try to load trained model
        model_path = Path(self.manager.config.model_path)
        if model_path.exists():
            self.manager.load(model_path)
            logger.info("[RL v3] Loaded trained model", path=str(model_path))
        else:
            logger.info("[RL v3] No trained model found, using untrained agent")
        
        # State tracking
        self.last_observation = None
        self.last_action = None
        self.last_value = None
        
        # Experience collection
        self.experiences = []
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info(
            "[RL Subscriber v3] Initialized",
            shadow_mode=shadow_mode,
            model_loaded=model_path.exists()
        )
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            self._handle_signal_generated
        )
        
        self.event_bus.subscribe(
            EventType.MARKET_DATA_UPDATED,
            self._handle_market_data_updated
        )
        
        self.event_bus.subscribe(
            EventType.POSITION_CLOSED,
            self._handle_position_closed
        )
        
        logger.info(
            "[RL Subscriber v3] Subscribed to events",
            events=[
                EventType.SIGNAL_GENERATED.value,
                EventType.MARKET_DATA_UPDATED.value,
                EventType.POSITION_CLOSED.value
            ]
        )
    
    async def _handle_signal_generated(self, event: Dict[str, Any]):
        """
        Handle SIGNAL_GENERATED event.
        
        Uses PPO agent to generate trading decision.
        
        Args:
            event: Signal generated event
        """
        try:
            data = event.get("data", {})
            
            # Build observation from signal data
            obs_dict = self._build_observation(data)
            
            # Get RL v3 prediction
            result = self.manager.predict(obs_dict)
            
            action = result['action']
            confidence = result['confidence']
            value = result['value']
            
            # Map action to trading decision
            action_name = self._map_action_to_name(action)
            
            # Store for experience collection
            self.last_observation = obs_dict
            self.last_action = action
            self.last_value = value
            
            # Publish RL v3 decision
            await self.event_bus.publish(
                EventType.RL_V3_DECISION,
                {
                    "symbol": data.get("symbol"),
                    "action": action_name,
                    "action_code": action,
                    "confidence": confidence,
                    "value": value,
                    "shadow_mode": self.shadow_mode,
                    "source": "rl_v3_ppo"
                }
            )
            
            logger.info(
                "[RL v3] Generated decision",
                symbol=data.get("symbol"),
                action=action_name,
                confidence=confidence,
                value=value,
                shadow_mode=self.shadow_mode
            )
            
        except Exception as e:
            logger.error(
                "[RL v3] Error handling signal",
                error=str(e),
                exc_info=True
            )
    
    async def _handle_market_data_updated(self, event: Dict[str, Any]):
        """
        Handle MARKET_DATA_UPDATED event.
        
        Updates internal market state for next prediction.
        
        Args:
            event: Market data updated event
        """
        try:
            data = event.get("data", {})
            
            # Store latest market data for observation building
            # This can be used to augment signal-based observations
            
            logger.debug(
                "[RL v3] Market data updated",
                symbol=data.get("symbol"),
                price=data.get("price")
            )
            
        except Exception as e:
            logger.error(
                "[RL v3] Error handling market data",
                error=str(e)
            )
    
    async def _handle_position_closed(self, event: Dict[str, Any]):
        """
        Handle POSITION_CLOSED event.
        
        Calculates reward and stores experience for future training.
        
        Args:
            event: Position closed event
        """
        try:
            data = event.get("data", {})
            
            # Calculate reward from position result
            pnl = data.get("pnl", 0.0)
            roi = data.get("roi", 0.0)
            
            # Simple reward based on PnL
            reward = pnl * 100  # Scale for better learning
            
            # Store experience
            if self.last_observation is not None and self.last_action is not None:
                experience = {
                    "observation": self.last_observation,
                    "action": self.last_action,
                    "reward": reward,
                    "value": self.last_value,
                    "pnl": pnl,
                    "roi": roi,
                    "symbol": data.get("symbol")
                }
                
                self.experiences.append(experience)
                
                logger.info(
                    "[RL v3] Stored experience",
                    symbol=data.get("symbol"),
                    action=self.last_action,
                    reward=reward,
                    pnl=pnl,
                    total_experiences=len(self.experiences)
                )
            
        except Exception as e:
            logger.error(
                "[RL v3] Error handling position closed",
                error=str(e)
            )
    
    def _build_observation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build observation dict from signal data.
        
        Args:
            data: Signal data from event
            
        Returns:
            Observation dictionary for RL v3
        """
        return {
            "price_change_1m": data.get("price_change_1m", 0.0),
            "price_change_5m": data.get("price_change_5m", 0.0),
            "price_change_15m": data.get("price_change_15m", 0.0),
            "volatility": data.get("volatility", 0.02),
            "rsi": data.get("rsi", 50.0),
            "macd": data.get("macd", 0.0),
            "position_size": data.get("position_size", 0.0),
            "position_side": data.get("position_side", 0.0),
            "balance": data.get("balance", 10000.0),
            "equity": data.get("equity", 10000.0),
            "regime": data.get("regime", "TREND"),
            "trend_strength": data.get("trend_strength", 0.5),
            "volume_ratio": data.get("volume_ratio", 1.0),
            "spread": data.get("spread", 0.001),
            "time_of_day": data.get("time_of_day", 0.5)
        }
    
    def _map_action_to_name(self, action: int) -> str:
        """
        Map action code to human-readable name.
        
        Args:
            action: Action code (0-5)
            
        Returns:
            Action name
        """
        action_map = {
            0: "HOLD",
            1: "LONG",
            2: "SHORT",
            3: "REDUCE",
            4: "CLOSE",
            5: "EMERGENCY_FLATTEN"
        }
        return action_map.get(action, "UNKNOWN")
    
    def get_experiences(self) -> list:
        """Get collected experiences for offline training."""
        return self.experiences
    
    def clear_experiences(self):
        """Clear collected experiences."""
        self.experiences = []
        logger.info("[RL v3] Cleared experiences")
    
    def save_model(self, path: Optional[Path] = None):
        """Save RL v3 model."""
        self.manager.save(path)
        logger.info("[RL v3] Model saved", path=str(path or self.manager.config.model_path))
    
    def train_from_experiences(self, num_epochs: int = 10):
        """
        Train RL v3 model from collected experiences.
        
        Note: This is a simplified training approach.
        For full PPO training, use the manager's train() method.
        
        Args:
            num_epochs: Number of training epochs
        """
        if len(self.experiences) < 100:
            logger.warning(
                "[RL v3] Not enough experiences for training",
                count=len(self.experiences),
                required=100
            )
            return
        
        logger.info(
            "[RL v3] Training from experiences",
            count=len(self.experiences),
            epochs=num_epochs
        )
        
        # This would require implementing experience replay for PPO
        # For now, just log the intent
        logger.info("[RL v3] Experience-based training not yet implemented")
