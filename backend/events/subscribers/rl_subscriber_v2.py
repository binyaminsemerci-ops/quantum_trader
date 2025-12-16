"""
RL Subscriber v2 - Event-Driven RL Integration (Domain Architecture)
=====================================================================

Integrates RL v2 agents with EventFlow system using domain-based architecture.

Handles:
- SIGNAL_GENERATED events
- TRADE_EXECUTED events
- POSITION_CLOSED events

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0 (Domain Architecture)
"""

from typing import Dict, Any
import structlog

from backend.core.event_bus import EventBus
from backend.events.event_types import EventType
from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
from backend.domains.learning.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2

logger = structlog.get_logger(__name__)


class RLSubscriberV2:
    """
    RL subscriber v2 for event-driven RL integration (Domain Architecture).
    
    Subscribes to:
    - SIGNAL_GENERATED: Use RL to adjust strategy/model
    - TRADE_EXECUTED: Use RL to adjust position sizing
    - POSITION_CLOSED: Update RL agents with results
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        meta_agent: MetaStrategyAgentV2,
        sizing_agent: PositionSizingAgentV2
    ):
        """
        Initialize RL Subscriber v2.
        
        Args:
            event_bus: Event bus (EventBus from backend.core.event_bus)
            meta_agent: Meta strategy agent
            sizing_agent: Position sizing agent
        """
        self.event_bus = event_bus
        self.meta_agent = meta_agent
        self.sizing_agent = sizing_agent
        
        # Subscribe to events
        self._subscribe_to_events()
        
        logger.info("[RL Subscriber v2] Initialized and subscribed")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        self.event_bus.subscribe(
            EventType.SIGNAL_GENERATED,
            self._handle_signal_generated
        )
        
        self.event_bus.subscribe(
            EventType.TRADE_EXECUTED,
            self._handle_trade_executed
        )
        
        self.event_bus.subscribe(
            EventType.POSITION_CLOSED,
            self._handle_position_closed
        )
        
        logger.info(
            "[RL Subscriber v2] Subscribed to events",
            events=[
                EventType.SIGNAL_GENERATED.value,
                EventType.TRADE_EXECUTED.value,
                EventType.POSITION_CLOSED.value
            ]
        )
    
    async def _handle_signal_generated(self, event: Dict[str, Any]):
        """
        Handle SIGNAL_GENERATED event.
        
        Uses meta strategy agent to optimize strategy/model selection.
        
        Args:
            event: Signal generated event
        """
        try:
            data = event.get("data", {})
            
            # Extract market data
            market_data = {
                "symbol": data.get("symbol"),
                "prediction": data.get("prediction"),
                "confidence": data.get("confidence", 0.5),
                "model": data.get("model", "ensemble"),
                "strategy": data.get("strategy", "dual_momentum"),
                "market_features": data.get("market_features", {})
            }
            
            # Select meta strategy action
            action = self.meta_agent.select_action(market_data)
            
            logger.info(
                "[RL Subscriber v2] Meta strategy action selected",
                symbol=market_data["symbol"],
                strategy=action.strategy,
                model=action.model,
                weight=action.weight
            )
            
            # Publish RL decision event (optional)
            await self.event_bus.publish(
                EventType.CUSTOM,
                {
                    "event_name": "rl_meta_strategy_decision",
                    "symbol": market_data["symbol"],
                    "strategy": action.strategy,
                    "model": action.model,
                    "weight": action.weight
                }
            )
            
        except Exception as e:
            logger.error(
                "[RL Subscriber v2] Error in prediction handler",
                error=str(e)
            )
    
    async def _handle_trade_executed(self, event: Dict[str, Any]):
        """
        Handle TRADE_EXECUTED event.
        
        Uses position sizing agent to optimize size/leverage.
        
        Args:
            event: Trade executed event
        """
        try:
            data = event.get("data", {})
            
            # Extract market data
            market_data = {
                "symbol": data.get("symbol"),
                "side": data.get("side"),
                "quantity": data.get("quantity"),
                "price": data.get("price"),
                "leverage": data.get("leverage", 20),
                "confidence": data.get("confidence", 0.5),
                "portfolio_exposure": data.get("portfolio_exposure", 0.0),
                "account_balance": data.get("account_balance", 0.0)
            }
            
            # Select position sizing action
            action = self.sizing_agent.select_action(market_data)
            
            logger.info(
                "[RL Subscriber v2] Position sizing action selected",
                symbol=market_data["symbol"],
                size_multiplier=action.size_multiplier,
                leverage=action.leverage
            )
            
            # Publish RL decision event (optional)
            await self.event_bus.publish(
                EventType.CUSTOM,
                {
                    "event_name": "rl_position_sizing_decision",
                    "symbol": market_data["symbol"],
                    "size_multiplier": action.size_multiplier,
                    "leverage": action.leverage
                }
            )
            
        except Exception as e:
            logger.error(
                "[RL Subscriber v2] Error in trade handler",
                error=str(e)
            )
    
    async def _handle_position_closed(self, event: Dict[str, Any]):
        """
        Handle POSITION_CLOSED event.
        
        Updates both RL agents with results.
        
        Args:
            event: Position closed event
        """
        try:
            data = event.get("data", {})
            
            # Extract result data
            result_data = {
                "symbol": data.get("symbol"),
                "pnl": data.get("pnl", 0.0),
                "pnl_percentage": data.get("pnl_percentage", 0.0),
                "holding_time": data.get("holding_time", 0),
                "entry_price": data.get("entry_price", 0.0),
                "exit_price": data.get("exit_price", 0.0),
                "side": data.get("side"),
                "quantity": data.get("quantity"),
                "leverage": data.get("leverage", 20),
                "close_reason": data.get("close_reason", "unknown"),
                "drawdown": data.get("max_drawdown", 0.0),
                "account_balance": data.get("account_balance", 0.0),
                "portfolio_exposure": data.get("portfolio_exposure", 0.0),
                "volatility": data.get("volatility", 0.0)
            }
            
            # Update both agents
            self.meta_agent.update(result_data)
            self.sizing_agent.update(result_data)
            
            logger.info(
                "[RL Subscriber v2] Agents updated",
                symbol=result_data["symbol"],
                pnl=result_data["pnl"],
                pnl_percentage=result_data["pnl_percentage"]
            )
            
        except Exception as e:
            logger.error(
                "[RL Subscriber v2] Error in position closed handler",
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RL subscriber statistics.
        
        Returns:
            Statistics
        """
        return {
            "meta_agent_stats": self.meta_agent.get_stats(),
            "sizing_agent_stats": self.sizing_agent.get_stats()
        }
    
    def shutdown(self):
        """Shutdown subscriber and save Q-tables."""
        logger.info("[RL Subscriber v2] Shutting down")
        
        # Save Q-tables
        self.meta_agent.save_q_table()
        self.sizing_agent.save_q_table()
        
        logger.info("[RL Subscriber v2] Shutdown complete")
