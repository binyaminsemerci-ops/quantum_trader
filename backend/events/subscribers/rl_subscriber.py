"""
RL Event Listener - Subscriber for Reinforcement Learning Agents
Listens to trading events and updates RL agents with state, action, and reward data.

Architecture v2 Component - Production Ready
"""
from typing import Dict, Any, Optional
import structlog

from backend.events.schemas import (
    SignalGeneratedEvent,
    TradeExecutedEvent,
    PositionClosedEvent
)
from backend.core.event_bus import EventBus
from backend.core.logger import get_logger
from backend.services.policy_store import PolicyStore


logger = get_logger(__name__)


class RLEventListener:
    """
    Reinforcement Learning Event Listener
    
    Subscribes to trading events and updates RL agents:
    - signal.generated → RL Meta Strategy (state update)
    - trade.executed → RL Position Sizing (action confirmation)
    - position.closed → Both agents (reward update)
    
    Integrates with:
    - EventBus v2 (Redis Streams)
    - Logger v2 (structlog + trace_id)
    - PolicyStore v2 (enable_rl flag)
    - RL Meta Strategy Agent
    - RL Position Sizing Agent
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        policy_store: PolicyStore,
        meta_strategy_agent: Optional[Any] = None,
        position_sizing_agent: Optional[Any] = None
    ):
        """
        Initialize RL Event Listener.
        
        Args:
            event_bus: EventBus v2 instance for subscribing to events
            policy_store: PolicyStore v2 for checking enable_rl flag
            meta_strategy_agent: RL Meta Strategy Agent instance
            position_sizing_agent: RL Position Sizing Agent instance
        """
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.meta_agent = meta_strategy_agent
        self.size_agent = position_sizing_agent
        
        # Subscription IDs for cleanup
        self._subscription_ids: list[str] = []
        
        logger.info(
            "[RL Listener] Initialized",
            meta_agent_available=meta_strategy_agent is not None,
            size_agent_available=position_sizing_agent is not None
        )
    
    async def start(self) -> None:
        """
        Start listening to events.
        Subscribes to signal.generated, trade.executed, position.closed.
        """
        try:
            # Subscribe to signal.generated events
            sub_id_signal = await self.event_bus.subscribe(
                event_type="signal.generated",
                handler=self._handle_signal_generated,
                consumer_group="rl_listener",
                consumer_name="rl_signal_handler"
            )
            self._subscription_ids.append(sub_id_signal)
            logger.info("[RL Listener] Subscribed to signal.generated")
            
            # Subscribe to trade.executed events
            sub_id_trade = await self.event_bus.subscribe(
                event_type="trade.executed",
                handler=self._handle_trade_executed,
                consumer_group="rl_listener",
                consumer_name="rl_trade_handler"
            )
            self._subscription_ids.append(sub_id_trade)
            logger.info("[RL Listener] Subscribed to trade.executed")
            
            # Subscribe to position.closed events
            sub_id_position = await self.event_bus.subscribe(
                event_type="position.closed",
                handler=self._handle_position_closed,
                consumer_group="rl_listener",
                consumer_name="rl_position_handler"
            )
            self._subscription_ids.append(sub_id_position)
            logger.info("[RL Listener] Subscribed to position.closed")
            
            logger.info(
                "[RL Listener] Started successfully",
                subscriptions=len(self._subscription_ids)
            )
            
        except Exception as e:
            logger.error(
                "[RL Listener] Failed to start",
                error=str(e),
                exc_info=True
            )
            # Publish error event
            await self._publish_error_event(
                trace_id="rl_listener_startup",
                error_type="RL_LISTENER_START_FAILED",
                error_message=str(e)
            )
            raise
    
    async def stop(self) -> None:
        """
        Stop listening to events and cleanup subscriptions.
        """
        try:
            for sub_id in self._subscription_ids:
                await self.event_bus.unsubscribe(sub_id)
            
            self._subscription_ids.clear()
            logger.info("[RL Listener] Stopped successfully")
            
        except Exception as e:
            logger.error(
                "[RL Listener] Error during shutdown",
                error=str(e),
                exc_info=True
            )
    
    async def _check_rl_enabled(self, trace_id: str) -> bool:
        """
        Check if RL is enabled in PolicyStore v2.
        
        Args:
            trace_id: Trace ID for logging context
            
        Returns:
            True if RL is enabled, False otherwise
        """
        try:
            profile = await self.policy_store.get_active_risk_profile()
            
            if not profile:
                logger.debug("[RL Listener] No active profile found", trace_id=trace_id)
                return False
            
            # Handle both object and dict profiles
            enable_rl = profile.get("enable_rl", False) if isinstance(profile, dict) else getattr(profile, "enable_rl", False)
            
            if not enable_rl:
                logger.debug(
                    "[RL Listener] RL disabled in policy",
                    trace_id=trace_id,
                    profile=profile
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(
                "[RL Listener] Failed to check RL policy",
                trace_id=trace_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def _handle_signal_generated(self, event_data: Dict[str, Any]) -> None:
        """
        Handle signal.generated event.
        Updates RL Meta Strategy Agent with current state.
        
        Event flow:
        signal.generated → Meta Agent state = {symbol, confidence, timeframe}
        
        Args:
            event_data: Event data dictionary
        """
        trace_id = event_data.get("trace_id", "unknown")
        
        try:
            # Extract data payload (handle both flat and nested formats)
            if "data" in event_data:
                # Nested format: {event_type, trace_id, timestamp, data: {...}}
                payload = {**event_data.get("data", {}), "trace_id": trace_id}
                if "timestamp" in event_data:
                    payload["timestamp"] = event_data["timestamp"]
            else:
                # Flat format: {trace_id, timestamp, symbol, side, ...}
                payload = event_data
            
            # Parse event
            event = SignalGeneratedEvent(**payload)
            
            logger.info(
                "[RL Listener] Received signal.generated",
                trace_id=trace_id,
                symbol=event.symbol,
                side=event.side,
                confidence=event.confidence
            )
            
            # Check if RL is enabled
            if not await self._check_rl_enabled(trace_id):
                return
            
            # Check if meta agent is available
            if not self.meta_agent:
                logger.warning(
                    "[RL Listener] Meta Strategy Agent not available",
                    trace_id=trace_id
                )
                return
            
            # Build state for Meta Strategy Agent
            state = {
                "symbol": event.symbol,
                "confidence": event.confidence,
                "timeframe": event.timeframe
            }
            
            # Update Meta Strategy Agent state
            if hasattr(self.meta_agent, 'set_current_state'):
                self.meta_agent.set_current_state(trace_id, state)
                logger.info(
                    "[RL Listener] Meta Agent state updated",
                    trace_id=trace_id,
                    state=state
                )
            else:
                logger.warning(
                    "[RL Listener] Meta Agent missing set_current_state method",
                    trace_id=trace_id
                )
            
        except Exception as e:
            logger.error(
                "[RL Listener] Error handling signal.generated",
                trace_id=trace_id,
                error=str(e),
                exc_info=True
            )
            await self._publish_error_event(
                trace_id=trace_id,
                error_type="RL_SIGNAL_HANDLER_ERROR",
                error_message=str(e)
            )
    
    async def _handle_trade_executed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle trade.executed event.
        Updates RL Position Sizing Agent with executed action.
        
        Event flow:
        trade.executed → Sizing Agent action = {leverage, size_usd}
        
        Args:
            event_data: Event data dictionary
        """
        trace_id = event_data.get("trace_id", "unknown")
        
        try:
            # Extract data payload (handle both flat and nested formats)
            if "data" in event_data:
                # Nested format: {event_type, trace_id, timestamp, data: {...}}
                payload = {**event_data.get("data", {}), "trace_id": trace_id}
                if "timestamp" in event_data:
                    payload["timestamp"] = event_data["timestamp"]
            else:
                # Flat format: {trace_id, timestamp, symbol, side, ...}
                payload = event_data
            
            # Parse event
            event = TradeExecutedEvent(**payload)
            
            logger.info(
                "[RL Listener] Received trade.executed",
                trace_id=trace_id,
                symbol=event.symbol,
                side=event.side,
                position_size_usd=event.position_size_usd
            )
            
            # Check if RL is enabled
            if not await self._check_rl_enabled(trace_id):
                return
            
            # Check if sizing agent is available
            if not self.size_agent:
                logger.warning(
                    "[RL Listener] Position Sizing Agent not available",
                    trace_id=trace_id
                )
                return
            
            # Build action for Position Sizing Agent
            action = {
                "leverage": event.leverage,
                "size_usd": event.position_size_usd  # Field name in TradeExecutedEvent
            }
            
            # Update Position Sizing Agent with executed action
            if hasattr(self.size_agent, 'set_executed_action'):
                self.size_agent.set_executed_action(trace_id, action)
                logger.info(
                    "[RL Listener] Sizing Agent action recorded",
                    trace_id=trace_id,
                    action=action
                )
            else:
                logger.warning(
                    "[RL Listener] Sizing Agent missing set_executed_action method",
                    trace_id=trace_id
                )
            
        except Exception as e:
            logger.error(
                "[RL Listener] Error handling trade.executed",
                trace_id=trace_id,
                error=str(e),
                exc_info=True
            )
            await self._publish_error_event(
                trace_id=trace_id,
                error_type="RL_TRADE_HANDLER_ERROR",
                error_message=str(e)
            )
    
    async def _handle_position_closed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position.closed event.
        Updates both RL agents with reward based on P&L.
        
        Event flow:
        position.closed → Calculate rewards → Update both agents
        
        Reward formulas:
        - Meta Strategy: reward = pnl_pct - max_drawdown_pct * 0.5
        - Position Sizing: reward = pnl_pct
        
        Args:
            event_data: Event data dictionary
        """
        trace_id = event_data.get("trace_id", "unknown")
        
        try:
            # Extract data payload (handle both flat and nested formats)
            if "data" in event_data:
                # Nested format: {event_type, trace_id, timestamp, data: {...}}
                payload = {**event_data.get("data", {}), "trace_id": trace_id}
                if "timestamp" in event_data:
                    payload["timestamp"] = event_data["timestamp"]
            else:
                # Flat format: {trace_id, timestamp, symbol, entry_price, ...}
                payload = event_data
            
            # Parse event
            event = PositionClosedEvent(**payload)
            
            logger.info(
                "[RL Listener] Received position.closed",
                trace_id=trace_id,
                symbol=event.symbol,
                pnl_usd=event.pnl_usd,
                pnl_pct=event.pnl_pct
            )
            
            # Check if RL is enabled
            if not await self._check_rl_enabled(trace_id):
                return
            
            # Calculate max drawdown (use from event model)
            max_drawdown_pct = event.max_drawdown_pct if event.max_drawdown_pct is not None else 0.0
            
            # Calculate rewards
            # Meta Strategy reward: penalize drawdown
            meta_reward = event.pnl_pct - (max_drawdown_pct * 0.5)
            
            # Position Sizing reward: pure P&L performance
            size_reward = event.pnl_pct
            
            logger.info(
                "[RL Listener] Calculated rewards",
                trace_id=trace_id,
                meta_reward=meta_reward,
                size_reward=size_reward,
                pnl_pct=event.pnl_pct,
                max_drawdown_pct=max_drawdown_pct
            )
            
            # Update Meta Strategy Agent
            if self.meta_agent and hasattr(self.meta_agent, 'update'):
                try:
                    self.meta_agent.update(trace_id, reward=meta_reward)
                    logger.info(
                        "[RL Listener] Meta Agent updated with reward",
                        trace_id=trace_id,
                        reward=meta_reward
                    )
                except Exception as e:
                    logger.error(
                        "[RL Listener] Failed to update Meta Agent",
                        trace_id=trace_id,
                        error=str(e)
                    )
            else:
                logger.debug(
                    "[RL Listener] Meta Agent not available or missing update method",
                    trace_id=trace_id
                )
            
            # Update Position Sizing Agent
            if self.size_agent and hasattr(self.size_agent, 'update'):
                try:
                    self.size_agent.update(trace_id, reward=size_reward)
                    logger.info(
                        "[RL Listener] Sizing Agent updated with reward",
                        trace_id=trace_id,
                        reward=size_reward
                    )
                except Exception as e:
                    logger.error(
                        "[RL Listener] Failed to update Sizing Agent",
                        trace_id=trace_id,
                        error=str(e)
                    )
            else:
                logger.debug(
                    "[RL Listener] Sizing Agent not available or missing update method",
                    trace_id=trace_id
                )
            
        except Exception as e:
            logger.error(
                "[RL Listener] Error handling position.closed",
                trace_id=trace_id,
                error=str(e),
                exc_info=True
            )
            await self._publish_error_event(
                trace_id=trace_id,
                error_type="RL_POSITION_HANDLER_ERROR",
                error_message=str(e)
            )
    
    async def _publish_error_event(
        self,
        trace_id: str,
        error_type: str,
        error_message: str
    ) -> None:
        """
        Publish system.event_error event when exception occurs.
        
        Args:
            trace_id: Trace ID for error context
            error_type: Type of error (e.g., RL_SIGNAL_HANDLER_ERROR)
            error_message: Error message/description
        """
        try:
            error_event_data = {
                "event_type": "system.event_error",
                "trace_id": trace_id,
                "timestamp": structlog.stdlib.get_logger().bind().info("get_timestamp"),
                "error_type": error_type,
                "error_message": error_message,
                "source": "RLEventListener"
            }
            
            await self.event_bus.publish(
                event_type="system.event_error",
                event_data=error_event_data
            )
            
            logger.info(
                "[RL Listener] Published error event",
                trace_id=trace_id,
                error_type=error_type
            )
            
        except Exception as e:
            # Don't raise - we're already in error handling
            logger.error(
                "[RL Listener] Failed to publish error event",
                trace_id=trace_id,
                error=str(e)
            )
