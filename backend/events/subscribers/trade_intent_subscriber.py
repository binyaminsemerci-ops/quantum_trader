"""
Trade Intent Subscriber
Consumes trade.intent events from orchestrators and routes to execution.
"""
import logging
from typing import Dict, Any, Optional
import asyncio

from backend.core.event_bus import EventBus
from backend.services.execution.execution import BinanceFuturesExecutionAdapter
from backend.services.risk.risk_guard import RiskGuardService
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore


class TradeIntentSubscriber:
    """Subscriber for trade.intent events."""
    
    def __init__(
        self,
        event_bus: EventBus,
        execution_adapter: BinanceFuturesExecutionAdapter,
        risk_guard: Optional[RiskGuardService] = None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        self.event_bus = event_bus
        self.execution_adapter = execution_adapter
        self.risk_guard = risk_guard
        self.logger = logger_instance or logging.getLogger(__name__)
        self.metrics_store = RLv3MetricsStore.instance()
        self._running = False
    
    async def start(self):
        """Start subscriber."""
        if self._running:
            self.logger.warning("[trade_intent] Already running")
            return
        
        self._running = True
        self.event_bus.subscribe("trade.intent", self._handle_trade_intent)
        self.logger.info("[trade_intent] Subscribed to trade.intent")
    
    async def stop(self):
        """Stop subscriber."""
        self._running = False
        self.logger.info("[trade_intent] Stopping subscriber")
    
    async def _handle_trade_intent(self, payload: Dict[str, Any]):
        """Handle trade.intent event."""
        try:
            trace_id = payload.get("trace_id", "")
            symbol = payload.get("symbol", "BTCUSDT")
            side = payload.get("side", "HOLD")
            size_pct = payload.get("size_pct", 0.1)
            source = payload.get("source", "UNKNOWN")
            confidence = payload.get("confidence", 0.0)
            leverage = payload.get("leverage", 10)
            
            self.logger.info(
                "[trade_intent] Received trade intent",
                symbol=symbol,
                side=side,
                size_pct=size_pct,
                source=source,
                confidence=confidence,
                trace_id=trace_id,
            )
            
            # Skip HOLD/FLAT actions
            if side in ["HOLD", "FLAT"]:
                self.logger.info(
                    "[trade_intent] Skipping HOLD/FLAT action",
                    side=side,
                    trace_id=trace_id,
                )
                return
            
            # Get account balance
            balance = await self.execution_adapter.get_cash_balance()
            
            # Calculate position size
            position_size_usd = balance * size_pct
            
            # Calculate quantity (approximate)
            # This should be more sophisticated in production
            current_price = await self._get_current_price(symbol)
            if not current_price:
                self.logger.error(
                    "[trade_intent] Failed to get current price",
                    symbol=symbol,
                    trace_id=trace_id,
                )
                return
            
            quantity = position_size_usd / current_price
            
            # Determine order side
            if side == "LONG":
                order_side = "BUY"
            elif side == "SHORT":
                order_side = "SELL"
            else:
                self.logger.warning(
                    "[trade_intent] Unknown side",
                    side=side,
                    trace_id=trace_id,
                )
                return
            
            # Set leverage
            try:
                await self.execution_adapter.set_leverage(symbol, leverage)
            except Exception as e:
                self.logger.error(
                    "[trade_intent] Failed to set leverage",
                    symbol=symbol,
                    leverage=leverage,
                    error=str(e),
                    trace_id=trace_id,
                )
            
            # Submit order
            try:
                order_result = await self.execution_adapter.submit_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    order_type="MARKET",
                )
                
                self.logger.info(
                    "[trade_intent] Order submitted",
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    order_result=order_result,
                    trace_id=trace_id,
                )
                
                # Publish trade.executed event
                await self.event_bus.publish(
                    "trade.executed",
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "source": source,
                        "confidence": confidence,
                        "order_id": order_result.get("orderId") if order_result else None,
                        "executed_at": order_result.get("updateTime") if order_result else None,
                    },
                    trace_id=trace_id,
                )
                
                # Update metrics store (mark as executed)
                # This is a simplified approach - in production, should track by trace_id
                self.logger.info(
                    "[trade_intent] Trade executed successfully",
                    symbol=symbol,
                    source=source,
                    trace_id=trace_id,
                )
                
            except Exception as e:
                self.logger.error(
                    "[trade_intent] Failed to submit order",
                    symbol=symbol,
                    side=order_side,
                    quantity=quantity,
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )
                
                # Publish trade.failed event
                await self.event_bus.publish(
                    "trade.failed",
                    {
                        "symbol": symbol,
                        "side": side,
                        "source": source,
                        "error": str(e),
                    },
                    trace_id=trace_id,
                )
        
        except Exception as e:
            self.logger.error(
                "[trade_intent] Error handling trade intent",
                error=str(e),
                trace_id=payload.get("trace_id"),
                exc_info=True,
            )
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price."""
        try:
            ticker = await self.execution_adapter.get_ticker_price(symbol)
            if ticker:
                return float(ticker.get("price", 0))
            return None
        except Exception as e:
            self.logger.error(
                "[trade_intent] Failed to get ticker price",
                symbol=symbol,
                error=str(e),
            )
            return None
