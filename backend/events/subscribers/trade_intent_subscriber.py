"""
Trade Intent Subscriber
Consumes trade.intent events from orchestrators and routes to execution.
"""
import logging
from typing import Dict, Any, Optional
import asyncio
import redis.asyncio as redis

from backend.core.event_bus import EventBus
from backend.services.execution.execution import BinanceFuturesExecutionAdapter
from backend.services.risk.risk_guard import RiskGuardService
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore
from backend.domains.exits.exit_brain_v3.v35_integration import ExitBrainV35Integration


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
        self.exitbrain_v35 = ExitBrainV35Integration(enabled=True)  # 🔥 Enable ILF integration
        self._running = False
        
        # 🛡️ SAFE_DRAIN mode: process backlog WITHOUT executing trades
        # Set TRADE_INTENT_SAFE_DRAIN=true to drain old events safely
        # Set TRADE_INTENT_MAX_AGE_MINUTES=5 to only execute recent trades (default: 5 min)
        import os
        self.safe_drain_mode = os.getenv("TRADE_INTENT_SAFE_DRAIN", "false").lower() == "true"
        self.max_age_minutes = int(os.getenv("TRADE_INTENT_MAX_AGE_MINUTES", "5"))
        
        if self.safe_drain_mode:
            self.logger.warning("[trade_intent] 🛡️ SAFE_DRAIN mode ENABLED - will NOT execute trades, only consume events")
        else:
            self.logger.info(f"[trade_intent] ⚡ LIVE mode - will execute trades within {self.max_age_minutes} min of event timestamp")
        
        # Initialize Redis client for ILF metadata storage
        redis_host = os.getenv("REDIS_HOST", "quantum_redis")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    
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
            timestamp = payload.get("timestamp")  # Event creation timestamp
            
            # 🛡️ SAFE_DRAIN: Check event age to avoid executing stale trades
            import time
            current_time_ms = int(time.time() * 1000)
            event_time_ms = timestamp if timestamp else current_time_ms
            age_minutes = (current_time_ms - event_time_ms) / 1000 / 60
            
            is_stale = age_minutes > self.max_age_minutes
            should_skip_execution = self.safe_drain_mode or is_stale
            
            if should_skip_execution:
                if self.safe_drain_mode:
                    self.logger.info(
                        f"[trade_intent] 🛡️ SAFE_DRAIN: Skipping execution (mode=DRAIN)",
                        symbol=symbol,
                        side=side,
                        age_minutes=f"{age_minutes:.1f}",
                        trace_id=trace_id,
                    )
                else:
                    self.logger.warning(
                        f"[trade_intent] ⏰ Skipping STALE trade (age={age_minutes:.1f} min > max={self.max_age_minutes} min)",
                        symbol=symbol,
                        side=side,
                        timestamp=event_time_ms,
                        trace_id=trace_id,
                    )
                
                # Mark as handled WITHOUT executing
                await self.event_bus.publish(
                    "execution.result",
                    {
                        "symbol": symbol,
                        "side": side,
                        "status": "skipped_stale" if is_stale else "skipped_drain",
                        "reason": f"Event age {age_minutes:.1f} min (max {self.max_age_minutes} min)" if is_stale else "SAFE_DRAIN mode active",
                        "trace_id": trace_id,
                        "timestamp": current_time_ms,
                    },
                    trace_id=trace_id,
                )
                return
            
            # 🔥 AI-CALCULATED VALUES (AUTONOMOUS)
            position_size_usd = payload.get("position_size_usd")  # AI-calculated USD amount
            leverage = payload.get("leverage", 1)  # AI-calculated leverage
            
            # 🔥 ILF METADATA (for ExitBrain v3.5 adaptive leverage)
            atr_value = payload.get("atr_value")
            volatility_factor = payload.get("volatility_factor", 1.0)
            exchange_divergence = payload.get("exchange_divergence")
            funding_rate = payload.get("funding_rate")
            regime = payload.get("regime")
            
            # Fallback to size_pct if position_size_usd not provided (backwards compatibility)
            size_pct = payload.get("size_pct", 0.1)
            
            source = payload.get("source", "UNKNOWN")
            confidence = payload.get("confidence", 0.0)
            
            self.logger.info(
                "[trade_intent] Received AI trade intent with ILF metadata",
                symbol=symbol,
                side=side,
                position_size_usd=position_size_usd,
                leverage=leverage,
                volatility_factor=volatility_factor,
                atr_value=atr_value,
                size_pct=size_pct if not position_size_usd else "N/A",
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
            
            # Calculate position size from AI or fallback to % of balance
            if position_size_usd:
                # Use AI-calculated position size (PREFERRED)
                pass
            else:
                # Fallback: Calculate from balance percentage
                balance = await self.execution_adapter.get_cash_balance()
                position_size_usd = balance * size_pct
                self.logger.warning(
                    "[trade_intent] No AI position_size_usd, using fallback calculation",
                    balance=balance,
                    size_pct=size_pct,
                    calculated=position_size_usd,
                )
            
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
                
                # 🔥 COMPUTE ADAPTIVE TP/SL LEVELS using ILF metadata
                if volatility_factor is not None and order_result:
                    try:
                        adaptive_levels = self.exitbrain_v35.compute_adaptive_levels(
                            leverage=leverage,
                            volatility_factor=volatility_factor,
                            confidence=confidence
                        )
                        
                        self.logger.info(
                            "[trade_intent] 🎯 ExitBrain v3.5 Adaptive Levels Calculated",
                            symbol=symbol,
                            leverage=leverage,
                            volatility_factor=volatility_factor,
                            tp1=f"{adaptive_levels['tp1']:.3%}",
                            tp2=f"{adaptive_levels['tp2']:.3%}",
                            tp3=f"{adaptive_levels['tp3']:.3%}",
                            sl=f"{adaptive_levels['sl']:.3%}",
                            lsf=adaptive_levels['LSF'],
                            harvest_scheme=adaptive_levels['harvest_scheme'],
                            adjustment=adaptive_levels['adjustment'],
                            trace_id=trace_id,
                        )
                        
                        # Store ILF metadata in Redis for ExitBrain to use
                        await self._store_ilf_metadata(
                            symbol=symbol,
                            order_id=str(order_result.get("orderId")),
                            ilf_metadata={
                                "atr_value": atr_value,
                                "volatility_factor": volatility_factor,
                                "exchange_divergence": exchange_divergence,
                                "funding_rate": funding_rate,
                                "regime": regime,
                            },
                            adaptive_levels=adaptive_levels
                        )
                        
                        # Publish event for ExitBrain to consume
                        await self.event_bus.publish(
                            "exitbrain.adaptive_levels",
                            {
                                "symbol": symbol,
                                "order_id": order_result.get("orderId"),
                                "leverage": leverage,
                                "volatility_factor": volatility_factor,
                                "adaptive_levels": adaptive_levels,
                                "ilf_metadata": {
                                    "atr_value": atr_value,
                                    "exchange_divergence": exchange_divergence,
                                    "funding_rate": funding_rate,
                                    "regime": regime,
                                }
                            },
                            trace_id=trace_id,
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            "[trade_intent] ❌ Failed to compute adaptive levels",
                            symbol=symbol,
                            leverage=leverage,
                            volatility_factor=volatility_factor,
                            error=str(e),
                            trace_id=trace_id,
                            exc_info=True,
                        )
                else:
                    self.logger.warning(
                        "[trade_intent] ⚠️ No ILF metadata available, skipping adaptive levels",
                        symbol=symbol,
                        volatility_factor=volatility_factor,
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
    
    async def _store_ilf_metadata(
        self,
        symbol: str,
        order_id: str,
        ilf_metadata: Dict[str, Any],
        adaptive_levels: Dict[str, Any]
    ):
        """
        Store ILF metadata in Redis for ExitBrain to use
        
        Args:
            symbol: Trading pair
            order_id: Order ID
            ilf_metadata: ILF metadata dictionary
            adaptive_levels: Computed adaptive levels
        """
        try:
            # Store in Redis hash for position
            redis_key = f"quantum:position:ilf:{symbol}:{order_id}"
            
            # Prepare data
            data = {
                "symbol": symbol,
                "order_id": order_id,
                "atr_value": ilf_metadata.get("atr_value", 0),
                "volatility_factor": ilf_metadata.get("volatility_factor", 1.0),
                "exchange_divergence": ilf_metadata.get("exchange_divergence", 0),
                "funding_rate": ilf_metadata.get("funding_rate", 0),
                "regime": ilf_metadata.get("regime", "unknown"),
                "tp1": adaptive_levels.get("tp1", 0),
                "tp2": adaptive_levels.get("tp2", 0),
                "tp3": adaptive_levels.get("tp3", 0),
                "sl": adaptive_levels.get("sl", 0),
                "lsf": adaptive_levels.get("LSF", 1.0),
                "adjustment": adaptive_levels.get("adjustment", 1.0),
            }
            
            # Store in Redis
            await self.redis.hset(redis_key, mapping=data)
            await self.redis.expire(redis_key, 86400)  # Expire after 24 hours
            
            self.logger.info(
                "[trade_intent] ✅ ILF metadata stored in Redis",
                redis_key=redis_key,
                volatility_factor=ilf_metadata.get("volatility_factor"),
            )
            
        except Exception as e:
            self.logger.error(
                "[trade_intent] Failed to store ILF metadata in Redis",
                symbol=symbol,
                order_id=order_id,
                error=str(e),
                exc_info=True,
            )
