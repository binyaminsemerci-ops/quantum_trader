"""
Simple Trading Bot - Continuously generates trade signals from AI Engine.

Flow:
1. Fetch market data (price, volume, etc.)
2. Call AI Engine for predictions
3. Publish trade.intent to EventBus
4. Execution service receives and executes
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)

# 🔥 ILF INTEGRATION: Import RL Position Sizing Agent
try:
    from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
    RL_AGENT_AVAILABLE = True
    logger.info("[TRADING-BOT] ✅ RL Position Sizing Agent available")
except ImportError as e:
    RL_AGENT_AVAILABLE = False
    logger.warning(f"[TRADING-BOT] ⚠️ RL Position Sizing Agent not available: {e}")


class SimpleTradingBot:
    """
    Minimal trading bot that bridges AI Engine and Execution Service.
    
    Features:
    - Fetches market data from Binance
    - Calls AI Engine for predictions
    - Publishes trade signals via EventBus
    - No order execution (delegated to Execution Service)
    """
    
    def __init__(
        self,
        ai_engine_url: str = "http://ai-engine:8001",
        symbols: List[str] = None,
        check_interval_seconds: int = 60,
        min_confidence: float = 0.70,
        event_bus = None,
        binance_api_key: str = None,
        binance_api_secret: str = None
    ):
        self.ai_engine_url = ai_engine_url
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.check_interval = check_interval_seconds
        self.min_confidence = min_confidence
        self.event_bus = event_bus
        
        self.binance_api_key = binance_api_key
        self.binance_api_secret = binance_api_secret
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.signals_generated = 0
        
        # 🔥 ILF INTEGRATION: Initialize RL Position Sizing Agent
        self.rl_sizing_agent = None
        if RL_AGENT_AVAILABLE:
            try:
                self.rl_sizing_agent = RLPositionSizingAgent()
                logger.info("[TRADING-BOT] ✅ RL Position Sizing Agent initialized")
            except Exception as e:
                logger.warning(f"[TRADING-BOT] ⚠️ Failed to initialize RL Agent: {e}")
        
        logger.info(
            f"[TRADING-BOT] Initialized: {len(self.symbols)} symbols, "
            f"check every {check_interval_seconds}s, min_confidence={min_confidence:.0%}, "
            f"RL_Agent={'ACTIVE' if self.rl_sizing_agent else 'DISABLED'}"
        )
    
    async def _get_latest_regime(self, symbol: str) -> str:
        """
        Get latest regime from Redis stream.
        
        Note: Meta regime is GLOBAL (not per-symbol). Represents overall
        market conditions.
        
        Returns:
            Regime string: RANGE, TREND, BULL, BEAR, NORMAL, or UNKNOWN
        """
        if not self.event_bus or "redis" not in self.event_bus:
            return "UNKNOWN"
        
        try:
            redis_client = self.event_bus["redis"]
            
            # Read last regime event (global, not symbol-specific)
            regime_events = await redis_client.xrevrange(
                b"quantum:stream:meta.regime",
                count=1
            )
            
            if not regime_events:
                return "UNKNOWN"
            
            # Parse the most recent regime event
            event_id, event_data = regime_events[0]
            
            try:
                # Direct format: {b'regime': b'RANGE', b'volatility': ..., ...}
                if b'regime' in event_data:
                    regime = event_data[b'regime'].decode()
                    confidence = float(event_data.get(b'confidence', b'0').decode())
                    logger.debug(
                        f"[TRADING-BOT] Market regime: {regime} (confidence={confidence:.2f})"
                    )
                    return regime
                
                # Fallback: payload format
                if b'payload' in event_data:
                    import json
                    payload = json.loads(event_data[b'payload'].decode())
                    return payload.get("regime", "UNKNOWN")
                
            except Exception as parse_err:
                logger.warning(f"[TRADING-BOT] Could not parse regime: {parse_err}")
            
            return "UNKNOWN"
            
        except Exception as e:
            logger.warning(f"[TRADING-BOT] Failed to get regime: {e}")
            return "UNKNOWN"
    
    async def start(self):
        """Start trading bot loop."""
        if self.running:
            logger.warning("[TRADING-BOT] Already running")
            return
        
        self.running = True
        
        # DISABLED: No need to listen to AI engine - we generate signals ourselves
        # if self.event_bus and "redis" in self.event_bus:
        #     asyncio.create_task(self._redis_listener())
        #     logger.info("[TRADING-BOT] 📡 Starting Redis listener for ai.decision.made")
        
        self._task = asyncio.create_task(self._trading_loop())
        logger.info("[TRADING-BOT] ✅ Started")
    
    async def stop(self):
        """Stop trading bot loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[TRADING-BOT] ✅ Stopped")
    
    async def _trading_loop(self):
        """Main trading loop - active polling mode with fallback signal generation."""
        # ALWAYS use polling mode - EventBus is only for publishing
        logger.info(f"[TRADING-BOT] 🔄 Polling mode - monitoring {len(self.symbols)} symbols with fallback strategy")
        try:
            while self.running:
                try:
                    # Process all symbols in parallel for speed
                    tasks = [self._process_symbol(symbol) for symbol in self.symbols]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.check_interval)
                
                except Exception as e:
                    logger.error(f"[TRADING-BOT] Error in trading loop: {e}", exc_info=True)
                    await asyncio.sleep(10)  # Wait 10s on error
        
        except asyncio.CancelledError:
            logger.info("[TRADING-BOT] Trading loop cancelled")
    
    async def _process_symbol(self, symbol: str):
        """Process one symbol - fetch data, get prediction, publish signal."""
        try:
            # 1. Fetch market data from Binance
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                return
            
            # 2. Call AI Engine for prediction
            prediction = await self._get_ai_prediction(symbol, market_data)
            if not prediction:
                return
            
            # 3. Check if signal meets confidence threshold
            confidence = prediction.get("confidence", 0)
            if confidence < self.min_confidence:
                logger.debug(
                    f"[TRADING-BOT] {symbol}: Low confidence {confidence:.2%} < {self.min_confidence:.0%}"
                )
                return
            
            # 4. Publish trade signal
            await self._publish_trade_signal(symbol, prediction, market_data)
        
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error processing {symbol}: {e}")
    
    async def _fetch_market_data(self, symbol: str) -> Optional[dict]:
        """Fetch current market data from Binance."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get ticker (24h stats)
                url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={symbol}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        ticker = await resp.json()
                        
                        return {
                            "symbol": symbol,
                            "price": float(ticker["lastPrice"]),
                            "volume_24h": float(ticker["volume"]),
                            "price_change_24h": float(ticker["priceChangePercent"]),
                            "high_24h": float(ticker["highPrice"]),
                            "low_24h": float(ticker["lowPrice"]),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        logger.warning(f"[TRADING-BOT] Failed to fetch {symbol}: HTTP {resp.status}")
                        return None
        
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error fetching market data for {symbol}: {e}")
            return None
    
    async def _get_ai_prediction(self, symbol: str, market_data: dict) -> Optional[dict]:
        """Get prediction from AI Engine with fallback to simple strategy."""
        try:
            # Try AI Engine first
            async with aiohttp.ClientSession() as session:
                url = f"{self.ai_engine_url}/api/ai/signal"
                payload = {
                    "symbol": symbol,
                    "price": market_data.get("price"),
                    "volume": market_data.get("volume_24h"),
                    "timeframe": "1h"
                }
                
                logger.debug(f"[TRADING-BOT] Calling AI Engine: {url} with payload: {payload}")
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(
                            f"[TRADING-BOT] ✅ AI Engine: {symbol} {result.get('action', 'UNKNOWN')} "
                            f"confidence={result.get('confidence', 0):.2%}"
                        )
                        return result
                    else:
                        response_text = await resp.text()
                        logger.warning(
                            f"[TRADING-BOT] AI Engine unavailable (HTTP {resp.status}): {response_text[:200]}, "
                            f"using fallback strategy"
                        )
                        return await self._generate_fallback_signal(symbol, market_data)
        
        except Exception as e:
            logger.warning(f"[TRADING-BOT] AI Engine error: {e}, using fallback strategy")
            return await self._generate_fallback_signal(symbol, market_data)
    
    async def _generate_fallback_signal(self, symbol: str, market_data: dict) -> Optional[dict]:
        """
        Simple fallback signal generator using trend-following strategy.
        
        Strategy:
        - BUY if 24h price change > +1% (uptrend)
        - SELL if 24h price change < -1% (downtrend)
        - HOLD otherwise
        
        Confidence based on momentum strength.
        """
        try:
            price_change_pct = market_data.get("price_change_24h", 0)
            volume = market_data.get("volume_24h", 0)
            price = market_data.get("price", 0)
            
            # Determine action based on trend (lowered threshold to ±1% for more signals)
            if price_change_pct > 1.0:
                action = "BUY"
                # Confidence increases with stronger momentum (cap at 80%)
                confidence = min(0.50 + (price_change_pct / 100), 0.80)
            elif price_change_pct < -1.0:
                action = "SELL"
                confidence = min(0.50 + (abs(price_change_pct) / 100), 0.80)
            else:
                action = "HOLD"
                confidence = 0.30  # Low confidence for hold
            
            # Calculate simple TP/SL (2% for SL, 4% for TP = 2:1 R:R)
            if action == "BUY":
                stop_loss = price * 0.98  # 2% below entry
                take_profit = price * 1.04  # 4% above entry
            elif action == "SELL":
                stop_loss = price * 1.02  # 2% above entry (short)
                take_profit = price * 0.96  # 4% below entry (short)
            else:
                stop_loss = price * 0.98
                take_profit = price * 1.02
            
            signal = {
                "symbol": symbol,
                "side": action,
                "action": action,  # For compatibility
                "confidence": confidence,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size_usd": 150,  # Increased to $150 to meet Binance min notional
                "leverage": 1,
                "model": "fallback-trend-following",
                "reason": f"24h change: {price_change_pct:+.2f}% (fallback strategy)"
            }
            
            logger.info(
                f"[TRADING-BOT] 🔄 Fallback signal: {symbol} {action} @ ${price:.2f} "
                f"(24h: {price_change_pct:+.2f}%, confidence={confidence:.0%})"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error generating fallback signal: {e}")
            return None
    
    async def _redis_listener(self):
        """Listen to Redis Stream for AI decisions."""
        import json
        import asyncio
        
        try:
            redis_client = self.event_bus["redis"]
            stream_name = "quantum:stream:ai.decision.made"  # FIXED: Use EventBus stream prefix
            consumer_group = "trading-bot"
            consumer_name = "bot-001"
            last_id = ">"  # Read only new messages
            
            # Create consumer group (ignore error if exists)
            try:
                await redis_client.xgroup_create(stream_name, consumer_group, id="0", mkstream=True)
                logger.info(f"[TRADING-BOT] 📋 Created consumer group: {consumer_group}")
            except Exception as e:
                if "BUSYGROUP" not in str(e):
                    logger.warning(f"[TRADING-BOT] Consumer group setup: {e}")
            
            logger.info(f"[TRADING-BOT] 🎧 Listening to Redis Stream: {stream_name}")
            
            while self.running:
                try:
                    # Read from stream (blocking with 5s timeout)
                    messages = await redis_client.xreadgroup(
                        groupname=consumer_group,
                        consumername=consumer_name,
                        streams={stream_name: last_id},
                        count=10,
                        block=5000  # 5 seconds timeout
                    )
                    
                    if not messages:
                        continue  # Timeout, try again
                    
                    for stream, stream_messages in messages:
                        for msg_id, msg_data in stream_messages:
                            try:
                                # msg_data is dict with byte keys: {b'payload': b'{"symbol": ...}', ...}
                                # Extract and parse the payload JSON
                                payload_bytes = msg_data.get(b'payload', msg_data.get('payload', b'{}'))
                                if isinstance(payload_bytes, bytes):
                                    payload_str = payload_bytes.decode('utf-8')
                                else:
                                    payload_str = payload_bytes
                                
                                event_data = json.loads(payload_str)
                                
                                # Process the event
                                await self._handle_ai_decision(event_data)
                                
                                # Acknowledge message
                                await redis_client.xack(stream_name, consumer_group, msg_id)
                            except Exception as e:
                                logger.error(f"[TRADING-BOT] Error processing stream message {msg_id}: {e}", exc_info=True)
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[TRADING-BOT] Redis stream read error: {e}")
                    await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"[TRADING-BOT] Redis listener error: {e}", exc_info=True)
    
    async def _handle_ai_decision(self, event_data: dict):
        """Handle incoming AI decision events from EventBus."""
        try:
            symbol = event_data.get("symbol")
            # AI Engine uses "side" not "action" in ai.decision.made
            action = event_data.get("side", event_data.get("action", "")).upper()
            confidence = event_data.get("confidence", 0)
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                logger.debug(
                    f"[TRADING-BOT] {symbol}: Low confidence {confidence:.2%} < {self.min_confidence:.0%}"
                )
                return
            
            # Skip HOLD signals
            if action == "HOLD":
                return
            
            # Build trade intent from AI decision
            trade_signal = {
                "symbol": symbol,
                "side": action,  # BUY/SELL
                "confidence": confidence,
                "entry_price": event_data.get("entry_price", 0),
                "stop_loss": event_data.get("stop_loss"),
                "take_profit": event_data.get("take_profit"),
                "position_size_usd": event_data.get("position_size_usd", 100),
                "leverage": event_data.get("leverage", 1),
                "timestamp": datetime.utcnow().isoformat(),
                "model": "ai-engine-ensemble",
                "reason": f"AI confidence={confidence:.2%}"
            }
            
            self.signals_generated += 1
            logger.info(
                f"[TRADING-BOT] ✅ Signal received: {symbol} {action} @ ${trade_signal['entry_price']:.2f} "
                f"(confidence={confidence:.2%}, size=${trade_signal['position_size_usd']:.0f})"
            )
            
            # Publish trade intent to Redis Stream for Execution Service
            if self.event_bus and "redis" in self.event_bus:
                try:
                    redis_client = self.event_bus["redis"]
                    import json
                    message_id = await redis_client.xadd(
                        b"quantum:stream:trade.intent",
                        {b"event_type": b"trade.intent", b"payload": json.dumps(trade_signal).encode(), b"timestamp": datetime.utcnow().isoformat().encode(), b"source": b"trading-bot"}
                    )
                    logger.info(f"[TRADING-BOT] 📤 Trade intent published: {symbol} {action} (id={message_id.decode()})")
                except Exception as e:
                    logger.error(f"[TRADING-BOT] ❌ Failed to publish trade.intent: {e}")
        
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error handling AI decision: {e}", exc_info=True)
    
    async def _publish_trade_signal(self, symbol: str, prediction: dict, market_data: dict):
        """Publish trade.intent signal to EventBus with ILF metadata."""
        try:
            confidence = prediction.get("confidence", 0)
            
            # 🔥 ILF INTEGRATION: Calculate AI position sizing with RL Agent
            position_size_usd = prediction.get("position_size_usd", 200.0)
            leverage = prediction.get("leverage", 1)
            
            # ILF Metadata for ExitBrain v3
            atr_value = 0.0
            volatility_factor = 1.0
            exchange_divergence = 0.0
            funding_rate = 0.0
            
            if self.rl_sizing_agent:
                try:
                    # Calculate ATR from market data (simplified)
                    price_change_24h = abs(market_data.get("price_change_24h", 0.0)) / 100.0
                    atr_value = max(0.02, min(0.10, price_change_24h))  # Clamp to [2%, 10%]
                    
                    # Volatility factor from price range
                    high = market_data.get("high_24h", market_data["price"])
                    low = market_data.get("low_24h", market_data["price"])
                    price = market_data["price"]
                    if low > 0:
                        volatility_factor = ((high - low) / low) * 10.0  # Scale to ~1.0-3.0 range
                        volatility_factor = max(0.5, min(5.0, volatility_factor))
                    
                    # Call RL Agent for position sizing
                    sizing_decision = await asyncio.to_thread(
                        self.rl_sizing_agent.decide_sizing,
                        symbol=symbol,
                        confidence=confidence,
                        atr_pct=atr_value,
                        current_exposure_pct=0.0,  # TODO: Get from portfolio tracker
                        equity_usd=10000.0,  # TODO: Get from execution service
                        adx=None,
                        trend_strength=None
                    )
                    
                    # Use RL Agent's position size and leverage from Math AI
                    position_size_usd = sizing_decision.position_size_usd
                    leverage = sizing_decision.leverage  # ✅ Use Math AI's calculated leverage
                    
                    logger.info(
                        f"[TRADING-BOT] [RL-SIZING] {symbol}: ${position_size_usd:.0f} @ {leverage}x "
                        f"(ATR={atr_value:.2%}, volatility={volatility_factor:.2f})"
                    )
                    
                except Exception as rl_error:
                    logger.warning(f"[TRADING-BOT] RL Agent failed: {rl_error}, using defaults")
            
            # Get current market regime from meta.regime stream
            regime = await self._get_latest_regime(symbol)
            
            # Build trade intent event with ILF metadata
            signal = {
                "symbol": symbol,
                "side": prediction.get("action", prediction.get("side", "HOLD")).upper(),  # BUY/SELL/HOLD
                "confidence": confidence,
                "entry_price": market_data["price"],
                "stop_loss": prediction.get("stop_loss", market_data["price"] * 0.98),
                "take_profit": prediction.get("take_profit", market_data["price"] * 1.02),
                "position_size_usd": position_size_usd,
                "leverage": leverage,
                "timestamp": datetime.utcnow().isoformat(),
                "model": prediction.get("model", "ensemble"),
                "reason": prediction.get("reason", "AI signal"),
                # 🔥 ILF METADATA for ExitBrain v3:
                "atr_value": atr_value,
                "volatility_factor": volatility_factor,
                "exchange_divergence": exchange_divergence,
                "funding_rate": funding_rate,
                "regime": regime  # ✅ FIXED: Now reads from quantum:stream:meta.regime
            }
            
            # Skip HOLD signals
            if signal["side"] == "HOLD":
                return
            
            # Log signal
            self.signals_generated += 1
            logger.info(
                f"[TRADING-BOT] 📡 Signal: {symbol} {signal['side']} @ ${signal['entry_price']:.2f} "
                f"(confidence={signal['confidence']:.2%}, size=${signal['position_size_usd']:.0f})"
            )
            
            # Publish to EventBus/Redis
            if self.event_bus and "redis" in self.event_bus:
                try:
                    redis_client = self.event_bus["redis"]
                    # Publish to quantum:stream:trade.intent for execution service
                    import json
                    message_id = await redis_client.xadd(
                        b"quantum:stream:trade.intent",
                        {b"event_type": b"trade.intent", b"payload": json.dumps(signal).encode(), b"timestamp": datetime.utcnow().isoformat().encode(), b"source": b"trading-bot"}
                    )
                    logger.info(f"[TRADING-BOT] ✅ Published trade.intent for {symbol} (id={message_id.decode()})")
                except Exception as e:
                    logger.error(f"[TRADING-BOT] ❌ Failed to publish trade.intent: {e}")
        
        except Exception as e:
            logger.error(f"[TRADING-BOT] Error publishing signal: {e}", exc_info=True)
    
    def get_status(self) -> dict:
        """Get bot status."""
        return {
            "running": self.running,
            "symbols": self.symbols,
            "check_interval_seconds": self.check_interval,
            "min_confidence": self.min_confidence,
            "signals_generated": self.signals_generated
        }
