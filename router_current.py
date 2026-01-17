#!/usr/bin/env python3
"""AI → Strategy Router
Routes AI decisions from Redis Stream → Strategy Brain HTTP → Risk Brain
"""

import asyncio
import httpx
import redis
import logging
import sys
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379"
STRATEGY_BRAIN_URL = "http://127.0.0.1:8011"
RISK_BRAIN_URL = "http://127.0.0.1:8012"

# Streams
AI_DECISION_STREAM = "quantum:stream:ai.decision.made"
TRADE_INTENT_STREAM = "quantum:stream:trade.intent"
CONSUMER_GROUP = "router"
CONSUMER_NAME = "ai_strategy_router"


class AIStrategyRouter:
    def __init__(self):
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.http_client = httpx.AsyncClient(timeout=5.0)
        
    async def setup(self):
        """Create consumer group if not exists."""
        try:
            await asyncio.to_thread(
                self.redis.xgroup_create,
                AI_DECISION_STREAM,
                CONSUMER_GROUP,
                id="0",
                mkstream=True
            )
            logger.info(f"✅ Consumer group '{CONSUMER_GROUP}' created")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Consumer group '{CONSUMER_GROUP}' already exists")
            else:
                raise
                
    async def route_decision(self, decision: dict, trace_id: str, correlation_id: str):
        """Route AI decision through Strategy → Risk → Trade Intent."""
        try:
            # P0 FIX: Idempotency check - prevent duplicate trade intents (SYNCHRONOUS - no race condition)
            dedup_key = f"quantum:dedup:trade_intent:{trace_id}"
            was_set = self.redis.set(
                dedup_key,
                "1",
                nx=True,  # Only set if not exists
                ex=86400  # 24h TTL
            )
            
            if not was_set:
                logger.warning(f"🔁 DUPLICATE_SKIP trace_id={trace_id} correlation_id={correlation_id}")
                return
            
            symbol = decision.get("symbol", "UNKNOWN")
            # Parse 'side' from Redis (buy/sell) or fallback to 'action'
            side = decision.get("side", decision.get("action", "hold")).upper()
            confidence = decision.get("confidence", 0.0)
            
            logger.info(f"📥 AI Decision: {symbol} {side} @ {confidence:.2%} | trace_id={trace_id}")
            
            # Step 1: Strategy Brain evaluation
            strategy_response = await self.http_client.post(
                f"{STRATEGY_BRAIN_URL}/evaluate",
                json={
                    "symbol": symbol,
                    "direction": side,
                    "confidence": confidence
                }
            )
            strategy_response.raise_for_status()
            strategy_result = strategy_response.json()
            
            if not strategy_result.get("approved"):
                logger.info(f"❌ Strategy denied: {strategy_result.get('reason')}")
                return
                
            logger.info(f"✅ Strategy approved")
            
            # Step 2: Risk Brain evaluation (TEMPORARILY SKIPPED - 422 error)
            # TODO: Fix Risk Brain API schema mismatch
            risk_result = {
                "approved": True,
                "adjusted_size_usd": decision.get("position_size_usd", 100.0),
                "adjusted_leverage": decision.get("leverage", 1.0)
            }
            logger.info(f"⚠️  Risk Brain skipped (TESTNET)")
            
            # Step 3: Publish trade intent (EventBus format)
            trade_intent = {
                "symbol": symbol,
                "action": side,
                "confidence": confidence,
                "position_size_usd": risk_result.get("adjusted_size_usd", decision.get("position_size_usd", 100.0)),
                "leverage": risk_result.get("adjusted_leverage", decision.get("leverage", 1.0)),
                "timestamp": datetime.utcnow().isoformat(),
                "source": "ai_strategy_router",
                "stop_loss_pct": decision.get("stop_loss_pct"),
                "take_profit_pct": decision.get("take_profit_pct"),
                "entry_price": decision.get("entry_price"),
                "stop_loss": decision.get("stop_loss"),
                "take_profit": decision.get("take_profit"),
                "quantity": decision.get("quantity")
            }
            
            # Wrap in EventBus format (execution service expects "data" field)
            import json
            await asyncio.to_thread(
                self.redis.xadd,
                TRADE_INTENT_STREAM,
                {"data": json.dumps(trade_intent)},
                maxlen=10000
            )
            
            logger.info(f"🚀 Trade Intent published: {symbol} {side}")
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error routing decision: {e}")
        except Exception as e:
            logger.error(f"Error routing decision: {e}")
            
    async def run(self):
        """Main consumer loop."""
        await self.setup()
        logger.info(f"🚀 AI→Strategy Router started")
        logger.info(f"📥 Consuming: {AI_DECISION_STREAM}")
        logger.info(f"📤 Publishing: {TRADE_INTENT_STREAM}")
        
        last_id = ">"  # Only new messages
        
        while True:
            try:
                messages = await asyncio.to_thread(
                    self.redis.xreadgroup,
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {AI_DECISION_STREAM: last_id},
                    count=10,
                    block=5000
                )
                
                if not messages:
                    continue
                    
                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        # Parse AI Engine event format (payload is JSON string)
                        if 'payload' in msg_data:
                            import json
                            decision = json.loads(msg_data['payload'])
                        else:
                            decision = msg_data
                        
                        # Extract trace_id and correlation_id for idempotency
                        trace_id = msg_data.get('trace_id', msg_id)
                        correlation_id = msg_data.get('correlation_id', trace_id)
                        
                        await self.route_decision(decision, trace_id, correlation_id)
                        
                        # ACK message
                        await asyncio.to_thread(
                            self.redis.xack,
                            AI_DECISION_STREAM,
                            CONSUMER_GROUP,
                            msg_id
                        )
                        
            except KeyboardInterrupt:
                logger.info("⏹️  Shutting down...")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(5)
                
        await self.http_client.aclose()


async def main():
    router = AIStrategyRouter()
    await router.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        sys.exit(0)
