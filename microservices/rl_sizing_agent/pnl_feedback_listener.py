"""
PnL Feedback Listener - Phase 4O+
Continuous learning loop for RL Position Sizing Agent

Listens to quantum:stream:exitbrain.pnl Redis stream and feeds
experiences to the RL agent for continuous policy improvement.

Stream Format:
{
    "timestamp": float,
    "symbol": str,
    "side": str,
    "confidence": float,
    "dynamic_leverage": float,
    "take_profit_pct": float,
    "stop_loss_pct": float,
    "volatility": float,
    "exch_divergence": float,
    "funding_rate": float,
    "pnl_trend": float,
    "margin_util": float
}

Retraining triggers:
- Every 100 trades
- Mean absolute PnL < 0.001 over last 50 trades
"""

import time
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime, timezone

try:
    from redis import Redis
except ImportError:
    Redis = None

from .rl_agent import get_rl_agent

logger = logging.getLogger(__name__)


class PnLFeedbackListener:
    """
    Listens to ExitBrain PnL stream and trains RL agent
    
    Architecture:
        Redis Stream → PnL Listener → RL Agent → Policy Update
    """
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        stream_key: str = "quantum:stream:exitbrain.pnl",
        config: Optional[Dict] = None
    ):
        """
        Initialize PnL feedback listener
        
        Args:
            redis_client: Redis client for stream reading
            stream_key: Redis stream key to listen to
            config: Configuration overrides
        """
        self.redis = redis_client
        self.stream_key = stream_key
        self.config = config or {}
        
        # Get RL agent
        self.rl_agent = get_rl_agent(
            model_path=self.config.get("model_path", "/models/rl_sizing_agent_v3.pth"),
            config=self.config.get("rl_agent", {})
        )
        
        # Listener state
        self.running = False
        self.last_id = "0-0"  # Start from beginning
        self.messages_processed = 0
        self.errors = 0
        
        # Performance tracking
        self.avg_pnl = 0.0
        self.pnl_history = []
        
        logger.info(
            f"[PnL-Listener] Initialized | "
            f"Stream: {self.stream_key} | "
            f"RL Model: {self.config.get('model_path', 'default')}"
        )
    
    async def start(self):
        """Start listening to PnL stream (async)"""
        if not self.redis:
            logger.error("[PnL-Listener] No Redis client provided, cannot start")
            return
        
        self.running = True
        logger.info(f"[PnL-Listener] Started listening to {self.stream_key}")
        
        try:
            while self.running:
                # Read from stream (blocking with timeout)
                try:
                    messages = self.redis.xread(
                        {self.stream_key: self.last_id},
                        count=10,
                        block=5000  # 5 second timeout
                    )
                    
                    if messages:
                        for stream, message_list in messages:
                            for message_id, data in message_list:
                                await self._process_message(message_id, data)
                                self.last_id = message_id
                    
                    # Small sleep to prevent tight loop
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"[PnL-Listener] Error reading stream: {e}")
                    self.errors += 1
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("[PnL-Listener] Cancelled, shutting down")
        finally:
            self.running = False
    
    def start_blocking(self):
        """Start listening to PnL stream (blocking)"""
        if not self.redis:
            logger.error("[PnL-Listener] No Redis client provided, cannot start")
            return
        
        self.running = True
        logger.info(f"[PnL-Listener] Started listening to {self.stream_key} (blocking)")
        
        try:
            while self.running:
                try:
                    # Read from stream
                    messages = self.redis.xread(
                        {self.stream_key: self.last_id},
                        count=10,
                        block=5000  # 5 second timeout
                    )
                    
                    if messages:
                        for stream, message_list in messages:
                            for message_id, data in message_list:
                                self._process_message_sync(message_id, data)
                                self.last_id = message_id
                    
                    # Small sleep
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    logger.info("[PnL-Listener] Keyboard interrupt, stopping")
                    break
                except Exception as e:
                    logger.error(f"[PnL-Listener] Error reading stream: {e}")
                    self.errors += 1
                    time.sleep(1)
                    
        finally:
            self.running = False
            logger.info("[PnL-Listener] Stopped")
    
    async def _process_message(self, message_id: str, data: Dict):
        """Process a single PnL message (async)"""
        try:
            # Parse message data
            timestamp = float(data.get("timestamp", time.time()))
            symbol = data.get("symbol", "UNKNOWN")
            confidence = float(data.get("confidence", 0.5))
            leverage = float(data.get("dynamic_leverage", 20.0))
            volatility = float(data.get("volatility", 1.0))
            exch_divergence = float(data.get("exch_divergence", 0.0))
            funding_rate = float(data.get("funding_rate", 0.0))
            pnl_trend = float(data.get("pnl_trend", 0.0))
            margin_util = float(data.get("margin_util", 0.0))
            
            # Build state dict
            state = {
                "confidence": confidence,
                "volatility": volatility,
                "pnl_trend": pnl_trend,
                "exch_divergence": exch_divergence,
                "funding_rate": funding_rate,
                "margin_util": margin_util
            }
            
            # Update RL agent (deprecated method, kept for compatibility)
            self.rl_agent.update_policy(
                pnl_trend=pnl_trend,
                leverage=leverage,
                confidence=confidence,
                exch_divergence=exch_divergence,
                funding_rate=funding_rate
            )
            
            # Update statistics
            self.messages_processed += 1
            
            if self.messages_processed % 100 == 0:
                logger.info(
                    f"[PnL-Listener] Processed {self.messages_processed} messages | "
                    f"RL Stats: {self.rl_agent.get_statistics()}"
                )
            
        except Exception as e:
            logger.error(f"[PnL-Listener] Error processing message {message_id}: {e}")
            self.errors += 1
    
    def _process_message_sync(self, message_id: str, data: Dict):
        """Process a single PnL message (sync)"""
        try:
            # Parse message data
            timestamp = float(data.get("timestamp", time.time()))
            symbol = data.get("symbol", "UNKNOWN")
            confidence = float(data.get("confidence", 0.5))
            leverage = float(data.get("dynamic_leverage", 20.0))
            volatility = float(data.get("volatility", 1.0))
            exch_divergence = float(data.get("exch_divergence", 0.0))
            funding_rate = float(data.get("funding_rate", 0.0))
            pnl_trend = float(data.get("pnl_trend", 0.0))
            margin_util = float(data.get("margin_util", 0.0))
            
            # Build state dict
            state = {
                "confidence": confidence,
                "volatility": volatility,
                "pnl_trend": pnl_trend,
                "exch_divergence": exch_divergence,
                "funding_rate": funding_rate,
                "margin_util": margin_util
            }
            
            # Update RL agent
            self.rl_agent.update_policy(
                pnl_trend=pnl_trend,
                leverage=leverage,
                confidence=confidence,
                exch_divergence=exch_divergence,
                funding_rate=funding_rate
            )
            
            # Update statistics
            self.messages_processed += 1
            
            if self.messages_processed % 100 == 0:
                logger.info(
                    f"[PnL-Listener] Processed {self.messages_processed} messages | "
                    f"RL Stats: {self.rl_agent.get_statistics()}"
                )
            
        except Exception as e:
            logger.error(f"[PnL-Listener] Error processing message {message_id}: {e}")
            self.errors += 1
    
    def stop(self):
        """Stop listening"""
        logger.info("[PnL-Listener] Stopping...")
        self.running = False
    
    def get_statistics(self) -> Dict:
        """Get listener statistics"""
        return {
            "running": self.running,
            "messages_processed": self.messages_processed,
            "errors": self.errors,
            "last_id": self.last_id,
            "rl_agent_stats": self.rl_agent.get_statistics()
        }


# CLI entry point for standalone service
def main():
    """Run PnL feedback listener as standalone service"""
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Connect to Redis
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    try:
        redis_client = Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Test connection
        redis_client.ping()
        logger.info(f"[PnL-Listener] Connected to Redis at {redis_host}:{redis_port}")
        
    except Exception as e:
        logger.error(f"[PnL-Listener] Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Create and start listener
    listener = PnLFeedbackListener(redis_client=redis_client)
    
    try:
        logger.info("[PnL-Listener] Starting service...")
        listener.start_blocking()
    except KeyboardInterrupt:
        logger.info("[PnL-Listener] Keyboard interrupt received")
    finally:
        listener.stop()
        logger.info("[PnL-Listener] Service stopped")


if __name__ == "__main__":
    main()
