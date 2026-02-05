#!/usr/bin/env python3
"""
RL Feedback V2 Daemon
═════════════════════════════════════════════════════════════════

Continuous learning loop: listens to PnL stream and computes
variable reward signals for RL training.

Stream architecture:
  quantum:stream:exitbrain.pnl → (this daemon) → reward computation
  → published to quantum:stream:rl_rewards
  → consumed by quantum-rl-trainer.service

Requirements:
- Non-constant outputs (reward varies with PnL)
- Continuous loop (must not exit)
- Variable leverage adjustments logged
- Active Redis publishing

Status: CRITICAL INVARIANT ENFORCER
Date: Feb 4, 2026
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [RL-FEEDBACK-V2] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "rl_feedback_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RL_FEEDBACK_V2")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.error("Redis not available - running in simulation mode")


class RLFeedbackV2Daemon:
    """
    RL Feedback V2: Computes variable rewards from PnL stream
    
    Architecture:
        PnL Stream → Reward Computation → Policy Adjustment Stream
    
    Variables tracked:
    - Reward signal (pnl-based, normalized)
    - Confidence adjustment (based on prediction accuracy)
    - Dynamic leverage multiplier
    - Trade win rate
    """
    
    def __init__(self):
        """Initialize daemon"""
        self.running = False
        self.message_count = 0
        self.error_count = 0
        self.reward_history = []  # Track recent rewards
        self.leverage_history = []  # Track recent leverage adjustments
        
        # Configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = int(os.getenv("REDIS_DB", "0"))
        
        # Stream configuration
        self.pnl_stream_key = "quantum:stream:exitbrain.pnl"
        self.reward_stream_key = "quantum:stream:rl_rewards"
        self.heartbeat_key = "quantum:svc:rl_feedback_v2:heartbeat"
        self.heartbeat_ttl = int(os.getenv("RL_FEEDBACK_HEARTBEAT_TTL", "30"))
        self.adjustment_key = "quantum:ai_policy_adjustment"
        
        # Reward computation parameters
        self.pnl_scale = 1000.0  # Scale PnL to reward range
        self.confidence_boost = 0.1  # Confidence impact on leverage
        self.max_leverage = 2.0  # Cap leverage
        self.min_leverage = 0.5  # Floor leverage
        
        self.last_stream_id = "0-0"
        
        # Redis connection
        self.redis = None
        self._connect_redis()
        
        logger.info("=" * 80)
        logger.info("RL FEEDBACK V2 DAEMON INITIALIZED")
        logger.info(f"  Redis: {self.redis_host}:{self.redis_port}/{self.redis_db}")
        logger.info(f"  Input Stream: {self.pnl_stream_key}")
        logger.info(f"  Output Stream: {self.reward_stream_key}")
        logger.info("=" * 80)
    
    def _connect_redis(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, simulation mode enabled")
            return
        
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis.ping()
            logger.info(f"✅ Redis connected: {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("Running in simulation mode")
            self.redis = None
    
    def _compute_reward(self, pnl_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute reward signal from PnL event
        
        INVARIANT: Output must be NON-CONSTANT
        
        Args:
            pnl_data: PnL event from stream
        
        Returns:
            Reward computation result
        """
        try:
            # Extract PnL components
            symbol = pnl_data.get("symbol", "UNKNOWN")
            pnl = float(pnl_data.get("pnl", 0.0))
            confidence = float(pnl_data.get("confidence", 0.5))
            volatility = float(pnl_data.get("volatility", 0.01))
            
            # Base reward: normalized PnL
            base_reward = min(max(pnl / self.pnl_scale, -1.0), 1.0)
            
            # Confidence adjustment: boost reward if confident was high
            confidence_factor = 0.5 + (confidence * 0.5)  # [0.5, 1.0]
            
            # Volatility adjustment: reduce confidence in high-vol environments
            vol_factor = 1.0 / (1.0 + volatility)
            
            # Compute final reward
            reward = base_reward * confidence_factor * vol_factor
            
            # Dynamic leverage: adjust based on reward and confidence
            # Higher reward + higher confidence = higher leverage
            leverage = self.min_leverage + (reward + 1) * 0.25  # [0.5, 1.5]
            leverage = max(self.min_leverage, min(self.max_leverage, leverage))
            
            # Track history
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            
            self.leverage_history.append(leverage)
            if len(self.leverage_history) > 100:
                self.leverage_history.pop(0)
            
            # Compute statistics (prove non-constant)
            avg_reward = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
            reward_std = (
                (sum((r - avg_reward) ** 2 for r in self.reward_history) / len(self.reward_history)) ** 0.5
                if len(self.reward_history) > 1 else 0.0
            )
            
            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "pnl": pnl,
                "confidence": confidence,
                "volatility": volatility,
                "base_reward": float(base_reward),
                "confidence_factor": float(confidence_factor),
                "vol_factor": float(vol_factor),
                "reward": float(reward),
                "leverage": float(leverage),
                "avg_reward": float(avg_reward),
                "reward_std": float(reward_std),
                "message_count": self.message_count
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return None
    
    def _process_message(self, message_id: str, data: Dict[str, str]):
        """
        Process PnL message and publish reward
        
        Args:
            message_id: Redis message ID
            data: Message data from stream
        """
        try:
            self.message_count += 1
            
            # Parse message
            pnl_data = {k: v for k, v in data.items()}
            
            # Compute reward (NON-CONSTANT)
            reward = self._compute_reward(pnl_data)
            
            if reward is None:
                self.error_count += 1
                return
            
            # Publish reward to stream
            if self.redis:
                try:
                    self.redis.xadd(
                        self.reward_stream_key,
                        reward,
                        maxlen=10000
                    )
                except Exception as e:
                    logger.error(f"Error publishing reward: {e}")
                    self.error_count += 1
            
            # Also publish latest adjustment
            if self.redis:
                try:
                    self.redis.hset(
                        self.adjustment_key,
                        mapping={
                            "reward": str(reward["reward"]),
                            "leverage": str(reward["leverage"]),
                            "timestamp": reward["timestamp"],
                            "symbol": reward["symbol"],
                            "msg_count": str(self.message_count)
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating adjustment: {e}")
            
            # Log periodically
            if self.message_count % 10 == 0:
                logger.info(
                    f"[MSG {self.message_count}] "
                    f"Symbol: {reward['symbol']}, "
                    f"PnL: ${reward['pnl']:.4f}, "
                    f"Reward: {reward['reward']:.4f}, "
                    f"Leverage: {reward['leverage']:.2f}x, "
                    f"AvgReward: {reward['avg_reward']:.4f}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message {message_id}: {e}")
            self.error_count += 1
    
    def _generate_test_event(self):
        """Generate test PnL event (for simulation/testing)"""
        import random
        
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        symbol = random.choice(symbols)
        
        # Variable PnL (non-constant)
        pnl = random.uniform(-100, 200)  # -$100 to +$200
        confidence = random.uniform(0.3, 0.95)
        volatility = random.uniform(0.01, 0.05)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "pnl": str(pnl),
            "confidence": str(confidence),
            "volatility": str(volatility),
            "side": random.choice(["BUY", "SELL"])
        }
    
    def start(self):
        """Start daemon loop"""
        self.running = True
        
        # Handle signals
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("🚀 Starting daemon loop...")
        
        while self.running:
            try:
                if self.redis:
                    # Heartbeat (learning plane health signal) - MUST be before blocking operations
                    try:
                        ts = int(time.time())
                        self.redis.set(self.heartbeat_key, str(ts), ex=self.heartbeat_ttl)
                    except Exception as e:
                        logger.error(f"✗ Heartbeat failed: {e}")

                    # Read from PnL stream (non-blocking to ensure heartbeat continues)
                    try:
                        messages = self.redis.xread(
                            {self.pnl_stream_key: self.last_stream_id},
                            count=10,
                            block=1000  # 1 second timeout (shorter to emit heartbeat frequently)
                        )
                        
                        if messages:
                            for stream_key, message_list in messages:
                                for message_id, data in message_list:
                                    self._process_message(message_id, data)
                                    self.last_stream_id = message_id
                    
                    except Exception as e:
                        logger.error(f"Error reading stream: {e}")
                        self.error_count += 1
                        time.sleep(1)
                else:
                    # Simulation mode: generate test events
                    if self.message_count == 0:
                        logger.info("Running in simulation mode (no Redis)")
                    
                    test_event = self._generate_test_event()
                    self._process_message("sim-0", test_event)
                    time.sleep(2)  # Simulate 2-second processing
                
                # Small sleep to prevent tight loop
                time.sleep(0.01)
            
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt, shutting down...")
                self.running = False
            
            except Exception as e:
                logger.error(f"Unexpected error in daemon loop: {e}")
                self.error_count += 1
                time.sleep(1)
        
        self._shutdown()
    
    def _shutdown(self):
        """Shutdown daemon"""
        logger.info("=" * 80)
        logger.info("DAEMON SHUTDOWN")
        logger.info(f"  Messages processed: {self.message_count}")
        logger.info(f"  Errors encountered: {self.error_count}")
        if self.reward_history:
            avg = sum(self.reward_history) / len(self.reward_history)
            logger.info(f"  Average reward: {avg:.4f}")
        logger.info("=" * 80)
        
        if self.redis:
            try:
                self.redis.close()
            except:
                pass


def main():
    """Main entry point"""
    daemon = RLFeedbackV2Daemon()
    daemon.start()


if __name__ == "__main__":
    main()
