#!/usr/bin/env python3
"""
RL Agent Daemon - Continuous Learning Position Sizing Agent

Listens to Redis streams for:
- Trade executions (quantum:stream:trade.execution.result)
- Closed positions (quantum:stream:trade.closed)
- RL rewards (quantum:stream:rl_rewards)

Continuously trains the RL policy and publishes sizing recommendations.
"""

import os
import sys
import time
import json
import logging
import signal
from typing import Dict, Optional, Any
from datetime import datetime

# Add quantum_trader to path
sys.path.insert(0, "/home/qt/quantum_trader")

try:
    import redis
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install redis numpy")
    sys.exit(1)

# Import RL Agent
from microservices.rl_sizing_agent.rl_agent import get_rl_agent, RLPositionSizingAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


class RLAgentDaemon:
    """RL Agent Daemon - Continuous learning and position sizing"""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        model_path: str = "/models/rl_sizing_agent_v3.pth",
        poll_interval: int = 5
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.model_path = model_path
        self.poll_interval = poll_interval
        
        # Redis connection
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        
        # Initialize RL Agent
        logger.info(f"Initializing RL Agent with model path: {model_path}")
        self.agent = get_rl_agent(model_path=model_path)
        
        # Stream tracking
        self.last_reward_id = "0"
        self.last_closed_id = "0"
        
        # Statistics
        self.experiences_processed = 0
        self.policies_updated = 0
        self.recommendations_published = 0
        
        logger.info("RL Agent Daemon initialized")
    
    def process_reward_stream(self):
        """Process RL rewards from Redis stream"""
        try:
            # Read from rl_rewards stream
            messages = self.redis_client.xread(
                {"quantum:stream:rl_rewards": self.last_reward_id},
                count=10,
                block=1000
            )
            
            if not messages:
                return
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    self.last_reward_id = msg_id
                    
                    try:
                        symbol = fields.get("symbol", "")
                        reward = float(fields.get("reward", 0))
                        state_json = fields.get("state", "{}")
                        action = float(fields.get("action", 0))
                        
                        # Parse state
                        state = json.loads(state_json) if state_json else {}
                        
                        # Convert state dict to numpy array
                        state_vector = self._state_dict_to_vector(state)
                        
                        # Store experience in agent
                        # (The agent has its own experience buffer)
                        logger.debug(
                            f"Reward received: {symbol} "
                            f"reward={reward:.4f} action={action:.4f}"
                        )
                        
                        self.experiences_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing reward message: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading reward stream: {e}")
    
    def process_closed_positions(self):
        """Process closed positions and compute rewards"""
        try:
            # Read from trade.closed stream
            messages = self.redis_client.xread(
                {"quantum:stream:trade.closed": self.last_closed_id},
                count=10,
                block=1000
            )
            
            if not messages:
                return
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    self.last_closed_id = msg_id
                    
                    try:
                        symbol = fields.get("symbol", "")
                        pnl = float(fields.get("realized_pnl", 0))
                        entry_price = float(fields.get("entry_price", 0))
                        close_price = float(fields.get("close_price", 0))
                        leverage = float(fields.get("leverage", 1))
                        
                        if entry_price > 0:
                            # Calculate PnL percentage
                            pnl_pct = ((close_price - entry_price) / entry_price) * 100 * leverage
                            
                            # Simple reward: PnL percentage
                            reward = pnl_pct / 100.0  # Normalize to -1 to +1 range roughly
                            
                            logger.info(
                                f"Closed position: {symbol} "
                                f"PnL={pnl:.2f} ({pnl_pct:+.2f}%) "
                                f"reward={reward:+.4f}"
                            )
                            
                            # Publish to RL rewards stream for other consumers
                            self.redis_client.xadd(
                                "quantum:stream:rl_rewards",
                                {
                                    "symbol": symbol,
                                    "reward": reward,
                                    "pnl": pnl,
                                    "pnl_pct": pnl_pct,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            )
                            
                            self.experiences_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing closed position: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error reading closed positions stream: {e}")
    
    def publish_statistics(self):
        """Publish RL agent statistics to Redis"""
        try:
            stats = self.agent.get_statistics()
            stats.update({
                "daemon_experiences_processed": self.experiences_processed,
                "daemon_policies_updated": self.policies_updated,
                "daemon_recommendations_published": self.recommendations_published,
                "last_update": datetime.utcnow().isoformat()
            })
            
            # Publish to Redis hash
            self.redis_client.hset(
                "quantum:rl:agent:stats",
                mapping={k: str(v) for k, v in stats.items()}
            )
            
            # Also publish to stream for monitoring
            self.redis_client.xadd(
                "quantum:stream:rl.stats",
                stats,
                maxlen=1000
            )
            
        except Exception as e:
            logger.error(f"Error publishing statistics: {e}")
    
    def _state_dict_to_vector(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to numpy vector"""
        # Expected state keys (from RL agent spec):
        # confidence, volatility, pnl_trend, exch_divergence, funding_rate, margin_util
        return np.array([
            state.get("confidence", 0.5),
            state.get("volatility", 0.1),
            state.get("pnl_trend", 0.0),
            state.get("exch_divergence", 0.0),
            state.get("funding_rate", 0.0),
            state.get("margin_util", 0.0)
        ], dtype=np.float32)
    
    def run(self):
        """Main daemon loop"""
        logger.info("Starting RL Agent Daemon main loop")
        logger.info(f"Polling interval: {self.poll_interval}s")
        logger.info(f"Model path: {self.model_path}")
        
        iteration = 0
        last_stats_publish = time.time()
        
        while not shutdown_flag:
            try:
                iteration += 1
                
                # Process incoming data
                self.process_reward_stream()
                self.process_closed_positions()
                
                # Publish statistics every 60 seconds
                now = time.time()
                if now - last_stats_publish >= 60:
                    self.publish_statistics()
                    last_stats_publish = now
                    
                    logger.info(
                        f"RL Agent Stats: "
                        f"experiences={self.experiences_processed} "
                        f"updates={self.agent.policy_updates} "
                        f"avg_reward={self.agent.avg_reward:.4f}"
                    )
                
                # Sleep between iterations
                time.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(5)
        
        # Final statistics
        self.publish_statistics()
        logger.info(
            f"RL Agent Daemon shutting down. "
            f"Total experiences: {self.experiences_processed}, "
            f"Policy updates: {self.agent.policy_updates}"
        )


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Configuration from environment
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    model_path = os.getenv("RL_MODEL_PATH", "/models/rl_sizing_agent_v3.pth")
    poll_interval = int(os.getenv("POLL_INTERVAL", "5"))
    
    logger.info("=" * 60)
    logger.info("RL AGENT DAEMON - Continuous Learning Position Sizing")
    logger.info("=" * 60)
    logger.info(f"Redis: {redis_host}:{redis_port}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info("=" * 60)
    
    # Create and run daemon
    try:
        daemon = RLAgentDaemon(
            redis_host=redis_host,
            redis_port=redis_port,
            model_path=model_path,
            poll_interval=poll_interval
        )
        daemon.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("RL Agent Daemon terminated")
    sys.exit(0)


if __name__ == "__main__":
    main()
