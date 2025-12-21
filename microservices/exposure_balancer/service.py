"""
Exposure Balancer Service - Phase 4P
Background service for real-time portfolio rebalancing

Runs continuously and:
- Monitors portfolio exposure every 10 seconds
- Executes rebalancing actions automatically
- Sends alerts to Redis streams
- Provides health endpoint
"""

import time
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from redis import Redis
from exposure_balancer import get_exposure_balancer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/exposure_balancer.log')
    ]
)

logger = logging.getLogger(__name__)


class ExposureBalancerService:
    """Background service for continuous exposure monitoring"""
    
    def __init__(self):
        """Initialize service"""
        self.running = False
        self.redis_client = None
        self.balancer = None
        
        # Configuration from environment
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.check_interval = int(os.getenv("REBALANCE_INTERVAL", 10))
        
        logger.info("[Service] Exposure Balancer Service initialized")
    
    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            
            logger.info(f"[Service] Connected to Redis at {self.redis_host}:{self.redis_port}")
            return True
            
        except Exception as e:
            logger.error(f"[Service] Failed to connect to Redis: {e}")
            return False
    
    def start(self):
        """Start the service"""
        logger.info("[Service] Starting Exposure Balancer Service...")
        
        # Connect to Redis
        if not self.connect_redis():
            logger.error("[Service] Cannot start without Redis connection")
            sys.exit(1)
        
        # Initialize balancer
        config = {
            "max_margin_util": float(os.getenv("MAX_MARGIN_UTIL", 0.85)),
            "max_symbol_exposure": float(os.getenv("MAX_SYMBOL_EXPOSURE", 0.15)),
            "min_diversification": int(os.getenv("MIN_DIVERSIFICATION", 5)),
            "divergence_threshold": float(os.getenv("DIVERGENCE_THRESHOLD", 0.03)),
            "rebalance_interval": self.check_interval
        }
        
        self.balancer = get_exposure_balancer(
            redis_client=self.redis_client,
            config=config
        )
        
        logger.info(
            f"[Service] Balancer configured | "
            f"Max Margin: {config['max_margin_util']*100:.0f}% | "
            f"Check Interval: {self.check_interval}s"
        )
        
        # Start monitoring loop
        self.running = True
        self.run_loop()
    
    def run_loop(self):
        """Main monitoring loop"""
        logger.info("[Service] Monitoring loop started")
        
        cycle_count = 0
        
        try:
            while self.running:
                cycle_count += 1
                
                try:
                    # Run rebalancing check
                    self.balancer.rebalance()
                    
                    # Log statistics every 10 cycles
                    if cycle_count % 10 == 0:
                        stats = self.balancer.get_statistics()
                        logger.info(
                            f"[Service] Cycle #{cycle_count} | "
                            f"Actions: {stats['actions_taken']} | "
                            f"Margin: {stats['last_metrics'].get('margin_utilization', 0)*100:.1f}% | "
                            f"Symbols: {stats['last_metrics'].get('symbol_count', 0)}"
                        )
                    
                except Exception as e:
                    logger.error(f"[Service] Error in rebalance cycle: {e}", exc_info=True)
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("[Service] Keyboard interrupt received")
        except Exception as e:
            logger.error(f"[Service] Fatal error in run loop: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """Stop the service"""
        logger.info("[Service] Stopping Exposure Balancer Service...")
        self.running = False
        
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("[Service] Redis connection closed")
            except Exception as e:
                logger.error(f"[Service] Error closing Redis: {e}")
        
        logger.info("[Service] Service stopped")


def main():
    """Main entry point"""
    logger.info("="*70)
    logger.info("EXPOSURE BALANCER SERVICE - PHASE 4P")
    logger.info("Real-time Adaptive Portfolio Risk Management")
    logger.info("="*70)
    
    service = ExposureBalancerService()
    
    try:
        service.start()
    except Exception as e:
        logger.error(f"[Service] Failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
