"""
Portfolio Governance Service
=============================
Main service for Portfolio Governance Agent.

This service:
- Runs the governance controller in continuous mode
- Provides REST API for policy queries
- Handles graceful shutdown
- Integrates with Redis for distributed state management
"""

from governance_controller import PortfolioGovernanceAgent
import signal
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)
    
    def request_shutdown(self, signum, frame):
        """Request shutdown."""
        logger.info(f"Received signal {signum}, requesting shutdown...")
        self.shutdown_requested = True


def main():
    """Main entry point for Portfolio Governance Service."""
    
    logger.info("=" * 60)
    logger.info("Portfolio Governance Agent - Starting")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.utcnow().isoformat()}")
    
    # Setup graceful shutdown
    shutdown_handler = GracefulShutdown()
    
    try:
        # Get Redis URL from environment
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info(f"Connecting to Redis: {redis_url}")
        
        # Initialize governance agent
        agent = PortfolioGovernanceAgent(redis_url=redis_url)
        logger.info("Governance agent initialized successfully")
        
        # Display initial policy
        initial_policy = agent.get_current_policy()
        logger.info(f"Initial policy: {initial_policy['policy']}")
        logger.info(f"Initial score: {initial_policy['score']}")
        
        # Run governance loop
        logger.info("Starting governance loop...")
        agent.run(interval=30)  # Adjust policy every 30 seconds
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Portfolio Governance Agent shutting down")
        logger.info(f"End time: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
