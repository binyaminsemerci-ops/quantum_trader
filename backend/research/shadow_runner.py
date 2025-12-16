"""
Shadow Testing & Deployment Runner.

Monitors shadow strategies, evaluates forward-test performance,
and promotes winners to live trading.

Deployment: Docker service that runs 24/7 in production.
"""

import logging
import time
import signal
from datetime import datetime, timedelta
from typing import List, Tuple
from binance.client import Client

from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.binance_market_data import BinanceMarketDataClient
from backend.research.shadow import ShadowTestManager
from backend.research.deployment import DeploymentManager
from backend.research.models import StrategyConfig, StrategyStatus
from backend.research.metrics import (
    record_shadow_test_metrics,
    record_deployment_metrics,
    update_status_counts,
    shadow_test_errors,
    deployment_errors
)
from backend.research.error_recovery import retry_with_backoff

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShadowTestRunner:
    """
    Shadow testing and deployment service.
    
    Runs shadow tests every 15 minutes and evaluates deployment
    candidates every hour.
    """
    
    def __init__(
        self,
        session,
        binance_client: Client,
        shadow_interval_minutes: int = 15,
        deployment_interval_hours: int = 1
    ):
        """
        Initialize shadow runner.
        
        Args:
            session: SQLAlchemy database session
            binance_client: Binance API client
            shadow_interval_minutes: Minutes between shadow tests
            deployment_interval_hours: Hours between deployment checks
        """
        self.session = session
        self.shadow_interval = timedelta(minutes=shadow_interval_minutes)
        self.deployment_interval = timedelta(hours=deployment_interval_hours)
        self.running = True
        
        # Initialize components
        self.repository = PostgresStrategyRepository(session)
        self.market_data = BinanceMarketDataClient(binance_client)
        self.shadow_manager = ShadowTestManager(self.repository, self.market_data)
        self.deployment_manager = DeploymentManager(self.repository)
        
        # Track state
        self.last_shadow_test: datetime = None
        self.last_deployment_check: datetime = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🔬 Shadow Test Runner initialized")
        logger.info(f"   Shadow test interval: {shadow_interval_minutes}m")
        logger.info(f"   Deployment check interval: {deployment_interval_hours}h")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def should_run_shadow_test(self) -> bool:
        """Check if it's time to run shadow tests."""
        if self.last_shadow_test is None:
            return True
        
        time_since_last = datetime.utcnow() - self.last_shadow_test
        return time_since_last >= self.shadow_interval
    
    def should_check_deployment(self) -> bool:
        """Check if it's time to evaluate deployment candidates."""
        if self.last_deployment_check is None:
            return True
        
        time_since_last = datetime.utcnow() - self.last_deployment_check
        return time_since_last >= self.deployment_interval
    
    def run_shadow_tests(self) -> dict:
        """
        Run shadow tests on all SHADOW strategies.
        
        Returns:
            dict: Summary of shadow test results
        """
        logger.info("🔬 Running shadow tests...")
        
        start_time = time.time()
        
        try:
            # Get all SHADOW strategies
            shadow_strategies = self.repository.get_strategies_by_status(StrategyStatus.SHADOW)
            
            if not shadow_strategies:
                logger.info("   No SHADOW strategies to test")
                return {'strategies_tested': 0}
            
            logger.info(f"   Testing {len(shadow_strategies)} SHADOW strategies")
            
            # Run tests on each strategy
            results = []
            for strategy in shadow_strategies:
                try:
                    result = self.shadow_manager.run_shadow_test(strategy.strategy_id, days=7)
                    results.append(result)
                    
                    logger.info(f"   ✅ {strategy.name}: PF={result.profit_factor:.2f}, "
                               f"WR={result.win_rate:.1%}, Fitness={result.fitness_score:.1f}")
                    
                except Exception as e:
                    logger.error(f"   ❌ {strategy.name}: {e}")
            
            elapsed = time.time() - start_time
            
            summary = {
                'strategies_tested': len(results),
                'avg_profit_factor': sum(r.profit_factor for r in results) / len(results) if results else 0,
                'avg_win_rate': sum(r.win_rate for r in results) / len(results) if results else 0,
                'avg_fitness': sum(r.fitness_score for r in results) / len(results) if results else 0,
                'elapsed_seconds': elapsed
            }
            
            logger.info(f"✅ Shadow tests complete: {len(results)} strategies, {elapsed:.1f}s")
            
            # Record metrics
            record_shadow_test_metrics(results, elapsed)
            update_status_counts(self.repository)
            
            self.last_shadow_test = datetime.utcnow()
            return summary
            
        except Exception as e:
            logger.error(f"❌ Shadow tests failed: {e}", exc_info=True)
            shadow_test_errors.labels(error_type=type(e).__name__).inc()
            return {'error': str(e)}
    
    def check_deployments(self) -> dict:
        """
        Check if any SHADOW strategies should be promoted to LIVE.
        
        Returns:
            dict: Summary of deployment decisions
        """
        logger.info("🚀 Checking deployment candidates...")
        
        start_time = time.time()
        
        try:
            # Get deployment recommendations
            candidates = self.deployment_manager.get_deployment_candidates(min_shadow_days=7)
            
            if not candidates:
                logger.info("   No strategies ready for deployment")
                return {'candidates': 0, 'promoted': 0}
            
            logger.info(f"   Found {len(candidates)} deployment candidates")
            
            promoted = []
            for strategy_id, score in candidates:
                # Get strategy
                strategies = self.repository.get_strategies_by_status(StrategyStatus.SHADOW)
                strategy = next((s for s in strategies if s.strategy_id == strategy_id), None)
                
                if not strategy:
                    continue
                
                logger.info(f"   Evaluating: {strategy.name} (score: {score:.2f})")
                
                # Promote if score is high enough
                if score >= 70.0:  # Deployment threshold
                    self.repository.update_status(strategy_id, StrategyStatus.LIVE)
                    promoted.append(strategy.name)
                    logger.info(f"   ✅ PROMOTED TO LIVE: {strategy.name}")
            
            elapsed = time.time() - start_time
            
            summary = {
                'candidates': len(candidates),
                'promoted': len(promoted),
                'promoted_names': promoted,
                'elapsed_seconds': elapsed
            }
            
            logger.info(f"✅ Deployment check complete: {len(promoted)} promoted, {elapsed:.1f}s")
            
            # Record metrics
            record_deployment_metrics(len(candidates), len(promoted))
            update_status_counts(self.repository)
            
            self.last_deployment_check = datetime.utcnow()
            return summary
            
        except Exception as e:
            logger.error(f"❌ Deployment check failed: {e}", exc_info=True)
            deployment_errors.labels(error_type=type(e).__name__).inc()
            return {'error': str(e)}
    
    def run(self):
        """Main loop - runs continuously until shutdown."""
        logger.info("🏁 Starting shadow test & deployment loop")
        logger.info(f"   Checking every 60 seconds")
        
        while self.running:
            try:
                # Check if shadow tests should run
                if self.should_run_shadow_test():
                    self.run_shadow_tests()
                
                # Check if deployment evaluation should run
                if self.should_check_deployment():
                    self.check_deployments()
                
                # Sleep for 60 seconds before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep before retrying
        
        logger.info("✅ Shadow runner shutdown complete")


def main():
    """Entry point for Docker container."""
    import os
    
    # Get configuration from environment
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    SHADOW_INTERVAL_MINUTES = int(os.getenv("SHADOW_INTERVAL_MINUTES", "15"))
    DEPLOYMENT_INTERVAL_HOURS = int(os.getenv("DEPLOYMENT_INTERVAL_HOURS", "1"))
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.warning("⚠️  No Binance API keys provided, using testnet")
        client = Client()
    else:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        logger.info("✅ Binance API client initialized")
    
    # Create database session
    session = SessionLocal()
    
    try:
        # Create and run shadow runner
        runner = ShadowTestRunner(
            session=session,
            binance_client=client,
            shadow_interval_minutes=SHADOW_INTERVAL_MINUTES,
            deployment_interval_hours=DEPLOYMENT_INTERVAL_HOURS
        )
        
        runner.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        import sys
        sys.exit(1)
    finally:
        session.close()
        logger.info("Database session closed")


if __name__ == "__main__":
    main()
