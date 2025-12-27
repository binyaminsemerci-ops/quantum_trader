"""
Continuous Strategy Generator Runner.

Runs evolutionary strategy generation in a loop, creating new generations
and promoting strategies based on backtest performance.

Deployment: Docker service that runs 24/7 in production.
"""

import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
from binance.client import Client

from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.binance_market_data import BinanceMarketDataClient
from backend.research.ensemble_backtest import EnsembleBacktester
from backend.research.search import StrategySearchEngine
from backend.research.models import StrategyStatus
from backend.research.metrics import (
    record_generation_metrics,
    update_status_counts,
    generation_errors
)
from backend.research.error_recovery import (
    retry_with_backoff,
    generation_error_budget
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousStrategyRunner:
    """
    Continuous strategy generation service.
    
    Runs evolutionary search every N hours, creating new candidate strategies
    and promoting high-performers to shadow testing.
    
    NOW INTEGRATED with OpportunityRanker: Only generates strategies for
    top-ranked symbols with high opportunity scores.
    """
    
    def __init__(
        self,
        session,
        binance_client: Client,
        generation_interval_hours: int = 24,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.4,
        opportunity_ranker=None  # NEW: OpportunityRanker from main app
    ):
        """
        Initialize continuous runner.
        
        Args:
            session: SQLAlchemy database session
            binance_client: Binance API client
            generation_interval_hours: Hours between generation runs
            population_size: Number of strategies per generation
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            opportunity_ranker: OpportunityRanker instance (optional)
        """
        self.session = session
        self.generation_interval = timedelta(hours=generation_interval_hours)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.running = True
        
        # Initialize components
        self.repository = PostgresStrategyRepository(session)
        self.market_data = BinanceMarketDataClient(binance_client)
        self.backtest_engine = EnsembleBacktester(self.market_data)
        
        # NEW: OpportunityRanker integration
        from backend.research.opportunity_integration import OpportunityFilteredSymbols
        self.opportunity_filter = OpportunityFilteredSymbols(
            opportunity_ranker=opportunity_ranker,
            top_n=10,  # Top 10 symbols
            min_score=0.65  # Minimum opportunity score
        )
        
        # Get initial symbols from OpportunityRanker
        initial_symbols = self.opportunity_filter.get_top_symbols()
        
        # Initialize search engine
        self.search_engine = StrategySearchEngine(
            repository=self.repository,
            backtest_engine=self.backtest_engine,
            backtest_symbols=initial_symbols,  # NOW USING TOP-RANKED SYMBOLS
            backtest_days=90
        )
        
        logger.info(f"[SG AI] Using OpportunityRanker-filtered symbols: {initial_symbols}")
        
        # Track state
        self.last_generation: Optional[datetime] = None
        self.generation_count: int = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Continuous Strategy Runner initialized")
        logger.info(f"   Generation interval: {generation_interval_hours}h")
        logger.info(f"   Population size: {population_size}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def should_generate(self) -> bool:
        """Check if it's time to run a new generation."""
        if self.last_generation is None:
            return True
        
        time_since_last = datetime.utcnow() - self.last_generation
        return time_since_last >= self.generation_interval
    
    def run_generation(self) -> dict:
        """
        Run one generation of evolutionary strategy search.
        
        NOW USES REFRESHED SYMBOLS from OpportunityRanker each generation.
        
        Returns:
            dict: Summary of generation results
        """
        logger.info("=" * 70)
        logger.info(f"GENERATION {self.generation_count + 1} STARTING")
        logger.info("=" * 70)
        
        # NEW: Refresh symbols from OpportunityRanker before each generation
        updated_symbols = self.opportunity_filter.get_top_symbols()
        if updated_symbols != self.search_engine.backtest_symbols:
            logger.info(
                f"[SG AI] Opportunity symbols updated: "
                f"{self.search_engine.backtest_symbols} → {updated_symbols}"
            )
            self.search_engine.backtest_symbols = updated_symbols
        
        start_time = time.time()
        
        try:
            # Run evolutionary search
            strategies = self.search_engine.run_generation(
                population_size=self.population_size,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate
            )
            
            # Calculate statistics
            fitness_scores = [stats.fitness_score for _, stats in strategies]
            profit_factors = [stats.profit_factor for _, stats in strategies]
            win_rates = [stats.win_rate for _, stats in strategies]
            
            # Count promotions (strategies that exceeded thresholds)
            promoted = sum(1 for _, stats in strategies 
                          if stats.profit_factor > 1.5 and stats.win_rate > 0.45)
            
            elapsed = time.time() - start_time
            
            summary = {
                'generation': self.generation_count + 1,
                'strategies_created': len(strategies),
                'promoted_to_shadow': promoted,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
                'avg_profit_factor': sum(profit_factors) / len(profit_factors) if profit_factors else 0,
                'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else 0,
                'elapsed_seconds': elapsed
            }
            
            logger.info("=" * 70)
            logger.info(f"GENERATION {self.generation_count + 1} COMPLETE")
            logger.info(f"   Strategies Created: {summary['strategies_created']}")
            logger.info(f"   Promoted to Shadow: {summary['promoted_to_shadow']}")
            logger.info(f"   Avg Fitness: {summary['avg_fitness']:.2f}")
            logger.info(f"   Avg PF: {summary['avg_profit_factor']:.2f}")
            logger.info(f"   Avg WR: {summary['avg_win_rate']:.1%}")
            logger.info(f"   Elapsed: {elapsed:.1f}s")
            logger.info("=" * 70)
            
            # Record metrics
            record_generation_metrics(
                self.generation_count + 1,
                strategies,
                elapsed
            )
            update_status_counts(self.repository)
            generation_error_budget.record(success=True)
            
            self.generation_count += 1
            self.last_generation = datetime.utcnow()
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Generation {self.generation_count + 1} failed: {e}", exc_info=True)
            generation_errors.labels(error_type=type(e).__name__).inc()
            generation_error_budget.record(success=False)
            
            # Check error budget
            if generation_error_budget.is_budget_exhausted():
                logger.critical(
                    f"⚠️  Error budget exhausted! "
                    f"Error rate: {generation_error_budget.get_error_rate():.2f}%"
                )
            
            return {
                'generation': self.generation_count + 1,
                'error': str(e)
            }
    
    def run(self):
        """Main loop - runs continuously until shutdown."""
        logger.info("🏁 Starting continuous strategy generation loop")
        logger.info(f"   Checking every 60 seconds")
        
        while self.running:
            try:
                if self.should_generate():
                    self.run_generation()
                else:
                    # Log time until next generation
                    time_until_next = self.generation_interval - (datetime.utcnow() - self.last_generation)
                    hours_remaining = time_until_next.total_seconds() / 3600
                    logger.info(f"⏳ Next generation in {hours_remaining:.1f} hours")
                
                # Sleep for 60 seconds before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Sleep before retrying
        
        logger.info("✅ Continuous runner shutdown complete")


def main():
    """Entry point for Docker container."""
    import os
    
    # Get configuration from environment
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    GENERATION_INTERVAL_HOURS = int(os.getenv("GENERATION_INTERVAL_HOURS", "24"))
    POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", "20"))
    MUTATION_RATE = float(os.getenv("MUTATION_RATE", "0.3"))
    CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", "0.4"))
    
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.warning("⚠️  No Binance API keys provided, using testnet")
        client = Client()  # Testnet or public endpoints
    else:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        logger.info("✅ Binance API client initialized")
    
    # Create database session
    session = SessionLocal()
    
    try:
        # Create and run continuous runner
        runner = ContinuousStrategyRunner(
            session=session,
            binance_client=client,
            generation_interval_hours=GENERATION_INTERVAL_HOURS,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE
        )
        
        runner.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        session.close()
        logger.info("Database session closed")


if __name__ == "__main__":
    main()
