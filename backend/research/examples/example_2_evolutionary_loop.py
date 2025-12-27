"""
Example 2: Evolutionary loop.

Runs multiple generations with selection, crossover, and mutation.
"""

import logging
from datetime import datetime, timedelta

from backend.research.models import StrategyStatus
from backend.research.backtest import StrategyBacktester
from backend.research.search import StrategySearchEngine
from backend.research.repositories import StrategyRepository, MarketDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Use stub implementations from example 1
from example_1_first_generation import InMemoryStrategyRepository, StubMarketDataClient


def main():
    """Run evolutionary loop over multiple generations"""
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Evolutionary Loop")
    logger.info("=" * 60)
    
    # Setup
    repo = InMemoryStrategyRepository()
    market_data = StubMarketDataClient()
    backtester = StrategyBacktester(market_data)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ARBUSDT"]
    search = StrategySearchEngine(
        backtester=backtester,
        repository=repo,
        backtest_symbols=symbols,
        backtest_days=90
    )
    
    # Evolution parameters
    num_generations = 5
    population_size = 20
    parent_count = 5  # Top 5 become parents
    
    # Evolution tracking
    best_fitness_history = []
    avg_fitness_history = []
    
    # Initial population
    logger.info("\n🧬 Generation 1: Creating initial population...\n")
    
    current_gen = search.run_generation(
        population_size=population_size,
        generation=1
    )
    
    # Track stats
    stats = [repo.get_stats(s.strategy_id, "BACKTEST")[0] for s in current_gen]
    best_fitness = max(s.fitness_score for s in stats)
    avg_fitness = sum(s.fitness_score for s in stats) / len(stats)
    
    best_fitness_history.append(best_fitness)
    avg_fitness_history.append(avg_fitness)
    
    logger.info(
        f"   Best Fitness: {best_fitness:.3f} | "
        f"Avg Fitness: {avg_fitness:.3f}\n"
    )
    
    # Evolution loop
    for gen in range(2, num_generations + 1):
        logger.info(f"🧬 Generation {gen}: Evolving...\n")
        
        # Select parents (top N by fitness)
        parents = current_gen[:parent_count]
        
        logger.info(f"   Parents (Top {parent_count}):")
        for i, parent in enumerate(parents, 1):
            parent_stats = repo.get_stats(parent.strategy_id, "BACKTEST")[0]
            logger.info(
                f"     {i}. {parent.name} | Fitness: {parent_stats.fitness_score:.3f}"
            )
        
        # Generate offspring
        logger.info(f"\n   Generating {population_size} offspring...\n")
        
        current_gen = search.run_generation(
            population_size=population_size,
            generation=gen,
            parent_strategies=parents
        )
        
        # Track stats
        stats = [repo.get_stats(s.strategy_id, "BACKTEST")[0] for s in current_gen]
        best_fitness = max(s.fitness_score for s in stats)
        avg_fitness = sum(s.fitness_score for s in stats) / len(stats)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        logger.info(
            f"   Best Fitness: {best_fitness:.3f} | "
            f"Avg Fitness: {avg_fitness:.3f}\n"
        )
    
    # Final results
    logger.info("=" * 60)
    logger.info("EVOLUTION COMPLETE")
    logger.info("=" * 60)
    
    logger.info("\n📈 Fitness Evolution:\n")
    for gen in range(num_generations):
        logger.info(
            f"   Gen {gen+1}: Best={best_fitness_history[gen]:.3f}, "
            f"Avg={avg_fitness_history[gen]:.3f}"
        )
    
    improvement = (
        (best_fitness_history[-1] - best_fitness_history[0]) / 
        best_fitness_history[0] * 100
    )
    
    logger.info(f"\n   Improvement: {improvement:+.1f}%")
    
    # Best strategy
    logger.info("\n🏆 Best Strategy:\n")
    
    best_config = current_gen[0]
    best_stats = repo.get_stats(best_config.strategy_id, "BACKTEST")[0]
    
    logger.info(
        f"   Name: {best_config.name}\n"
        f"   Fitness: {best_stats.fitness_score:.3f}\n"
        f"   PF: {best_stats.profit_factor:.2f}\n"
        f"   WR: {best_stats.win_rate:.1%}\n"
        f"   DD: {best_stats.max_drawdown_pct:.1%}\n"
        f"   Trades: {best_stats.total_trades}\n"
        f"   Total P&L: ${best_stats.total_pnl:.2f}\n"
        f"   Entry: {best_config.entry_type.value}\n"
        f"   Regime: {best_config.regime_filter.value}\n"
        f"   Min Confidence: {best_config.min_confidence:.0%}\n"
        f"   TP: {best_config.take_profit_pct:.1%} | SL: {best_config.stop_loss_pct:.1%}\n"
        f"   Leverage: {best_config.leverage}x | Risk: {best_config.risk_per_trade_pct:.1%}\n"
    )
    
    # Save top strategies
    logger.info("\n💾 Saving top 3 strategies as CANDIDATES...\n")
    
    for i, config in enumerate(current_gen[:3], 1):
        config.status = StrategyStatus.CANDIDATE
        repo.save_strategy(config)
        
        stats = repo.get_stats(config.strategy_id, "BACKTEST")[0]
        logger.info(
            f"   {i}. {config.name} | Fitness: {stats.fitness_score:.3f}"
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Evolutionary loop complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
