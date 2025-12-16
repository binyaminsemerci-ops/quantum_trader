"""
Example 1: Running first generation.

Creates initial population, backtests, and ranks by fitness.
"""

import logging
from datetime import datetime, timedelta

from backend.research.models import StrategyStatus, RegimeFilter
from backend.research.backtest import StrategyBacktester
from backend.research.search import StrategySearchEngine
from backend.research.repositories import StrategyRepository, MarketDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Stub implementations for demo ---

class InMemoryStrategyRepository:
    """Simple in-memory strategy storage"""
    
    def __init__(self):
        self.strategies = {}
        self.stats = {}
    
    def save_strategy(self, config):
        self.strategies[config.strategy_id] = config
        logger.info(f"Saved strategy: {config.name}")
    
    def get_strategies_by_status(self, status):
        return [s for s in self.strategies.values() if s.status == status]
    
    def update_status(self, strategy_id, status):
        if strategy_id in self.strategies:
            self.strategies[strategy_id].status = status
    
    def save_stats(self, stats):
        key = (stats.strategy_id, stats.source, stats.timestamp)
        self.stats[key] = stats
        logger.info(f"Saved stats: {stats.strategy_id} ({stats.source})")
    
    def get_stats(self, strategy_id, source=None, days=None):
        results = []
        for (sid, src, ts), stats in self.stats.items():
            if sid != strategy_id:
                continue
            if source and src != source:
                continue
            if days:
                cutoff = datetime.utcnow() - timedelta(days=days)
                if ts < cutoff:
                    continue
            results.append(stats)
        return sorted(results, key=lambda s: s.timestamp, reverse=True)


class StubMarketDataClient:
    """Stub market data for demo (returns dummy OHLCV)"""
    
    def get_history(self, symbol, timeframe, start, end):
        import pandas as pd
        import numpy as np
        
        # Generate dummy data
        periods = int((end - start).total_seconds() / 900)  # 15min bars
        dates = pd.date_range(start, periods=periods, freq='15min')
        
        # Random walk
        np.random.seed(hash(symbol) % 2**32)
        close = 100 + np.random.randn(periods).cumsum()
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(periods) * 0.1,
            'high': close + abs(np.random.randn(periods)) * 0.5,
            'low': close - abs(np.random.randn(periods)) * 0.5,
            'close': close,
            'volume': np.random.uniform(1000, 10000, periods)
        })
        
        return df


# --- Main example ---

def main():
    """Run first generation of strategy evolution"""
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: First Generation")
    logger.info("=" * 60)
    
    # Setup
    repo = InMemoryStrategyRepository()
    market_data = StubMarketDataClient()
    backtester = StrategyBacktester(market_data)
    
    # Search engine
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    search = StrategySearchEngine(
        backtester=backtester,
        repository=repo,
        backtest_symbols=symbols,
        backtest_days=90
    )
    
    # Generate first population
    logger.info("\n🧬 Generating first population...")
    
    population_size = 10
    
    strategies = search.run_generation(
        population_size=population_size,
        generation=1
    )
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nGenerated {len(strategies)} strategies")
    
    logger.info("\n📊 Top 5 by Fitness:\n")
    
    for i, (config, stats) in enumerate(strategies[:5], 1):
        logger.info(
            f"{i}. {config.name}\n"
            f"   Fitness: {stats.fitness_score:.3f}\n"
            f"   PF: {stats.profit_factor:.2f} | WR: {stats.win_rate:.1%} | "
            f"DD: {stats.max_drawdown_pct:.1%}\n"
            f"   Trades: {stats.total_trades} | Total P&L: ${stats.total_pnl:.2f}\n"
            f"   Entry: {config.entry_type} | Regime: {config.regime_filter}\n"
        )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ First generation complete!")
    logger.info("=" * 60)
    
    # Save best strategies
    logger.info("\n💾 Saving top candidates...")
    
    for config, stats in strategies[:3]:
        repo.save_strategy(config)
    
    logger.info(f"\nSaved top 3 strategies to repository")
    logger.info(f"Status: {StrategyStatus.CANDIDATE.value}")


if __name__ == "__main__":
    main()
