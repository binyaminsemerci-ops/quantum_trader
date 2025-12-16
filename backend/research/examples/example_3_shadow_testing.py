"""
Example 3: Shadow testing setup.

Forward-tests strategies on live data without real orders.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from backend.research.models import StrategyStatus, StrategyConfig, EntryType, RegimeFilter
from backend.research.shadow import StrategyShadowTester
from backend.research.repositories import StrategyRepository, MarketDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Use stub implementations from example 1
from example_1_first_generation import InMemoryStrategyRepository, StubMarketDataClient


async def main():
    """Run shadow testing on candidate strategies"""
    
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Shadow Testing")
    logger.info("=" * 60)
    
    # Setup
    repo = InMemoryStrategyRepository()
    market_data = StubMarketDataClient()
    
    # Create some candidate strategies manually for demo
    logger.info("\n📝 Creating mock CANDIDATE strategies...\n")
    
    strategies = [
        StrategyConfig(
            name="Aggressive Momentum",
            entry_type=EntryType.MOMENTUM,
            regime_filter=RegimeFilter.TRENDING,
            min_confidence=0.70,
            take_profit_pct=0.025,
            stop_loss_pct=0.015,
            trailing_stop_enabled=True,
            risk_per_trade_pct=0.02,
            leverage=30.0,
            status=StrategyStatus.SHADOW,  # Promote to shadow
            generation=1,
            parent_ids=[]
        ),
        StrategyConfig(
            name="Conservative Mean Rev",
            entry_type=EntryType.MEAN_REVERSION,
            regime_filter=RegimeFilter.RANGING,
            min_confidence=0.80,
            take_profit_pct=0.015,
            stop_loss_pct=0.010,
            trailing_stop_enabled=False,
            risk_per_trade_pct=0.01,
            leverage=20.0,
            status=StrategyStatus.SHADOW,
            generation=1,
            parent_ids=[]
        )
    ]
    
    for config in strategies:
        repo.save_strategy(config)
        logger.info(f"   Created: {config.name} ({config.strategy_id})")
    
    # Setup shadow tester
    shadow_tester = StrategyShadowTester(
        repository=repo,
        market_data_client=market_data
    )
    
    # Option 1: Run once (for cron/scheduled execution)
    logger.info("\n🔍 Running single shadow test iteration...\n")
    
    await shadow_tester.run_once(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        lookback_days=7
    )
    
    # Check results
    logger.info("\n📊 Shadow Test Results:\n")
    
    for config in strategies:
        stats_list = repo.get_stats(config.strategy_id, source="SHADOW")
        
        if stats_list:
            stats = stats_list[0]
            logger.info(
                f"   {config.name}:\n"
                f"     Trades: {stats.total_trades}\n"
                f"     P&L: ${stats.total_pnl:.2f}\n"
                f"     PF: {stats.profit_factor:.2f}\n"
                f"     WR: {stats.win_rate:.1%}\n"
                f"     DD: {stats.max_drawdown_pct:.1%}\n"
                f"     Fitness: {stats.fitness_score:.3f}\n"
            )
        else:
            logger.info(f"   {config.name}: No stats yet")
    
    # Option 2: Continuous monitoring (commented out for demo)
    logger.info("\n" + "=" * 60)
    logger.info("💡 Continuous Monitoring Example (commented out)")
    logger.info("=" * 60)
    
    logger.info("""
    # To run continuous shadow testing:
    
    await shadow_tester.run_shadow_loop(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        lookback_days=7,
        interval_minutes=15  # Test every 15 minutes
    )
    
    # This will run indefinitely, testing all SHADOW strategies
    # every 15 minutes on the last 7 days of data.
    """)
    
    # For demo, run a few iterations
    logger.info("\n🔄 Running 3 iterations (45 minutes simulated)...\n")
    
    for i in range(1, 4):
        logger.info(f"Iteration {i}/3:")
        
        await shadow_tester.run_once(
            symbols=["BTCUSDT", "ETHUSDT"],
            lookback_days=7
        )
        
        # In real scenario, would await asyncio.sleep(900)  # 15 min
        logger.info("")
    
    # Final stats
    logger.info("=" * 60)
    logger.info("FINAL SHADOW STATS")
    logger.info("=" * 60)
    
    for config in strategies:
        stats_list = repo.get_stats(config.strategy_id, source="SHADOW", days=1)
        
        if stats_list:
            total_trades = sum(s.total_trades for s in stats_list)
            total_pnl = sum(s.total_pnl for s in stats_list)
            
            logger.info(
                f"\n   {config.name}:\n"
                f"     Total Trades: {total_trades}\n"
                f"     Total P&L: ${total_pnl:.2f}\n"
                f"     Iterations: {len(stats_list)}\n"
            )
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Shadow testing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
