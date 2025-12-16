"""
Example 4: Full pipeline.

Complete workflow: Generation → Shadow → Promotion → Demotion.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from backend.research.models import StrategyStatus
from backend.research.backtest import StrategyBacktester
from backend.research.search import StrategySearchEngine
from backend.research.shadow import StrategyShadowTester
from backend.research.deployment import StrategyDeploymentManager
from backend.research.repositories import StrategyRepository, MarketDataClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Use stub implementations
from example_1_first_generation import InMemoryStrategyRepository, StubMarketDataClient


async def main():
    """Run complete SG AI pipeline"""
    
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Full Pipeline")
    logger.info("=" * 80)
    
    # Setup
    repo = InMemoryStrategyRepository()
    market_data = StubMarketDataClient()
    backtester = StrategyBacktester(market_data)
    
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    search = StrategySearchEngine(
        backtester,
        repo,
        backtest_symbols=symbols,
        backtest_days=90
    )
    shadow_tester = StrategyShadowTester(repo, market_data)
    deployment_manager = StrategyDeploymentManager(
        repository=repo,
        # Relaxed thresholds for demo
        candidate_min_pf=1.2,
        candidate_min_trades=10,
        candidate_max_dd=0.30,
        shadow_min_pf=1.1,
        shadow_min_trades=5,
        shadow_min_days=0,  # Immediate for demo
        live_min_pf=1.0,
        live_max_dd=0.40,
        live_check_days=1
    )
    
    # Config (symbols and dates now in search engine constructor)
    
    # === STEP 1: Generate initial population ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Generate Initial Population")
    logger.info("=" * 80 + "\n")
    
    strategies = search.run_generation(
        population_size=15,
        generation=1
    )
    
    logger.info(f"\n✅ Generated {len(strategies)} CANDIDATE strategies\n")
    
    # Show top 3
    for i, config in enumerate(strategies[:3], 1):
        stats = repo.get_stats(config.strategy_id, "BACKTEST")[0]
        logger.info(
            f"   {i}. {config.name} | "
            f"Fitness: {stats.fitness_score:.3f} | "
            f"PF: {stats.profit_factor:.2f}"
        )
    
    # === STEP 2: Promote to SHADOW ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Promote Qualifying Strategies to SHADOW")
    logger.info("=" * 80 + "\n")
    
    promoted_ids = deployment_manager.review_and_promote()
    
    if promoted_ids:
        logger.info(f"\n✅ Promoted {len(promoted_ids)} strategies to SHADOW\n")
        
        for strategy_id in promoted_ids:
            config = repo.strategies[strategy_id]
            logger.info(f"   - {config.name}")
    else:
        logger.info("\n⚠️  No strategies qualified for SHADOW yet\n")
    
    # === STEP 3: Run shadow testing ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Forward Test SHADOW Strategies")
    logger.info("=" * 80 + "\n")
    
    shadow_count = len(repo.get_strategies_by_status(StrategyStatus.SHADOW))
    
    if shadow_count > 0:
        logger.info(f"Testing {shadow_count} SHADOW strategies...\n")
        
        # Run 3 iterations
        for iteration in range(1, 4):
            logger.info(f"Shadow Iteration {iteration}/3:")
            
            await shadow_tester.run_once(
                symbols=symbols,
                lookback_days=7
            )
            
            logger.info("")
        
        logger.info("✅ Shadow testing complete\n")
        
        # Show shadow results
        logger.info("📊 Shadow Performance:\n")
        
        for config in repo.get_strategies_by_status(StrategyStatus.SHADOW):
            stats_list = repo.get_stats(config.strategy_id, "SHADOW")
            
            if stats_list:
                total_trades = sum(s.total_trades for s in stats_list)
                total_pnl = sum(s.total_pnl for s in stats_list)
                
                logger.info(
                    f"   {config.name}: "
                    f"{total_trades} trades, ${total_pnl:.2f} P&L"
                )
    else:
        logger.info("⚠️  No SHADOW strategies to test\n")
    
    # === STEP 4: Promote to LIVE ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Promote SHADOW → LIVE")
    logger.info("=" * 80 + "\n")
    
    promoted_ids = deployment_manager.review_and_promote()
    
    live_strategies = repo.get_strategies_by_status(StrategyStatus.LIVE)
    
    if live_strategies:
        logger.info(f"\n✅ {len(live_strategies)} LIVE strategies\n")
        
        for config in live_strategies:
            # Get combined stats
            backtest_stats = repo.get_stats(config.strategy_id, "BACKTEST")[0]
            shadow_stats = repo.get_stats(config.strategy_id, "SHADOW")
            
            shadow_trades = sum(s.total_trades for s in shadow_stats)
            shadow_pnl = sum(s.total_pnl for s in shadow_stats)
            
            logger.info(
                f"   🚀 {config.name}\n"
                f"      Backtest: PF={backtest_stats.profit_factor:.2f}, "
                f"Fitness={backtest_stats.fitness_score:.3f}\n"
                f"      Shadow: {shadow_trades} trades, ${shadow_pnl:.2f} P&L\n"
            )
    else:
        logger.info("⚠️  No strategies qualified for LIVE yet\n")
    
    # === STEP 5: Monitor and demote underperformers ===
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Monitor LIVE Strategies")
    logger.info("=" * 80 + "\n")
    
    if live_strategies:
        logger.info("Simulating live trading performance...\n")
        
        # Simulate some "live" stats (in real system, these come from actual trading)
        for config in live_strategies:
            # Create mock live performance
            from backend.research.models import StrategyStats
            
            # Simulated underperformance
            live_stats = StrategyStats(
                strategy_id=config.strategy_id,
                source="LIVE",
                total_trades=15,
                winning_trades=7,
                losing_trades=8,
                gross_profit=1200.0,
                gross_loss=-1500.0,  # Net loss
                total_pnl=-300.0,
                win_rate=0.467,
                profit_factor=0.80,  # Below threshold
                avg_win=171.43,
                avg_loss=-187.50,
                max_drawdown_pct=0.15,
                sharpe_ratio=0.5,
                period_start=datetime.utcnow() - timedelta(days=7),
                period_end=datetime.utcnow()
            )
            
            repo.save_stats(live_stats)
        
        logger.info("📉 Checking for underperformance...\n")
        
        disabled_ids = deployment_manager.review_and_disable()
        
        if disabled_ids:
            logger.info(f"⚠️  Disabled {len(disabled_ids)} underperforming strategies:\n")
            
            for strategy_id in disabled_ids:
                config = repo.strategies[strategy_id]
                logger.info(f"   - {config.name}")
        else:
            logger.info("✅ All LIVE strategies performing well")
    else:
        logger.info("No LIVE strategies to monitor yet\n")
    
    # === FINAL SUMMARY ===
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE - STATUS SUMMARY")
    logger.info("=" * 80 + "\n")
    
    for status in StrategyStatus:
        strategies = repo.get_strategies_by_status(status)
        logger.info(f"   {status.value}: {len(strategies)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Full pipeline demonstration complete!")
    logger.info("=" * 80)
    
    logger.info("""
    
    NEXT STEPS FOR PRODUCTION:
    
    1. Implement concrete StrategyRepository (PostgreSQL/MongoDB)
    2. Connect MarketDataClient to real data source
    3. Integrate with existing Quantum Trader ensemble
    4. Set up continuous shadow testing (cron or async loop)
    5. Configure production thresholds (stricter than demo)
    6. Add monitoring/alerting for strategy lifecycle events
    7. Implement strategy versioning and rollback
    8. Add analytics dashboard for strategy performance
    
    """)


if __name__ == "__main__":
    asyncio.run(main())
