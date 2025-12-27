"""
Performance & Analytics Layer (PAL) - Examples & Demonstrations

Complete working examples using fake repositories.
"""

import logging
from datetime import datetime, timedelta

from .analytics_service import PerformanceAnalyticsService
from .fake_repositories import (
    FakeTradeRepository,
    FakeStrategyStatsRepository,
    FakeSymbolStatsRepository,
    FakeMetricsRepository,
    FakeEventLogRepository,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_global_performance():
    """Example: Global account performance analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Global Performance Summary")
    logger.info("=" * 60)
    
    # Create service
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # Get 90-day summary
    summary = analytics.get_global_performance_summary(days=90)
    
    logger.info("\n📊 GLOBAL 90-DAY SUMMARY:")
    logger.info(f"  Period: {summary['period']['start_date']} to {summary['period']['end_date']}")
    
    if "balance" in summary:
        logger.info(f"\n  💰 Balance:")
        logger.info(f"    Initial: ${summary['balance']['initial']:,.2f}")
        logger.info(f"    Current: ${summary['balance']['current']:,.2f}")
        logger.info(f"    Total PnL: ${summary['balance']['pnl_total']:,.2f} ({summary['balance']['pnl_pct']*100:.2f}%)")
        
        logger.info(f"\n  📈 Trades:")
        logger.info(f"    Total: {summary['trades']['total']}")
        logger.info(f"    Winning: {summary['trades']['winning']}")
        logger.info(f"    Losing: {summary['trades']['losing']}")
        logger.info(f"    Win Rate: {summary['trades']['win_rate']*100:.1f}%")
        
        logger.info(f"\n  ⚠️ Risk:")
        logger.info(f"    Max Drawdown: {summary['risk']['max_drawdown']*100:.2f}%")
        logger.info(f"    Sharpe Ratio: {summary['risk']['sharpe_ratio']:.2f}")
        logger.info(f"    Profit Factor: {summary['risk']['profit_factor']:.2f}")
        logger.info(f"    Avg R-Multiple: {summary['risk']['avg_r_multiple']:.2f}R")
        
        logger.info(f"\n  🏆 Best/Worst:")
        logger.info(f"    Best Trade: ${summary['best_worst']['best_trade_pnl']:,.2f}")
        logger.info(f"    Worst Trade: ${summary['best_worst']['worst_trade_pnl']:,.2f}")
        logger.info(f"    Best Day: ${summary['best_worst']['best_day_pnl']:,.2f}")
        logger.info(f"    Worst Day: ${summary['best_worst']['worst_day_pnl']:,.2f}")
    
    return summary


def example_strategy_analytics():
    """Example: Strategy performance analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Strategy Performance Analysis")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # Get top strategies
    logger.info("\n🏆 TOP 5 STRATEGIES (180 days):")
    top_strategies = analytics.get_top_strategies(days=180, limit=5)
    
    for i, strat in enumerate(top_strategies, 1):
        logger.info(f"\n  {i}. {strat['strategy_id']}")
        logger.info(f"     PnL: ${strat['pnl_total']:,.2f}")
        logger.info(f"     Trades: {strat['trade_count']}")
        logger.info(f"     Win Rate: {strat['win_rate']*100:.1f}%")
    
    # Detailed analysis for best strategy
    if top_strategies:
        best_strategy_id = top_strategies[0]['strategy_id']
        logger.info(f"\n📊 DETAILED ANALYSIS: {best_strategy_id}")
        
        detailed = analytics.get_strategy_performance(best_strategy_id, days=365)
        
        if "performance" in detailed:
            logger.info(f"\n  Performance:")
            logger.info(f"    Total PnL: ${detailed['performance']['pnl_total']:,.2f}")
            logger.info(f"    Win Rate: {detailed['performance']['win_rate']*100:.1f}%")
            logger.info(f"    Profit Factor: {detailed['performance']['profit_factor']:.2f}")
            logger.info(f"    Avg R-Multiple: {detailed['performance']['avg_r_multiple']:.2f}R")
            
            logger.info(f"\n  Per-Symbol Breakdown:")
            for symbol, stats in detailed['by_symbol'].items():
                logger.info(f"    {symbol}: ${stats['pnl_total']:,.2f} ({stats['trade_count']} trades, {stats['win_rate']*100:.1f}% WR)")
    
    return top_strategies


def example_symbol_analytics():
    """Example: Symbol performance analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Symbol Performance Analysis")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # Get top symbols
    logger.info("\n🪙 TOP SYMBOLS (365 days):")
    top_symbols = analytics.get_top_symbols(days=365, limit=10)
    
    for i, sym in enumerate(top_symbols, 1):
        logger.info(f"  {i}. {sym['symbol']}: ${sym['pnl_total']:,.2f} ({sym['trade_count']} trades)")
    
    # Detailed analysis for BTCUSDT
    logger.info("\n📊 DETAILED ANALYSIS: BTCUSDT")
    btc_stats = analytics.get_symbol_performance("BTCUSDT", days=365)
    
    if "performance" in btc_stats:
        logger.info(f"\n  Performance:")
        logger.info(f"    Total PnL: ${btc_stats['performance']['pnl_total']:,.2f}")
        logger.info(f"    Win Rate: {btc_stats['performance']['win_rate']*100:.1f}%")
        logger.info(f"    Profit Factor: {btc_stats['performance']['profit_factor']:.2f}")
        
        logger.info(f"\n  Volume:")
        logger.info(f"    Total: ${btc_stats['volume']['total']:,.2f}")
        logger.info(f"    Avg per Trade: ${btc_stats['volume']['avg_per_trade']:,.2f}")
        
        logger.info(f"\n  By Regime:")
        for regime, stats in btc_stats['by_regime'].items():
            logger.info(f"    {regime}: ${stats['pnl_total']:,.2f} ({stats['trade_count']} trades)")
    
    return btc_stats


def example_regime_analytics():
    """Example: Regime-based performance analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Regime Performance Analysis")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    regime_stats = analytics.get_regime_performance(days=365)
    
    logger.info("\n🌍 PERFORMANCE BY MARKET REGIME:")
    for regime, stats in regime_stats['by_regime'].items():
        logger.info(f"\n  {regime}:")
        logger.info(f"    Trades: {stats['trade_count']}")
        logger.info(f"    PnL: ${stats['pnl_total']:,.2f}")
        logger.info(f"    Win Rate: {stats['win_rate']*100:.1f}%")
    
    logger.info("\n📊 PERFORMANCE BY VOLATILITY:")
    for vol, stats in regime_stats['by_volatility'].items():
        logger.info(f"\n  {vol}:")
        logger.info(f"    Trades: {stats['trade_count']}")
        logger.info(f"    PnL: ${stats['pnl_total']:,.2f}")
        logger.info(f"    Win Rate: {stats['win_rate']*100:.1f}%")
    
    return regime_stats


def example_risk_analytics():
    """Example: Risk and drawdown analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Risk & Drawdown Analysis")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # R-multiple distribution
    logger.info("\n📈 R-MULTIPLE DISTRIBUTION:")
    r_dist = analytics.get_r_distribution(days=365)
    
    if "summary" in r_dist:
        logger.info(f"\n  Summary:")
        logger.info(f"    Avg R: {r_dist['summary']['avg_r']:.2f}R")
        logger.info(f"    Median R: {r_dist['summary']['median_r']:.2f}R")
        logger.info(f"    Max R: {r_dist['summary']['max_r']:.2f}R")
        logger.info(f"    Min R: {r_dist['summary']['min_r']:.2f}R")
        logger.info(f"    Positive: {r_dist['summary']['positive_count']}")
        logger.info(f"    Negative: {r_dist['summary']['negative_count']}")
        
        logger.info(f"\n  Distribution:")
        for bucket, count in r_dist['buckets'].items():
            logger.info(f"    {bucket}: {count} trades")
    
    # Drawdown stats
    logger.info("\n⬇️ DRAWDOWN STATISTICS:")
    dd_stats = analytics.get_drawdown_stats(days=365)
    
    if "max_drawdown" in dd_stats:
        logger.info(f"  Max Drawdown: {dd_stats['max_drawdown']*100:.2f}%")
        logger.info(f"  Max DD Date: {dd_stats['max_drawdown_date']}")
        logger.info(f"  Avg Drawdown: {dd_stats['avg_drawdown']*100:.2f}%")
        logger.info(f"  Current Drawdown: {dd_stats['current_drawdown']*100:.2f}%")
        logger.info(f"  DD Periods: {dd_stats['drawdown_periods']}")
    
    return r_dist, dd_stats


def example_safety_analytics():
    """Example: Emergency stop and health analysis"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Safety & Health Analysis")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # Emergency stop history
    logger.info("\n🚨 EMERGENCY STOP HISTORY:")
    ess_events = analytics.get_emergency_stop_history(days=365)
    
    if ess_events:
        logger.info(f"  Total ESS events: {len(ess_events)}")
        for event in ess_events[:5]:  # Show first 5
            logger.info(f"\n  {event['timestamp']}:")
            logger.info(f"    Type: {event['event_type']}")
            logger.info(f"    Description: {event['description']}")
            if event.get('equity_at_event'):
                logger.info(f"    Equity: ${event['equity_at_event']:,.2f}")
    else:
        logger.info("  No emergency stop events in period")
    
    # System health timeline
    logger.info("\n💚 SYSTEM HEALTH TIMELINE:")
    health_events = analytics.get_system_health_timeline(days=90)
    
    if health_events:
        logger.info(f"  Total health events: {len(health_events)}")
        for event in health_events[:5]:  # Show first 5
            logger.info(f"\n  {event['timestamp']}:")
            logger.info(f"    Type: {event['event_type']}")
            logger.info(f"    Severity: {event['severity']}")
            logger.info(f"    Description: {event['description']}")
    else:
        logger.info("  No health events in period")
    
    return ess_events, health_events


def example_equity_curve():
    """Example: Equity curve visualization data"""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Equity Curve Data")
    logger.info("=" * 60)
    
    analytics = PerformanceAnalyticsService(
        trades=FakeTradeRepository(),
        strategies=FakeStrategyStatsRepository(),
        symbols=FakeSymbolStatsRepository(),
        metrics=FakeMetricsRepository(),
        events=FakeEventLogRepository(),
    )
    
    # Get equity curve
    equity_curve = analytics.get_global_equity_curve(days=90)
    
    logger.info(f"\n📈 EQUITY CURVE (90 days):")
    logger.info(f"  Data points: {len(equity_curve)}")
    
    if equity_curve:
        logger.info(f"  Start: {equity_curve[0][0].strftime('%Y-%m-%d')} - ${equity_curve[0][1]:,.2f}")
        logger.info(f"  End: {equity_curve[-1][0].strftime('%Y-%m-%d')} - ${equity_curve[-1][1]:,.2f}")
        logger.info(f"  Change: ${equity_curve[-1][1] - equity_curve[0][1]:,.2f}")
        
        # Show sample points
        logger.info(f"\n  Sample points:")
        for i in range(0, len(equity_curve), len(equity_curve) // 10):
            ts, equity = equity_curve[i]
            logger.info(f"    {ts.strftime('%Y-%m-%d')}: ${equity:,.2f}")
    
    return equity_curve


def main():
    """Run all examples"""
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE & ANALYTICS LAYER (PAL) - COMPREHENSIVE DEMONSTRATION")
    logger.info("=" * 80)
    
    # Run all examples
    example_global_performance()
    example_strategy_analytics()
    example_symbol_analytics()
    example_regime_analytics()
    example_risk_analytics()
    example_safety_analytics()
    example_equity_curve()
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY ✅")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
