"""
Prometheus metrics for Strategy Generator AI.

Tracks generation performance, strategy lifecycle, and system health.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary
import logging

logger = logging.getLogger(__name__)

# === Strategy Generation Metrics ===

generation_counter = Counter(
    'strategy_generator_generations_total',
    'Total number of generations completed'
)

generation_duration = Histogram(
    'strategy_generator_generation_duration_seconds',
    'Time taken to complete a generation',
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

strategies_created = Counter(
    'strategy_generator_strategies_created_total',
    'Total number of strategies created',
    ['generation']
)

strategies_promoted = Counter(
    'strategy_generator_strategies_promoted_total',
    'Total number of strategies promoted to SHADOW',
    ['generation']
)

# === Strategy Status Metrics ===

strategies_by_status = Gauge(
    'strategy_generator_strategies_by_status',
    'Number of strategies by status',
    ['status']
)

# === Performance Metrics ===

strategy_fitness = Histogram(
    'strategy_generator_fitness_score',
    'Fitness scores of generated strategies',
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
)

strategy_profit_factor = Histogram(
    'strategy_generator_profit_factor',
    'Profit factors of generated strategies',
    buckets=(0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
)

strategy_win_rate = Histogram(
    'strategy_generator_win_rate',
    'Win rates of generated strategies',
    buckets=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# === Shadow Testing Metrics ===

shadow_tests_counter = Counter(
    'strategy_generator_shadow_tests_total',
    'Total number of shadow tests completed'
)

shadow_test_duration = Histogram(
    'strategy_generator_shadow_test_duration_seconds',
    'Time taken to complete shadow tests',
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

shadow_promotions = Counter(
    'strategy_generator_shadow_promotions_total',
    'Total number of strategies promoted from SHADOW to LIVE'
)

shadow_fitness = Histogram(
    'strategy_generator_shadow_fitness_score',
    'Fitness scores from shadow testing',
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
)

# === Deployment Metrics ===

deployment_checks = Counter(
    'strategy_generator_deployment_checks_total',
    'Total number of deployment candidate checks'
)

strategies_deployed = Counter(
    'strategy_generator_strategies_deployed_total',
    'Total number of strategies deployed to LIVE'
)

active_live_strategies = Gauge(
    'strategy_generator_active_live_strategies',
    'Number of currently active LIVE strategies'
)

# === Error Metrics ===

generation_errors = Counter(
    'strategy_generator_generation_errors_total',
    'Total number of generation errors',
    ['error_type']
)

shadow_test_errors = Counter(
    'strategy_generator_shadow_test_errors_total',
    'Total number of shadow test errors',
    ['error_type']
)

deployment_errors = Counter(
    'strategy_generator_deployment_errors_total',
    'Total number of deployment errors',
    ['error_type']
)

# === Backtest Metrics ===

backtest_duration = Summary(
    'strategy_generator_backtest_duration_seconds',
    'Time taken to backtest a strategy'
)

backtest_trades = Histogram(
    'strategy_generator_backtest_trades',
    'Number of trades in backtest',
    buckets=(0, 10, 50, 100, 500, 1000, 5000, 10000)
)

# === Market Data Metrics ===

market_data_fetches = Counter(
    'strategy_generator_market_data_fetches_total',
    'Total number of market data fetches',
    ['symbol', 'timeframe']
)

market_data_cache_hits = Counter(
    'strategy_generator_market_data_cache_hits_total',
    'Total number of market data cache hits'
)

market_data_cache_misses = Counter(
    'strategy_generator_market_data_cache_misses_total',
    'Total number of market data cache misses'
)

# === Helper Functions ===

def update_status_counts(repo):
    """Update strategy status gauges from repository."""
    try:
        from backend.research.models import StrategyStatus
        
        for status in StrategyStatus:
            count = len(repo.get_strategies_by_status(status))
            strategies_by_status.labels(status=status.value).set(count)
            
            if status == StrategyStatus.LIVE:
                active_live_strategies.set(count)
    
    except Exception as e:
        logger.error(f"Failed to update status counts: {e}")


def record_generation_metrics(generation_num: int, strategies: list, elapsed: float):
    """Record metrics for a completed generation."""
    try:
        # Count metrics
        generation_counter.inc()
        generation_duration.observe(elapsed)
        strategies_created.labels(generation=str(generation_num)).inc(len(strategies))
        
        # Performance metrics
        for _, stats in strategies:
            strategy_fitness.observe(stats.fitness_score)
            strategy_profit_factor.observe(stats.profit_factor)
            strategy_win_rate.observe(stats.win_rate)
            
            # Track promotions (PF >1.5, WR >0.45)
            if stats.profit_factor > 1.5 and stats.win_rate > 0.45:
                strategies_promoted.labels(generation=str(generation_num)).inc()
    
    except Exception as e:
        logger.error(f"Failed to record generation metrics: {e}")
        generation_errors.labels(error_type='metric_recording').inc()


def record_shadow_test_metrics(results: list, elapsed: float):
    """Record metrics for shadow tests."""
    try:
        shadow_tests_counter.inc(len(results))
        shadow_test_duration.observe(elapsed)
        
        for result in results:
            shadow_fitness.observe(result.fitness_score)
            
            # Track promotions (fitness >= 70)
            if result.fitness_score >= 70.0:
                shadow_promotions.inc()
    
    except Exception as e:
        logger.error(f"Failed to record shadow test metrics: {e}")
        shadow_test_errors.labels(error_type='metric_recording').inc()


def record_deployment_metrics(candidates: int, promoted: int):
    """Record metrics for deployment checks."""
    try:
        deployment_checks.inc()
        strategies_deployed.inc(promoted)
    
    except Exception as e:
        logger.error(f"Failed to record deployment metrics: {e}")
        deployment_errors.labels(error_type='metric_recording').inc()
