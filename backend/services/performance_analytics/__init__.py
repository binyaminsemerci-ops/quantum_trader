"""
Performance & Analytics Layer (PAL) - Centralized Analytics Backend

The Performance & Analytics Layer provides a read-only, queryable interface
for comprehensive performance analysis across all dimensions of Quantum Trader:
strategies, symbols, regimes, risk metrics, and system health.

It aggregates data from trades, metrics, and events to support:
- Dashboard visualizations
- Performance reports
- Risk monitoring
- Decision-making for strategy tuning and risk management

Key Components:
- models.py: Core data models (Trade, StrategyStats, EventLog, etc.)
- repositories.py: Data access protocols
- analytics_service.py: Main service with analytics methods
- fake_repositories.py: Testing implementations
- examples.py: Complete usage demonstrations
"""

from .models import (
    Trade, TradeDirection, TradeExitReason,
    MarketRegime, VolatilityLevel, RiskMode,
    EventType, StrategyStats, SymbolStats, EventLog,
    EquityPoint, DrawdownPeriod, PerformanceSummary,
)

from .repositories import (
    TradeRepository,
    StrategyStatsRepository,
    SymbolStatsRepository,
    MetricsRepository,
    EventLogRepository,
)

from .analytics_service import PerformanceAnalyticsService

from .fake_repositories import (
    FakeTradeRepository,
    FakeStrategyStatsRepository,
    FakeSymbolStatsRepository,
    FakeMetricsRepository,
    FakeEventLogRepository,
)

# Real repository implementations
try:
    from .real_repositories import (
        DatabaseTradeRepository,
        DatabaseStrategyStatsRepository,
        DatabaseSymbolStatsRepository,
        DatabaseMetricsRepository,
        DatabaseEventLogRepository,
    )
    REAL_REPOSITORIES_AVAILABLE = True
except ImportError:
    REAL_REPOSITORIES_AVAILABLE = False

__all__ = [
    # Models
    "Trade",
    "TradeDirection",
    "TradeExitReason",
    "MarketRegime",
    "VolatilityLevel",
    "RiskMode",
    "EventType",
    "StrategyStats",
    "SymbolStats",
    "EventLog",
    "EquityPoint",
    "DrawdownPeriod",
    "PerformanceSummary",
    
    # Repositories
    "TradeRepository",
    "StrategyStatsRepository",
    "SymbolStatsRepository",
    "MetricsRepository",
    "EventLogRepository",
    
    # Service
    "PerformanceAnalyticsService",
    
    # Fake Implementations
    "FakeTradeRepository",
    "FakeStrategyStatsRepository",
    "FakeSymbolStatsRepository",
    "FakeMetricsRepository",
    "FakeEventLogRepository",
]

# Add real implementations to __all__ if available
if REAL_REPOSITORIES_AVAILABLE:
    __all__.extend([
        "DatabaseTradeRepository",
        "DatabaseStrategyStatsRepository",
        "DatabaseSymbolStatsRepository",
        "DatabaseMetricsRepository",
        "DatabaseEventLogRepository",
    ])
