"""
Tests for TPDashboardService

Covers:
- list_tp_entities() with various scenarios
- get_tp_dashboard_entry() for single pairs
- get_top_best_and_worst() ranking logic
- Edge cases: no metrics, missing profiles, optimizer failures
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from backend.services.dashboard.tp_dashboard_service import (
    TPDashboardService,
    TPDashboardKey,
    TPDashboardEntry,
    TPDashboardSummary
)
from backend.services.monitoring.tp_performance_tracker import TPMetrics
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    TPProfile,
    TPProfileLeg,
    TrailingProfile,
    TPKind,
    MarketRegime
)
from backend.services.monitoring.tp_optimizer_v3 import (
    TPAdjustmentRecommendation,
    AdjustmentDirection
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_tp_tracker():
    """Mock TPPerformanceTracker"""
    tracker = Mock()
    return tracker


@pytest.fixture
def mock_metrics_btc():
    """Mock TPMetrics for BTC"""
    return TPMetrics(
        strategy_id="RL_V3",
        symbol="BTCUSDT",
        tp_attempts=50,
        tp_hits=32,
        tp_misses=18,
        tp_hit_rate=0.64,
        total_slippage_pct=0.256,
        avg_slippage_pct=0.008,
        max_slippage_pct=0.025,
        avg_time_to_tp_minutes=35.2,
        total_tp_profit_usd=1580.00,
        avg_tp_profit_usd=49.38,
        premature_exits=4,
        missed_opportunities_usd=180.00,
        last_updated=datetime(2025, 12, 10, 16, 0, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def mock_metrics_eth():
    """Mock TPMetrics for ETH"""
    return TPMetrics(
        strategy_id="RL_V3",
        symbol="ETHUSDT",
        tp_attempts=38,
        tp_hits=22,
        tp_misses=16,
        tp_hit_rate=0.579,
        avg_slippage_pct=0.011,
        avg_time_to_tp_minutes=41.5,
        total_tp_profit_usd=920.50,
        premature_exits=3,
        last_updated=datetime(2025, 12, 10, 15, 45, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def mock_metrics_sol():
    """Mock TPMetrics for SOL (poor performance)"""
    return TPMetrics(
        strategy_id="RL_V3",
        symbol="SOLUSDT",
        tp_attempts=25,
        tp_hits=8,
        tp_misses=17,
        tp_hit_rate=0.32,
        avg_slippage_pct=0.015,
        total_tp_profit_usd=120.00,
        premature_exits=7,
        last_updated=datetime(2025, 12, 10, 14, 30, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def mock_profile_trend():
    """Mock TP profile"""
    return TPProfile(
        name="TREND_DEFAULT",
        tp_legs=[
            TPProfileLeg(r_multiple=0.5, size_fraction=0.15, kind=TPKind.SOFT),
            TPProfileLeg(r_multiple=1.0, size_fraction=0.20, kind=TPKind.HARD),
            TPProfileLeg(r_multiple=2.0, size_fraction=0.30, kind=TPKind.HARD),
        ],
        trailing=TrailingProfile(
            callback_pct=0.020,
            activation_r=1.5,
            tightening_curve=[(3.0, 0.015), (5.0, 0.010)]
        ),
        description="Trend-following profile"
    )


@pytest.fixture
def mock_recommendation():
    """Mock TPAdjustmentRecommendation"""
    return TPAdjustmentRecommendation(
        strategy_id="RL_V3",
        symbol="BTCUSDT",
        profile_name="TREND_DEFAULT_ADJUSTED",
        current_profile_id="TREND_DEFAULT",
        direction=AdjustmentDirection.CLOSER,
        suggested_scale_factor=0.95,
        reason="Hit rate above target, bringing TPs closer",
        confidence=0.70,
        metrics_snapshot={},
        targets={}
    )


# ============================================================================
# Test: list_tp_entities()
# ============================================================================

@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_list_tp_entities_multiple_pairs(
    mock_get_tracker,
    mock_tp_tracker,
    mock_metrics_btc,
    mock_metrics_eth
):
    """Test list_tp_entities returns all tracked pairs"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc, mock_metrics_eth]
    mock_get_tracker.return_value = mock_tracker
    
    service = TPDashboardService()
    
    # Execute
    entities = service.list_tp_entities()
    
    # Verify
    assert len(entities) == 2
    assert entities[0].strategy_id == "RL_V3"
    assert entities[0].symbol == "BTCUSDT"
    assert entities[1].symbol == "ETHUSDT"


@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_list_tp_entities_empty(mock_get_tracker):
    """Test list_tp_entities returns empty list when no metrics"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = []
    mock_get_tracker.return_value = mock_tracker
    
    service = TPDashboardService()
    
    # Execute
    entities = service.list_tp_entities()
    
    # Verify
    assert len(entities) == 0


# ============================================================================
# Test: get_tp_dashboard_entry()
# ============================================================================

@patch('backend.services.dashboard.tp_dashboard_service.TPOptimizerV3')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_and_trailing_profile')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_tp_dashboard_entry_complete(
    mock_get_tracker,
    mock_get_profile,
    mock_optimizer_class,
    mock_metrics_btc,
    mock_profile_trend,
    mock_recommendation
):
    """Test get_tp_dashboard_entry returns complete entry"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, mock_profile_trend.trailing)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = mock_recommendation
    mock_optimizer_class.return_value = mock_optimizer
    
    service = TPDashboardService()
    
    # Execute
    entry = service.get_tp_dashboard_entry("RL_V3", "BTCUSDT")
    
    # Verify
    assert entry is not None
    assert entry.key.strategy_id == "RL_V3"
    assert entry.key.symbol == "BTCUSDT"
    
    # Check metrics
    assert entry.metrics.tp_hit_rate == 0.64
    assert entry.metrics.tp_attempts == 50
    assert entry.metrics.tp_hits == 32
    assert entry.metrics.avg_r_multiple is not None  # Should be calculated
    
    # Check profile
    assert entry.profile.profile_id == "TREND_DEFAULT"
    assert len(entry.profile.legs) == 3
    assert entry.profile.legs[0].label == "TP1"
    assert entry.profile.legs[0].r_multiple == 0.5
    assert entry.profile.trailing_profile_id is not None
    
    # Check recommendation
    assert entry.recommendation.has_recommendation is True
    assert entry.recommendation.suggested_scale_factor == 0.95
    assert entry.recommendation.direction == "CLOSER"


@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_tp_dashboard_entry_not_found(mock_get_tracker):
    """Test get_tp_dashboard_entry returns None when no metrics"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = []
    mock_get_tracker.return_value = mock_tracker
    
    service = TPDashboardService()
    
    # Execute
    entry = service.get_tp_dashboard_entry("UNKNOWN", "UNKNOWN")
    
    # Verify
    assert entry is None


@patch('backend.services.dashboard.tp_dashboard_service.TPOptimizerV3')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_and_trailing_profile')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_tp_dashboard_entry_no_recommendation(
    mock_get_tracker,
    mock_get_profile,
    mock_optimizer_class,
    mock_metrics_btc,
    mock_profile_trend
):
    """Test get_tp_dashboard_entry handles no recommendation"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None  # No recommendation
    mock_optimizer_class.return_value = mock_optimizer
    
    service = TPDashboardService()
    
    # Execute
    entry = service.get_tp_dashboard_entry("RL_V3", "BTCUSDT")
    
    # Verify
    assert entry is not None
    assert entry.recommendation.has_recommendation is False
    assert entry.recommendation.suggested_scale_factor is None


@patch('backend.services.dashboard.tp_dashboard_service.TPOptimizerV3')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_and_trailing_profile')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_tp_dashboard_entry_profile_error(
    mock_get_tracker,
    mock_get_profile,
    mock_optimizer_class,
    mock_metrics_btc
):
    """Test get_tp_dashboard_entry handles profile lookup errors"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    # Profile lookup fails
    mock_get_profile.side_effect = Exception("Profile not found")
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None
    mock_optimizer_class.return_value = mock_optimizer
    
    service = TPDashboardService()
    
    # Execute
    entry = service.get_tp_dashboard_entry("RL_V3", "BTCUSDT")
    
    # Verify - entry still created with fallback profile
    assert entry is not None
    assert entry.profile.profile_id == "UNKNOWN"
    assert len(entry.profile.legs) == 0


# ============================================================================
# Test: get_top_best_and_worst()
# ============================================================================

@patch('backend.services.dashboard.tp_dashboard_service.TPOptimizerV3')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_and_trailing_profile')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_top_best_and_worst_ranking(
    mock_get_tracker,
    mock_get_profile,
    mock_optimizer_class,
    mock_metrics_btc,
    mock_metrics_eth,
    mock_metrics_sol,
    mock_profile_trend
):
    """Test get_top_best_and_worst ranks correctly"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [
        mock_metrics_btc,  # Best: 64% hit, $1580 profit
        mock_metrics_eth,  # Medium: 58% hit, $920 profit
        mock_metrics_sol   # Worst: 32% hit, $120 profit
    ]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None
    mock_optimizer_class.return_value = mock_optimizer
    
    service = TPDashboardService()
    
    # Execute
    summary = service.get_top_best_and_worst(limit=2)
    
    # Verify
    assert summary.total_entries == 3
    assert len(summary.best) <= 2
    assert len(summary.worst) <= 2
    
    # Best should be BTC (highest hit rate and profit)
    assert summary.best[0].key.symbol == "BTCUSDT"
    
    # Worst should be SOL (lowest hit rate and profit)
    assert summary.worst[0].key.symbol == "SOLUSDT"


@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_top_best_and_worst_empty(mock_get_tracker):
    """Test get_top_best_and_worst handles no metrics"""
    # Setup
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = []
    mock_get_tracker.return_value = mock_tracker
    
    service = TPDashboardService()
    
    # Execute
    summary = service.get_top_best_and_worst(limit=10)
    
    # Verify
    assert summary.total_entries == 0
    assert len(summary.best) == 0
    assert len(summary.worst) == 0


@patch('backend.services.dashboard.tp_dashboard_service.TPOptimizerV3')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_and_trailing_profile')
@patch('backend.services.dashboard.tp_dashboard_service.get_tp_tracker')
def test_get_top_best_and_worst_limit_respected(
    mock_get_tracker,
    mock_get_profile,
    mock_optimizer_class,
    mock_profile_trend
):
    """Test get_top_best_and_worst respects limit parameter"""
    # Setup - create 20 mock metrics
    mock_metrics_list = []
    for i in range(20):
        metrics = TPMetrics(
            strategy_id="RL_V3",
            symbol=f"PAIR{i}USDT",
            tp_attempts=30,
            tp_hits=15 + i,  # Varying hit rates
            tp_misses=15 - i,
            tp_hit_rate=(15 + i) / 30,
            total_tp_profit_usd=500.0 + (i * 50),
            last_updated=datetime.now(timezone.utc)
        )
        mock_metrics_list.append(metrics)
    
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = mock_metrics_list
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None
    mock_optimizer_class.return_value = mock_optimizer
    
    service = TPDashboardService()
    
    # Execute
    summary = service.get_top_best_and_worst(limit=5)
    
    # Verify
    assert summary.total_entries == 20
    assert len(summary.best) == 5
    assert len(summary.worst) == 5


# ============================================================================
# Test: Performance Score Calculation
# ============================================================================

def test_calculate_performance_score():
    """Test performance score calculation logic"""
    from backend.services.dashboard.tp_dashboard_service import (
        TPDashboardService,
        TPDashboardKey,
        TPDashboardEntry,
        TPDashboardProfile,
        TPDashboardMetrics,
        TPDashboardRecommendation
    )
    
    service = TPDashboardService()
    
    # Create mock entries with different characteristics
    # High hit rate, high R, high profit
    high_performer = TPDashboardEntry(
        key=TPDashboardKey(strategy_id="RL_V3", symbol="BTCUSDT"),
        profile=TPDashboardProfile(
            profile_id="TREND_DEFAULT",
            legs=[],
            trailing_profile_id=None
        ),
        metrics=TPDashboardMetrics(
            tp_hit_rate=0.75,
            tp_attempts=50,
            tp_hits=38,
            tp_misses=12,
            avg_r_multiple=2.5,
            total_tp_profit_usd=10000.0
        ),
        recommendation=TPDashboardRecommendation(has_recommendation=False)
    )
    
    # Low hit rate, low R, low profit
    low_performer = TPDashboardEntry(
        key=TPDashboardKey(strategy_id="RL_V3", symbol="SOLUSDT"),
        profile=TPDashboardProfile(
            profile_id="RANGE_DEFAULT",
            legs=[],
            trailing_profile_id=None
        ),
        metrics=TPDashboardMetrics(
            tp_hit_rate=0.30,
            tp_attempts=30,
            tp_hits=9,
            tp_misses=21,
            avg_r_multiple=1.0,
            total_tp_profit_usd=100.0
        ),
        recommendation=TPDashboardRecommendation(has_recommendation=False)
    )
    
    # Calculate scores
    high_score = service._calculate_performance_score(high_performer)
    low_score = service._calculate_performance_score(low_performer)
    
    # Verify
    assert high_score > low_score
    assert high_score > 50.0  # Should be significantly positive
    assert low_score < 45.0   # Should be lower (adjusted threshold)
