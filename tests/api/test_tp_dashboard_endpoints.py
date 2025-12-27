"""
Tests for TP Dashboard API Endpoints

Covers:
- GET /api/dashboard/tp/summary
- GET /api/dashboard/tp/{strategy_id}/{symbol}
- Edge cases: no metrics, missing profiles, optimizer errors
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app
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
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_metrics_btc():
    """Mock TPMetrics for BTC"""
    return TPMetrics(
        strategy_id="RL_V3",
        symbol="BTCUSDT",
        tp_attempts=45,
        tp_hits=28,
        tp_misses=17,
        tp_hit_rate=0.622,
        total_slippage_pct=0.224,
        avg_slippage_pct=0.008,
        max_slippage_pct=0.025,
        avg_time_to_tp_minutes=38.5,
        fastest_tp_minutes=12.0,
        slowest_tp_minutes=125.0,
        total_tp_profit_usd=1245.50,
        avg_tp_profit_usd=44.48,
        premature_exits=5,
        missed_opportunities_usd=230.0,
        last_updated=datetime(2025, 12, 10, 15, 30, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def mock_metrics_eth():
    """Mock TPMetrics for ETH"""
    return TPMetrics(
        strategy_id="RL_V3",
        symbol="ETHUSDT",
        tp_attempts=32,
        tp_hits=18,
        tp_misses=14,
        tp_hit_rate=0.563,
        avg_slippage_pct=0.012,
        avg_time_to_tp_minutes=42.3,
        total_tp_profit_usd=856.20,
        premature_exits=3,
        last_updated=datetime(2025, 12, 10, 14, 20, 0, tzinfo=timezone.utc)
    )


@pytest.fixture
def mock_profile_trend():
    """Mock TP profile for TREND regime"""
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
        description="Trend-following: Let profits run with wide trailing"
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
        reason="Hit rate 62% above target 70%, but avg R 1.61 below 1.80. Bringing TPs closer.",
        confidence=0.65,
        metrics_snapshot={
            'tp_hit_rate': 0.622,
            'avg_r_multiple': 1.61,
            'tp_attempts': 45,
            'tp_hits': 28,
            'tp_misses': 17
        },
        targets={}
    )


# ============================================================================
# Test: GET /api/dashboard/tp/summary
# ============================================================================

@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
@patch('backend.api.dashboard.tp_dashboard_router.TPOptimizerV3')
def test_get_tp_summary_with_multiple_pairs(
    mock_optimizer_class,
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc,
    mock_metrics_eth,
    mock_profile_trend
):
    """Test /api/dashboard/tp/summary with multiple strategy/symbol pairs"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc, mock_metrics_eth]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None  # No recommendations
    mock_optimizer_class.return_value = mock_optimizer
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert "rows" in data
    assert "total_pairs" in data
    assert "generated_at" in data
    
    assert data["total_pairs"] == 2
    assert len(data["rows"]) == 2
    
    # Check first row (BTC)
    btc_row = data["rows"][0]
    assert btc_row["strategy_id"] == "RL_V3"
    assert btc_row["symbol"] == "BTCUSDT"
    assert btc_row["metrics"]["tp_hit_rate"] == 0.622
    assert btc_row["metrics"]["tp_attempts"] == 45
    assert btc_row["profile"]["profile_id"] == "TREND_DEFAULT"
    assert len(btc_row["profile"]["legs"]) == 3
    assert btc_row["recommendation"] is None  # No recommendation in this test
    
    # Check second row (ETH)
    eth_row = data["rows"][1]
    assert eth_row["symbol"] == "ETHUSDT"
    assert eth_row["metrics"]["tp_attempts"] == 32


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
def test_get_tp_summary_no_metrics(mock_get_tracker, client):
    """Test /api/dashboard/tp/summary when no metrics exist"""
    # Setup mock
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = []
    mock_get_tracker.return_value = mock_tracker
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_pairs"] == 0
    assert len(data["rows"]) == 0


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
@patch('backend.api.dashboard.tp_dashboard_router.TPOptimizerV3')
def test_get_tp_summary_with_recommendation(
    mock_optimizer_class,
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc,
    mock_profile_trend,
    mock_recommendation
):
    """Test /api/dashboard/tp/summary includes recommendations when available"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = mock_recommendation
    mock_optimizer_class.return_value = mock_optimizer
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    
    # Check recommendation is included
    assert row["recommendation"] is not None
    rec = row["recommendation"]
    assert rec["profile_id"] == "TREND_DEFAULT"
    assert rec["suggested_scale_factor"] == 0.95
    assert rec["direction"] == "CLOSER"
    assert rec["confidence"] == 0.65
    assert "Hit rate" in rec["reason"]


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
def test_get_tp_summary_profile_missing(
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc
):
    """Test /api/dashboard/tp/summary when profile lookup fails"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    # Simulate profile not found
    mock_get_profile.side_effect = Exception("Profile not found")
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    
    # Metrics should still be present
    assert row["metrics"]["tp_hit_rate"] == 0.622
    # Profile should be null
    assert row["profile"] is None


# ============================================================================
# Test: GET /api/dashboard/tp/{strategy_id}/{symbol}
# ============================================================================

@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
@patch('backend.api.dashboard.tp_dashboard_router.TPOptimizerV3')
def test_get_tp_for_pair_success(
    mock_optimizer_class,
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc,
    mock_profile_trend
):
    """Test /api/dashboard/tp/{strategy_id}/{symbol} returns correct data"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = None
    mock_optimizer_class.return_value = mock_optimizer
    
    # Make request
    response = client.get("/api/dashboard/tp/RL_V3/BTCUSDT")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert data["strategy_id"] == "RL_V3"
    assert data["symbol"] == "BTCUSDT"
    assert data["regime"] == "NORMAL"
    
    # Check metrics
    metrics = data["metrics"]
    assert metrics["tp_hit_rate"] == 0.622
    assert metrics["tp_attempts"] == 45
    assert metrics["tp_hits"] == 28
    assert metrics["tp_misses"] == 17
    assert metrics["avg_slippage_pct"] == 0.008
    assert metrics["total_tp_profit_usd"] == 1245.50
    assert metrics["premature_exit_rate"] == pytest.approx(5 / 45, rel=1e-3)
    
    # Check profile
    profile = data["profile"]
    assert profile["profile_id"] == "TREND_DEFAULT"
    assert len(profile["legs"]) == 3
    assert profile["legs"][0]["label"] == "TP1"
    assert profile["legs"][0]["r_multiple"] == 0.5
    assert profile["legs"][0]["size_fraction"] == 0.15
    assert profile["legs"][0]["kind"] == "SOFT"
    
    # Check trailing
    trailing = profile["trailing"]
    assert trailing is not None
    assert trailing["callback_pct"] == 0.020
    assert trailing["activation_r"] == 1.5
    assert len(trailing["tightening_curve"]) == 2


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
def test_get_tp_for_pair_not_found(mock_get_tracker, client):
    """Test /api/dashboard/tp/{strategy_id}/{symbol} returns 404 when no metrics"""
    # Setup mock
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = []  # No metrics
    mock_get_tracker.return_value = mock_tracker
    
    # Make request
    response = client.get("/api/dashboard/tp/RL_V3/ADAUSDT")
    
    # Assertions
    assert response.status_code == 404
    assert "No TP metrics found" in response.json()["detail"]


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
@patch('backend.api.dashboard.tp_dashboard_router.TPOptimizerV3')
def test_get_tp_for_pair_with_recommendation(
    mock_optimizer_class,
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc,
    mock_profile_trend,
    mock_recommendation
):
    """Test /api/dashboard/tp/{strategy_id}/{symbol} includes recommendation"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.return_value = mock_recommendation
    mock_optimizer_class.return_value = mock_optimizer
    
    # Make request
    response = client.get("/api/dashboard/tp/RL_V3/BTCUSDT")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    # Check recommendation
    rec = data["recommendation"]
    assert rec is not None
    assert rec["profile_id"] == "TREND_DEFAULT"
    assert rec["suggested_scale_factor"] == 0.95
    assert rec["direction"] == "CLOSER"
    assert rec["metrics_snapshot"]["tp_hit_rate"] == 0.622


# ============================================================================
# Test: Edge Cases
# ============================================================================

@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
def test_get_tp_summary_tracker_error(mock_get_tracker, client):
    """Test /api/dashboard/tp/summary handles tracker errors gracefully"""
    # Setup mock to raise exception
    mock_get_tracker.side_effect = Exception("Tracker initialization failed")
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 500
    assert "Failed to generate TP summary" in response.json()["detail"]


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
@patch('backend.api.dashboard.tp_dashboard_router.TPOptimizerV3')
def test_get_tp_summary_optimizer_error_does_not_break(
    mock_optimizer_class,
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc,
    mock_profile_trend
):
    """Test /api/dashboard/tp/summary continues when optimizer fails"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    # Optimizer raises exception
    mock_optimizer = Mock()
    mock_optimizer.evaluate_profile.side_effect = Exception("Optimizer error")
    mock_optimizer_class.return_value = mock_optimizer
    
    # Make request
    response = client.get("/api/dashboard/tp/summary")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    # Row should still be present, just without recommendation
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert row["metrics"]["tp_hit_rate"] == 0.622
    assert row["profile"]["profile_id"] == "TREND_DEFAULT"
    assert row["recommendation"] is None  # Optimizer failed, no recommendation


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
def test_get_tp_for_pair_profile_no_trailing(
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_metrics_btc
):
    """Test /api/dashboard/tp/{strategy_id}/{symbol} handles profile without trailing"""
    # Setup mocks
    mock_tracker = Mock()
    mock_tracker.get_metrics.return_value = [mock_metrics_btc]
    mock_get_tracker.return_value = mock_tracker
    
    # Profile without trailing
    profile_no_trailing = TPProfile(
        name="RANGE_DEFAULT",
        tp_legs=[
            TPProfileLeg(r_multiple=0.3, size_fraction=0.30, kind=TPKind.SOFT),
            TPProfileLeg(r_multiple=0.6, size_fraction=0.40, kind=TPKind.HARD),
        ],
        trailing=None,  # No trailing
        description="Range mode"
    )
    mock_get_profile.return_value = (profile_no_trailing, None)
    
    # Make request
    response = client.get("/api/dashboard/tp/RL_V3/BTCUSDT")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    profile = data["profile"]
    assert profile["profile_id"] == "RANGE_DEFAULT"
    assert profile["trailing"] is None


@patch('backend.api.dashboard.tp_dashboard_router.get_tp_tracker')
@patch('backend.api.dashboard.tp_dashboard_router.get_tp_and_trailing_profile')
def test_metrics_with_zero_values(
    mock_get_profile,
    mock_get_tracker,
    client,
    mock_profile_trend
):
    """Test /api/dashboard/tp/{strategy_id}/{symbol} handles metrics with zero values"""
    # Setup mocks
    mock_tracker = Mock()
    
    # Metrics with many zero values
    sparse_metrics = TPMetrics(
        strategy_id="RL_V3",
        symbol="BTCUSDT",
        tp_attempts=10,
        tp_hits=5,
        tp_misses=5,
        tp_hit_rate=0.5,
        avg_slippage_pct=0.0,  # Zero
        max_slippage_pct=0.0,  # Zero
        avg_time_to_tp_minutes=0.0,  # Zero
        total_tp_profit_usd=0.0,  # Zero
        premature_exits=0,  # Zero
        last_updated=datetime.now(timezone.utc)
    )
    mock_tracker.get_metrics.return_value = [sparse_metrics]
    mock_get_tracker.return_value = mock_tracker
    
    mock_get_profile.return_value = (mock_profile_trend, None)
    
    # Make request
    response = client.get("/api/dashboard/tp/RL_V3/BTCUSDT")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    metrics = data["metrics"]
    # Zero values should be returned as None for cleaner frontend handling
    assert metrics["avg_slippage_pct"] is None
    assert metrics["max_slippage_pct"] is None
    assert metrics["avg_time_to_tp_minutes"] is None
    assert metrics["total_tp_profit_usd"] is None
    assert metrics["premature_exit_rate"] == 0.0  # This is a valid zero
