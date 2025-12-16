"""
Tests for Dashboard TP API Routes

Covers:
- GET /api/dashboard/tp/entities
- GET /api/dashboard/tp/entry
- GET /api/dashboard/tp/summary
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from backend.main import app
from backend.services.dashboard.tp_dashboard_service import (
    TPDashboardKey,
    TPDashboardEntry,
    TPDashboardProfile,
    TPDashboardMetrics,
    TPDashboardRecommendation,
    TPDashboardSummary,
    TPDashboardProfileLeg
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_service():
    """Mock TPDashboardService"""
    return Mock()


@pytest.fixture
def mock_entities():
    """Mock entity list"""
    return [
        TPDashboardKey(strategy_id="RL_V3", symbol="BTCUSDT"),
        TPDashboardKey(strategy_id="RL_V3", symbol="ETHUSDT"),
        TPDashboardKey(strategy_id="RL_V3", symbol="SOLUSDT"),
    ]


@pytest.fixture
def mock_entry():
    """Mock dashboard entry"""
    return TPDashboardEntry(
        key=TPDashboardKey(strategy_id="RL_V3", symbol="BTCUSDT"),
        profile=TPDashboardProfile(
            profile_id="TREND_DEFAULT",
            legs=[
                TPDashboardProfileLeg(
                    label="TP1",
                    r_multiple=0.5,
                    size_fraction=0.15,
                    kind="SOFT"
                ),
                TPDashboardProfileLeg(
                    label="TP2",
                    r_multiple=1.0,
                    size_fraction=0.20,
                    kind="HARD"
                ),
            ],
            trailing_profile_id="TREND_TRAILING",
            description="Trend-following profile"
        ),
        metrics=TPDashboardMetrics(
            tp_hit_rate=0.64,
            tp_attempts=50,
            tp_hits=32,
            tp_misses=18,
            avg_r_multiple=1.56,
            avg_slippage_pct=0.008,
            max_slippage_pct=0.025,
            avg_time_to_tp_minutes=35.2,
            total_tp_profit_usd=1580.00,
            avg_tp_profit_usd=49.38,
            premature_exits=4,
            missed_opportunities_usd=180.00
        ),
        recommendation=TPDashboardRecommendation(
            has_recommendation=True,
            profile_id="TREND_DEFAULT",
            suggested_scale_factor=0.95,
            reason="Hit rate above target",
            confidence=0.70,
            direction="CLOSER"
        )
    )


@pytest.fixture
def mock_summary():
    """Mock summary with best/worst"""
    best_entry = TPDashboardEntry(
        key=TPDashboardKey(strategy_id="RL_V3", symbol="BTCUSDT"),
        profile=TPDashboardProfile(
            profile_id="TREND_DEFAULT",
            legs=[],
            trailing_profile_id=None
        ),
        metrics=TPDashboardMetrics(
            tp_hit_rate=0.70,
            tp_attempts=50,
            tp_hits=35,
            tp_misses=15,
            total_tp_profit_usd=2000.0
        ),
        recommendation=TPDashboardRecommendation(has_recommendation=False)
    )
    
    worst_entry = TPDashboardEntry(
        key=TPDashboardKey(strategy_id="RL_V3", symbol="SOLUSDT"),
        profile=TPDashboardProfile(
            profile_id="RANGE_DEFAULT",
            legs=[],
            trailing_profile_id=None
        ),
        metrics=TPDashboardMetrics(
            tp_hit_rate=0.30,
            tp_attempts=25,
            tp_hits=8,
            tp_misses=17,
            total_tp_profit_usd=120.0
        ),
        recommendation=TPDashboardRecommendation(has_recommendation=False)
    )
    
    return TPDashboardSummary(
        best=[best_entry],
        worst=[worst_entry],
        total_entries=5
    )


# ============================================================================
# Test: GET /api/dashboard/tp/entities
# ============================================================================

@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entities_success(mock_get_service, client, mock_entities):
    """Test GET /entities returns list of tracked pairs"""
    # Setup
    mock_service = Mock()
    mock_service.list_tp_entities.return_value = mock_entities
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/entities")
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    
    assert len(data) == 3
    assert data[0]["strategy_id"] == "RL_V3"
    assert data[0]["symbol"] == "BTCUSDT"
    assert data[1]["symbol"] == "ETHUSDT"
    assert data[2]["symbol"] == "SOLUSDT"


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entities_empty(mock_get_service, client):
    """Test GET /entities with no tracked pairs"""
    # Setup
    mock_service = Mock()
    mock_service.list_tp_entities.return_value = []
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/entities")
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entities_service_error(mock_get_service, client):
    """Test GET /entities handles service errors"""
    # Setup
    mock_service = Mock()
    mock_service.list_tp_entities.side_effect = Exception("Service error")
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/entities")
    
    # Verify
    assert response.status_code == 500
    assert "Failed to list TP entities" in response.json()["detail"]


# ============================================================================
# Test: GET /api/dashboard/tp/entry
# ============================================================================

@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entry_success(mock_get_service, client, mock_entry):
    """Test GET /entry returns complete dashboard entry"""
    # Setup
    mock_service = Mock()
    mock_service.get_tp_dashboard_entry.return_value = mock_entry
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get(
        "/api/dashboard/tp/entry",
        params={"strategy_id": "RL_V3", "symbol": "BTCUSDT"}
    )
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    
    # Check key
    assert data["key"]["strategy_id"] == "RL_V3"
    assert data["key"]["symbol"] == "BTCUSDT"
    
    # Check metrics
    assert data["metrics"]["tp_hit_rate"] == 0.64
    assert data["metrics"]["tp_attempts"] == 50
    assert data["metrics"]["tp_hits"] == 32
    assert data["metrics"]["avg_r_multiple"] == 1.56
    assert data["metrics"]["total_tp_profit_usd"] == 1580.00
    
    # Check profile
    assert data["profile"]["profile_id"] == "TREND_DEFAULT"
    assert len(data["profile"]["legs"]) == 2
    assert data["profile"]["legs"][0]["label"] == "TP1"
    assert data["profile"]["legs"][0]["r_multiple"] == 0.5
    assert data["profile"]["trailing_profile_id"] == "TREND_TRAILING"
    
    # Check recommendation
    assert data["recommendation"]["has_recommendation"] is True
    assert data["recommendation"]["suggested_scale_factor"] == 0.95
    assert data["recommendation"]["direction"] == "CLOSER"


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entry_not_found(mock_get_service, client):
    """Test GET /entry returns 404 when pair not tracked"""
    # Setup
    mock_service = Mock()
    mock_service.get_tp_dashboard_entry.return_value = None
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get(
        "/api/dashboard/tp/entry",
        params={"strategy_id": "UNKNOWN", "symbol": "UNKNOWN"}
    )
    
    # Verify
    assert response.status_code == 404
    assert "No TP metrics found" in response.json()["detail"]


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entry_missing_params(client):
    """Test GET /entry requires query parameters"""
    # Execute - missing both params
    response = client.get("/api/dashboard/tp/entry")
    
    # Verify
    assert response.status_code == 422  # Validation error


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_entry_service_error(mock_get_service, client):
    """Test GET /entry handles service errors"""
    # Setup
    mock_service = Mock()
    mock_service.get_tp_dashboard_entry.side_effect = Exception("Service error")
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get(
        "/api/dashboard/tp/entry",
        params={"strategy_id": "RL_V3", "symbol": "BTCUSDT"}
    )
    
    # Verify
    assert response.status_code == 500
    assert "Failed to get TP entry" in response.json()["detail"]


# ============================================================================
# Test: GET /api/dashboard/tp/summary
# ============================================================================

@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_summary_success(mock_get_service, client, mock_summary):
    """Test GET /summary returns best and worst entries"""
    # Setup
    mock_service = Mock()
    mock_service.get_top_best_and_worst.return_value = mock_summary
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/summary")
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    
    assert "best" in data
    assert "worst" in data
    assert "total_entries" in data
    
    assert data["total_entries"] == 5
    assert len(data["best"]) == 1
    assert len(data["worst"]) == 1
    
    # Check best entry
    best = data["best"][0]
    assert best["key"]["symbol"] == "BTCUSDT"
    assert best["metrics"]["tp_hit_rate"] == 0.70
    
    # Check worst entry
    worst = data["worst"][0]
    assert worst["key"]["symbol"] == "SOLUSDT"
    assert worst["metrics"]["tp_hit_rate"] == 0.30


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_summary_with_limit(mock_get_service, client, mock_summary):
    """Test GET /summary respects limit parameter"""
    # Setup
    mock_service = Mock()
    mock_service.get_top_best_and_worst.return_value = mock_summary
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/summary?limit=5")
    
    # Verify
    assert response.status_code == 200
    mock_service.get_top_best_and_worst.assert_called_once_with(limit=5)


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_summary_limit_validation(client):
    """Test GET /summary validates limit parameter"""
    # Execute - limit too high
    response = client.get("/api/dashboard/tp/summary?limit=100")
    
    # Verify
    assert response.status_code == 422  # Validation error
    
    # Execute - limit too low
    response = client.get("/api/dashboard/tp/summary?limit=0")
    
    # Verify
    assert response.status_code == 422  # Validation error


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_summary_empty(mock_get_service, client):
    """Test GET /summary with no tracked pairs"""
    # Setup
    empty_summary = TPDashboardSummary(
        best=[],
        worst=[],
        total_entries=0
    )
    mock_service = Mock()
    mock_service.get_top_best_and_worst.return_value = empty_summary
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/summary")
    
    # Verify
    assert response.status_code == 200
    data = response.json()
    
    assert data["total_entries"] == 0
    assert len(data["best"]) == 0
    assert len(data["worst"]) == 0


@patch('backend.api.routes.dashboard_tp.get_tp_dashboard_service')
def test_get_tp_summary_service_error(mock_get_service, client):
    """Test GET /summary handles service errors"""
    # Setup
    mock_service = Mock()
    mock_service.get_top_best_and_worst.side_effect = Exception("Service error")
    mock_get_service.return_value = mock_service
    
    # Execute
    response = client.get("/api/dashboard/tp/summary")
    
    # Verify
    assert response.status_code == 500
    assert "Failed to generate TP summary" in response.json()["detail"]
