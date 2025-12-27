"""
Dashboard V3.0 API Contract Tests - Overview Endpoint
QA Test Suite: Backend BFF /api/dashboard/overview

Tests:
- Response schema validation
- Numeric safety (no NaN values)
- GO-LIVE status reflection
- ESS status integration
- Portfolio data aggregation
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.main import app

client = TestClient(app)


@pytest.fixture
def mock_portfolio_data():
    """Fixture: Mock portfolio service response"""
    return {
        "total_equity": 816.61,
        "cash_balance": 892.94,
        "daily_pnl": -76.33,
        "unrealized_pnl": -76.33,
        "realized_pnl_today": 0.0,
        "total_exposure": 41498.5,
        "num_positions": 10,
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "size": 0.001,
                "entry_price": 95000.0,
                "current_price": 96360.0,
                "unrealized_pnl": -8.36,
                "leverage": 20
            }
        ]
    }


@pytest.fixture
def mock_ess_status():
    """Fixture: Mock ESS status"""
    return {
        "status": "INACTIVE",
        "triggers_today": 0,
        "last_trigger_timestamp": None,
        "daily_loss_threshold": -5.0,
        "current_daily_loss": 0.0
    }


class TestOverviewEndpointContract:
    """Contract tests for /api/dashboard/overview"""
    
    def test_endpoint_exists(self):
        """TEST-OV-001: Endpoint is registered and accessible"""
        response = client.get("/api/dashboard/overview")
        assert response.status_code in [200, 500], "Endpoint should exist"
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_response_schema_complete(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-002: Response contains all required fields"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Top-level required fields
        required_fields = [
            "timestamp",
            "environment",
            "go_live_active",
            "global_pnl",
            "positions_count",
            "exposure_per_exchange",
            "risk_state",
            "ess_status",
            "capital_profiles_summary"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_global_pnl_structure(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-003: global_pnl object has correct structure"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        pnl = data["global_pnl"]
        assert isinstance(pnl, dict)
        
        # Check all PnL fields
        assert "equity" in pnl
        assert "cash" in pnl
        assert "daily_pnl" in pnl
        assert "daily_pnl_pct" in pnl
        assert "weekly_pnl" in pnl
        assert "monthly_pnl" in pnl
        assert "total_pnl" in pnl
        
        # Verify types are numeric
        assert isinstance(pnl["equity"], (int, float))
        assert isinstance(pnl["cash"], (int, float))
        assert isinstance(pnl["daily_pnl"], (int, float))
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_no_nan_values(self, mock_ess, mock_fetch, mock_ess_status):
        """TEST-OV-004: Response never contains NaN values"""
        # Mock portfolio with missing/None values
        mock_fetch.return_value = {
            "total_equity": None,
            "cash_balance": None,
            "daily_pnl": None
        }
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that numeric fields are valid numbers (not NaN or None)
        pnl = data["global_pnl"]
        assert pnl["equity"] is not None
        assert pnl["cash"] is not None
        assert pnl["daily_pnl"] is not None
        assert pnl["daily_pnl_pct"] is not None
        
        # Verify they are actual numbers
        assert isinstance(pnl["equity"], (int, float))
        assert isinstance(pnl["cash"], (int, float))
        
        # Verify no NaN (NaN != NaN in Python)
        assert pnl["equity"] == pnl["equity"]  # NaN != NaN would fail
        assert pnl["cash"] == pnl["cash"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_environment_detection(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-005: Environment is correctly identified"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert data["environment"] in ["STAGING", "PRODUCTION", "TESTNET"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_positions_count_matches_portfolio(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-006: positions_count matches portfolio service data"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert data["positions_count"] == 10
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_status_structure(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-007: ESS status has correct structure"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        ess = data["ess_status"]
        assert isinstance(ess, dict)
        assert "status" in ess
        assert "triggers_today" in ess
        assert "daily_loss" in ess
        assert "threshold" in ess
        
        assert ess["status"] in ["ACTIVE", "INACTIVE", "UNKNOWN", "ERROR"]
        assert isinstance(ess["triggers_today"], int)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_exposure_per_exchange_list(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-008: exposure_per_exchange is a list"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert isinstance(data["exposure_per_exchange"], list)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_risk_state_valid(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-009: risk_state is a valid enum value"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert data["risk_state"] in ["OK", "WARNING", "CRITICAL", "UNKNOWN"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_go_live_status_boolean(self, mock_ess, mock_fetch, mock_portfolio_data, mock_ess_status):
        """TEST-OV-010: go_live_active is boolean"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert isinstance(data["go_live_active"], bool)


class TestOverviewErrorHandling:
    """Test error handling for overview endpoint"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_portfolio_service_timeout(self, mock_ess, mock_fetch, mock_ess_status):
        """TEST-OV-ERR-001: Graceful handling when portfolio service times out"""
        mock_fetch.return_value = None  # Simulate timeout
        mock_ess.return_value = mock_ess_status
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200  # Should still return 200 with defaults
        
        data = response.json()
        assert "global_pnl" in data
        assert data["global_pnl"]["equity"] >= 0  # Should have default value
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_service_error(self, mock_ess, mock_fetch, mock_portfolio_data):
        """TEST-OV-ERR-002: Graceful handling when ESS service fails"""
        mock_fetch.return_value = mock_portfolio_data
        mock_ess.side_effect = Exception("ESS connection failed")
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        assert "ess_status" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
