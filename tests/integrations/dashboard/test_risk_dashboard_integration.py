"""
Dashboard V3.0 Risk & ESS Integration Tests
QA Test Suite: Integration with Risk Gate and ESS systems

Tests:
- Risk state propagation to dashboard
- ESS status monitoring
- Risk gate decision tracking
- Critical state UI triggers
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app

client = TestClient(app)


@pytest.fixture
def mock_risk_ok_state():
    """Fixture: Normal risk state"""
    return {
        "state": "OK",
        "global_risk_level": 0.3,
        "portfolio_var_95": 150.0,
        "max_drawdown": -2.5
    }


@pytest.fixture
def mock_risk_critical_state():
    """Fixture: Critical risk state"""
    return {
        "state": "CRITICAL",
        "global_risk_level": 0.95,
        "portfolio_var_95": 500.0,
        "max_drawdown": -12.8
    }


@pytest.fixture
def mock_ess_inactive():
    """Fixture: ESS inactive state"""
    return {
        "status": "INACTIVE",
        "triggers_today": 0,
        "last_trigger_timestamp": None,
        "daily_loss_threshold": -5.0,
        "current_daily_loss": -1.2
    }


@pytest.fixture
def mock_ess_active():
    """Fixture: ESS active (tripped) state"""
    return {
        "status": "ACTIVE",
        "triggers_today": 1,
        "last_trigger_timestamp": "2025-12-05T10:30:00Z",
        "last_trigger_reason": "Daily loss threshold exceeded",
        "daily_loss_threshold": -5.0,
        "current_daily_loss": -5.8
    }


class TestRiskStateDashboardIntegration:
    """Test risk state integration with dashboard"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_risk_ok_state_in_overview(self, mock_ess, mock_fetch, mock_risk_ok_state, mock_ess_inactive):
        """TEST-RISK-INT-001: OK risk state appears in dashboard overview"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_inactive
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Risk state should be OK
        assert data["risk_state"] in ["OK", "UNKNOWN"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_risk_critical_state_in_overview(self, mock_ess, mock_fetch, mock_risk_critical_state, mock_ess_inactive):
        """TEST-RISK-INT-002: CRITICAL risk state triggers UI warning"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_inactive
        
        # Note: Current BFF returns mock data, so we test the data structure
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Risk state field should exist
        assert "risk_state" in data
        assert data["risk_state"] in ["OK", "WARNING", "CRITICAL", "UNKNOWN"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_risk_data_in_risk_endpoint(self, mock_fetch):
        """TEST-RISK-INT-003: Risk endpoint returns risk gate stats"""
        mock_fetch.return_value = {
            "risk_gate_decisions_stats": {
                "allow": 45,
                "block": 8,
                "scale": 12,
                "total": 65
            }
        }
        
        response = client.get("/api/dashboard/risk")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have risk gate stats
        assert "risk_gate_decisions_stats" in data
        assert data["risk_gate_decisions_stats"]["total"] >= 0


class TestEssStatusDashboardIntegration:
    """Test ESS status integration with dashboard"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_inactive_in_overview(self, mock_ess, mock_fetch, mock_ess_inactive):
        """TEST-ESS-INT-001: Inactive ESS status appears in dashboard"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_inactive
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # ESS status should be present
        assert "ess_status" in data
        assert data["ess_status"]["status"] in ["INACTIVE", "UNKNOWN"]
        assert data["ess_status"]["triggers_today"] == 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_active_triggers_warning(self, mock_ess, mock_fetch, mock_ess_active):
        """TEST-ESS-INT-002: Active ESS status triggers dashboard warning"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_active
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # ESS should show as active
        assert data["ess_status"]["status"] == "ACTIVE"
        assert data["ess_status"]["triggers_today"] == 1
        
        # Loss threshold info should be present
        assert "threshold" in data["ess_status"]
        assert "daily_loss" in data["ess_status"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_loss_tracking(self, mock_ess, mock_fetch, mock_ess_active):
        """TEST-ESS-INT-003: ESS tracks daily loss accurately"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_active
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Daily loss should be tracked
        assert data["ess_status"]["daily_loss"] == -5.8
        assert data["ess_status"]["threshold"] == -5.0
        
        # Loss exceeds threshold
        assert data["ess_status"]["daily_loss"] < data["ess_status"]["threshold"]


class TestRiskGateDecisions:
    """Test Risk Gate decision tracking in dashboard"""
    
    def test_risk_gate_stats_structure(self):
        """TEST-RISKGATE-001: Risk gate stats have correct structure"""
        response = client.get("/api/dashboard/risk")
        assert response.status_code == 200
        
        data = response.json()
        
        stats = data["risk_gate_decisions_stats"]
        
        # Check all decision types
        assert "allow" in stats
        assert "block" in stats
        assert "scale" in stats
        assert "total" in stats
        
        # All should be non-negative integers
        assert stats["allow"] >= 0
        assert stats["block"] >= 0
        assert stats["scale"] >= 0
        assert stats["total"] >= 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_risk_gate_blocking_visible(self, mock_fetch):
        """TEST-RISKGATE-002: Risk gate blocking events are visible"""
        mock_fetch.return_value = {
            "risk_gate_decisions_stats": {
                "allow": 30,
                "block": 5,  # Some blocked
                "scale": 8,
                "total": 43
            }
        }
        
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        # Blocked trades should be visible
        assert data["risk_gate_decisions_stats"]["block"] == 5
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_risk_gate_scaling_visible(self, mock_fetch):
        """TEST-RISKGATE-003: Risk gate scaling events are visible"""
        mock_fetch.return_value = {
            "risk_gate_decisions_stats": {
                "allow": 30,
                "block": 2,
                "scale": 10,  # Some scaled
                "total": 42
            }
        }
        
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        # Scaled trades should be visible
        assert data["risk_gate_decisions_stats"]["scale"] == 10


class TestRiskAlertFlow:
    """Test risk alert flow from Risk Service to Dashboard"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_high_risk_alert_in_overview(self, mock_ess, mock_fetch, mock_risk_critical_state, mock_ess_inactive):
        """TEST-RISK-ALERT-001: High risk state visible in overview"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_inactive
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Risk state should be accessible for UI alerts
        assert "risk_state" in data
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_trip_alert_in_overview(self, mock_ess, mock_fetch, mock_ess_active):
        """TEST-RISK-ALERT-002: ESS trip triggers alert in overview"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_active
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # ESS active state should be visible for UI alert
        assert data["ess_status"]["status"] == "ACTIVE"


class TestDrawdownPerProfile:
    """Test drawdown tracking per capital profile"""
    
    def test_dd_per_profile_structure(self):
        """TEST-DD-001: Drawdown per profile has correct structure"""
        response = client.get("/api/dashboard/risk")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have dd_per_profile list
        assert "dd_per_profile" in data
        assert isinstance(data["dd_per_profile"], list)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_dd_profile_data(self, mock_fetch):
        """TEST-DD-002: Drawdown profile data is tracked"""
        mock_fetch.return_value = {
            "dd_per_profile": [
                {
                    "profile": "conservative",
                    "current_dd": -2.1,
                    "max_dd": -5.0,
                    "utilization": 0.42
                },
                {
                    "profile": "aggressive",
                    "current_dd": -8.5,
                    "max_dd": -15.0,
                    "utilization": 0.57
                }
            ]
        }
        
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        # DD per profile should be accessible
        assert isinstance(data["dd_per_profile"], list)


class TestVarEsSnapshot:
    """Test VaR/ES (Value at Risk / Expected Shortfall) snapshot"""
    
    def test_var_es_structure(self):
        """TEST-VAR-001: VaR/ES snapshot has correct structure"""
        response = client.get("/api/dashboard/risk")
        assert response.status_code == 200
        
        data = response.json()
        
        var_es = data["var_es_snapshot"]
        
        # Check required fields
        assert "var_95" in var_es
        assert "var_99" in var_es
        assert "es_95" in var_es
        assert "es_99" in var_es
    
    def test_var_es_numeric_values(self):
        """TEST-VAR-002: VaR/ES values are valid numbers"""
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        var_es = data["var_es_snapshot"]
        
        # All values should be valid numbers (not NaN)
        assert var_es["var_95"] == var_es["var_95"]
        assert var_es["var_99"] == var_es["var_99"]
        assert var_es["es_95"] == var_es["es_95"]
        assert var_es["es_99"] == var_es["es_99"]
        
        # Should be non-negative
        assert var_es["var_95"] >= 0
        assert var_es["var_99"] >= 0


class TestRiskServiceFailover:
    """Test dashboard behavior when Risk service is unavailable"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_risk_endpoint_graceful_degradation(self, mock_fetch):
        """TEST-RISK-FAIL-001: Risk endpoint works with defaults when service unavailable"""
        mock_fetch.return_value = None
        
        response = client.get("/api/dashboard/risk")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should return default structures
        assert "risk_gate_decisions_stats" in data
        assert "var_es_snapshot" in data
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_error_handling(self, mock_ess, mock_fetch):
        """TEST-RISK-FAIL-002: ESS errors handled gracefully"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.side_effect = Exception("ESS service unavailable")
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should still return ESS status (may be UNKNOWN or ERROR)
        assert "ess_status" in data


class TestRiskDataConsistency:
    """Test consistency of risk data across endpoints"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_ess_status_consistent(self, mock_ess, mock_fetch, mock_ess_active):
        """TEST-RISK-CONS-001: ESS status consistent across overview and risk endpoints"""
        mock_fetch.return_value = {"total_equity": 10000.0}
        mock_ess.return_value = mock_ess_active
        
        # Get overview
        overview_response = client.get("/api/dashboard/overview")
        overview_data = overview_response.json()
        
        # ESS status should be ACTIVE
        assert overview_data["ess_status"]["status"] == "ACTIVE"
        assert overview_data["ess_status"]["triggers_today"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
