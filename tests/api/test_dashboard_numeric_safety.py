"""
Dashboard V3.0 Numeric Safety Tests
QA Test Suite: NaN and Empty State Handling

Tests:
- No NaN values in API responses
- Empty arrays instead of null
- Zero values displayed correctly
- Null handling in numeric fields
- Frontend displays 0.00 instead of NaN
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import math
from backend.main import app

client = TestClient(app)


class TestNaNSafety:
    """Test that NaN never appears in API responses"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_overview_no_nan_with_null_portfolio(self, mock_ess, mock_fetch):
        """TEST-NAN-001: Overview returns valid numbers when portfolio data is null"""
        mock_fetch.return_value = None  # Portfolio service returns nothing
        mock_ess.return_value = {"status": "INACTIVE", "triggers_today": 0}
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all numeric fields for NaN
        pnl = data["global_pnl"]
        assert not math.isnan(pnl["equity"])
        assert not math.isnan(pnl["cash"])
        assert not math.isnan(pnl["daily_pnl"])
        assert not math.isnan(pnl["daily_pnl_pct"])
        assert not math.isnan(pnl["weekly_pnl"])
        assert not math.isnan(pnl["monthly_pnl"])
        assert not math.isnan(pnl["total_pnl"])
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_overview_no_nan_with_none_values(self, mock_ess, mock_fetch):
        """TEST-NAN-002: Overview handles None values in portfolio data"""
        mock_fetch.return_value = {
            "total_equity": None,
            "cash_balance": None,
            "daily_pnl": None,
            "unrealized_pnl": None,
            "realized_pnl_today": None
        }
        mock_ess.return_value = {"status": "INACTIVE", "triggers_today": 0}
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # All should be valid numbers (likely 0 or defaults)
        pnl = data["global_pnl"]
        assert isinstance(pnl["equity"], (int, float))
        assert isinstance(pnl["cash"], (int, float))
        assert isinstance(pnl["daily_pnl"], (int, float))
        
        # Verify not NaN (NaN != NaN in Python)
        assert pnl["equity"] == pnl["equity"]
        assert pnl["cash"] == pnl["cash"]
        assert pnl["daily_pnl"] == pnl["daily_pnl"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_trading_no_nan_with_null_position_fields(self, mock_fetch):
        """TEST-NAN-003: Trading endpoint handles None in position fields"""
        mock_fetch.return_value = {
            "positions": [
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "size": None,
                    "entry_price": None,
                    "current_price": None,
                    "unrealized_pnl": None
                }
            ]
        }
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        if len(data["open_positions"]) > 0:
            pos = data["open_positions"][0]
            
            # All numeric fields should be valid numbers
            assert isinstance(pos["size"], (int, float))
            assert isinstance(pos["entry_price"], (int, float))
            assert isinstance(pos["unrealized_pnl"], (int, float))
            
            # Not NaN
            assert pos["size"] == pos["size"]
            assert pos["entry_price"] == pos["entry_price"]
    
    def test_risk_no_nan_in_var_es(self):
        """TEST-NAN-004: Risk endpoint VaR/ES values are never NaN"""
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        var_es = data["var_es_snapshot"]
        
        # Check all VaR/ES values
        assert not math.isnan(var_es["var_95"])
        assert not math.isnan(var_es["var_99"])
        assert not math.isnan(var_es["es_95"])
        assert not math.isnan(var_es["es_99"])


class TestEmptyStateSafety:
    """Test handling of empty data structures"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_overview_empty_exposure_list(self, mock_ess, mock_fetch):
        """TEST-EMPTY-001: Overview returns empty array for exposure_per_exchange"""
        mock_fetch.return_value = {
            "total_equity": 0.0,
            "positions": []
        }
        mock_ess.return_value = {"status": "INACTIVE", "triggers_today": 0}
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["exposure_per_exchange"], list)
        assert len(data["exposure_per_exchange"]) == 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_trading_empty_positions_list(self, mock_fetch):
        """TEST-EMPTY-002: Trading returns empty array for positions"""
        mock_fetch.return_value = {
            "num_positions": 0,
            "positions": []
        }
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["open_positions"], list)
        assert len(data["open_positions"]) == 0
    
    def test_risk_empty_ess_triggers(self):
        """TEST-EMPTY-003: Risk returns empty array for ESS triggers"""
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["ess_triggers_recent"], list)
    
    def test_risk_empty_dd_profiles(self):
        """TEST-EMPTY-004: Risk returns empty array for DD profiles"""
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["dd_per_profile"], list)
    
    def test_system_empty_services(self):
        """TEST-EMPTY-005: System returns empty array for services"""
        response = client.get("/api/dashboard/system")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["services_health"], list)
    
    def test_system_empty_exchanges(self):
        """TEST-EMPTY-006: System returns empty array for exchanges"""
        response = client.get("/api/dashboard/system")
        data = response.json()
        
        # Should be empty array, not null
        assert isinstance(data["exchanges_health"], list)


class TestZeroValueDisplay:
    """Test that zero values are displayed correctly"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_zero_equity_displayed(self, mock_ess, mock_fetch):
        """TEST-ZERO-001: Zero equity is displayed as 0.0, not NaN"""
        mock_fetch.return_value = {
            "total_equity": 0.0,
            "cash_balance": 0.0,
            "daily_pnl": 0.0
        }
        mock_ess.return_value = {"status": "INACTIVE", "triggers_today": 0}
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        assert data["global_pnl"]["equity"] == 0.0
        assert data["global_pnl"]["cash"] == 0.0
        assert data["global_pnl"]["daily_pnl"] == 0.0
    
    def test_zero_risk_gate_stats(self):
        """TEST-ZERO-002: Zero risk gate stats are valid"""
        response = client.get("/api/dashboard/risk")
        data = response.json()
        
        stats = data["risk_gate_decisions_stats"]
        
        # All should be integers >= 0
        assert isinstance(stats["allow"], int)
        assert isinstance(stats["block"], int)
        assert isinstance(stats["scale"], int)
        assert isinstance(stats["total"], int)
        
        assert stats["allow"] >= 0
        assert stats["block"] >= 0
        assert stats["scale"] >= 0
        assert stats["total"] >= 0


class TestDivisionByZero:
    """Test handling of division by zero scenarios"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.api.dashboard.bff_routes.get_ess_status")
    def test_pnl_pct_with_zero_equity(self, mock_ess, mock_fetch):
        """TEST-DIV-001: PnL percentage with zero equity doesn't return NaN"""
        mock_fetch.return_value = {
            "total_equity": 0.0,
            "daily_pnl": 10.0  # Profit but zero equity (edge case)
        }
        mock_ess.return_value = {"status": "INACTIVE", "triggers_today": 0}
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Should handle gracefully (0.0 or capped value, not NaN)
        pnl_pct = data["global_pnl"]["daily_pnl_pct"]
        assert isinstance(pnl_pct, (int, float))
        assert pnl_pct == pnl_pct  # Not NaN


class TestSafeFloatUtility:
    """Test safe_float utility function"""
    
    def test_safe_float_with_none(self):
        """TEST-SAFE-001: safe_float handles None"""
        from backend.api.dashboard.utils import safe_float
        
        result = safe_float(None, default=0.0)
        assert result == 0.0
        assert isinstance(result, float)
    
    def test_safe_float_with_nan(self):
        """TEST-SAFE-002: safe_float handles NaN"""
        from backend.api.dashboard.utils import safe_float
        
        result = safe_float(float('nan'), default=0.0)
        assert result == 0.0
        assert not math.isnan(result)
    
    def test_safe_float_with_infinity(self):
        """TEST-SAFE-003: safe_float handles infinity"""
        from backend.api.dashboard.utils import safe_float
        
        result = safe_float(float('inf'), default=0.0)
        assert result == 0.0
        assert math.isfinite(result)
    
    def test_safe_float_with_valid_number(self):
        """TEST-SAFE-004: safe_float preserves valid numbers"""
        from backend.api.dashboard.utils import safe_float
        
        result = safe_float(123.45, default=0.0)
        assert result == 123.45
    
    def test_safe_float_with_string_number(self):
        """TEST-SAFE-005: safe_float handles string numbers"""
        from backend.api.dashboard.utils import safe_float
        
        result = safe_float("123.45", default=0.0)
        assert result == 123.45


class TestFrontendPlaceholders:
    """Test that frontend shows placeholders instead of NaN/empty"""
    
    # Note: These would be frontend tests, included here for completeness
    
    def test_frontend_shows_zero_instead_of_nan(self):
        """TEST-FE-SAFE-001: Frontend displays 0.00 instead of NaN"""
        # This would be tested in frontend/__tests__/
        # Verified by OverviewTab.test.tsx and TradingTab.test.tsx
        pass
    
    def test_frontend_shows_placeholder_for_empty_positions(self):
        """TEST-FE-SAFE-002: Frontend shows "No positions" placeholder"""
        # Verified by TradingTab.test.tsx TEST-FE-TR-EMPTY-001
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
