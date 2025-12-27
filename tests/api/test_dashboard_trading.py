"""
Dashboard V3.0 API Contract Tests - Trading Endpoint
QA Test Suite: Backend BFF /api/dashboard/trading

Tests:
- Response schema validation
- Position data structure
- Orders and signals lists
- Empty state handling
- Numeric safety
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from backend.main import app

client = TestClient(app)


@pytest.fixture
def mock_portfolio_with_positions():
    """Fixture: Mock portfolio with positions"""
    return {
        "total_equity": 816.61,
        "num_positions": 2,
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "size": 0.001,
                "entry_price": 95000.0,
                "current_price": 96360.0,
                "unrealized_pnl": -8.36,
                "unrealized_pnl_pct": -1.43,
                "exposure": 96.36,
                "leverage": 20
            },
            {
                "symbol": "ETHUSDT",
                "side": "BUY",
                "size": 0.1,
                "entry_price": 3500.0,
                "current_price": 3600.0,
                "unrealized_pnl": 10.0,
                "unrealized_pnl_pct": 2.86,
                "exposure": 360.0,
                "leverage": 10
            }
        ]
    }


@pytest.fixture
def mock_portfolio_empty():
    """Fixture: Mock portfolio with no positions"""
    return {
        "total_equity": 1000.0,
        "num_positions": 0,
        "positions": []
    }


class TestTradingEndpointContract:
    """Contract tests for /api/dashboard/trading"""
    
    def test_endpoint_exists(self):
        """TEST-TR-001: Endpoint is registered and accessible"""
        response = client.get("/api/dashboard/trading")
        assert response.status_code in [200, 500], "Endpoint should exist"
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_response_schema_complete(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-002: Response contains all required fields"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        
        # Required top-level fields
        required_fields = [
            "timestamp",
            "open_positions",
            "recent_orders",
            "recent_signals",
            "strategies_per_account"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_positions_list_structure(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-003: open_positions is a list with correct structure"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        assert isinstance(data["open_positions"], list)
        assert len(data["open_positions"]) == 2
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_position_object_fields(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-004: Each position has required fields"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        position = data["open_positions"][0]
        
        required_position_fields = [
            "symbol",
            "side",
            "size",
            "entry_price",
            "current_price",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "value",
            "leverage"
        ]
        
        for field in required_position_fields:
            assert field in position, f"Position missing field: {field}"
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_position_numeric_fields_valid(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-005: Position numeric fields are valid numbers"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        for position in data["open_positions"]:
            # Verify all numeric fields are actual numbers
            assert isinstance(position["size"], (int, float))
            assert isinstance(position["entry_price"], (int, float))
            assert isinstance(position["current_price"], (int, float))
            assert isinstance(position["unrealized_pnl"], (int, float))
            assert isinstance(position["leverage"], (int, float))
            
            # Verify no NaN
            assert position["size"] == position["size"]
            assert position["entry_price"] == position["entry_price"]
            assert position["unrealized_pnl"] == position["unrealized_pnl"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_position_side_valid(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-006: Position side is valid enum value"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        for position in data["open_positions"]:
            assert position["side"] in ["LONG", "SHORT", "BUY", "SELL"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_empty_positions_list(self, mock_fetch, mock_portfolio_empty):
        """TEST-TR-007: Empty positions list handled correctly"""
        mock_fetch.return_value = mock_portfolio_empty
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        assert isinstance(data["open_positions"], list)
        assert len(data["open_positions"]) == 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_recent_orders_list(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-008: recent_orders is a list"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        assert isinstance(data["recent_orders"], list)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_recent_signals_list(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-009: recent_signals is a list"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        assert isinstance(data["recent_signals"], list)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_strategies_per_account_list(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-010: strategies_per_account is a list"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        assert isinstance(data["strategies_per_account"], list)


class TestTradingErrorHandling:
    """Test error handling for trading endpoint"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_service_timeout(self, mock_fetch):
        """TEST-TR-ERR-001: Graceful handling when portfolio service times out"""
        mock_fetch.return_value = None
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["open_positions"], list)
        assert len(data["open_positions"]) == 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_malformed_position_data(self, mock_fetch):
        """TEST-TR-ERR-002: Handles malformed position data"""
        # Missing required fields
        mock_fetch.return_value = {
            "positions": [
                {"symbol": "BTCUSDT"}  # Missing most fields
            ]
        }
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data["open_positions"], list)
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_none_values_in_positions(self, mock_fetch):
        """TEST-TR-ERR-003: Handles None values in position fields"""
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
        assert response.status_code == 200
        
        data = response.json()
        if len(data["open_positions"]) > 0:
            pos = data["open_positions"][0]
            # Verify safe_float converted None to 0.0
            assert isinstance(pos["size"], (int, float))
            assert pos["size"] == pos["size"]  # Not NaN


class TestTradingDataMapping:
    """Test data mapping from Portfolio Service to Dashboard format"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_symbol_mapping(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-MAP-001: Symbol is correctly mapped"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        symbols = [pos["symbol"] for pos in data["open_positions"]]
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_pnl_calculation_preserved(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-MAP-002: PnL values are preserved from portfolio service"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        btc_pos = next(p for p in data["open_positions"] if p["symbol"] == "BTCUSDT")
        eth_pos = next(p for p in data["open_positions"] if p["symbol"] == "ETHUSDT")
        
        assert btc_pos["unrealized_pnl"] == -8.36
        assert eth_pos["unrealized_pnl"] == 10.0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_leverage_mapping(self, mock_fetch, mock_portfolio_with_positions):
        """TEST-TR-MAP-003: Leverage values are correctly mapped"""
        mock_fetch.return_value = mock_portfolio_with_positions
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        btc_pos = next(p for p in data["open_positions"] if p["symbol"] == "BTCUSDT")
        assert btc_pos["leverage"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
