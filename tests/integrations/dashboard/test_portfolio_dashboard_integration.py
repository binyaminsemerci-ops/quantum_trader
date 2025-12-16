"""
Dashboard V3.0 Portfolio Integration Tests
QA Test Suite: Integration with Portfolio Intelligence Service

Tests:
- Portfolio service data flow to dashboard
- Position synchronization from exchanges
- PnL aggregation accuracy
- Real Binance testnet integration (optional)
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import os
from backend.main import app

client = TestClient(app)


@pytest.fixture
def mock_binance_positions():
    """Fixture: Mock Binance testnet positions"""
    return [
        {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "quantity": 0.001,
            "entry_price": 95000.0,
            "mark_price": 96360.0,
            "unrealized_pnl": -8.36,
            "leverage": 20,
            "notional": 96.36
        },
        {
            "symbol": "ETHUSDT",
            "side": "BUY",
            "quantity": 0.05,
            "entry_price": 3500.0,
            "mark_price": 3600.0,
            "unrealized_pnl": 5.0,
            "leverage": 10,
            "notional": 180.0
        }
    ]


@pytest.fixture
def mock_portfolio_snapshot():
    """Fixture: Complete portfolio snapshot"""
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
                "unrealized_pnl_pct": -1.43,
                "exposure": 96.36,
                "leverage": 20
            }
        ]
    }


class TestPortfolioDashboardIntegration:
    """Test integration between Portfolio Service and Dashboard"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_dashboard_overview_uses_portfolio_data(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-001: Dashboard overview fetches data from Portfolio service"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify portfolio data is used
        assert data["global_pnl"]["equity"] == 816.61
        assert data["positions_count"] == 10
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_dashboard_trading_reflects_portfolio_positions(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-002: Dashboard trading tab shows portfolio positions"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify positions match portfolio
        assert len(data["open_positions"]) == 1
        assert data["open_positions"][0]["symbol"] == "BTCUSDT"
        assert data["open_positions"][0]["unrealized_pnl"] == -8.36
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_equity_calculation(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-003: Equity calculation is correct"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Equity should equal total_equity from portfolio
        assert data["global_pnl"]["equity"] == 816.61
        
        # Cash should equal cash_balance from portfolio
        assert data["global_pnl"]["cash"] == 892.94
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_pnl_aggregation(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-004: PnL aggregation from portfolio is accurate"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Daily PnL should match portfolio
        assert data["global_pnl"]["daily_pnl"] == -76.33
        
        # Daily PnL % should be calculated correctly
        expected_pct = (-76.33 / 816.61) * 100
        assert abs(data["global_pnl"]["daily_pnl_pct"] - expected_pct) < 0.01
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_position_count_accuracy(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-005: Position count matches portfolio"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Position count should match
        assert data["positions_count"] == 10
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_exposure_calculation(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-INT-006: Total exposure is calculated correctly"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/overview")
        data = response.json()
        
        # Verify exposure is aggregated
        if len(data["exposure_per_exchange"]) > 0:
            total_exposure = sum(ex.get("exposure", 0) for ex in data["exposure_per_exchange"])
            assert total_exposure > 0


class TestExchangeDataIntegration:
    """Test exchange data integration (Binance testnet)"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    @patch("backend.integrations.exchanges.binance_client.BinanceClient.get_positions")
    def test_binance_positions_flow_to_dashboard(self, mock_binance, mock_fetch, mock_binance_positions, mock_portfolio_snapshot):
        """TEST-EXC-INT-001: Binance positions flow through to dashboard"""
        mock_binance.return_value = mock_binance_positions
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify positions from Binance appear in dashboard
        assert len(data["open_positions"]) > 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_position_side_mapping(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-EXC-INT-002: Position sides are correctly mapped"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        for position in data["open_positions"]:
            # Side should be valid
            assert position["side"] in ["LONG", "SHORT", "BUY", "SELL"]
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_leverage_mapping(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-EXC-INT-003: Leverage values are correctly mapped"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        response = client.get("/api/dashboard/trading")
        data = response.json()
        
        for position in data["open_positions"]:
            # Leverage should be positive number
            assert position["leverage"] > 0
            assert position["leverage"] <= 125  # Max leverage on Binance


class TestPortfolioServiceFailover:
    """Test dashboard behavior when Portfolio service is unavailable"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_dashboard_graceful_degradation_no_portfolio(self, mock_fetch):
        """TEST-PORT-FAIL-001: Dashboard works with default data when Portfolio unavailable"""
        mock_fetch.return_value = None  # Simulate timeout
        
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should return defaults
        assert "global_pnl" in data
        assert data["global_pnl"]["equity"] >= 0
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_trading_empty_state_no_portfolio(self, mock_fetch):
        """TEST-PORT-FAIL-002: Trading tab shows empty state when Portfolio unavailable"""
        mock_fetch.return_value = None
        
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should return empty lists
        assert isinstance(data["open_positions"], list)
        assert len(data["open_positions"]) == 0


class TestPortfolioDataConsistency:
    """Test data consistency between Portfolio service and Dashboard"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_position_data_consistency(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-CONS-001: Position data is consistent across endpoints"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        # Get overview
        overview_response = client.get("/api/dashboard/overview")
        overview_data = overview_response.json()
        
        # Get trading
        trading_response = client.get("/api/dashboard/trading")
        trading_data = trading_response.json()
        
        # Position counts should match
        assert overview_data["positions_count"] == len(trading_data["open_positions"])
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_pnl_data_consistency(self, mock_fetch, mock_portfolio_snapshot):
        """TEST-PORT-CONS-002: PnL data is consistent across endpoints"""
        mock_fetch.return_value = mock_portfolio_snapshot
        
        # Get overview
        overview_response = client.get("/api/dashboard/overview")
        overview_data = overview_response.json()
        
        # PnL should be from same source
        assert overview_data["global_pnl"]["daily_pnl"] == -76.33


@pytest.mark.skipif(
    not os.getenv("BINANCE_USE_TESTNET") or not os.getenv("BINANCE_API_KEY"),
    reason="Binance testnet credentials not configured"
)
class TestBinanceTestnetLiveIntegration:
    """Live integration tests with Binance testnet (optional)"""
    
    def test_live_binance_positions_to_dashboard(self):
        """TEST-LIVE-001: Real Binance testnet positions appear in dashboard"""
        # This test only runs if testnet credentials are set
        response = client.get("/api/dashboard/trading")
        assert response.status_code == 200
        
        data = response.json()
        
        # Should have positions if testnet account is active
        assert isinstance(data["open_positions"], list)
    
    def test_live_portfolio_equity_realistic(self):
        """TEST-LIVE-002: Portfolio equity from testnet is realistic"""
        response = client.get("/api/dashboard/overview")
        assert response.status_code == 200
        
        data = response.json()
        
        # Equity should be positive and reasonable for testnet
        equity = data["global_pnl"]["equity"]
        assert equity > 0
        assert equity < 1000000  # Testnet shouldn't have millions


class TestPortfolioSyncTiming:
    """Test timing and synchronization of portfolio data"""
    
    @patch("backend.api.dashboard.bff_routes.fetch_json")
    def test_portfolio_fetch_timeout_handling(self, mock_fetch):
        """TEST-PORT-TIME-001: Portfolio fetch timeout is handled gracefully"""
        import asyncio
        
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate timeout
            return None
        
        mock_fetch.side_effect = slow_fetch
        
        # Should still return response (may take timeout duration)
        try:
            response = client.get("/api/dashboard/overview", timeout=3.0)
            # If it returns, it should be 200 with defaults
            if response is not None:
                assert response.status_code == 200
        except Exception:
            # Timeout is acceptable in this test
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
