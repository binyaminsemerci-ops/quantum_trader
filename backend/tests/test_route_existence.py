"""Test that critical routes exist and are properly imported."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app


def test_critical_routes_exist():
    """Test that all critical API routes exist and return valid responses."""
    client = TestClient(app)
    
    # Test base routes
    critical_routes = [
        "/",
        "/api/v1/system/status",
        "/api/v1/trades",
        "/api/v1/stats/summary",
        "/api/v1/chart/data",
        "/api/v1/settings",
        "/api/v1/prices/btc",
        "/api/v1/candles/BTCUSDT",
        "/api/v1/stress/summary",
        "/api/v1/watchlist",
        "/api/v1/portfolio/summary",
        "/api/v1/trading/status",
        "/api/v1/model/active",
        "/api/v1/metrics/",
    ]
    
    for route in critical_routes:
        response = client.get(route)
        # Routes should exist (not return 404) - they may return other status codes
        # due to missing data, auth, etc. but should be routed properly
        assert response.status_code != 404, f"Route {route} not found (404)"


def test_router_imports():
    """Test that all routers are properly imported in routes package."""
    from backend.routes import (
        trades,
        stats,
        chart,
        settings,
        binance,
        signals,
        prices,
        candles,
        stress,
        trade_logs,
        health,
        watchlist,
        layout,
        portfolio,
        trading,
        enhanced_api,
        ai_trading,
    )
    
    # Verify each router has the expected router attribute
    routers = [
        trades, stats, chart, settings, binance, signals, prices, 
        candles, stress, trade_logs, health, watchlist, layout, 
        portfolio, trading, enhanced_api, ai_trading
    ]
    
    for router_module in routers:
        assert hasattr(router_module, 'router'), f"Module {router_module.__name__} missing 'router' attribute"
        assert router_module.router is not None, f"Module {router_module.__name__} has None router"


def test_websocket_routes_exist():
    """Test WebSocket routes are properly registered."""
    client = TestClient(app)
    
    # WebSocket routes - these will return 426 (Upgrade Required) for GET requests
    # but should not return 404 if properly registered
    ws_routes = [
        "/api/v1/ws/dashboard",
        "/ws/dashboard",
    ]
    
    for route in ws_routes:
        response = client.get(route)
        # WebSocket routes should return 426 or similar, not 404
        assert response.status_code != 404, f"WebSocket route {route} not found (404)"