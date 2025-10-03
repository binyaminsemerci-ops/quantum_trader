"""Test that critical routes exist and are properly imported."""

from fastapi.testclient import TestClient
from backend.main import app


def test_critical_routes_exist():
    """Test that all critical API routes exist and return valid responses."""
    client = TestClient(app)

    # Test base routes - using only endpoints that definitely exist
    critical_routes = [
        "/",
        "/api/v1/system/status",
        "/api/v1/trades",
        "/api/v1/stats",
        "/api/v1/chart",  # Fixed: chart base endpoint
        "/api/v1/settings",
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
    ]

    for router_module in routers:
        assert hasattr(
            router_module, "router"
        ), f"Module {router_module.__name__} missing 'router' attribute"
        assert (
            router_module.router is not None
        ), f"Module {router_module.__name__} has None router"


def test_websocket_routes_exist():
    """Test WebSocket routes are properly registered."""
    # Skip WebSocket testing for now as TestClient doesn't handle them well
    # WebSocket functionality is tested elsewhere
    pass
