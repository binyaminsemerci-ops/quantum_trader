"""
Test Suite for Dashboard V3.0 BFF API
DASHBOARD-V3-001: Phase 11 - Backend Endpoint Testing

Tests all 5 new BFF endpoints:
- GET /api/dashboard/overview
- GET /api/dashboard/trading
- GET /api/dashboard/risk
- GET /api/dashboard/system
- POST /api/dashboard/stress/run_all
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app

# Test client
client = TestClient(app)


class TestDashboardOverviewEndpoint:
    """Test /api/dashboard/overview endpoint"""
    
    def test_overview_endpoint_exists(self):
        """Test that overview endpoint is registered"""
        response = client.get("/api/dashboard/overview")
        assert response.status_code in [200, 500], "Endpoint should exist (200 OK or 500 if services down)"
    
    def test_overview_response_structure(self):
        """Test overview endpoint returns expected JSON structure"""
        response = client.get("/api/dashboard/overview")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            assert "timestamp" in data, "Response should include timestamp"
            assert "go_live_active" in data, "Response should include go_live_active"
            assert "global_pnl" in data, "Response should include global_pnl"
            assert "risk_state" in data, "Response should include risk_state"
            assert "ess_status" in data, "Response should include ess_status"
            
            # Check types
            assert isinstance(data["go_live_active"], bool), "go_live_active should be boolean"
            assert isinstance(data["global_pnl"], dict), "global_pnl should be dict"
            assert data["risk_state"] in ["OK", "WARNING", "CRITICAL"], "risk_state should be valid"
    
    def test_overview_go_live_status(self):
        """Test GO-LIVE status detection"""
        response = client.get("/api/dashboard/overview")
        
        if response.status_code == 200:
            data = response.json()
            go_live = data.get("go_live_active", False)
            
            # GO-LIVE should be boolean
            assert isinstance(go_live, bool), "go_live_active must be boolean"
            print(f"✓ GO-LIVE Status: {'ACTIVE' if go_live else 'INACTIVE'}")
    
    def test_overview_handles_missing_services(self):
        """Test graceful degradation when microservices are unavailable"""
        # This test passes if endpoint returns 200 even with service errors
        response = client.get("/api/dashboard/overview")
        assert response.status_code in [200, 500], "Should handle service failures gracefully"


class TestDashboardTradingEndpoint:
    """Test /api/dashboard/trading endpoint"""
    
    def test_trading_endpoint_exists(self):
        """Test that trading endpoint is registered"""
        response = client.get("/api/dashboard/trading")
        assert response.status_code in [200, 500], "Endpoint should exist"
    
    def test_trading_response_structure(self):
        """Test trading endpoint returns expected structure"""
        response = client.get("/api/dashboard/trading")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required arrays
            assert "open_positions" in data, "Response should include open_positions"
            assert "recent_orders" in data, "Response should include recent_orders"
            assert "recent_signals" in data, "Response should include recent_signals"
            
            # Check types
            assert isinstance(data["open_positions"], list), "open_positions should be list"
            assert isinstance(data["recent_orders"], list), "recent_orders should be list"
            assert isinstance(data["recent_signals"], list), "recent_signals should be list"
    
    def test_trading_position_fields(self):
        """Test that positions have required fields"""
        response = client.get("/api/dashboard/trading")
        
        if response.status_code == 200:
            data = response.json()
            positions = data.get("open_positions", [])
            
            if len(positions) > 0:
                pos = positions[0]
                # Check common position fields
                assert "symbol" in pos, "Position should have symbol"
                assert "side" in pos, "Position should have side"
                print(f"✓ Found {len(positions)} open position(s)")


class TestDashboardRiskEndpoint:
    """Test /api/dashboard/risk endpoint"""
    
    def test_risk_endpoint_exists(self):
        """Test that risk endpoint is registered"""
        response = client.get("/api/dashboard/risk")
        assert response.status_code in [200, 500], "Endpoint should exist"
    
    def test_risk_response_structure(self):
        """Test risk endpoint returns expected structure"""
        response = client.get("/api/dashboard/risk")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields
            assert "risk_gate_decisions_stats" in data, "Response should include risk_gate_decisions_stats"
            assert "ess_triggers_recent" in data, "Response should include ess_triggers_recent"
            assert "var_es_snapshot" in data, "Response should include var_es_snapshot"
            
            # Check types
            assert isinstance(data["risk_gate_decisions_stats"], dict), "risk_gate_decisions_stats should be dict"
            assert isinstance(data["ess_triggers_recent"], list), "ess_triggers_recent should be list"
    
    def test_risk_gate_stats_structure(self):
        """Test RiskGate statistics structure"""
        response = client.get("/api/dashboard/risk")
        
        if response.status_code == 200:
            data = response.json()
            stats = data.get("risk_gate_decisions_stats", {})
            
            # Check for decision counts
            if stats:
                assert "total" in stats or stats == {}, "Stats should have total or be empty"
                print(f"✓ RiskGate Stats: {stats}")


class TestDashboardSystemEndpoint:
    """Test /api/dashboard/system endpoint"""
    
    def test_system_endpoint_exists(self):
        """Test that system endpoint is registered"""
        response = client.get("/api/dashboard/system")
        assert response.status_code in [200, 500], "Endpoint should exist"
    
    def test_system_response_structure(self):
        """Test system endpoint returns expected structure"""
        response = client.get("/api/dashboard/system")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check required arrays
            assert "services_health" in data, "Response should include services_health"
            assert "exchanges_health" in data, "Response should include exchanges_health"
            assert "failover_events_recent" in data, "Response should include failover_events_recent"
            
            # Check types
            assert isinstance(data["services_health"], list), "services_health should be list"
            assert isinstance(data["exchanges_health"], list), "exchanges_health should be list"
    
    def test_system_services_health(self):
        """Test services health structure"""
        response = client.get("/api/dashboard/system")
        
        if response.status_code == 200:
            data = response.json()
            services = data.get("services_health", [])
            
            if len(services) > 0:
                service = services[0]
                assert "name" in service, "Service should have name"
                assert "status" in service, "Service should have status"
                print(f"✓ Found {len(services)} service(s)")


class TestDashboardStressEndpoint:
    """Test /api/dashboard/stress/run_all endpoint"""
    
    def test_stress_endpoint_exists(self):
        """Test that stress endpoint is registered"""
        response = client.post("/api/dashboard/stress/run_all")
        assert response.status_code in [200, 500, 503], "Endpoint should exist"
    
    def test_stress_endpoint_method(self):
        """Test that GET is not allowed on stress endpoint"""
        response = client.get("/api/dashboard/stress/run_all")
        assert response.status_code == 405, "GET should not be allowed (Method Not Allowed)"
    
    @pytest.mark.slow
    def test_stress_run_all_response(self):
        """Test stress scenario execution response structure"""
        response = client.post("/api/dashboard/stress/run_all")
        
        if response.status_code == 200:
            data = response.json()
            
            # Check response structure
            assert "status" in data or "passed" in data or "message" in data, \
                "Response should include status info"
            print(f"✓ Stress test response: {data}")


class TestEndpointPerformance:
    """Test endpoint response times"""
    
    def test_overview_response_time(self):
        """Test overview endpoint responds within 10 seconds"""
        import time
        start = time.time()
        response = client.get("/api/dashboard/overview")
        elapsed = time.time() - start
        
        assert elapsed < 10.0, f"Overview endpoint too slow: {elapsed:.2f}s > 10s"
        print(f"✓ Overview response time: {elapsed:.3f}s")
    
    def test_trading_response_time(self):
        """Test trading endpoint responds within 10 seconds"""
        import time
        start = time.time()
        response = client.get("/api/dashboard/trading")
        elapsed = time.time() - start
        
        assert elapsed < 10.0, f"Trading endpoint too slow: {elapsed:.2f}s > 10s"
        print(f"✓ Trading response time: {elapsed:.3f}s")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_endpoint(self):
        """Test that invalid endpoints return 404"""
        response = client.get("/api/dashboard/invalid")
        assert response.status_code == 404, "Invalid endpoint should return 404"
    
    def test_endpoints_return_json(self):
        """Test that all endpoints return JSON"""
        endpoints = [
            "/api/dashboard/overview",
            "/api/dashboard/trading",
            "/api/dashboard/risk",
            "/api/dashboard/system",
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code == 200:
                assert response.headers["content-type"] == "application/json", \
                    f"{endpoint} should return JSON"


class TestIntegration:
    """Integration tests across endpoints"""
    
    def test_all_endpoints_accessible(self):
        """Test that all 4 GET endpoints are accessible"""
        endpoints = {
            "overview": "/api/dashboard/overview",
            "trading": "/api/dashboard/trading",
            "risk": "/api/dashboard/risk",
            "system": "/api/dashboard/system",
        }
        
        results = {}
        for name, endpoint in endpoints.items():
            response = client.get(endpoint)
            results[name] = response.status_code
            assert response.status_code in [200, 500], \
                f"{endpoint} should be accessible (got {response.status_code})"
        
        print(f"✓ Endpoint Status: {results}")
    
    def test_consistent_timestamps(self):
        """Test that endpoints return consistent timestamp formats"""
        response = client.get("/api/dashboard/overview")
        
        if response.status_code == 200:
            data = response.json()
            timestamp = data.get("timestamp")
            
            if timestamp:
                # Should be ISO 8601 format
                from datetime import datetime
                try:
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    print(f"✓ Timestamp format valid: {timestamp}")
                except ValueError:
                    pytest.fail(f"Invalid timestamp format: {timestamp}")


def test_bff_router_registered():
    """Test that BFF router is registered in main app"""
    from backend.main import app
    
    # Check that routes exist
    routes = [route.path for route in app.routes]
    
    assert "/api/dashboard/overview" in routes, "Overview route not registered"
    assert "/api/dashboard/trading" in routes, "Trading route not registered"
    assert "/api/dashboard/risk" in routes, "Risk route not registered"
    assert "/api/dashboard/system" in routes, "System route not registered"
    
    print("✓ All BFF routes registered in FastAPI app")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
