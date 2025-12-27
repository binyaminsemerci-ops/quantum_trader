"""
Comprehensive End-to-End Test Suite for Quantum Trader
Tests the complete trading flow from signal generation to execution
"""

import asyncio
import pytest
import httpx
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


class TestSystemHealth:
    """Test system health and availability"""

    async def test_backend_health(self):
        """Verify backend is healthy"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    async def test_scheduler_health(self):
        """Verify scheduler is running"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            data = response.json()
            assert "scheduler" in data
            assert data["scheduler"]["running"] is True


class TestAIEngine:
    """Test AI model and predictions"""

    async def test_model_ready(self):
        """Verify AI model is loaded and ready"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/ai/model/info", timeout=TIMEOUT)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "Ready"
            assert "accuracy" in data
            assert data["accuracy"] > 0.5  # Minimum 50% accuracy

    async def test_signal_generation(self):
        """Verify AI generates trading signals"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/api/ai/signals/latest", timeout=TIMEOUT
            )
            assert response.status_code == 200
            signals = response.json()
            assert isinstance(signals, list)
            assert len(signals) > 0

            # Check signal structure
            signal = signals[0]
            assert "symbol" in signal
            assert "type" in signal
            assert "confidence" in signal
            assert signal["type"] in ["BUY", "SELL", "HOLD"]
            assert 0 <= signal["confidence"] <= 1

    async def test_ensemble_predictions(self):
        """Verify ensemble model is working"""
        async with httpx.AsyncClient() as client:
            # Test ensemble prediction
            response = await client.get(
                f"{BASE_URL}/api/ai/predict?symbol=BTCUSDT", timeout=TIMEOUT
            )
            assert response.status_code == 200
            prediction = response.json()
            assert "action" in prediction
            assert "confidence" in prediction
            assert "models_used" in prediction


class TestMarketData:
    """Test market data ingestion and caching"""

    async def test_price_fetch(self):
        """Verify price data fetching"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/api/prices/latest?symbol=BTCUSDT", timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "price" in data
            assert data["price"] > 0

    async def test_candles_fetch(self):
        """Verify OHLCV candle data"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/api/candles?symbol=BTCUSDT&limit=100", timeout=TIMEOUT
            )
            assert response.status_code == 200
            data = response.json()
            assert "candles" in data
            assert len(data["candles"]) > 0

            candle = data["candles"][0]
            assert all(
                key in candle
                for key in ["timestamp", "open", "high", "low", "close", "volume"]
            )

    async def test_market_cache(self):
        """Verify market data is cached and updated"""
        async with httpx.AsyncClient() as client:
            # Fetch twice
            response1 = await client.get(
                f"{BASE_URL}/api/prices/latest?symbol=ETHUSDT", timeout=TIMEOUT
            )
            await asyncio.sleep(1)
            response2 = await client.get(
                f"{BASE_URL}/api/prices/latest?symbol=ETHUSDT", timeout=TIMEOUT
            )

            assert response1.status_code == 200
            assert response2.status_code == 200

            # Prices should be relatively close (within 5% for cached data)
            price1 = response1.json()["price"]
            price2 = response2.json()["price"]
            assert abs(price1 - price2) / price1 < 0.05


class TestRiskManagement:
    """Test risk controls and position sizing"""

    async def test_position_limits(self):
        """Verify position size limits are enforced"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/risk", timeout=TIMEOUT)
            assert response.status_code == 200
            risk = response.json()
            assert "max_position_size" in risk
            assert risk["max_position_size"] > 0

    async def test_exposure_tracking(self):
        """Verify exposure tracking"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/positions", timeout=TIMEOUT)
            assert response.status_code == 200
            positions = response.json()

            # Calculate total exposure
            total_exposure = sum(
                abs(pos.get("size", 0) * pos.get("entry_price", 0)) for pos in positions
            )

            # Check against risk limits
            risk_response = await client.get(f"{BASE_URL}/api/risk", timeout=TIMEOUT)
            risk = risk_response.json()

            # Total exposure should not exceed max
            max_exposure = risk.get("max_total_exposure", float("inf"))
            assert total_exposure <= max_exposure


class TestExecutionFlow:
    """Test trade execution flow"""

    async def test_execution_cycle(self):
        """Verify execution cycle completes successfully"""
        async with httpx.AsyncClient() as client:
            # Get health before
            health_before = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            assert health_before.status_code == 200

            # Wait for execution cycle (configured interval)
            await asyncio.sleep(10)

            # Get health after
            health_after = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            assert health_after.status_code == 200

            # Check execution ran
            data = health_after.json()
            assert "execution" in data
            assert data["execution"]["status"] in ["ok", "idle"]

    async def test_dry_run_mode(self):
        """Verify dry-run mode works correctly"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            data = response.json()

            # In dry-run, orders should be planned but skipped
            if "execution" in data:
                execution = data["execution"]
                if execution.get("orders_planned", 0) > 0:
                    assert execution.get("orders_skipped", 0) == execution.get(
                        "orders_planned", 0
                    )


class TestMetricsAndAnalytics:
    """Test metrics collection and reporting"""

    async def test_metrics_endpoint(self):
        """Verify metrics are collected"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/metrics", timeout=TIMEOUT)
            assert response.status_code == 200
            metrics = response.json()
            assert "total_trades" in metrics
            assert "win_rate" in metrics

    async def test_stats_overview(self):
        """Verify statistics overview"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/api/stats/overview", timeout=TIMEOUT)
            assert response.status_code == 200
            stats = response.json()
            assert "pnl" in stats
            assert "total_trades" in stats


class TestWebSocketConnection:
    """Test WebSocket real-time updates"""

    async def test_websocket_dashboard(self):
        """Verify WebSocket dashboard stream"""
        # Note: This requires websockets library
        try:
            import websockets

            async with websockets.connect(
                "ws://localhost:8000/ws/dashboard", timeout=10
            ) as websocket:
                # Receive first message
                message = await websocket.recv()
                import json

                data = json.loads(message)

                assert "stats" in data
                assert "trades" in data
                assert "logs" in data
        except ImportError:
            pytest.skip("websockets library not installed")


class TestPerformance:
    """Test system performance and latency"""

    async def test_api_response_time(self):
        """Verify API response times are acceptable"""
        async with httpx.AsyncClient() as client:
            start = datetime.now()
            response = await client.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            end = datetime.now()

            assert response.status_code == 200
            latency = (end - start).total_seconds()
            assert latency < 1.0  # Should respond within 1 second

    async def test_concurrent_requests(self):
        """Verify system handles concurrent requests"""
        async with httpx.AsyncClient() as client:
            # Make 10 concurrent requests
            tasks = [
                client.get(f"{BASE_URL}/api/prices/latest?symbol=BTCUSDT", timeout=TIMEOUT)
                for _ in range(10)
            ]
            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert all(r.status_code == 200 for r in responses)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
