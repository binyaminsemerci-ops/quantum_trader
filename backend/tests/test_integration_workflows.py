"""
Integration tests for main trading workflows in Quantum Trader.
Tests complete end-to-end scenarios and API interactions.
"""

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_complete_trading_workflow():
    """Test complete trading workflow from signal to execution."""

    # 1. Check system health
    health_response = client.get("/")
    assert health_response.status_code == 200
    assert "message" in health_response.json()

    # 2. Fetch current market prices
    prices_response = client.get("/prices/recent")
    assert prices_response.status_code == 200
    prices = prices_response.json()
    assert isinstance(prices, list)

    # 3. Get trading signals
    signals_response = client.get("/signals")
    assert signals_response.status_code == 200
    signals_data = signals_response.json()

    # API returns paginated format with 'items' key
    if isinstance(signals_data, dict) and "items" in signals_data:
        signals = signals_data["items"]
        assert isinstance(signals, list)
    else:
        signals = signals_data
        assert isinstance(signals, list)

    # 4. Execute a buy trade based on signals
    trade_payload = {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.1, "price": 45000.0}

    trade_response = client.post("/trades", json=trade_payload)
    assert trade_response.status_code == 201
    trade_data = trade_response.json()
    assert trade_data["symbol"] == "BTCUSDT"
    assert trade_data["side"] == "BUY"
    assert "id" in trade_data
    trade_id = trade_data["id"]

    # 5. Verify trade appears in trade history
    trades_response = client.get("/trades")
    assert trades_response.status_code == 200
    trades = trades_response.json()
    assert isinstance(trades, list)

    # Find our trade
    our_trade = next((t for t in trades if t["id"] == trade_id), None)
    assert our_trade is not None
    assert our_trade["symbol"] == "BTCUSDT"

    # 6. Check updated statistics
    stats_response = client.get("/stats/overview")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert isinstance(stats, dict)


def test_settings_persistence_workflow():
    """Test settings configuration and persistence workflow."""

    # 1. Get current settings
    get_response = client.get("/settings")
    assert get_response.status_code == 200
    # Verify settings are accessible but don't need to store them
    assert isinstance(get_response.json(), dict)

    # 2. Update settings
    new_settings = {
        "risk_percentage": 2.5,
        "max_position_size": 5000.0,
        "trading_enabled": True,
        "default_symbol": "ETHUSDT",
    }

    update_response = client.post("/settings", json=new_settings)
    assert update_response.status_code == 200

    # 3. Verify settings were saved
    verify_response = client.get("/settings")
    assert verify_response.status_code == 200
    saved_settings = verify_response.json()

    for key, value in new_settings.items():
        assert key in saved_settings
        assert saved_settings[key] == value


def test_market_data_workflow():
    """Test market data retrieval and chart generation workflow."""

    # 1. Get candle data for different timeframes
    timeframes = ["1h", "4h", "1d"]
    symbols = ["BTCUSDT", "ETHUSDT"]

    for symbol in symbols:
        for interval in timeframes:
            candles_response = client.get(
                f"/candles?symbol={symbol}&interval={interval}"
            )
            assert candles_response.status_code == 200

            candles_data = candles_response.json()

            # API returns {"candles": [...], "symbol": "..."} format
            if isinstance(candles_data, dict) and "candles" in candles_data:
                candles = candles_data["candles"]
                assert isinstance(candles, list)

                # Verify candle data structure
                if candles:  # If we have data
                    candle = candles[0]
                    required_fields = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    for field in required_fields:
                        assert field in candle

    # 2. Get chart data
    chart_response = client.get("/chart")
    assert chart_response.status_code == 200
    chart_data = chart_response.json()
    # Chart endpoint returns array of values
    assert isinstance(chart_data, (list, dict))


def test_error_recovery_workflow():
    """Test system behavior and recovery after various error conditions."""

    # 1. Cause validation errors and ensure recovery
    invalid_requests = [
        {"symbol": "", "side": "BUY", "qty": 1, "price": 100},
        {"symbol": "BTCUSDT", "side": "INVALID", "qty": 1, "price": 100},
        {"symbol": "BTCUSDT", "side": "BUY", "qty": -1, "price": 100},
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 1, "price": 0},
    ]

    for invalid_request in invalid_requests:
        response = client.post("/trades", json=invalid_request)
        assert response.status_code == 422

    # 2. Test system still works after errors
    valid_request = {"symbol": "BTCUSDT", "side": "SELL", "qty": 0.05, "price": 44000.0}

    response = client.post("/trades", json=valid_request)
    assert response.status_code == 201

    # 3. Verify all endpoints still respond
    endpoints = ["/", "/trades", "/stats/overview", "/prices/recent", "/signals"]

    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200


def test_concurrent_trading_operations():
    """Test handling of multiple concurrent trading operations."""

    # Simulate multiple simultaneous trade requests
    trade_requests = [
        {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.1, "price": 45000.0},
        {"symbol": "ETHUSDT", "side": "SELL", "qty": 1.0, "price": 3000.0},
        {"symbol": "ADAUSDT", "side": "BUY", "qty": 100.0, "price": 0.5},
    ]

    trade_ids = []

    # Execute trades
    for trade_request in trade_requests:
        response = client.post("/trades", json=trade_request)
        assert response.status_code == 201

        trade_data = response.json()
        trade_ids.append(trade_data["id"])

    # Verify all trades were created
    trades_response = client.get("/trades")
    assert trades_response.status_code == 200
    trades = trades_response.json()

    created_trades = [t for t in trades if t["id"] in trade_ids]
    assert len(created_trades) == len(trade_requests)

    # Verify each trade has correct data
    for i, trade in enumerate(created_trades):
        original_request = next(
            r for r in trade_requests if r["symbol"] == trade["symbol"]
        )
        assert trade["side"] == original_request["side"]
        assert trade["qty"] == original_request["qty"]
        assert trade["price"] == original_request["price"]


def test_api_performance_baseline():
    """Test basic API performance to ensure response times are reasonable."""
    import time

    endpoints_to_test = [
        ("/", "GET"),
        ("/trades", "GET"),
        ("/stats/overview", "GET"),
        ("/prices/recent", "GET"),
        ("/signals", "GET"),
    ]

    for endpoint, method in endpoints_to_test:
        start_time = time.time()

        if method == "GET":
            response = client.get(endpoint)

        end_time = time.time()
        response_time = end_time - start_time

        # Ensure response is successful
        assert response.status_code == 200

        # Ensure response time is reasonable (under 2 seconds)
        assert response_time < 2.0, f"Endpoint {endpoint} took {response_time:.2f}s"
