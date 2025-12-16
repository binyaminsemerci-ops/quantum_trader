"""
Tests for exception handling and error boundaries in the API.
"""

from fastapi.testclient import TestClient
from backend.main import app

ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}

client = TestClient(app)
client.headers.update(ADMIN_HEADERS)


def test_validation_error_handling():
    """Test validation error responses for malformed requests."""
    # Test invalid trade data
    invalid_payload = {
        "symbol": "",  # Empty symbol should fail validation
        "side": "INVALID",  # Invalid side should fail validation
        "qty": -1.0,  # Negative quantity should fail validation
        "price": 0,  # Zero price should fail validation
    }

    response = client.post("/trades", json=invalid_payload)

    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"] == "Validation Error"
    assert "details" in data
    assert "timestamp" in data
    assert "path" in data


def test_unsupported_symbol_error():
    """Test error handling for unsupported trading symbols."""
    payload = {"symbol": "INVALID_SYMBOL", "side": "BUY", "qty": 1.0, "price": 100.0}

    response = client.post("/trades", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Unsupported trading symbol" in data["message"]
    assert "timestamp" in data


def test_invalid_json_handling():
    """Test handling of invalid JSON in request body."""
    response = client.post(
        "/trades",
        content="invalid json",  # Not JSON
        headers={**ADMIN_HEADERS, "content-type": "application/json"},
    )

    assert response.status_code == 422
    data = response.json()
    assert "error" in data


def test_missing_fields_validation():
    """Test validation when required fields are missing."""
    incomplete_payload = {
        "symbol": "BTCUSDT",
        # Missing side, qty, price
    }

    response = client.post("/trades", json=incomplete_payload)

    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "details" in data
    assert len(data["details"]) > 0  # Should have validation errors


def test_http_404_error():
    """Test 404 error handling for non-existent endpoints."""
    response = client.get("/non-existent-endpoint")

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "timestamp" in data


def test_successful_error_recovery():
    """Test that the API recovers properly after errors."""
    # First, cause an error
    client.post("/trades", json={"invalid": "data"})

    # Then verify normal operations still work
    valid_payload = {"symbol": "BTCUSDT", "side": "BUY", "qty": 1.0, "price": 50000.0}

    response = client.post("/trades", json=valid_payload)
    assert response.status_code == 201

    # And that we can still fetch trades
    response = client.get("/trades")
    assert response.status_code == 200


def test_consistent_error_format():
    """Test that all errors follow consistent response format."""
    test_cases = [
        ("/non-existent", "get", {}, 404),
        ("/trades", "post", {"invalid": "data"}, 422),
        (
            "/trades",
            "post",
            {"symbol": "INVALID", "side": "BUY", "qty": 1, "price": 100},
            400,
        ),
    ]

    for endpoint, method, payload, expected_status in test_cases:
        if method == "get":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint, json=payload)

        assert response.status_code == expected_status

        data = response.json()
        # All errors should have these fields
        required_fields = ["error", "timestamp", "path"]
        for field in required_fields:
            assert field in data, f"Missing {field} in error response for {endpoint}"
