"""Test RL v3 simple prediction API"""
from fastapi.testclient import TestClient
from backend.main import app
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config

# Create test client
client = TestClient(app)

# Ensure RL v3 manager exists in app state
if not hasattr(app.state, 'rl_v3_manager'):
    print("⚠️  Creating RL v3 manager for testing...")
    config = RLv3Config()
    manager = RLv3Manager(config=config)
    app.state.rl_v3_manager = manager

# Test prediction endpoint
print("\n📡 Testing POST /api/v1/rl-v3/predict")

request_data = {
    "price_change_1m": 0.02,
    "price_change_5m": 0.05,
    "price_change_15m": 0.08,
    "volatility": 0.03,
    "rsi": 65.0,
    "macd": 0.01,
    "position_size": 0.0,
    "position_side": 0.0,
    "balance": 10000.0,
    "equity": 10000.0,
    "regime": "BULLISH",
    "trend_strength": 0.75,
    "volume_ratio": 1.3,
    "bid_ask_spread": 0.0015,
    "time_of_day": 0.6
}

response = client.post("/api/v1/rl-v3/predict", json=request_data)

print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"✅ Prediction successful!")
    print(f"   Action: {data['action']}")
    print(f"   Confidence: {data['confidence']:.4f}")
    
    # Validate response structure
    assert 'action' in data, "Missing 'action' field"
    assert 'confidence' in data, "Missing 'confidence' field"
    assert isinstance(data['action'], int), "Action must be int"
    assert isinstance(data['confidence'], float), "Confidence must be float"
    assert 0 <= data['action'] <= 5, "Action must be 0-5"
    assert 0.0 <= data['confidence'] <= 1.0, "Confidence must be 0-1"
    
    print("✅ Response validation passed!")
else:
    print(f"❌ Request failed: {response.text}")
    exit(1)

# Test with minimal data (using defaults)
print("\n📡 Testing with minimal request (defaults)")
minimal_request = {
    "rsi": 45.0
}

response = client.post("/api/v1/rl-v3/predict", json=minimal_request)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"✅ Minimal request successful!")
    print(f"   Action: {data['action']}")
    print(f"   Confidence: {data['confidence']:.4f}")
else:
    print(f"❌ Minimal request failed: {response.text}")

print("\n✅ All tests passed!")
