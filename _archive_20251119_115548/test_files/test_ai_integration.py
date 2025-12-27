# Test AI Integration
# Run this after backend is started to verify AI components

import requests
import json

BASE_URL = "http://localhost:8000"
ADMIN_TOKEN = "your-secret-admin-token"  # Update with actual token

headers = {
    "X-Admin-Token": ADMIN_TOKEN,
    "Content-Type": "application/json"
}

def test_ai_status():
    """Test AI live status endpoint."""
    print("\n=== Testing AI Status ===")
    try:
        resp = requests.get(f"{BASE_URL}/ai/live-status", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Model loaded: {data.get('model', {}).get('loaded')}")
        print(f"Model type: {data.get('model', {}).get('type')}")
        
        predictions = data.get('predictions', {})
        total = predictions.get('total', 0)
        buy = predictions.get('buy_signals', 0)
        sell = predictions.get('sell_signals', 0)
        hold = predictions.get('hold_signals', 0)
        
        print(f"\nPredictions: {total} total")
        if total > 0:
            print(f"  BUY:  {buy:2d} ({100*buy/total:.1f}%)")
            print(f"  SELL: {sell:2d} ({100*sell/total:.1f}%)")
            print(f"  HOLD: {hold:2d} ({100*hold/total:.1f}%)")
        
        recent = data.get('recent_signals', [])
        if recent:
            print(f"\nTop {len(recent)} signals:")
            for sig in recent[:5]:
                print(f"  {sig['symbol']:12s} weight={sig['weight']:.3f} score={sig['score']:.2f}")
                print(f"    reason: {sig['reason'][:70]}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_ai_predict():
    """Test AI prediction endpoint."""
    print("\n=== Testing AI Predictions ===")
    try:
        payload = {
            "symbols": ["BTCUSDC", "ETHUSDC", "SOLUSDC"]
        }
        resp = requests.post(f"{BASE_URL}/ai/predict", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        print(f"Predictions for {len(data.get('predictions', []))} symbols:")
        for pred in data.get('predictions', []):
            symbol = pred.get('symbol')
            action = pred.get('action')
            confidence = pred.get('confidence', 0)
            score = pred.get('score', 0)
            print(f"  {symbol:12s} {action:4s} confidence={confidence:.2f} score={score:+.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_liquidity_status():
    """Test liquidity refresh status."""
    print("\n=== Testing Liquidity Status ===")
    try:
        resp = requests.get(f"{BASE_URL}/liquidity", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Universe size: {data.get('universe_size', 0)}")
        print(f"Selection size: {data.get('selection_size', 0)}")
        print(f"Last refresh: {data.get('last_refresh_time', 'never')}")
        
        if data.get('top_symbols'):
            print(f"\nTop symbols:")
            for sym in data['top_symbols'][:5]:
                print(f"  {sym}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_risk_status():
    """Test risk guard status."""
    print("\n=== Testing Risk Status ===")
    try:
        resp = requests.get(f"{BASE_URL}/risk", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        state = data.get('state', {})
        print(f"Kill-switch: {'ENABLED' if state.get('kill_switch_enabled') else 'DISABLED'}")
        print(f"Daily loss: {state.get('daily_loss', 0):.2f} / {state.get('max_daily_loss', 0):.2f}")
        print(f"Gross exposure: {state.get('gross_exposure', 0):.2f} / {state.get('max_gross_exposure', 0):.2f}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Quantum Trader AI Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("AI Status", test_ai_status),
        ("AI Predictions", test_ai_predict),
        ("Liquidity Status", test_liquidity_status),
        ("Risk Status", test_risk_status)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            break
        except Exception as e:
            print(f"\n{name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:7s} {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
