#!/usr/bin/env python3
"""
Schema Validation Test - TRADE_INTENT_SCHEMA_CONTRACT.md
Pre-deployment validation gate for trade intent messages.

Usage:
    python tests/test_trade_intent_schema.py
    pytest tests/test_trade_intent_schema.py -v
"""
import sys
import os
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.services.eventbus_bridge import validate_trade_intent, TradeIntent


def test_valid_trade_intent():
    """Test that valid trade intent passes validation"""
    valid_intent = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.95,
        "position_size_usd": 100.0,
        "leverage": 10,
        "timestamp": "2026-01-21T00:00:00Z",
        "entry_price": 88000.0,
        "stop_loss": 86000.0,
        "take_profit": 90000.0
    }
    
    errors = validate_trade_intent(valid_intent)
    assert errors == [], f"Valid intent should have no errors, got: {errors}"
    print("✅ test_valid_trade_intent PASSED")


def test_missing_required_fields():
    """Test that missing required fields are caught"""
    invalid_intent = {
        "symbol": "BTCUSDT",
        # Missing: action, confidence, position_size_usd, leverage, timestamp
    }
    
    errors = validate_trade_intent(invalid_intent)
    assert len(errors) == 5, f"Expected 5 errors for missing fields, got {len(errors)}"
    assert any("action" in err for err in errors), "Should report missing 'action'"
    assert any("confidence" in err for err in errors), "Should report missing 'confidence'"
    print("✅ test_missing_required_fields PASSED")


def test_invalid_symbol_format():
    """Test that invalid symbol format is caught"""
    invalid_intent = {
        "symbol": "BTCUSD",  # Should end with USDT
        "action": "BUY",
        "confidence": 0.95,
        "position_size_usd": 100.0,
        "leverage": 10,
        "timestamp": "2026-01-21T00:00:00Z"
    }
    
    errors = validate_trade_intent(invalid_intent)
    assert any("symbol" in err.lower() for err in errors), "Should report invalid symbol format"
    print("✅ test_invalid_symbol_format PASSED")


def test_invalid_action():
    """Test that invalid action is caught"""
    invalid_intent = {
        "symbol": "BTCUSDT",
        "action": "HODL",  # Invalid action
        "confidence": 0.95,
        "position_size_usd": 100.0,
        "leverage": 10,
        "timestamp": "2026-01-21T00:00:00Z"
    }
    
    errors = validate_trade_intent(invalid_intent)
    assert any("action" in err.lower() for err in errors), "Should report invalid action"
    print("✅ test_invalid_action PASSED")


def test_confidence_out_of_range():
    """Test that confidence out of range is caught"""
    invalid_intent = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 1.5,  # Out of range [0, 1]
        "position_size_usd": 100.0,
        "leverage": 10,
        "timestamp": "2026-01-21T00:00:00Z"
    }
    
    errors = validate_trade_intent(invalid_intent)
    assert any("confidence" in err.lower() for err in errors), "Should report confidence out of range"
    print("✅ test_confidence_out_of_range PASSED")


def test_leverage_out_of_range():
    """Test that leverage out of range is caught"""
    invalid_intent = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.95,
        "position_size_usd": 100.0,
        "leverage": 150,  # Out of range [1, 125]
        "timestamp": "2026-01-21T00:00:00Z"
    }
    
    errors = validate_trade_intent(invalid_intent)
    assert any("leverage" in err.lower() for err in errors), "Should report leverage out of range"
    print("✅ test_leverage_out_of_range PASSED")


def test_deprecated_side_field():
    """Test that deprecated 'side' field triggers warning"""
    intent_with_side = {
        "symbol": "BTCUSDT",
        "side": "BUY",  # Deprecated field
        "confidence": 0.95,
        "position_size_usd": 100.0,
        "leverage": 10,
        "timestamp": "2026-01-21T00:00:00Z"
    }
    
    errors = validate_trade_intent(intent_with_side)
    # Should pass but log warning (action is missing)
    assert any("action" in err for err in errors), "Should report missing 'action' when only 'side' provided"
    print("✅ test_deprecated_side_field PASSED")


def test_trade_intent_dataclass_creation():
    """Test that TradeIntent can be created from valid data"""
    valid_data = {
        "symbol": "ETHUSDT",
        "action": "SELL",
        "confidence": 0.92,
        "position_size_usd": 200.0,
        "leverage": 7.0,
        "timestamp": "2026-01-21T00:00:00Z",
        "source": "ai-engine",
        "entry_price": 2950.0,
        "stop_loss": 3000.0,
        "take_profit": 2900.0
    }
    
    try:
        intent = TradeIntent(**valid_data)
        assert intent.symbol == "ETHUSDT"
        assert intent.action == "SELL"
        assert intent.confidence == 0.92
        print("✅ test_trade_intent_dataclass_creation PASSED")
    except Exception as e:
        raise AssertionError(f"Failed to create TradeIntent: {e}")


def test_mock_publish_and_consume():
    """Test mock publish → validate → consume flow (no Redis)"""
    # Simulate a publisher creating a trade intent
    mock_intent = {
        "symbol": "SOLUSDT",
        "action": "BUY",
        "confidence": 0.88,
        "position_size_usd": 150.0,
        "leverage": 15,
        "timestamp": "2026-01-21T00:00:00Z",
        "source": "strategy-router",
        "entry_price": 45.5,
        "stop_loss": 44.0,
        "take_profit": 47.0,
        "model": "ensemble",  # Extra field (should be ignored by consumer)
        "reason": "AI signal"  # Extra field (should be ignored by consumer)
    }
    
    # Step 1: Validate before "publishing"
    errors = validate_trade_intent(mock_intent)
    assert errors == [], f"Mock intent should be valid, got errors: {errors}"
    
    # Step 2: Simulate consumer filtering (only allowed fields)
    allowed_fields = {
        'symbol', 'action', 'confidence', 'position_size_usd', 'leverage', 
        'timestamp', 'source', 'stop_loss_pct', 'take_profit_pct', 
        'entry_price', 'stop_loss', 'take_profit', 'quantity'
    }
    filtered_data = {k: v for k, v in mock_intent.items() if k in allowed_fields}
    
    # Step 3: Create TradeIntent from filtered data
    try:
        intent = TradeIntent(**filtered_data)
        assert intent.symbol == "SOLUSDT"
        assert intent.action == "BUY"
        assert not hasattr(intent, 'model'), "Extra fields should not be in TradeIntent"
        print("✅ test_mock_publish_and_consume PASSED")
    except Exception as e:
        raise AssertionError(f"Mock consume failed: {e}")


def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_valid_trade_intent,
        test_missing_required_fields,
        test_invalid_symbol_format,
        test_invalid_action,
        test_confidence_out_of_range,
        test_leverage_out_of_range,
        test_deprecated_side_field,
        test_trade_intent_dataclass_creation,
        test_mock_publish_and_consume
    ]
    
    print("="*60)
    print("TRADE INTENT SCHEMA VALIDATION TESTS")
    print("="*60)
    print()
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1
    
    print()
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All schema validation tests PASSED - safe to deploy\n")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
