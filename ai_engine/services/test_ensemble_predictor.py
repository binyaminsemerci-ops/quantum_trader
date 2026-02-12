"""
Test script for Ensemble Predictor Service (PATH 2.2)

Validates:
1. Validator rejects forbidden fields
2. Validator accepts valid signals
3. Service produces signals to quantum:stream:signal.score
4. Fail-mode degrades confidence (not halt)
"""
import asyncio
import sys
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.services.ensemble_predictor_service import (
    EnsemblePredictorService,
    SignalValidator,
    SignalScore
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_validator():
    """Test validator enforcement."""
    print("\n" + "="*60)
    print("TEST 1: VALIDATOR ENFORCEMENT")
    print("="*60)
    
    validator = SignalValidator()
    
    # Valid signal
    valid_signal = {
        "symbol": "BTCUSDT",
        "horizon": "exit",
        "suggested_action": "CLOSE",
        "confidence": 0.75,
        "expected_edge": -0.12,
        "risk_context": "trend_exhaustion",
        "ensemble_version": "v1.0.0",
        "models_used": "lgbm,patchtst",
        "timestamp": "2026-02-11T12:00:00Z"
    }
    
    is_valid, reason = validator.validate(valid_signal)
    assert is_valid, f"Valid signal rejected: {reason}"
    print("✅ PASS: Valid signal accepted")
    
    # Invalid: forbidden field (quantity)
    invalid_signal_1 = valid_signal.copy()
    invalid_signal_1["quantity"] = 0.5
    is_valid, reason = validator.validate(invalid_signal_1)
    assert not is_valid, "Forbidden field 'quantity' not caught!"
    assert "FORBIDDEN_FIELDS" in reason
    print(f"✅ PASS: Forbidden field rejected - {reason}")
    
    # Invalid: wrong action
    invalid_signal_2 = valid_signal.copy()
    invalid_signal_2["suggested_action"] = "BUY"
    is_valid, reason = validator.validate(invalid_signal_2)
    assert not is_valid, "Invalid action 'BUY' not caught!"
    print(f"✅ PASS: Invalid action rejected - {reason}")
    
    # Invalid: wrong horizon
    invalid_signal_3 = valid_signal.copy()
    invalid_signal_3["horizon"] = "entry"
    is_valid, reason = validator.validate(invalid_signal_3)
    assert not is_valid, "Invalid horizon 'entry' not caught!"
    print(f"✅ PASS: Invalid horizon rejected - {reason}")
    
    # Invalid: confidence out of bounds
    invalid_signal_4 = valid_signal.copy()
    invalid_signal_4["confidence"] = 1.5
    is_valid, reason = validator.validate(invalid_signal_4)
    assert not is_valid, "Out-of-bounds confidence not caught!"
    print(f"✅ PASS: Invalid confidence rejected - {reason}")
    
    # Print stats
    stats = validator.get_stats()
    print(f"\n📊 Validator Stats:")
    print(f"   Total validated: {stats['total_validated']}")
    print(f"   Total dropped: {stats['total_dropped']}")
    print(f"   Drop rate: {stats['drop_rate']:.2%}")
    print(f"   Drop reasons: {stats['drop_reasons']}")


async def test_service():
    """Test service signal production."""
    print("\n" + "="*60)
    print("TEST 2: SERVICE SIGNAL PRODUCTION")
    print("="*60)
    
    service = EnsemblePredictorService()
    await service.connect()
    
    try:
        # Test signal production
        features = {
            "price": 50000.0,
            "volatility_20": 0.03,
            "trend_strength": 0.6
        }
        
        success = await service.produce_signal("BTCUSDT", features)
        assert success, "Signal production failed!"
        print("✅ PASS: Signal produced to quantum:stream:signal.score")
        
        # Check stats
        stats = service.get_stats()
        print(f"\n📊 Service Stats:")
        print(f"   Signals produced: {stats['signals_produced']}")
        print(f"   Signals dropped: {stats['signals_dropped']}")
        print(f"   Models loaded: {stats['models_loaded']}")
        
        # Test health check
        health = await service.health_check()
        print(f"\n🏥 Health Check:")
        print(f"   Status: {health['status']}")
        print(f"   Authority: {health['authority']}")
        print(f"   Output stream: {health['output_stream']}")
        
    finally:
        await service.disconnect()


async def test_fail_mode():
    """Test fail-mode produces degraded confidence."""
    print("\n" + "="*60)
    print("TEST 3: FAIL-MODE (DEGRADED CONFIDENCE)")
    print("="*60)
    
    service = EnsemblePredictorService()
    
    # Force fail-mode by not loading models
    signal = service._fail_mode_signal("BTCUSDT", "test_failure")
    
    assert signal.confidence == 0.0, "Fail-mode confidence should be 0.0"
    assert signal.suggested_action == "HOLD", "Fail-mode action should be HOLD"
    print(f"✅ PASS: Fail-mode signal: confidence={signal.confidence} action={signal.suggested_action}")


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTOR SERVICE - VALIDATION TEST SUITE")
    print("PATH 2.2 - SCORER AUTHORITY ONLY")
    print("="*70)
    
    try:
        await test_validator()
        await test_service()
        await test_fail_mode()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        print("\n🔒 Authority: SCORER ONLY")
        print("📤 Output: quantum:stream:signal.score")
        print("🚫 NO EXECUTION CAPABILITY")
        print("📋 Governance: NO_AUTHORITY_ENSEMBLE_PREDICTOR_FEB11_2026.md")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
