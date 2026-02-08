#!/usr/bin/env python3
"""
SimpleCLM Verification Script

Tests all required behaviors:
1. Valid trade acceptance
2. Invalid trade rejection
3. Outcome labeling
4. Persistence
5. Crash recovery
6. Observability
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_clm import SimpleCLM, OutcomeLabel


def create_valid_trade(symbol="BTCUSDT", pnl=2.5) -> dict:
    """Create a valid trade payload"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "side": "BUY",
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "pnl_percent": pnl,
        "confidence": 0.85,
        "model_id": "ensemble_v1",
        "strategy_id": "momentum",
        "position_size": 1000.0,
        "duration_seconds": 3600.0,
        "exit_reason": "take_profit"
    }


def test_valid_trade_acceptance():
    """Test 1: Valid trades are accepted and stored"""
    print("\n" + "="*60)
    print("TEST 1: Valid Trade Acceptance")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        clm = SimpleCLM(storage_path=str(storage_path))
        
        # Record 3 valid trades
        trades = [
            create_valid_trade("BTCUSDT", 2.5),   # WIN
            create_valid_trade("ETHUSDT", -1.2),  # LOSS
            create_valid_trade("BNBUSDT", 0.1),   # NEUTRAL
        ]
        
        for trade in trades:
            success, error = clm.record_trade(trade)
            assert success, f"Trade rejected: {error}"
            print(f"✅ Trade accepted: {trade['symbol']} PnL={trade['pnl_percent']}%")
        
        # Verify counters
        assert clm.total_received == 3
        assert clm.total_stored == 3
        assert clm.total_rejected == 0
        
        # Verify file exists and contains 3 lines
        assert storage_path.exists()
        with open(storage_path) as f:
            lines = f.readlines()
            assert len(lines) == 3
        
        print(f"✅ Storage file: {storage_path}")
        print(f"✅ Counters: received={clm.total_received}, stored={clm.total_stored}")
        print("✅ TEST 1 PASSED")


def test_invalid_trade_rejection():
    """Test 2: Invalid trades are rejected and logged"""
    print("\n" + "="*60)
    print("TEST 2: Invalid Trade Rejection")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        clm = SimpleCLM(storage_path=str(storage_path))
        
        # Invalid trades
        invalid_trades = [
            ({**create_valid_trade(), "symbol": ""}, "empty symbol"),
            ({**create_valid_trade(), "entry_price": -100}, "negative price"),
            ({**create_valid_trade(), "confidence": 1.5}, "confidence > 1"),
            ({k: v for k, v in create_valid_trade().items() if k != "timestamp"}, "missing timestamp"),
            ({**create_valid_trade(), "side": "INVALID"}, "invalid side"),
        ]
        
        for trade, reason in invalid_trades:
            success, error = clm.record_trade(trade)
            assert not success, f"Invalid trade accepted: {reason}"
            print(f"✅ Trade rejected: {reason} → {error}")
        
        # Verify all rejected
        assert clm.total_received == 5
        assert clm.total_stored == 0
        assert clm.total_rejected == 5
        
        # Verify no file created (no valid trades)
        assert not storage_path.exists()
        
        print(f"✅ Counters: received={clm.total_received}, rejected={clm.total_rejected}")
        print("✅ TEST 2 PASSED")


def test_outcome_labeling():
    """Test 3: Outcome labeling is deterministic"""
    print("\n" + "="*60)
    print("TEST 3: Outcome Labeling")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        clm = SimpleCLM(storage_path=str(storage_path), win_threshold=0.5, loss_threshold=-0.5)
        
        # Test cases
        test_cases = [
            (2.0, OutcomeLabel.WIN),
            (0.6, OutcomeLabel.WIN),
            (-1.5, OutcomeLabel.LOSS),
            (-0.6, OutcomeLabel.LOSS),
            (0.3, OutcomeLabel.NEUTRAL),
            (-0.2, OutcomeLabel.NEUTRAL),
            (0.0, OutcomeLabel.NEUTRAL),
        ]
        
        for pnl, expected_label in test_cases:
            label = clm.label_outcome(pnl)
            assert label == expected_label, f"PnL={pnl}: expected {expected_label}, got {label}"
            print(f"✅ PnL={pnl:+.1f}% → {label.value}")
        
        print("✅ TEST 3 PASSED")


def test_persistence_and_recovery():
    """Test 4: Persistence survives restart"""
    print("\n" + "="*60)
    print("TEST 4: Persistence & Crash Recovery")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        
        # Session 1: Record 2 trades
        clm1 = SimpleCLM(storage_path=str(storage_path))
        clm1.record_trade(create_valid_trade("BTCUSDT", 1.5))
        clm1.record_trade(create_valid_trade("ETHUSDT", -0.8))
        
        assert clm1.total_stored == 2
        print(f"✅ Session 1: Stored 2 trades")
        
        # Simulate crash (destroy object)
        del clm1
        
        # Session 2: Load from disk
        clm2 = SimpleCLM(storage_path=str(storage_path))
        
        # Should recover 2 trades
        assert clm2.total_stored == 2
        assert clm2.last_trade_timestamp is not None
        print(f"✅ Session 2: Recovered 2 trades from disk")
        
        # Add 1 more trade
        clm2.record_trade(create_valid_trade("BNBUSDT", 0.5))
        assert clm2.total_stored == 3
        print(f"✅ Session 2: Added 1 more trade (total: 3)")
        
        # Verify file has 3 lines
        with open(storage_path) as f:
            lines = f.readlines()
            assert len(lines) == 3
        
        print("✅ TEST 4 PASSED")


def test_observability():
    """Test 5: Observability metrics are exposed"""
    print("\n" + "="*60)
    print("TEST 5: Observability")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        clm = SimpleCLM(storage_path=str(storage_path))
        
        # Record trades
        clm.record_trade(create_valid_trade("BTCUSDT", 2.0))
        clm.record_trade({**create_valid_trade("ETHUSDT"), "symbol": ""})  # Invalid
        clm.record_trade(create_valid_trade("BNBUSDT", -1.0))
        
        # Get status
        status = clm.get_status()
        
        # Verify all metrics present
        required_keys = [
            "running", "persistence_enabled", "storage_path", "file_size_mb",
            "total_trades_received", "total_trades_stored", "total_trades_rejected",
            "last_trade_timestamp", "idle_hours", "starving", "thresholds"
        ]
        
        for key in required_keys:
            assert key in status, f"Missing metric: {key}"
            print(f"✅ Metric present: {key}={status[key]}")
        
        # Verify values
        assert status["total_trades_received"] == 3
        assert status["total_trades_stored"] == 2
        assert status["total_trades_rejected"] == 1
        assert status["last_trade_timestamp"] is not None
        
        print("✅ TEST 5 PASSED")


def test_stored_data_structure():
    """Test 6: Stored data contains all required fields"""
    print("\n" + "="*60)
    print("TEST 6: Stored Data Structure")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        clm = SimpleCLM(storage_path=str(storage_path))
        
        # Record 1 trade
        clm.record_trade(create_valid_trade("BTCUSDT", 3.0))
        
        # Read and verify
        with open(storage_path) as f:
            stored_trade = json.loads(f.readline())
        
        # Required fields
        required_fields = [
            "timestamp", "symbol", "side", "entry_price", "exit_price",
            "pnl_percent", "confidence", "model_id", "outcome_label"
        ]
        
        for field in required_fields:
            assert field in stored_trade, f"Missing field: {field}"
            print(f"✅ Field present: {field}={stored_trade[field]}")
        
        # Verify outcome label
        assert stored_trade["outcome_label"] in ["WIN", "LOSS", "NEUTRAL"]
        assert stored_trade["outcome_label"] == "WIN"  # PnL=3.0 > 0.5
        
        print("✅ TEST 6 PASSED")


async def test_starvation_detection():
    """Test 7: Starvation detection works"""
    print("\n" + "="*60)
    print("TEST 7: Starvation Detection")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "test_trades.jsonl"
        
        # Set very short starvation threshold for testing
        clm = SimpleCLM(
            storage_path=str(storage_path),
            starvation_hours=0.001,  # ~3.6 seconds
            stats_log_interval_seconds=2
        )
        
        # Start monitoring
        await clm.start()
        
        # Record 1 trade
        clm.record_trade(create_valid_trade("BTCUSDT", 1.0))
        print(f"✅ Recorded 1 trade at {clm.last_trade_timestamp}")
        
        # Wait for starvation detection
        print("⏳ Waiting 5 seconds for starvation alert...")
        await asyncio.sleep(5)
        
        # Check status
        status = clm.get_status()
        assert status["starving"] == True, "Starvation not detected"
        print(f"✅ Starvation detected: idle_hours={status['idle_hours']}")
        
        await clm.stop()
        print("✅ TEST 7 PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("SimpleCLM Verification Suite")
    print("="*60)
    
    try:
        test_valid_trade_acceptance()
        test_invalid_trade_rejection()
        test_outcome_labeling()
        test_persistence_and_recovery()
        test_observability()
        test_stored_data_structure()
        
        # Async test
        asyncio.run(test_starvation_detection())
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED")
        print("="*60)
        print("\nSimpleCLM is OPERATIONAL:")
        print("✅ Validates input")
        print("✅ Labels outcomes (WIN/LOSS/NEUTRAL)")
        print("✅ Persists to disk (atomic writes)")
        print("✅ Survives crashes (recovery on startup)")
        print("✅ Exposes observability metrics")
        print("✅ Detects starvation")
        print("✅ Rejects invalid trades loudly")
        
        return 0
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
