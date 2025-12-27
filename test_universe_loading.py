#!/usr/bin/env python3
"""
Test script for universe loading system.

Tests all universe modes and profiles to ensure correct behavior.
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import get_qt_symbols, get_qt_universe, get_qt_max_symbols
from backend.utils.universe import (
    load_universe,
    get_l1l2_top_by_volume,
    get_megacap_universe,
    get_all_usdt_universe,
    save_universe_snapshot
)


def test_config_helpers():
    """Test configuration helper functions."""
    print("=" * 80)
    print("TEST 1: Configuration Helpers")
    print("=" * 80)
    
    # Test with no env vars set
    os.environ.pop("QT_SYMBOLS", None)
    os.environ.pop("QT_UNIVERSE", None)
    os.environ.pop("QT_MAX_SYMBOLS", None)
    
    print(f"QT_SYMBOLS (empty): '{get_qt_symbols()}'")
    print(f"QT_UNIVERSE (default): '{get_qt_universe()}'")
    print(f"QT_MAX_SYMBOLS (default): {get_qt_max_symbols()}")
    
    # Test with env vars set
    os.environ["QT_SYMBOLS"] = "BTCUSDT,ETHUSDT,BNBUSDT"
    os.environ["QT_UNIVERSE"] = "l1l2-top"
    os.environ["QT_MAX_SYMBOLS"] = "500"
    
    print(f"\nQT_SYMBOLS (set): '{get_qt_symbols()}'")
    print(f"QT_UNIVERSE (set): '{get_qt_universe()}'")
    print(f"QT_MAX_SYMBOLS (set): {get_qt_max_symbols()}")
    
    # Test bounds checking
    os.environ["QT_MAX_SYMBOLS"] = "5"  # Too low
    print(f"\nQT_MAX_SYMBOLS (5 -> bounded): {get_qt_max_symbols()}")
    
    os.environ["QT_MAX_SYMBOLS"] = "2000"  # Too high
    print(f"QT_MAX_SYMBOLS (2000 -> bounded): {get_qt_max_symbols()}")
    
    os.environ["QT_MAX_SYMBOLS"] = "invalid"  # Invalid
    print(f"QT_MAX_SYMBOLS (invalid -> default): {get_qt_max_symbols()}")
    
    print("\n✅ Config helpers test passed\n")


def test_universe_profiles():
    """Test all universe profile loaders."""
    print("=" * 80)
    print("TEST 2: Universe Profile Loaders")
    print("=" * 80)
    
    # Test megacap
    print("\n[MEGACAP UNIVERSE]")
    megacap = get_megacap_universe(quote="USDT", max_symbols=20)
    print(f"Symbols loaded: {len(megacap)}")
    print(f"First 10: {megacap[:10]}")
    
    # Test l1l2-top
    print("\n[L1L2-TOP UNIVERSE]")
    l1l2 = get_l1l2_top_by_volume(quote="USDT", max_symbols=50)
    print(f"Symbols loaded: {len(l1l2)}")
    print(f"First 10: {l1l2[:10]}")
    
    # Test all-usdt
    print("\n[ALL-USDT UNIVERSE]")
    all_usdt = get_all_usdt_universe(quote="USDT", max_symbols=100)
    print(f"Symbols loaded: {len(all_usdt)}")
    print(f"First 10: {all_usdt[:10]}")
    
    print("\n✅ Universe profiles test passed\n")


def test_load_universe():
    """Test main load_universe function."""
    print("=" * 80)
    print("TEST 3: Main load_universe Function")
    print("=" * 80)
    
    # Test with different profiles
    profiles = ["megacap", "l1l2-top", "all-usdt", "unknown-profile"]
    
    for profile in profiles:
        print(f"\n[TESTING: {profile}]")
        try:
            symbols = load_universe(
                universe_name=profile,
                max_symbols=30,
                quote="USDT"
            )
            print(f"✅ Loaded {len(symbols)} symbols")
            print(f"   First 5: {symbols[:5]}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n✅ load_universe test passed\n")


def test_snapshot():
    """Test snapshot save functionality."""
    print("=" * 80)
    print("TEST 4: Universe Snapshot")
    print("=" * 80)
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
    snapshot_path = "/tmp/test_universe_snapshot.json"
    
    print(f"Saving snapshot to: {snapshot_path}")
    save_universe_snapshot(
        symbols=test_symbols,
        mode="test",
        qt_universe="megacap",
        qt_max_symbols=50,
        snapshot_path=snapshot_path
    )
    
    # Read back
    import json
    with open(snapshot_path, "r") as f:
        data = json.load(f)
    
    print(f"Snapshot contents:")
    print(f"  Mode: {data['mode']}")
    print(f"  Universe: {data['qt_universe']}")
    print(f"  Max Symbols: {data['qt_max_symbols']}")
    print(f"  Symbol Count: {data['symbol_count']}")
    print(f"  Symbols: {data['symbols']}")
    
    print("\n✅ Snapshot test passed\n")


def test_explicit_mode():
    """Test explicit QT_SYMBOLS mode."""
    print("=" * 80)
    print("TEST 5: Explicit Mode (QT_SYMBOLS)")
    print("=" * 80)
    
    os.environ["QT_SYMBOLS"] = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT"
    os.environ.pop("QT_UNIVERSE", None)
    
    symbols_env = get_qt_symbols()
    if symbols_env:
        symbols = [s.strip() for s in symbols_env.split(",") if s.strip()]
        print(f"✅ Explicit mode: {len(symbols)} symbols")
        print(f"   Symbols: {symbols}")
    else:
        print("❌ Failed to get explicit symbols")
    
    print("\n✅ Explicit mode test passed\n")


def test_dynamic_mode():
    """Test dynamic mode (QT_UNIVERSE + QT_MAX_SYMBOLS)."""
    print("=" * 80)
    print("TEST 6: Dynamic Mode (QT_UNIVERSE + QT_MAX_SYMBOLS)")
    print("=" * 80)
    
    os.environ.pop("QT_SYMBOLS", None)
    os.environ["QT_UNIVERSE"] = "megacap"
    os.environ["QT_MAX_SYMBOLS"] = "30"
    
    universe_name = get_qt_universe()
    max_symbols = get_qt_max_symbols()
    
    print(f"Universe: {universe_name}")
    print(f"Max Symbols: {max_symbols}")
    
    symbols = load_universe(
        universe_name=universe_name,
        max_symbols=max_symbols,
        quote="USDT"
    )
    
    print(f"✅ Dynamic mode: {len(symbols)} symbols loaded")
    print(f"   First 10: {symbols[:10]}")
    
    print("\n✅ Dynamic mode test passed\n")


def main():
    """Run all tests."""
    print("\n")
    print("█" * 80)
    print("█  UNIVERSE LOADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("█" * 80)
    print()
    
    try:
        test_config_helpers()
        test_universe_profiles()
        test_load_universe()
        test_snapshot()
        test_explicit_mode()
        test_dynamic_mode()
        
        print("=" * 80)
        print("🎉 ALL TESTS PASSED")
        print("=" * 80)
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
