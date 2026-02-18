#!/usr/bin/env python3
"""
Portfolio Top-N Gate - Unit Test
Verifies buffer, filtering, and Top-N selection logic
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any


class MockTopNGate:
    """Simplified version of Top-N Gate for testing"""
    
    def __init__(self, top_n_limit: int = 10, min_confidence: float = 0.55):
        self._prediction_buffer: List[Dict[str, Any]] = []
        self._prediction_buffer_lock = asyncio.Lock()
        self._top_n_limit = top_n_limit
        self._min_confidence = min_confidence
        self._published_predictions: List[Dict] = []  # Track published for testing
    
    async def buffer_prediction(self, symbol: str, action: str, confidence: float):
        """Simulate buffering a prediction"""
        async with self._prediction_buffer_lock:
            self._prediction_buffer.append({
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "raw_event": {"symbol": symbol, "action": action, "confidence": confidence}
            })
    
    async def process_buffer(self):
        """Process buffer and apply Top-N filtering"""
        async with self._prediction_buffer_lock:
            if not self._prediction_buffer:
                return []
            
            buffered_predictions = self._prediction_buffer.copy()
            self._prediction_buffer.clear()
        
        # Filter HOLD
        non_hold = [p for p in buffered_predictions if p["action"] != "HOLD"]
        
        # Filter by confidence
        eligible = [p for p in non_hold if p["confidence"] >= self._min_confidence]
        
        # Sort descending
        eligible.sort(key=lambda p: p["confidence"], reverse=True)
        
        # Select top N
        selected = eligible[:self._top_n_limit]
        
        # Simulate publish
        self._published_predictions.extend(selected)
        
        return {
            "total": len(buffered_predictions),
            "eligible": len(eligible),
            "selected": len(selected),
            "published": [{"symbol": p["symbol"], "conf": p["confidence"]} for p in selected]
        }


async def test_basic_filtering():
    """Test 1: Basic filtering - all eligible"""
    print("\n=== TEST 1: Basic Filtering ===")
    gate = MockTopNGate(top_n_limit=5)
    
    # Add 3 predictions (all eligible)
    await gate.buffer_prediction("BTCUSDT", "BUY", 0.75)
    await gate.buffer_prediction("ETHUSDT", "SELL", 0.68)
    await gate.buffer_prediction("SOLUSDT", "BUY", 0.82)
    
    result = await gate.process_buffer()
    
    assert result["total"] == 3, f"Expected 3 total, got {result['total']}"
    assert result["eligible"] == 3, f"Expected 3 eligible, got {result['eligible']}"
    assert result["selected"] == 3, f"Expected 3 selected, got {result['selected']}"
    assert result["published"][0]["symbol"] == "SOLUSDT", "Highest confidence should be first"
    assert result["published"][0]["conf"] == 0.82, "Highest confidence should be 0.82"
    
    print(f"✅ PASS: {result}")


async def test_hold_filtering():
    """Test 2: HOLD actions filtered out"""
    print("\n=== TEST 2: HOLD Filtering ===")
    gate = MockTopNGate(top_n_limit=10)
    
    await gate.buffer_prediction("BTCUSDT", "HOLD", 0.95)  # Even high confidence HOLD → filtered
    await gate.buffer_prediction("ETHUSDT", "BUY", 0.70)
    await gate.buffer_prediction("SOLUSDT", "HOLD", 0.88)
    
    result = await gate.process_buffer()
    
    assert result["total"] == 3, "Should process all 3 predictions"
    assert result["eligible"] == 1, "Only 1 non-HOLD prediction"
    assert result["selected"] == 1, "Should select 1"
    assert result["published"][0]["symbol"] == "ETHUSDT", "Only ETHUSDT should be published"
    
    print(f"✅ PASS: {result}")


async def test_confidence_threshold():
    """Test 3: Confidence threshold filtering"""
    print("\n=== TEST 3: Confidence Threshold ===")
    gate = MockTopNGate(top_n_limit=10, min_confidence=0.55)
    
    await gate.buffer_prediction("BTC", "BUY", 0.45)   # Below threshold
    await gate.buffer_prediction("ETH", "BUY", 0.50)   # Below threshold
    await gate.buffer_prediction("SOL", "BUY", 0.60)   # Above threshold
    await gate.buffer_prediction("AVAX", "SELL", 0.72) # Above threshold
    
    result = await gate.process_buffer()
    
    assert result["total"] == 4, "Should buffer all 4"
    assert result["eligible"] == 2, "Only 2 above threshold"
    assert result["selected"] == 2, "Should select 2"
    assert result["published"][0]["conf"] == 0.72, "Highest should be 0.72"
    
    print(f"✅ PASS: {result}")


async def test_topn_selection():
    """Test 4: Top-N selection when more eligible than limit"""
    print("\n=== TEST 4: Top-N Selection ===")
    gate = MockTopNGate(top_n_limit=3, min_confidence=0.55)
    
    # Add 7 eligible predictions
    await gate.buffer_prediction("A", "BUY", 0.91)
    await gate.buffer_prediction("B", "BUY", 0.58)
    await gate.buffer_prediction("C", "BUY", 0.77)
    await gate.buffer_prediction("D", "BUY", 0.62)
    await gate.buffer_prediction("E", "BUY", 0.85)
    await gate.buffer_prediction("F", "BUY", 0.69)
    await gate.buffer_prediction("G", "BUY", 0.73)
    
    result = await gate.process_buffer()
    
    assert result["total"] == 7, "Should buffer all 7"
    assert result["eligible"] == 7, "All 7 above threshold"
    assert result["selected"] == 3, "Should select only top 3"
    
    # Verify top 3 are correctly selected and sorted
    published_confs = [p["conf"] for p in result["published"]]
    assert published_confs == [0.91, 0.85, 0.77], f"Top 3 should be [0.91, 0.85, 0.77], got {published_confs}"
    
    print(f"✅ PASS: {result}")
    print(f"   Selected: {result['published']}")
    print(f"   Rejected: 4 predictions with conf < 0.77")


async def test_empty_buffer():
    """Test 5: Empty buffer handling"""
    print("\n=== TEST 5: Empty Buffer ===")
    gate = MockTopNGate(top_n_limit=10)
    
    result = await gate.process_buffer()
    
    assert result == [], "Empty buffer should return empty result"
    
    print(f"✅ PASS: Empty buffer handled correctly")


async def test_mixed_scenario():
    """Test 6: Mixed scenario with HOLD, low confidence, and Top-N"""
    print("\n=== TEST 6: Mixed Scenario (Realistic) ===")
    gate = MockTopNGate(top_n_limit=5, min_confidence=0.55)
    
    # Simulate 15 predictions from different symbols
    predictions = [
        ("BTC", "BUY", 0.92),      # Selected (rank 1)
        ("ETH", "HOLD", 0.88),     # Filtered: HOLD
        ("SOL", "SELL", 0.81),     # Selected (rank 2)
        ("AVAX", "BUY", 0.45),     # Filtered: low confidence
        ("MATIC", "BUY", 0.78),    # Selected (rank 3)
        ("LINK", "HOLD", 0.95),    # Filtered: HOLD
        ("DOT", "BUY", 0.72),      # Selected (rank 4)
        ("ATOM", "SELL", 0.50),    # Filtered: low confidence
        ("NEAR", "BUY", 0.67),     # Selected (rank 5)
        ("FTM", "BUY", 0.64),      # Eligible but not selected (rank 6)
        ("ALGO", "BUY", 0.61),     # Eligible but not selected (rank 7)
        ("XRP", "HOLD", 0.99),     # Filtered: HOLD
        ("ADA", "BUY", 0.58),      # Eligible but not selected (rank 8)
        ("UNI", "BUY", 0.48),      # Filtered: low confidence
        ("AAVE", "SELL", 0.55),    # Eligible but not selected (rank 9)
    ]
    
    for symbol, action, conf in predictions:
        await gate.buffer_prediction(symbol, action, conf)
    
    result = await gate.process_buffer()
    
    assert result["total"] == 15, "Should buffer all 15"
    assert result["eligible"] == 9, "9 non-HOLD + above threshold"
    assert result["selected"] == 5, "Should select top 5"
    
    # Verify correct top 5
    expected_symbols = ["BTC", "SOL", "MATIC", "DOT", "NEAR"]
    published_symbols = [p["symbol"] for p in result["published"]]
    assert published_symbols == expected_symbols, f"Expected {expected_symbols}, got {published_symbols}"
    
    print(f"✅ PASS: {result}")
    print(f"   Published: {result['published']}")
    print(f"   Filtered out:")
    print(f"     - 3 HOLD actions (even with high confidence)")
    print(f"     - 3 below threshold")
    print(f"     - 4 eligible but lower confidence than top 5")


async def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("PORTFOLIO TOP-N CONFIDENCE GATE - UNIT TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_filtering,
        test_hold_filtering,
        test_confidence_threshold,
        test_topn_selection,
        test_empty_buffer,
        test_mixed_scenario,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ FAIL: {test.__name__}")
            print(f"   Error: {e}")
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR: {test.__name__}")
            print(f"   Exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Portfolio Top-N Gate logic verified ✅")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
