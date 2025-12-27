#!/usr/bin/env python3
"""Test live signal generation with backend running"""

import sys
import os
sys.path.insert(0, r"c:\quantum_trader")

import asyncio
import aiohttp


async def test_live_signals():
    """Fetch signals from running backend"""
    
    print("\n=== Testing Live Signal Generation ===")
    print("Fetching from: http://localhost:8000/api/ai/signals/latest")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8000/api/ai/signals/latest",
                params={"limit": 10, "profile": "mixed"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    signals = await response.json()
                    
                    print(f"\n[OK] Received {len(signals)} signals:")
                    print("-" * 80)
                    
                    for i, sig in enumerate(signals, 1):
                        symbol = sig.get("symbol", "N/A")
                        action = sig.get("type", "N/A")
                        confidence = sig.get("confidence", 0.0)
                        source = sig.get("source", "unknown")
                        model = sig.get("model", "unknown")
                        price = sig.get("price", 0.0)
                        reason = sig.get("reason", "")[:50]
                        
                        print(f"[{i:2d}] {symbol:10s} | {action:4s} | conf={confidence:.3f} | "
                              f"${price:8.2f} | {source:15s} | {model:10s}")
                        if reason:
                            print(f"     └─ {reason}")
                    
                    print("-" * 80)
                    
                    # Count by source
                    sources = {}
                    for sig in signals:
                        src = sig.get("source", "unknown")
                        sources[src] = sources.get(src, 0) + 1
                    
                    print("\n[CHART] Signal Sources:")
                    for src, count in sources.items():
                        print(f"   {src}: {count}")
                    
                    return True
                else:
                    print(f"❌ Error: HTTP {response.status}")
                    text = await response.text()
                    print(text[:500])
                    return False
                    
    except aiohttp.ClientConnectorError:
        print("❌ Could not connect to backend at http://localhost:8000")
        print("\n💡 Start the backend first:")
        print("   cd C:/quantum_trader/backend")
        print("   uvicorn main:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_signals_endpoint():
    """Test /signals/recent endpoint"""
    
    print("\n\n=== Testing /signals/recent Endpoint ===")
    print("Fetching from: http://localhost:8000/signals/recent")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "http://localhost:8000/signals/recent",
                params={"limit": 5, "profile": "mixed"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    signals = await response.json()
                    
                    print(f"\n[OK] Received {len(signals)} signals")
                    
                    for i, sig in enumerate(signals[:3], 1):
                        print(f"\n[{i}] {sig.get('symbol')} - {sig.get('side', 'N/A').upper()}")
                        print(f"    Confidence: {sig.get('confidence', 0):.3f}")
                        print(f"    Source: {sig.get('details', {}).get('source', 'N/A')}")
                    
                    return True
                else:
                    print(f"❌ HTTP {response.status}")
                    return False
                    
    except aiohttp.ClientConnectorError:
        print("❌ Backend not running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    print("=" * 80)
    print("LIVE SIGNAL GENERATION TEST")
    print("=" * 80)
    
    # Test main endpoint
    test1 = await test_live_signals()
    
    # Test signals endpoint
    test2 = await test_signals_endpoint()
    
    print("\n" + "=" * 80)
    if test1 or test2:
        print("[OK] At least one endpoint working!")
        return 0
    else:
        print("❌ All tests failed - ensure backend is running")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
