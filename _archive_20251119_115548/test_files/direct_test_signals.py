#!/usr/bin/env python3
"""Direct test of get_live_ai_signals function"""

import sys
import asyncio
sys.path.insert(0, r"c:\quantum_trader")

async def test_direct():
    from backend.routes.live_ai_signals import get_live_ai_signals
    
    print("Calling get_live_ai_signals directly...")
    signals = await get_live_ai_signals(limit=15, profile="mixed")
    
    print(f"\nReturned {len(signals)} signals:\n")
    
    for i, sig in enumerate(signals, 1):
        symbol = sig.get("symbol", "N/A")
        action = sig.get("side", sig.get("type", "HOLD"))
        conf = sig.get("confidence", sig.get("score", 0))
        source = sig.get("source", "?")
        model = sig.get("model", "?")
        price = sig.get("price", 0)
        
        print(f"{i:2d}. {symbol:10s} {action:<5s} conf={conf:.4f} ${price:>10.2f} src={source:15s} model={model}")
        
        # Show details
        details = sig.get("details")
        if details:
            print(f"    Details: {details}")
    
    # Count sources
    sources = {}
    for sig in signals:
        src = sig.get("source", sig.get("details", {}).get("source", "unknown"))
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\n[CHART] Source breakdown:")
    for src, count in sources.items():
        print(f"   {src}: {count}")

if __name__ == "__main__":
    asyncio.run(test_direct())
