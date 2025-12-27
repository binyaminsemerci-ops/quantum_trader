#!/usr/bin/env python3
"""Test heuristic signal structure"""

import sys
import asyncio
import json
sys.path.insert(0, r"c:\quantum_trader")

async def test():
    from backend.routes.live_ai_signals import SimpleAITrader
    
    trader = SimpleAITrader()
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    signals = await trader.generate_signals(symbols, limit=5)
    
    print(f"\nGenerated {len(signals)} heuristic signals\n")
    print("="*80)
    
    for i, sig in enumerate(signals, 1):
        print(f"\nSignal #{i}:")
        print(json.dumps(sig, indent=2, default=str))
        print("-"*80)
    
    # Now test normalization
    sys.path.insert(0, r"c:\quantum_trader\backend")
    from main import _normalise_signals
    
    normalised = _normalise_signals(signals, len(signals))
    
    print("\n\n" + "="*80)
    print("AFTER NORMALIZATION:")
    print("="*80)
    
    for i, sig in enumerate(normalised, 1):
        print(f"\nNormalised #{i}:")
        print(f"  symbol: {sig['symbol']}")
        print(f"  type: {sig['type']}")
        print(f"  confidence: {sig['confidence']}")
        print(f"  source: '{sig['source']}'")
        print(f"  model: '{sig['model']}'")
        print(f"  reason: {sig['reason']}")

if __name__ == "__main__":
    asyncio.run(test())
