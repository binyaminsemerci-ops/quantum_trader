#!/usr/bin/env python3
"""Test normalization logic"""

import sys
import asyncio
import json
sys.path.insert(0, r"c:\quantum_trader")

async def test():
    from backend.routes.live_ai_signals import get_live_ai_signals
    
    signals = await get_live_ai_signals(limit=5, profile="mixed")
    
    print(f"\nGot {len(signals)} signals\n")
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
        print(json.dumps(sig, indent=2, default=str))
        print("-"*80)

if __name__ == "__main__":
    asyncio.run(test())
