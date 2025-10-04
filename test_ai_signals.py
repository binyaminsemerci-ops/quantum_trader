#!/usr/bin/env python3
"""Test AI signals generation"""

import sys
import os
sys.path.insert(0, r'c:\quantum_trader')

import asyncio
from backend.routes.live_ai_signals import get_live_ai_signals

async def test_signals():
    print("Testing AI signal generation...")
    signals = await get_live_ai_signals(3, "mixed")
    print(f"Generated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['side']} (score: {signal['score']})")
    return signals

if __name__ == "__main__":
    asyncio.run(test_signals())