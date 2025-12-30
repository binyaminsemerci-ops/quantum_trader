#!/usr/bin/env python3
"""Check which symbols are loaded from trading universe"""
from config.trading_universe import get_trading_universe

profiles = ["default", "conservative", "aggressive", "all"]

for profile in profiles:
    symbols = get_trading_universe(profile)
    print(f"\n{'='*60}")
    print(f"Profile: {profile.upper()}")
    print(f"Total: {len(symbols)} symbols")
    print(f"{'='*60}")
    for i, s in enumerate(symbols[:20], 1):
        print(f"  {i:2d}. {s}")
    if len(symbols) > 20:
        print(f"  ... and {len(symbols)-20} more")
