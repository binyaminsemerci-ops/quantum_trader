#!/usr/bin/env python3
"""
ATR normalization bug fix for /opt/quantum/microservices/ai_engine/service.py

ROOT CAUSE:
  volatility_structure_engine.calculate_atr() returns ABSOLUTE ATR in price units
  (e.g. $40 for ETH). But service.py passes it directly to rl_sizing_agent as
  atr_pct, which expects a DECIMAL FRACTION (e.g. 0.02 = 2%).

EFFECT:
  atr_pct = 40  -->  _ma_sl = 1.5 * 40 = 60  -->  sl = $1970 * (1+60) = $120,570
  take_profit becomes negative. SL never triggers. Positions churn indefinitely.
  This caused -412 USDT in 30 days, 1822 fills, fees > realized losses.
"""
import sys

TARGET = '/opt/quantum/microservices/ai_engine/service.py'

OLD = "                atr_pct = atr_value  # e.g. 0.02 = 2%"

NEW = (
    "                # BUG FIX: PHASE 2D volatility_structure_engine returns ABSOLUTE ATR\n"
    "                # (e.g. $40 for ETH). Must divide by price to get fraction (0.0203 = 2.03%).\n"
    "                # Without this fix: sl_percent=60 -> SL=$120k (never triggers), TP negative.\n"
    "                if atr_value is not None and atr_value > 0.5:\n"
    "                    _ref_price = float(features.get('price') or 1.0)\n"
    "                    atr_value = atr_value / _ref_price if _ref_price > 0 else 0.02\n"
    "                    atr_value = max(0.005, min(0.15, atr_value))\n"
    "                atr_pct = atr_value  # now guaranteed fraction, e.g. 0.02 = 2%"
)

content = open(TARGET).read()

if OLD not in content:
    # Check if already patched
    if 'PHASE 2D volatility_structure_engine returns ABSOLUTE ATR' in content:
        print("ALREADY_PATCHED")
        sys.exit(0)
    # Show context for debugging
    idx = content.find('atr_pct = atr_value')
    ctx = content[max(0,idx-120):idx+120] if idx >= 0 else "NOT FOUND"
    print(f"OLD_STRING_NOT_FOUND\nContext:\n{ctx}")
    sys.exit(1)

patched = content.replace(OLD, NEW, 1)
open(TARGET, 'w').write(patched)
print("PATCHED_OK")
