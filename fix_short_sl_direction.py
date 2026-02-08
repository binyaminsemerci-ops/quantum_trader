#!/usr/bin/env python3
"""
FIX: Correct SL direction for all SHORT positions

Problem: All SHORT positions have SL BELOW entry (wrong direction)
Solution: Set SL ABOVE entry = entry + abs(entry - current_sl)

Example: AAVEUSDT SHORT
  Entry: 103.15
  Current SL: 99.96 (3.17 BELOW)
  Corrected SL: 103.15 + 3.17 = 106.32 (ABOVE entry)
"""

import redis

r = redis.Redis(host='localhost', decode_responses=True)

print("🔧 FIXING SHORT POSITION SL DIRECTION\n")

fixed_count = 0
skipped_count = 0

for key in r.keys("quantum:position:*"):
    data = r.hgetall(key)
    
    if data.get("side") != "SHORT":
        continue
    
    symbol = data["symbol"]
    entry = float(data["entry_price"])
    sl_old = float(data["stop_loss"])
    tp_old = float(data["take_profit"])
    
    # Check if SL needs fixing
    if sl_old >= entry:
        print(f"✅ {symbol}: SL already correct ({sl_old:.6f} >= {entry:.6f})")
        skipped_count += 1
        continue
    
    # Calculate corrected SL (mirror distance to opposite side)
    distance = abs(entry - sl_old)
    sl_new = entry + distance
    
    print(f"\n🔧 {symbol} SHORT:")
    print(f"   Entry: {entry:.6f}")
    print(f"   SL OLD: {sl_old:.6f} (BELOW entry by {distance:.6f}) ❌")
    print(f"   SL NEW: {sl_new:.6f} (ABOVE entry by {distance:.6f}) ✅")
    print(f"   TP: {tp_old:.6f} (keeping same)")
    
    # Update Redis
    r.hset(key, "stop_loss", str(sl_new))
    
    print(f"   ✅ UPDATED!")
    fixed_count += 1

print(f"\n" + "="*60)
print(f"SUMMARY:")
print(f"  Fixed: {fixed_count} SHORT positions")
print(f"  Skipped: {skipped_count} (already correct)")
print(f"  Total: {fixed_count + skipped_count}")
print(f"="*60)
print()
print("✅ All SHORT positions now have SL ABOVE entry!")
print("🔥 Harvest Brain HARD SL TRIGGER will now protect these positions")
