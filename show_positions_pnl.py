#!/usr/bin/env python3
import redis
r = redis.Redis(decode_responses=True)

print("="*80)
print("CURRENT POSITIONS - UNREALIZED PNL ANALYSIS")
print("="*80)

positions = []
for key in r.keys("quantum:position:*"):
    data = r.hgetall(key)
    upnl = float(data.get("unrealized_pnl", 0))
    positions.append({
        "symbol": data.get("symbol"),
        "side": data.get("side"),
        "entry": float(data.get("entry_price", 0)),
        "upnl": upnl,
        "qty": float(data.get("quantity", 0))
    })

# Sort by unrealized PnL (worst first)
positions.sort(key=lambda x: x["upnl"])

total_upnl = 0
print(f"\n{'Symbol':<15} {'Side':<6} {'Entry':<12} {'Qty':<10} {'UnPnL USDT':<12}")
print("-"*80)

for p in positions:
    total_upnl += p["upnl"]
    print(f"{p['symbol']:<15} {p['side']:<6} {p['entry']:<12.6f} {p['qty']:<10.4f} {p['upnl']:<12.2f}")

print("-"*80)
print(f"{'TOTAL':<15} {'':<6} {'':<12} {'':<10} {total_upnl:<12.2f}")
print()
print(f"Number of positions: {len(positions)}")
print(f"Positions with losses: {len([p for p in positions if p['upnl'] < 0])}")
print(f"Positions with profits: {len([p for p in positions if p['upnl'] > 0])}")
print()
print("="*80)
