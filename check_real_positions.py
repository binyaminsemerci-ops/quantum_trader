import os
from binance.client import Client

# Use testnet keys from environment (docker-compose sets as BINANCE_API_KEY)
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if not api_key or not api_secret:
    print("ERROR: BINANCE_API_KEY or BINANCE_API_SECRET not set")
    print(f"Found env vars: {list(os.environ.keys())[:20]}")
    exit(1)

client = Client(api_key, api_secret, testnet=True)
positions = client.futures_position_information()
active = [p for p in positions if float(p["positionAmt"]) != 0]

print(f"\n{'='*60}")
print(f"ACTIVE POSITIONS (positionAmt != 0)")
print(f"{'='*60}")

for p in active:
    amt = float(p["positionAmt"])
    side = "LONG" if amt > 0 else "SHORT"
    pnl = float(p["unRealizedProfit"])
    entry = p["entryPrice"]
    mark = p["markPrice"]
    pnl_pct = (pnl / (abs(amt) * float(entry))) * 100 if amt != 0 else 0
    
    print(f"\n{p['symbol']:12s} {side:5s} {abs(amt):15.4f}")
    print(f"  Entry:  {entry:15s}  Mark: {mark}")
    print(f"  PnL:    ${pnl:10.2f}  ({pnl_pct:+.2f}%)")
    print(f"  Hedge:  {p.get('positionSide', 'N/A')}")
    print(f"  Lever:  {p.get('leverage', '?')}x")

print(f"\n{'='*60}")
print(f"Total: {len(active)} active positions")
print(f"{'='*60}\n")
