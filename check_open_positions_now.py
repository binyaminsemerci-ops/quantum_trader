"""Check current open positions"""

from binance.client import Client
import os

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)

print("🔍 CHECKING OPEN POSITIONS")
print("=" * 80)

positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]

print(f"\n📊 Total Open Positions: {len(open_positions)}")

if open_positions:
    print("\n📋 Position Details:")
    for p in open_positions:
        amt = float(p['positionAmt'])
        entry = float(p['entryPrice'])
        mark = float(p['markPrice'])
        pnl = float(p['unRealizedProfit'])
        pnl_pct = (pnl / (abs(amt) * entry)) * 100 if entry > 0 else 0
        
        side = "LONG" if amt > 0 else "SHORT"
        emoji = "🟢" if pnl > 0 else "🔴"
        
        print(f"\n{emoji} {p['symbol']} ({side}):")
        print(f"   Qty: {amt}")
        print(f"   Entry: ${entry}")
        print(f"   Mark: ${mark}")
        print(f"   PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
else:
    print("\n✅ INGEN ÅPNE POSISJONER")
    print("\n🤔 Mulige årsaker:")
    print("   1. Posisjoner ble stengt automatisk (TP/SL)")
    print("   2. Position Monitor stengte dem pga risk")
    print("   3. Manuelt stengt via Binance")
    print("   4. Liquidation (unlikely på testnet)")
    
print("\n" + "=" * 80)
