"""Check SOLUSDT position leverage from Binance API."""
from binance.client import Client
import os
import json

client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)

print("\n" + "="*80)
print("CHECKING SOLUSDT POSITION LEVERAGE")
print("="*80)

try:
    positions = client.futures_position_information(symbol='SOLUSDT')
    
    for pos in positions:
        amt = float(pos.get('positionAmt', 0))
        if amt != 0:
            print(f"\n📊 SOLUSDT Position Data:")
            print(f"   Position Amt: {amt}")
            print(f"   Entry Price: ${float(pos.get('entryPrice', 0))}")
            print(f"   Leverage (from API): {pos.get('leverage', 'NOT SET')}")
            print(f"   Notional: ${float(pos.get('notional', 0)):.2f}")
            print(f"   Initial Margin: ${float(pos.get('initialMargin', 0)):.2f}")
            
            # Calculate actual leverage
            notional = abs(float(pos.get('notional', 0)))
            initial_margin = float(pos.get('initialMargin', 0))
            
            if initial_margin > 0:
                actual_leverage = notional / initial_margin
                print(f"   Calculated Leverage: {actual_leverage:.1f}x")
            
            print(f"\n📋 Full Position Data:")
            print(json.dumps(pos, indent=2))

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
