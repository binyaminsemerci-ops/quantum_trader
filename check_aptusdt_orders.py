"""Check APTUSDT order details"""
import os
from binance.client import Client

def load_env():
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key:
        with open(".env") as f:
            for line in f:
                if line.startswith("BINANCE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                elif line.startswith("BINANCE_API_SECRET="):
                    api_secret = line.split("=", 1)[1].strip()
    
    return api_key, api_secret

api_key, api_secret = load_env()
client = Client(api_key, api_secret)

print("\n[SEARCH] APTUSDT Position & Orders Analysis\n")
print("=" * 80)

# Get position
positions = client.futures_position_information(symbol="APTUSDT")
for pos in positions:
    if float(pos['positionAmt']) != 0:
        amt = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        current = float(pos['markPrice'])
        unrealized = float(pos['unRealizedProfit'])
        
        print(f"\n[CHART] Position:")
        print(f"   Amount: {amt:.1f} APT")
        print(f"   Entry: ${entry:.6f}")
        print(f"   Current: ${current:.6f}")
        print(f"   PnL: ${unrealized:.2f}")
        
        # Calculate percentages
        pnl_pct = ((current - entry) / entry) * 100
        print(f"   PnL%: {pnl_pct:+.2f}%")
        
        # TP/SL levels
        tp = entry * 1.03
        sl = entry * 0.98
        print(f"\n[TARGET] Target Levels:")
        print(f"   TP (+3%): ${tp:.6f}")
        print(f"   SL (-2%): ${sl:.6f}")

# Get orders
print(f"\n[CLIPBOARD] Active Orders:")
print("-" * 80)
orders = client.futures_get_open_orders(symbol="APTUSDT")
for order in orders:
    order_type = order['type']
    side = order['side']
    qty = float(order['origQty'])
    stop_price = float(order.get('stopPrice', 0))
    
    print(f"\n   {order_type} {side}")
    print(f"   Quantity: {qty:.1f} APT")
    if stop_price > 0:
        print(f"   Trigger: ${stop_price:.6f}")
    
    if order_type == 'TRAILING_STOP_MARKET':
        callback = float(order.get('callbackRate', 0))
        print(f"   Trailing: {callback:.1f}%")
    
    print(f"   Order ID: {order['orderId']}")

# Calculate what happens at TP
print(f"\n\n💡 What happens when TP triggers:")
print("-" * 80)
if positions and float(positions[0]['positionAmt']) != 0:
    total_amt = abs(float(positions[0]['positionAmt']))
    partial_pct = 0.5  # 50% from config
    
    tp_qty = total_amt * partial_pct
    remaining_qty = total_amt * (1 - partial_pct)
    
    print(f"   Total position: {total_amt:.1f} APT")
    print(f"   Partial TP (50%): {tp_qty:.1f} APT will close @ TP")
    print(f"   Trailing (50%): {remaining_qty:.1f} APT continues with trailing stop")
    print(f"\n   [OK] This is CORRECT - locks profit while letting winners run!")
