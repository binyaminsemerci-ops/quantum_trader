"""Cancel all existing legacy orders on Binance to avoid conflicts with Exit Brain V3."""
from binance.client import Client
import os
import sys

try:
    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    
    print("\n🔍 Fetching all open orders...")
    orders = client.futures_get_open_orders()
    
    if len(orders) == 0:
        print("✅ No open orders found - nothing to cancel")
        sys.exit(0)
    
    print(f"\n📋 Found {len(orders)} open orders:\n")
    for o in orders:
        symbol = o['symbol']
        order_type = o['type']
        side = o['side']
        position_side = o.get('positionSide', 'N/A')
        price = o.get('stopPrice') or o.get('price', 'N/A')
        order_id = o['orderId']
        
        print(f"  {symbol:12} {side:4} {position_side:5} {order_type:15} @ ${price} (ID: {order_id})")
    
    # Auto-confirm in Docker environment
    print("\n🚀 Auto-confirming in Docker environment...")
    
    print("\n🗑️  Cancelling orders...")
    cancelled_count = 0
    failed_count = 0
    
    for o in orders:
        try:
            client.futures_cancel_order(
                symbol=o['symbol'],
                orderId=o['orderId']
            )
            print(f"  ✅ Cancelled {o['symbol']} order {o['orderId']}")
            cancelled_count += 1
        except Exception as e:
            print(f"  ❌ Failed to cancel {o['symbol']} order {o['orderId']}: {e}")
            failed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Cancelled: {cancelled_count}")
    print(f"  ❌ Failed: {failed_count}")
    print(f"  📝 Total: {len(orders)}")
    
    if cancelled_count > 0:
        print("\n🎯 Exit Brain V3 can now manage exits without conflicts!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
