#!/usr/bin/env python
"""Check and clean ALL orders including conditional/algo orders"""
from binance.client import Client
import os
import time

client = Client(
    os.getenv('BINANCE_TEST_API_KEY'),
    os.getenv('BINANCE_TEST_API_SECRET'),
    testnet=True
)

print("\n" + "="*60)
print("  🧹 QUANTUM TRADER - ORDER CLEANUP TOOL")
print("="*60 + "\n")

# Get all open positions
positions = client.futures_position_information()
open_positions = {p['symbol']: p for p in positions if float(p['positionAmt']) != 0}

print(f"📊 Open Positions: {len(open_positions)}")
for symbol, pos in open_positions.items():
    amt = float(pos['positionAmt'])
    side = "LONG" if amt > 0 else "SHORT"
    print(f"  - {symbol}: {side} {abs(amt)} @ ${float(pos['entryPrice']):.2f}")

print(f"\n{'='*60}\n")

# Get ALL orders (including conditional/algo orders)
recent_time = int(time.time() * 1000) - (7 * 24 * 60 * 60 * 1000)  # Last 7 days

total_orders = 0
by_symbol = {}

for symbol in [p['symbol'] for p in positions]:  # Check all symbols, not just open positions
    try:
        # Get all recent orders
        all_orders = client.futures_get_all_orders(symbol=symbol, startTime=recent_time, limit=500)
        
        # Filter to only OPEN orders (NEW or PARTIALLY_FILLED)
        open_orders = [o for o in all_orders if o.get('status') in ['NEW', 'PARTIALLY_FILLED']]
        
        if open_orders:
            by_symbol[symbol] = open_orders
            total_orders += len(open_orders)
    except Exception as e:
        print(f"⚠️  Error checking {symbol}: {e}")

print(f"📋 Total OPEN orders across all symbols: {total_orders}\n")

if total_orders == 0:
    print("✅ No open orders found!")
    exit(0)

# Show orders by symbol
print("Orders by symbol:")
for symbol, orders in sorted(by_symbol.items()):
    has_position = symbol in open_positions
    status_icon = "✅" if has_position else "🗑️"
    
    print(f"\n{status_icon} {symbol}: {len(orders)} orders {'(HAS POSITION)' if has_position else '(NO POSITION - ORPHANED)'}")
    
    # Group by order type
    type_counts = {}
    for o in orders:
        order_type = o['type']
        type_counts[order_type] = type_counts.get(order_type, 0) + 1
    
    for order_type, count in sorted(type_counts.items()):
        print(f"    - {order_type}: {count}")

print(f"\n{'='*60}\n")

# Ask user what to do
print("🔧 CLEANUP OPTIONS:")
print("  1. Delete ALL orphaned orders (symbols with NO position)")
print("  2. Delete DUPLICATE orders (keep only 1 TP and 1 SL per position)")  
print("  3. Delete EVERYTHING and let system recreate")
print("  4. Cancel (exit without changes)")

choice = input("\nYour choice (1-4): ").strip()

if choice == "1":
    # Delete only orphaned orders
    orphaned_symbols = [s for s in by_symbol.keys() if s not in open_positions]
    
    if not orphaned_symbols:
        print("\n✅ No orphaned orders found!")
        exit(0)
    
    print(f"\n🗑️  Deleting orders for {len(orphaned_symbols)} symbols with NO position...")
    
    deleted = 0
    for symbol in orphaned_symbols:
        try:
            result = client.futures_cancel_all_open_orders(symbol=symbol)
            deleted += len(by_symbol[symbol])
            print(f"  ✅ Deleted {len(by_symbol[symbol])} orders for {symbol}")
        except Exception as e:
            print(f"  ❌ Failed to delete {symbol}: {e}")
    
    print(f"\n✅ Total deleted: {deleted} orphaned orders")

elif choice == "2":
    # Keep only 1 TP and 1 SL per position, delete duplicates
    print(f"\n🔧 Removing duplicate orders (keeping newest TP and SL)...")
    
    deleted = 0
    for symbol, orders in by_symbol.items():
        if symbol not in open_positions:
            continue  # Skip symbols without position
        
        # Group by type
        tp_orders = [o for o in orders if o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']]
        sl_orders = [o for o in orders if o['type'] in ['STOP_MARKET', 'STOP', 'STOP_LOSS', 'TRAILING_STOP_MARKET']]
        
        # If we have duplicates, keep only the newest one
        to_delete = []
        
        if len(tp_orders) > 1:
            # Sort by time, keep newest
            tp_orders_sorted = sorted(tp_orders, key=lambda x: x['time'], reverse=True)
            to_delete.extend(tp_orders_sorted[1:])  # Delete all except first (newest)
            print(f"  {symbol}: Removing {len(tp_orders)-1} duplicate TP orders")
        
        if len(sl_orders) > 1:
            # Sort by time, keep newest
            sl_orders_sorted = sorted(sl_orders, key=lambda x: x['time'], reverse=True)
            to_delete.extend(sl_orders_sorted[1:])  # Delete all except first (newest)
            print(f"  {symbol}: Removing {len(sl_orders)-1} duplicate SL orders")
        
        # Delete the duplicates
        for order in to_delete:
            try:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                deleted += 1
            except Exception as e:
                print(f"    ❌ Failed to delete order {order['orderId']}: {e}")
    
    print(f"\n✅ Total deleted: {deleted} duplicate orders")

elif choice == "3":
    # Delete EVERYTHING
    print(f"\n⚠️  WARNING: This will delete ALL {total_orders} orders!")
    confirm = input("Type 'DELETE ALL' to confirm: ")
    
    if confirm == "DELETE ALL":
        deleted = 0
        for symbol in by_symbol.keys():
            try:
                result = client.futures_cancel_all_open_orders(symbol=symbol)
                deleted += len(by_symbol[symbol])
                print(f"  ✅ Deleted {len(by_symbol[symbol])} orders for {symbol}")
            except Exception as e:
                print(f"  ❌ Failed to delete {symbol}: {e}")
        
        print(f"\n✅ Total deleted: {deleted} orders")
        print("\n🔄 System will recreate TP/SL protection automatically...")
    else:
        print("\n❌ Cancelled - no changes made")

else:
    print("\n❌ Cancelled - no changes made")

print(f"\n{'='*60}")
print("  Done!")
print("="*60 + "\n")
