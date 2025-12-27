#!/usr/bin/env python3
"""
Deep precision analysis - find the REAL problem
"""

from binance.client import Client
import os
from decimal import Decimal

c = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'), testnet=True)

def get_proper_precision(symbol):
    """Get exact precision requirements"""
    exchange_info = c.futures_exchange_info()
    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
    
    if not symbol_info:
        return None
    
    # Price precision
    price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
    tick_size = float(price_filter['tickSize']) if price_filter else 0.01
    
    # Quantity precision
    lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
    step_size = float(lot_filter['stepSize']) if lot_filter else 0.01
    min_qty = float(lot_filter['minQty']) if lot_filter else 0.01
    
    # Price precision digits
    tick_str = f"{tick_size:.10f}".rstrip('0')
    price_decimals = len(tick_str.split('.')[-1]) if '.' in tick_str else 0
    
    # Quantity precision digits
    step_str = f"{step_size:.10f}".rstrip('0')
    qty_decimals = len(step_str.split('.')[-1]) if '.' in step_str else 0
    
    return {
        'tick_size': tick_size,
        'step_size': step_size,
        'min_qty': min_qty,
        'price_decimals': price_decimals,
        'qty_decimals': qty_decimals
    }

def round_to_precision(value, precision_type, precision_info):
    """Round value EXACTLY to required precision"""
    if precision_type == 'price':
        decimals = precision_info['price_decimals']
        tick = precision_info['tick_size']
        # Round to nearest tick
        rounded = round(value / tick) * tick
        # Format with exact decimals
        return float(f"{rounded:.{decimals}f}")
    else:  # quantity
        decimals = precision_info['qty_decimals']
        step = precision_info['step_size']
        # Round to nearest step
        rounded = round(value / step) * step
        # Ensure minimum
        if rounded < precision_info['min_qty']:
            rounded = precision_info['min_qty']
        # Format with exact decimals
        return float(f"{rounded:.{decimals}f}")

print("=" * 80)
print("PRECISION-PERFECT ORDER TEST")
print("=" * 80)

# Get position
positions = [p for p in c.futures_position_information() if float(p['positionAmt']) != 0]

if not positions:
    print("No positions to test")
    exit(0)

test_pos = positions[0]
symbol = test_pos['symbol']
amt = float(test_pos['positionAmt'])

print(f"\nTesting: {symbol}")
print(f"Position: {amt}")

# Get precision
prec = get_proper_precision(symbol)
print(f"\nPrecision info:")
print(f"  Price: {prec['price_decimals']} decimals, tick={prec['tick_size']}")
print(f"  Qty: {prec['qty_decimals']} decimals, step={prec['step_size']}, min={prec['min_qty']}")

# Calculate test values
mark_price = float(test_pos['markPrice'])
side = 'SELL' if amt > 0 else 'BUY'
test_price_mult = 1.10 if amt > 0 else 0.90
test_price_raw = mark_price * test_price_mult
test_qty_raw = abs(amt) * 0.05

print(f"\nRaw values:")
print(f"  Price: {test_price_raw}")
print(f"  Qty: {test_qty_raw}")

# Round properly
test_price = round_to_precision(test_price_raw, 'price', prec)
test_qty = round_to_precision(test_qty_raw, 'quantity', prec)

print(f"\nRounded values:")
print(f"  Price: {test_price} ({type(test_price)})")
print(f"  Qty: {test_qty} ({type(test_qty)})")

# Verify format
print(f"\nFormatted check:")
print(f"  Price string: '{test_price}' has {len(str(test_price).split('.')[-1]) if '.' in str(test_price) else 0} decimals")
print(f"  Qty string: '{test_qty}' has {len(str(test_qty).split('.')[-1]) if '.' in str(test_qty) else 0} decimals")

# Try to create order
print(f"\nAttempting order:")
print(f"  {side} TAKE_PROFIT_MARKET")
print(f"  stopPrice={test_price}")
print(f"  quantity={test_qty}")

try:
    order = c.futures_create_order(
        symbol=symbol,
        side=side,
        type='TAKE_PROFIT_MARKET',
        stopPrice=test_price,
        quantity=test_qty,
        workingType='MARK_PRICE',
        positionSide='BOTH'
    )
    
    print(f"\n✅ SUCCESS! Order created: {order['orderId']}")
    print(f"   This proves precision is NOT the problem!")
    
    # Cancel it
    c.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
    print(f"   Test order cancelled")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    
    if '-1111' in str(e):
        print(f"\n🔍 PRECISION ERROR - Let me debug further:")
        print(f"   Expected price decimals: {prec['price_decimals']}")
        print(f"   Expected qty decimals: {prec['qty_decimals']}")
        print(f"   Actual price: {test_price}")
        print(f"   Actual qty: {test_qty}")
        
        # Try with different formatting
        print(f"\n   Trying alternative formatting...")
        try:
            order = c.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=f"{test_price:.{prec['price_decimals']}f}",
                quantity=f"{test_qty:.{prec['qty_decimals']}f}",
                workingType='MARK_PRICE',
                positionSide='BOTH'
            )
            print(f"   ✅ String formatting worked! Order: {order['orderId']}")
            c.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
        except Exception as e2:
            print(f"   ❌ Still failed: {e2}")
    
    elif '-4045' in str(e):
        print(f"\n⚠️  This confirms -4045 is the real blocker")
        print(f"   Precision was correct, but order limit hit")

print("\n" + "=" * 80)
