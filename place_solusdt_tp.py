import os
from binance.client import Client

def place_tp_for_solusdt():
    """Place missing TP order for SOLUSDT LONG position"""
    
    api_key = os.getenv('BINANCE_TEST_API_KEY')
    api_secret = os.getenv('BINANCE_TEST_API_SECRET')
    
    client = Client(api_key, api_secret, testnet=True)
    
    symbol = 'SOLUSDT'
    
    # Get current position
    positions = client.futures_position_information(symbol=symbol)
    sol_pos = next((p for p in positions if p['symbol'] == symbol and float(p['positionAmt']) != 0), None)
    
    if not sol_pos:
        print("❌ No SOLUSDT position found!")
        return
    
    amt = float(sol_pos['positionAmt'])
    entry_price = float(sol_pos['entryPrice'])
    
    print(f"\n📊 SOLUSDT POSITION:")
    print(f"   Amount: {amt}")
    print(f"   Entry: ${entry_price:.2f}")
    
    is_long = amt > 0
    
    if not is_long:
        print("❌ Position is SHORT, not LONG!")
        return
    
    # Calculate TP at +3% for LONG
    tp_price = entry_price * 1.03
    tp_price = round(tp_price, 2)  # 2 decimals for SOLUSDT
    
    print(f"\n🎯 Placing TP order:")
    print(f"   TP Price: ${tp_price:.2f} (+3%)")
    
    try:
        # Place TAKE_PROFIT_MARKET order
        order = client.futures_create_order(
            symbol=symbol,
            side='SELL',  # Close LONG = SELL
            type='TAKE_PROFIT_MARKET',
            stopPrice=tp_price,
            closePosition=True,  # Close entire position
            workingType='MARK_PRICE',
            positionSide='LONG'
        )
        
        print(f"\n✅ TP ORDER PLACED!")
        print(f"   Order ID: {order['orderId']}")
        print(f"   Status: {order['status']}")
        
    except Exception as e:
        print(f"\n❌ Failed to place TP: {e}")

if __name__ == "__main__":
    place_tp_for_solusdt()
