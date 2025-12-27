#!/usr/bin/env python3
"""Verify all positions are on Binance Testnet and analyze current state"""

from binance.client import Client
import os
import sys

def main():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_testnet = os.getenv('BINANCE_USE_TESTNET', 'false').lower() == 'true'
    
    if not api_key or not api_secret:
        print("❌ API credentials not set!")
        sys.exit(1)
    
    print("="*70)
    print("BINANCE TESTNET POSITION VERIFICATION")
    print("="*70)
    print(f"Using Testnet: {use_testnet}")
    print(f"API Key: {api_key[:20]}...")
    print()
    
    client = Client(api_key, api_secret, testnet=use_testnet)
    
    # Get account info
    try:
        account = client.futures_account()
        print("✅ ACCOUNT INFORMATION")
        print(f"   Total Wallet Balance: {float(account['totalWalletBalance']):.2f} USDT")
        print(f"   Available Balance: {float(account['availableBalance']):.2f} USDT")
        print(f"   Total Unrealized PnL: {float(account['totalUnrealizedProfit']):.2f} USDT")
        print(f"   Margin Balance: {float(account['totalMarginBalance']):.2f} USDT")
        print()
    except Exception as e:
        print(f"❌ Failed to get account info: {e}")
        sys.exit(1)
    
    # Get all positions
    try:
        positions = client.futures_position_information()
        open_pos = [p for p in positions if float(p['positionAmt']) != 0]
        
        print(f"✅ OPEN POSITIONS: {len(open_pos)}")
        total_pnl = 0
        for pos in open_pos:
            symbol = pos['symbol']
            size = float(pos['positionAmt'])
            entry = float(pos['entryPrice'])
            mark = float(pos['markPrice'])
            pnl = float(pos['unRealizedProfit'])
            leverage = pos.get('leverage', 'N/A')
            liq_price = pos.get('liquidationPrice', 'N/A')
            
            side = "LONG" if size > 0 else "SHORT"
            pnl_pct = (pnl / (abs(size) * entry)) * 100 if entry > 0 else 0
            total_pnl += pnl
            
            status = "🟢" if pnl > 0 else "🔴"
            print(f"\n{status} {symbol}")
            print(f"   Side: {side} {abs(size)} @ ${entry:.4f}")
            print(f"   Mark Price: ${mark:.4f}")
            print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"   Leverage: {leverage}x")
            print(f"   Liquidation: {liq_price}")
        
        print(f"\n📊 TOTAL UNREALIZED PNL: ${total_pnl:.2f}")
        print()
    except Exception as e:
        print(f"❌ Failed to get positions: {e}")
        sys.exit(1)
    
    # Check for open orders (TP/SL)
    try:
        orders = client.futures_get_open_orders()
        print(f"✅ OPEN ORDERS: {len(orders)}")
        
        if len(orders) == 0:
            print("   ⚠️ NO ORDERS FOUND - This is the problem!")
        else:
            for order in orders:
                symbol = order['symbol']
                order_type = order['type']
                side = order['side']
                price = order.get('price', 'N/A')
                stop_price = order.get('stopPrice', 'N/A')
                
                print(f"\n   {symbol}: {order_type} {side}")
                print(f"      Price: {price}")
                print(f"      Stop Price: {stop_price}")
        print()
    except Exception as e:
        print(f"❌ Failed to get orders: {e}")
        sys.exit(1)
    
    # Analyze why no TP/SL
    print("="*70)
    print("🔍 ANALYSIS: Why No TP/SL Orders?")
    print("="*70)
    
    # Check position state file
    import json
    from pathlib import Path
    
    state_file = Path("/app/backend/data/position_state.json")
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print("📁 Position State File:")
        for symbol, data in state.items():
            print(f"\n   {symbol}:")
            print(f"      trail_percentage: {data.get('trail_percentage', 'NOT SET')}")
            print(f"      initial_stop_loss: {data.get('initial_stop_loss', 'NOT SET')}")
            print(f"      take_profit: {data.get('take_profit', 'NOT SET')}")
            print(f"      highest_profit_pct: {data.get('highest_profit_pct', 'NOT SET')}")
    else:
        print("⚠️ Position state file not found!")
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
