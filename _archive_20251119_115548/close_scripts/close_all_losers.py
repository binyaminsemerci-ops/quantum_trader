#!/usr/bin/env python3
"""
Emergency script to close ALL losing positions on Binance Futures.
Closes positions with negative unrealized P&L immediately at market price.
"""

import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Binance client
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

def close_all_losing_positions():
    """Close all positions with negative unrealized P&L."""
    print("[ALERT] EMERGENCY: Closing all losing positions...")
    print("=" * 60)
    
    # Get all open positions
    positions = client.futures_position_information()
    
    closed_count = 0
    total_loss = 0.0
    
    for position in positions:
        pos_amt = float(position['positionAmt'])
        if pos_amt == 0:
            continue
            
        symbol = position['symbol']
        unrealized_pnl = float(position['unRealizedProfit'])
        
        # Only close losing positions
        if unrealized_pnl < 0:
            print(f"\n📉 {symbol}")
            print(f"   Position Size: {pos_amt}")
            print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
            
            try:
                # Determine side (opposite of current position)
                side = 'SELL' if pos_amt > 0 else 'BUY'
                quantity = abs(pos_amt)
                
                # Close position at market price
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity,
                    reduceOnly=True
                )
                
                print(f"   [OK] CLOSED - Order ID: {order['orderId']}")
                closed_count += 1
                total_loss += unrealized_pnl
                
            except Exception as e:
                print(f"   ❌ ERROR closing {symbol}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"[CHECKERED_FLAG] RESULTS:")
    print(f"   Positions Closed: {closed_count}")
    print(f"   Total Realized Loss: ${total_loss:.2f}")
    print("=" * 60)
    
    # Get updated account balance
    try:
        account = client.futures_account()
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        print(f"\n[MONEY] UPDATED BALANCE:")
        print(f"   Total Balance: ${balance:.2f} USDT")
        print(f"   Available: ${available:.2f} USDT")
    except Exception as e:
        print(f"   [WARNING] Could not fetch balance: {str(e)}")

if __name__ == "__main__":
    close_all_losing_positions()
