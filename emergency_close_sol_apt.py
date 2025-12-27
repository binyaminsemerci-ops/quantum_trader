"""Emergency close SOLUSDT and APTUSDT positions - stopping the bleeding"""
import os
from binance.client import Client
from datetime import datetime

def close_positions():
    """Close SOLUSDT and APTUSDT positions immediately"""
    
    # Get API keys
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key:
        with open(".env") as f:
            for line in f:
                if line.startswith("BINANCE_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                elif line.startswith("BINANCE_API_SECRET="):
                    api_secret = line.split("=", 1)[1].strip()
    
    client = Client(api_key, api_secret)
    
    print("\n" + "=" * 80)
    print("[ALERT] EMERGENCY POSITION CLOSURE")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nClosing SOLUSDT and APTUSDT LONG positions to stop losses...")
    print("=" * 80)
    
    symbols_to_close = ["SOLUSDT", "APTUSDT"]
    
    for symbol in symbols_to_close:
        try:
            # Get current position
            positions = client.futures_position_information(symbol=symbol)
            position = positions[0] if positions else None
            
            if not position:
                print(f"\n❌ {symbol}: No position data")
                continue
            
            amt = float(position['positionAmt'])
            
            if amt == 0:
                print(f"\n[OK] {symbol}: No position to close")
                continue
            
            entry = float(position['entryPrice'])
            current = float(position['markPrice'])
            pnl = float(position['unRealizedProfit'])
            pnl_pct = (pnl / abs(amt * entry)) * 100 if entry else 0
            
            print(f"\n[CHART] {symbol}:")
            print(f"   Position: {abs(amt):.4f} {'LONG' if amt > 0 else 'SHORT'}")
            print(f"   Entry: ${entry:.6f}")
            print(f"   Current: ${current:.6f}")
            print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # Cancel all open orders first
            print(f"\n🗑️ Canceling all open orders for {symbol}...")
            try:
                result = client.futures_cancel_all_open_orders(symbol=symbol)
                print(f"   [OK] Canceled {len(result)} orders")
            except Exception as e:
                print(f"   [WARNING] Cancel orders error: {e}")
            
            # Close position with market order
            print(f"\n[RED_CIRCLE] CLOSING POSITION...")
            
            side = "SELL" if amt > 0 else "BUY"  # Opposite side to close
            
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=abs(amt),
                reduceOnly=True
            )
            
            print(f"   [OK] Market order executed!")
            print(f"   Order ID: {order['orderId']}")
            print(f"   Status: {order['status']}")
            print(f"   Filled: {order.get('executedQty', 'N/A')}")
            
            # Verify position is closed
            import time
            time.sleep(1)
            
            new_positions = client.futures_position_information(symbol=symbol)
            new_amt = float(new_positions[0]['positionAmt']) if new_positions else 0
            
            if new_amt == 0:
                print(f"\n   [OK][OK] {symbol} POSITION CLOSED!")
                print(f"   Final P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            else:
                print(f"\n   [WARNING] Warning: Position still shows {new_amt} amount")
            
        except Exception as e:
            print(f"\n❌ Error closing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("[CHART] SUMMARY")
    print("=" * 80)
    
    # Check remaining positions
    try:
        all_positions = client.futures_position_information()
        remaining = [p for p in all_positions if float(p['positionAmt']) != 0]
        
        if remaining:
            print(f"\n[OK] Remaining positions: {len(remaining)}")
            total_pnl = 0
            for pos in remaining:
                sym = pos['symbol']
                amt = float(pos['positionAmt'])
                pnl = float(pos['unRealizedProfit'])
                total_pnl += pnl
                direction = "LONG" if amt > 0 else "SHORT"
                print(f"   • {sym} {direction}: ${pnl:+.2f}")
            print(f"\n   Total remaining P&L: ${total_pnl:+.2f}")
        else:
            print("\n[OK] All positions closed!")
        
        # Show account balance
        account = client.futures_account()
        balance = float(account['totalWalletBalance'])
        available = float(account['availableBalance'])
        unrealized = float(account['totalUnrealizedProfit'])
        
        print(f"\n[MONEY] Account Status:")
        print(f"   Total Balance: ${balance:.2f}")
        print(f"   Available: ${available:.2f}")
        print(f"   Unrealized P&L: ${unrealized:+.2f}")
        
    except Exception as e:
        print(f"\n[WARNING] Could not fetch summary: {e}")
    
    print("\n" + "=" * 80)
    print("[OK] Emergency closure complete")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    close_positions()
