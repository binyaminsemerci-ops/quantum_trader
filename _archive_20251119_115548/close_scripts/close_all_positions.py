"""Emergency script to close all open positions on Binance Futures."""
import os
from binance.client import Client

def close_all_positions():
    """Close all open positions using Binance API directly."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ No Binance API credentials found!")
        return
    
    client = Client(api_key, api_secret)
    
    print("[SEARCH] Fetching open positions...")
    positions = client.futures_position_information()
    
    open_positions = [
        p for p in positions 
        if float(p.get('positionAmt', 0)) != 0
    ]
    
    if not open_positions:
        print("[OK] No open positions to close.")
        return
    
    print(f"\n[CHART] Found {len(open_positions)} open positions:")
    for pos in open_positions:
        sym = pos['symbol']
        qty = float(pos['positionAmt'])
        side = "LONG" if qty > 0 else "SHORT"
        entry = float(pos['entryPrice'])
        unrealized_pnl = float(pos['unRealizedProfit'])
        print(f"  - {sym}: {side} {abs(qty)} @ ${entry:.4f} (PnL: ${unrealized_pnl:.2f})")
    
    print("\n🛑 Closing all positions with MARKET orders...")
    
    results = []
    for pos in open_positions:
        try:
            sym = pos['symbol']
            qty = float(pos['positionAmt'])
            
            # Close side is opposite of position
            side = "SELL" if qty > 0 else "BUY"
            close_qty = abs(qty)
            
            print(f"  Closing {sym}: {side} {close_qty}...")
            
            # Submit MARKET order to close
            result = client.futures_create_order(
                symbol=sym,
                side=side,
                type="MARKET",
                quantity=close_qty,
                reduceOnly=True  # Ensure we only close, don't open new position
            )
            
            results.append((sym, "SUCCESS", result))
            print(f"    [OK] Closed {sym}")
            
        except Exception as e:
            results.append((sym, "FAILED", str(e)))
            print(f"    ❌ Failed {sym}: {e}")
    
    print("\n[CLIPBOARD] Summary:")
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    print(f"  [OK] Successfully closed: {success_count}/{len(results)}")
    if success_count < len(results):
        print(f"  ❌ Failed: {len(results) - success_count}")
    
    print("\n[OK] Done!")

if __name__ == "__main__":
    close_all_positions()
