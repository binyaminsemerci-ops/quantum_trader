"""Real-time monitoring of Position Monitor and Trailing Stop Manager"""
import asyncio
import os
from datetime import datetime
from binance.client import Client

def load_env():
    """Load API credentials from .env"""
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

async def monitor_positions():
    """Monitor positions with TP/SL and show real-time updates"""
    
    api_key, api_secret = load_env()
    client = Client(api_key, api_secret)
    
    print("\n" + "=" * 100)
    print("[SEARCH] POSITION MONITOR - TP/SL/TRAILING STOP TRACKER")
    print("=" * 100)
    print("Config: TP=3.0% | SL=2.0% | Trailing=1.5%")
    print("Monitoring every 10 seconds... Press Ctrl+C to stop\n")
    
    position_history = {}
    
    try:
        while True:
            positions = client.futures_position_information()
            active_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            if not active_positions:
                print(f"[{timestamp}] ⭕ No active positions")
            else:
                print(f"\n[{timestamp}] [CHART] Active Positions: {len(active_positions)}")
                print("-" * 100)
                
                for pos in active_positions:
                    symbol = pos['symbol']
                    qty = float(pos['positionAmt'])
                    entry_price = float(pos['entryPrice'])
                    current_price = float(pos['markPrice'])
                    unrealized_pnl = float(pos['unRealizedProfit'])
                    
                    # Calculate PnL percentage
                    if entry_price > 0:
                        if qty > 0:  # LONG
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100
                            direction = "[GREEN_CIRCLE] LONG"
                            tp_price = entry_price * 1.03  # +3%
                            sl_price = entry_price * 0.98  # -2%
                            trail_trigger = entry_price * 1.015  # +1.5%
                        else:  # SHORT
                            pnl_pct = ((entry_price - current_price) / entry_price) * 100
                            direction = "[RED_CIRCLE] SHORT"
                            tp_price = entry_price * 0.97  # -3%
                            sl_price = entry_price * 1.02  # +2%
                            trail_trigger = entry_price * 0.985  # -1.5%
                    else:
                        pnl_pct = 0
                        direction = "⚪"
                        tp_price = sl_price = trail_trigger = 0
                    
                    # Status indicators
                    status = []
                    if pnl_pct >= 3.0:
                        status.append("[TARGET] TP TARGET REACHED!")
                    elif pnl_pct <= -2.0:
                        status.append("🛑 SL TARGET REACHED!")
                    elif pnl_pct >= 1.5:
                        status.append("[CHART_UP] TRAILING ACTIVE")
                    elif pnl_pct > 0:
                        status.append("[OK] In Profit")
                    else:
                        status.append("[WARNING] In Loss")
                    
                    # Track if position changed
                    prev_data = position_history.get(symbol, {})
                    if prev_data:
                        prev_pnl = prev_data.get('pnl_pct', 0)
                        if abs(pnl_pct - prev_pnl) > 0.1:
                            if pnl_pct > prev_pnl:
                                status.append("⬆️")
                            else:
                                status.append("⬇️")
                    
                    position_history[symbol] = {'pnl_pct': pnl_pct, 'price': current_price}
                    
                    print(f"\n{direction} {symbol}")
                    print(f"  Entry:   ${entry_price:>12.6f}")
                    print(f"  Current: ${current_price:>12.6f}  |  PnL: {pnl_pct:>+7.2f}% (${unrealized_pnl:>+8.2f})")
                    print(f"  TP:      ${tp_price:>12.6f}  (+3.0%)")
                    print(f"  SL:      ${sl_price:>12.6f}  (-2.0%)")
                    print(f"  Trail:   ${trail_trigger:>12.6f}  (+1.5%)")
                    print(f"  Status:  {' '.join(status)}")
                    
                    # Check for orders (TP/SL orders set by system)
                    try:
                        open_orders = client.futures_get_open_orders(symbol=symbol)
                        if open_orders:
                            print(f"  Orders:  {len(open_orders)} active orders:")
                            for order in open_orders:
                                order_type = order['type']
                                side = order['side']
                                stop_price = float(order.get('stopPrice', 0))
                                print(f"           - {order_type} {side} @ ${stop_price:.6f}")
                    except:
                        pass
            
            print("-" * 100)
            
            # Wait 10 seconds
            await asyncio.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n[OK] Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(monitor_positions())
