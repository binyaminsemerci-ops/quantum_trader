"""Live monitoring of TP/SL/Partial profit system - følger med på alt"""
import asyncio
import os
import sys
from datetime import datetime
from binance.client import Client

sys.path.append(os.path.dirname(__file__))

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

async def monitor_system():
    """Continuous monitoring of all positions and protection orders"""
    
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
    
    check_count = 0
    tp_hit_tracker = {}  # Track when TP levels are hit
    sl_hit_tracker = {}  # Track when SL levels are hit
    
    while True:
        check_count += 1
        clear_screen()
        
        now = datetime.now().strftime("%H:%M:%S")
        print("=" * 100)
        print(f"[TARGET] QUANTUM TRADER - LIVE SYSTEM MONITOR | {now} | Check #{check_count}")
        print("=" * 100)
        
        try:
            # Get positions and orders
            positions = client.futures_position_information()
            active_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not active_positions:
                print("\n[OK] No active positions - system idle")
            else:
                print(f"\n[CHART] Active Positions: {len(active_positions)}")
                print("=" * 100)
                
                total_pnl = 0
                total_margin = 0
                
                for pos in active_positions:
                    symbol = pos['symbol']
                    amt = float(pos['positionAmt'])
                    entry = float(pos['entryPrice'])
                    current = float(pos['markPrice'])
                    pnl = float(pos['unRealizedProfit'])
                    leverage = int(pos['leverage'])
                    
                    direction = "[GREEN_CIRCLE] LONG" if amt > 0 else "[RED_CIRCLE] SHORT"
                    price_move = ((current - entry) / entry * 100) if entry else 0
                    
                    # Adjust price move for SHORT
                    if amt < 0:
                        price_move = -price_move
                    
                    margin_used = abs(amt * entry / leverage)
                    pnl_pct = (pnl / margin_used * 100) if margin_used else 0
                    
                    total_pnl += pnl
                    total_margin += margin_used
                    
                    # Get orders for this symbol
                    orders = client.futures_get_open_orders(symbol=symbol)
                    
                    tp_orders = [o for o in orders if o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']]
                    sl_orders = [o for o in orders if o['type'] in ['STOP_MARKET', 'STOP_LOSS']]
                    trail_orders = [o for o in orders if o['type'] == 'TRAILING_STOP_MARKET']
                    
                    # Calculate TP/SL levels
                    if amt > 0:  # LONG
                        tp_distance = ((float(tp_orders[0]['stopPrice']) - current) / current * 100) if tp_orders and tp_orders[0].get('stopPrice') else 0
                        sl_distance = ((current - float(sl_orders[0]['stopPrice'])) / current * 100) if sl_orders and sl_orders[0].get('stopPrice') else 0
                    else:  # SHORT
                        tp_distance = ((current - float(tp_orders[0]['stopPrice'])) / current * 100) if tp_orders and tp_orders[0].get('stopPrice') else 0
                        sl_distance = ((float(sl_orders[0]['stopPrice']) - current) / current * 100) if sl_orders and sl_orders[0].get('stopPrice') else 0
                    
                    print(f"\n{symbol} {direction} | {leverage}x leverage")
                    print(f"  Entry: ${entry:.6f} → Current: ${current:.6f} ({price_move:+.2f}%)")
                    print(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Margin: ${margin_used:.2f}")
                    
                    # Check if approaching TP or SL
                    if pnl_pct >= 2.5:
                        print(f"  [TARGET] NEARING TP! {pnl_pct:+.2f}% profit (target ~3%)")
                        if symbol not in tp_hit_tracker:
                            tp_hit_tracker[symbol] = datetime.now()
                    elif pnl_pct <= -1.8:
                        print(f"  [WARNING] NEARING SL! {pnl_pct:+.2f}% loss (stop -2%)")
                        if symbol not in sl_hit_tracker:
                            sl_hit_tracker[symbol] = datetime.now()
                    
                    # Protection status
                    print(f"\n  [SHIELD] Protection:")
                    
                    if tp_orders:
                        tp_qty = sum(float(o['origQty']) for o in tp_orders)
                        tp_pct = (tp_qty / abs(amt)) * 100
                        tp_price = float(tp_orders[0].get('stopPrice', 0))
                        print(f"    [OK] TP: {tp_qty:.1f} ({tp_pct:.0f}%) @ ${tp_price:.6f} | {tp_distance:.2f}% away")
                        
                        # Check for partial TP
                        if len(tp_orders) > 1 or tp_pct < 99:
                            print(f"    [CHART] PARTIAL TP ACTIVE: Will take {tp_pct:.0f}% profit first")
                    else:
                        print(f"    ❌ NO TP ORDER!")
                    
                    if trail_orders:
                        trail_qty = sum(float(o['origQty']) for o in trail_orders)
                        trail_pct = (trail_qty / abs(amt)) * 100
                        activation = float(trail_orders[0].get('activatePrice', 0))
                        callback = float(trail_orders[0].get('priceRate', 0))
                        print(f"    [OK] Trailing: {trail_qty:.1f} ({trail_pct:.0f}%) | Activation: ${activation:.6f} | Callback: {callback*100:.1f}%")
                    
                    if sl_orders:
                        sl_qty = sum(float(o['origQty']) for o in sl_orders)
                        sl_price = float(sl_orders[0].get('stopPrice', 0))
                        print(f"    [OK] SL: {sl_qty:.1f} (Full) @ ${sl_price:.6f} | {sl_distance:.2f}% away")
                    else:
                        print(f"    ❌ NO SL ORDER!")
                    
                    print("  " + "-" * 96)
                
                # Summary
                print("\n" + "=" * 100)
                total_pnl_pct = (total_pnl / total_margin * 100) if total_margin else 0
                print(f"[MONEY] TOTAL P&L: ${total_pnl:+.2f} ({total_pnl_pct:+.2f}%) on ${total_margin:.2f} margin")
                
            # Check recent trades for partial TP hits
            try:
                recent_trades = client.futures_account_trades(limit=10)
                for trade in recent_trades[-5:]:
                    symbol = trade['symbol']
                    if 'TAKE_PROFIT' in str(trade.get('positionSide', '')).upper() or trade.get('realizedPnl', 0) != '0':
                        time = datetime.fromtimestamp(trade['time'] / 1000).strftime("%H:%M:%S")
                        pnl = float(trade.get('realizedPnl', 0))
                        if pnl > 0:
                            print(f"\n[TARGET] Recent TP Hit: {symbol} @ {time} | Profit: ${pnl:+.2f}")
            except Exception as e:
                pass
            
            print("\n" + "=" * 100)
            print("[CHART] System Status:")
            print(f"  • Monitoring every 10 seconds")
            print(f"  • TP target: ~3% | SL protection: ~2%")
            print(f"  • Partial TP: Takes 50% profit, lets 50% run with trailing stop")
            print(f"  • Press Ctrl+C to stop")
            print("=" * 100)
            
        except KeyboardInterrupt:
            print("\n\n👋 Stopping monitor...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait 10 seconds
        await asyncio.sleep(10)

if __name__ == "__main__":
    print("[ROCKET] Starting Quantum Trader live monitor...")
    print("   Tracking: TP, SL, Partial Profit, P&L")
    print("   Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(monitor_system())
    except KeyboardInterrupt:
        print("\n\n[OK] Monitor stopped")
