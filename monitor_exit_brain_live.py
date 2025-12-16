#!/usr/bin/env python3
"""
Real-time monitoring of Exit Brain V3 LIVE mode
Shows positions, PnL, and AI decisions
"""
import requests
import time
import os
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_positions():
    try:
        resp = requests.get("http://localhost:8000/positions", timeout=3)
        return resp.json() if resp.status_code == 200 else []
    except:
        return []

def get_health():
    try:
        resp = requests.get("http://localhost:8000/health", timeout=3)
        return resp.json() if resp.status_code == 200 else {}
    except:
        return {}

def monitor_loop():
    print("🔴 EXIT BRAIN V3 LIVE MODE - REAL-TIME MONITORING")
    print("=" * 80)
    print("Monitoring 4 Binance Testnet positions...")
    print("Press CTRL+C to stop\n")
    
    cycle = 0
    
    while True:
        try:
            cycle += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get health
            health = get_health()
            status = health.get('status', '?')
            
            # Get positions
            positions = get_positions()
            
            # Clear and redraw
            if cycle > 1:  # Don't clear on first iteration
                clear_screen()
            
            print(f"🔴 EXIT BRAIN V3 LIVE MODE MONITOR - Cycle #{cycle} @ {timestamp}")
            print("=" * 80)
            print(f"Backend: {status.upper()}")
            print(f"Positions: {len(positions)}")
            print("-" * 80)
            
            if not positions:
                print("⚠️  No positions found")
                print("   Waiting for data...")
            else:
                # Calculate totals
                total_pnl = sum(float(p.get('unrealized_pnl', 0)) for p in positions)
                
                # Header
                print(f"{'Symbol':<12} {'Side':<6} {'Qty':<12} {'Entry $':<10} {'Current $':<10} {'PnL $':<10}")
                print("-" * 80)
                
                # Positions
                for pos in positions:
                    symbol = pos['symbol']
                    side = pos['side']
                    qty = float(pos['quantity'])
                    entry = float(pos.get('entry_price', 0))
                    current = float(pos.get('current_price', entry))
                    pnl = float(pos.get('unrealized_pnl', 0))
                    
                    # Color code by PnL
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    
                    print(f"{pnl_emoji} {symbol:<10} {side:<6} {qty:>12,.2f} ${entry:>8,.2f} ${current:>8,.2f} ${pnl:>9,.2f}")
                
                print("-" * 80)
                total_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
                print(f"{total_emoji} TOTAL PnL: ${total_pnl:>9,.2f}")
            
            print("-" * 80)
            print(f"Exit Brain checks positions every 10 seconds")
            print(f"Legacy modules BLOCKED by exit_order_gateway")
            print(f"Next update in 10 seconds... (CTRL+C to stop)")
            
            # Wait 10 seconds (matching Exit Brain cycle)
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped by user")
            print("=" * 80)
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_loop()
