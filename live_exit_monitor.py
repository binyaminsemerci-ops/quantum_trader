#!/usr/bin/env python3
"""
Live Exit Logic Monitor - Testnet Positions
Monitors exit monitor service and position changes in real-time
"""

import redis
import time
import subprocess
from datetime import datetime
from typing import Dict

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Track previous state
previous_positions: Dict[str, Dict] = {}

def get_active_positions():
    """Get all active positions"""
    keys = r.keys("quantum:position:snapshot:*")
    positions = {}
    
    for key in keys:
        data = r.hgetall(key)
        amt = float(data.get("position_amt", 0))
        
        if amt != 0:
            symbol = key.split(":")[-1]
            positions[symbol] = {
                "size": amt,
                "entry": float(data.get("entry_price", 0)),
                "mark": float(data.get("mark_price", 0)),
                "pnl": float(data.get("unrealized_pnl", 0)),
                "leverage": data.get("leverage", "?"),
                "side": "LONG" if amt > 0 else "SHORT"
            }
    
    return positions

def check_exit_monitor_activity():
    """Check exit monitor logs for recent activity"""
    try:
        result = subprocess.run([
            "journalctl", "-u", "quantum-exit-monitor",
            "--since", "10 seconds ago", "--no-pager", "-q"
        ], capture_output=True, text=True, timeout=5)
        
        if result.stdout:
            lines = [l for l in result.stdout.split("\n") if l.strip()]
            if lines:
                # Return last few relevant lines
                relevant = [l for l in lines if any(
                    word in l.lower() 
                    for word in ["exit", "stop", "evaluate", "triggered"]
                )]
                return relevant[-3:] if relevant else []
        return []
    except Exception as e:
        return [f"Error checking logs: {e}"]

def format_change(old_val, new_val, prefix=""):
    """Format value change with color indicator"""
    diff = new_val - old_val
    if abs(diff) < 0.01:
        return ""
    
    sign = "📈" if diff > 0 else "📉"
    return f"{prefix} {sign} {diff:+.2f}"

print("=" * 70)
print("🔍 LIVE EXIT LOGIC MONITOR - TESTNET POSITIONS")
print("=" * 70)
print("")
print("Monitoring:")
print("  • Exit Monitor V2 service logs")
print("  • Position changes (price, PnL)")
print("  • Exit signals and triggers")
print("")
print("Press Ctrl+C to stop")
print("=" * 70)
print("")

# Initial snapshot
previous_positions = get_active_positions()
print(f"📊 Tracking {len(previous_positions)} active positions")
print("")

iteration = 0
while True:
    try:
        iteration += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp}] Check #{iteration}")
        
        # Check exit monitor activity
        exit_logs = check_exit_monitor_activity()
        if exit_logs:
            print("  🚨 EXIT MONITOR ACTIVITY:")
            for log in exit_logs:
                print(f"     {log}")
        
        # Check position changes
        current_positions = get_active_positions()
        
        # Detect changes
        changes_detected = False
        for symbol, pos in current_positions.items():
            if symbol in previous_positions:
                prev = previous_positions[symbol]
                
                # Check for significant changes
                pnl_change = pos["pnl"] - prev["pnl"]
                mark_change = pos["mark"] - prev["mark"]
                mark_change_pct = (mark_change / prev["mark"]) * 100 if prev["mark"] > 0 else 0
                
                if abs(pnl_change) > 0.5 or abs(mark_change_pct) > 0.5:
                    if not changes_detected:
                        print("  📍 POSITION CHANGES:")
                        changes_detected = True
                    
                    print(f"     {symbol:15} {pos['side']:5}")
                    print(f"        Mark:  {pos['mark']:.6f} ({mark_change_pct:+.2f}%)")
                    print(f"        PnL:   {pos['pnl']:>8.2f} USDT ({pnl_change:+.2f})")
            else:
                # New position
                print(f"  ✨ NEW POSITION: {symbol} {pos['side']} {pos['leverage']}x")
        
        # Check for closed positions
        for symbol in previous_positions:
            if symbol not in current_positions:
                print(f"  ✅ CLOSED: {symbol}")
        
        if not exit_logs and not changes_detected:
            print("  💤 No activity")
        
        print("")
        
        # Update previous state
        previous_positions = current_positions
        
        # Sleep between checks
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print(f"Monitoring stopped after {iteration} iterations")
        print(f"Final active positions: {len(current_positions)}")
        print("=" * 70)
        break
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        time.sleep(5)
