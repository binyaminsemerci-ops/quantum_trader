"""
Live Exit Logic Monitor - Watch exit evaluations in real-time
"""
import time
import requests
import subprocess
from datetime import datetime

print("🔍 LIVE EXIT LOGIC MONITOR")
print("=" * 60)
print("")
print("Tracking 20 testnet positions with V2 dynamic exit math...")
print("Monitoring for exit triggers every 10 seconds")
print("")
print("Press Ctrl+C to stop")
print("=" * 60)
print("")

last_check_time = None
iteration = 0

while True:
    iteration += 1
    ts = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n[{ts}] Check #{iteration}")
    print("-" * 60)
    
    try:
        # Get health status
        health = requests.get("http://localhost:8007/health", timeout=5).json()
        
        tracked = health.get("tracked_positions", 0)
        exits = health.get("exits_triggered", 0)
        last_eval = health.get("last_exit_evaluation", "Never")
        
        # Exit breakdown
        liq_exits = health.get("liq_protection_exits", 0)
        risk_exits = health.get("risk_stop_exits", 0)
        trail_exits = health.get("trailing_stop_exits", 0)
        time_exits = health.get("time_exits", 0)
        
        print(f"📊 Status: {tracked} positions tracked | {exits} total exits triggered")
        
        if exits > 0:
            print(f"   Exit breakdown:")
            print(f"   - Liq Protection: {liq_exits}")
            print(f"   - Risk Stop: {risk_exits}")
            print(f"   - Trailing Stop: {trail_exits}")
            print(f"   - Time-based: {time_exits}")
        
        # Check for recent logs (exit evaluations)
        if iteration % 3 == 0:  # Every 30s check logs
            result = subprocess.run(
                ["journalctl", "-u", "quantum-exit-monitor", "--since", "30 seconds ago", "-n", "20", "--no-pager"],
                capture_output=True,
                text=True
            )
            
            if "evaluate_position_exit" in result.stdout or "EXIT" in result.stdout:
                print()
                print("🚨 RECENT EXIT ACTIVITY:")
                for line in result.stdout.split("\n")[-5:]:
                    if line.strip():
                        print(f"   {line[-100:]}")  # Last 100 chars
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    time.sleep(10)
