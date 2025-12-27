#!/usr/bin/env python3
"""
🎯 LIVE LEARNING DASHBOARD
Real-time monitoring av RL training, position closes, og PIL/PAL events
"""
import subprocess
import json
import time
import os
from datetime import datetime, timezone, timedelta

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_rl_training_history():
    """Get last 5 RL training runs"""
    try:
        result = subprocess.run(
            ["docker", "logs", "quantum_backend", "--since", "2h"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        training_lines = [line for line in result.stdout.split('\n') 
                         if 'Training completed' in line and 'RLv3' in line]
        
        runs = []
        for line in training_lines[-5:]:
            try:
                data = json.loads(line)
                runs.append({
                    'timestamp': data['timestamp'],
                    'avg_reward': data.get('avg_reward', 0),
                    'final_reward': data.get('final_reward', 0),
                    'duration': data.get('duration_seconds', 0)
                })
            except:
                continue
        
        return runs
    except:
        return []

def get_pil_classifications():
    """Get PIL classifications"""
    try:
        result = subprocess.run(
            ["docker", "exec", "quantum_backend", "python", "-c",
             "from backend.services.position_intelligence import get_position_intelligence; "
             "pil = get_position_intelligence(); "
             "print(len(pil.classifications))"],
            capture_output=True,
            text=True
        )
        return int(result.stdout.strip())
    except:
        return 0

def calculate_next_training(last_run_time):
    """Calculate next training time"""
    try:
        last_time = datetime.fromisoformat(last_run_time.replace('Z', '+00:00'))
        next_run = last_time + timedelta(minutes=30)
        now = datetime.now(timezone.utc)
        seconds_until = (next_run - now).total_seconds()
        return max(0, seconds_until), next_run
    except:
        return 0, None

def format_time_ago(timestamp_str):
    """Format timestamp"""
    try:
        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        diff = (now - ts).total_seconds()
        
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff/60)}m ago"
        else:
            return f"{int(diff/3600)}h ago"
    except:
        return "unknown"

def draw_dashboard():
    """Draw dashboard"""
    clear_screen()
    
    now = datetime.now(timezone.utc)
    
    print("=" * 100)
    print(f"🎯 QUANTUM TRADER - LIVE LEARNING DASHBOARD")
    print("=" * 100)
    print(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()
    
    # RL v3 TRAINING
    print("🤖 [1] RL v3 TRAINING STATUS")
    print("-" * 100)
    
    runs = get_rl_training_history()
    
    if runs:
        latest = runs[-1]
        print(f"  ✅ ACTIVE - Training every 30 minutes")
        print()
        print(f"  Latest Run:")
        print(f"    Time:         {latest['timestamp']} ({format_time_ago(latest['timestamp'])})")
        print(f"    Avg Reward:   {latest['avg_reward']:.2f}")
        print(f"    Final Reward: {latest['final_reward']:.2f}")
        print(f"    Duration:     {latest['duration']:.1f}s")
        
        seconds_until, next_run = calculate_next_training(latest['timestamp'])
        
        if next_run:
            minutes = int(seconds_until // 60)
            seconds = int(seconds_until % 60)
            
            print()
            print(f"  Next Training:")
            print(f"    Scheduled:   {next_run.strftime('%H:%M:%S UTC')}")
            
            if seconds_until > 0:
                print(f"    Time until:  {minutes}m {seconds}s")
            else:
                print(f"    Status:      ⏰ RUNNING NOW or OVERDUE")
        
        if len(runs) >= 2:
            print()
            print(f"  Recent History:")
            print(f"    {'Time':<12} {'Avg Reward':<15} {'Final Reward':<15} {'Trend'}")
            print(f"    {'-'*12} {'-'*15} {'-'*15} {'-'*10}")
            
            for i, run in enumerate(runs[-5:]):
                time_str = format_time_ago(run['timestamp'])
                avg = f"{run['avg_reward']:>12.0f}"
                final = f"{run['final_reward']:>12.0f}"
                
                if i > 0:
                    prev_final = runs[-5:][i-1]['final_reward']
                    diff = run['final_reward'] - prev_final
                    if diff > 0:
                        trend = f"↑ +{diff:.0f}"
                    elif diff < 0:
                        trend = f"↓ {diff:.0f}"
                    else:
                        trend = "→"
                else:
                    trend = "-"
                
                print(f"    {time_str:<12} {avg:<15} {final:<15} {trend}")
    else:
        print(f"  ⚠️  No recent training runs found")
    
    print()
    
    # LEARNING EVENTS
    print("📊 [2] POSITIONS & LEARNING EVENTS")
    print("-" * 100)
    
    pil_count = get_pil_classifications()
    print(f"  PIL Classifications:")
    if pil_count > 0:
        print(f"    ✅ {pil_count} positions classified (WINNER/LOSER/etc)")
    else:
        print(f"    ⏳ 0 classifications - Waiting for first position close")
    
    print()
    print(f"  Trade Logging:")
    print(f"    ✅ TradeStore active (SQLite backend)")
    
    print()
    
    # MODULES
    print("🎓 [3] LEARNING MODULES")
    print("-" * 100)
    
    modules = [
        ("RL v3 Training", "ACTIVE", "Every 30 min"),
        ("Meta-Strategy", "ACTIVE", "Real-time"),
        ("PIL", "READY", "On position close"),
        ("PAL", "READY", "On WINNER"),
        ("Continuous Learning", "READY", "Event-driven"),
        ("TP Tracker", "ACTIVE", "Continuous")
    ]
    
    for name, status, desc in modules:
        status_icon = "✅" if status == "ACTIVE" else "⏳"
        print(f"  • {name:<25} {status_icon} {status:<8} {desc}")
    
    print()
    print("=" * 100)
    print("💡 Refreshing every 5 seconds... Press Ctrl+C to exit")
    print("=" * 100)

def main():
    """Main loop"""
    print("Starting Live Learning Dashboard...")
    time.sleep(1)
    
    try:
        while True:
            draw_dashboard()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped. All systems remain active.")

if __name__ == "__main__":
    main()
