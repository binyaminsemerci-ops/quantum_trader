#!/usr/bin/env python3
"""
⏰ RL v3 TRAINING MONITOR
Watches for next training run and reports results in real-time
"""
import asyncio
import subprocess
import json
import time
from datetime import datetime, timezone, timedelta

print("=" * 80)
print("⏰ RL v3 TRAINING MONITOR - LIVE TRACKING")
print("=" * 80)
print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()

def get_last_training():
    """Get last training completion from logs"""
    try:
        result = subprocess.run(
            ["docker", "logs", "quantum_backend", "--tail", "500"],
            capture_output=True,
            text=True
        )
        
        training_lines = [line for line in result.stdout.split('\n') 
                         if 'Training completed' in line and 'RLv3' in line]
        
        if training_lines:
            last_line = training_lines[-1]
            data = json.loads(last_line)
            return {
                'timestamp': datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                'avg_reward': data.get('avg_reward', 0),
                'final_reward': data.get('final_reward', 0),
                'duration': data.get('duration_seconds', 0),
                'run_id': None
            }
    except Exception as e:
        print(f"⚠️  Error parsing logs: {e}")
    
    return None

def get_config():
    """Get training daemon config"""
    try:
        result = subprocess.run(
            ["docker", "exec", "quantum_backend", "python", "-c",
             "from backend.core.policy_store import PolicyStore; ps = PolicyStore.instance(); "
             "print(ps.get('rl_v3.training.interval_minutes', 30))"],
            capture_output=True,
            text=True
        )
        interval = int(result.stdout.strip())
        return {'interval_minutes': interval}
    except:
        return {'interval_minutes': 30}

# Get last training
last_training = get_last_training()
config = get_config()

if last_training:
    print(f"📊 LAST TRAINING RUN:")
    print(f"   Time: {last_training['timestamp'].strftime('%H:%M:%S UTC')}")
    print(f"   Avg Reward: {last_training['avg_reward']:.2f}")
    print(f"   Final Reward: {last_training['final_reward']:.2f}")
    print(f"   Duration: {last_training['duration']:.1f}s")
    print()
    
    # Calculate next run
    next_run = last_training['timestamp'] + timedelta(minutes=config['interval_minutes'])
    now = datetime.now(timezone.utc)
    time_until = (next_run - now).total_seconds()
    
    print(f"⏰ NEXT TRAINING RUN:")
    print(f"   Scheduled: {next_run.strftime('%H:%M:%S UTC')}")
    
    if time_until > 0:
        minutes = int(time_until // 60)
        seconds = int(time_until % 60)
        print(f"   Time until: {minutes}m {seconds}s")
        print()
        
        # Countdown
        print("⏳ COUNTDOWN TO NEXT TRAINING:")
        print("-" * 80)
        
        start_time = time.time()
        while time_until > 0:
            elapsed = time.time() - start_time
            remaining = max(0, time_until - elapsed)
            
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            
            print(f"\r   ⏰ {minutes:02d}:{seconds:02d} until next training run...", end='', flush=True)
            time.sleep(1)
            
            # Check if training started
            if remaining < 30:  # Last 30 seconds, check more frequently
                current_training = get_last_training()
                if current_training and current_training['timestamp'] > last_training['timestamp']:
                    print("\n\n🎉 NEW TRAINING RUN DETECTED!")
                    print("=" * 80)
                    break
        
        print("\n")
        
        # Wait a bit for training to complete
        print("⏳ Waiting for training to complete...")
        time.sleep(10)
        
        # Get new results
        new_training = get_last_training()
        
        if new_training and new_training['timestamp'] > last_training['timestamp']:
            print("\n✅ TRAINING COMPLETED!")
            print("=" * 80)
            print()
            print("📊 NEW RESULTS:")
            print(f"   Time: {new_training['timestamp'].strftime('%H:%M:%S UTC')}")
            print(f"   Avg Reward: {new_training['avg_reward']:.2f}")
            print(f"   Final Reward: {new_training['final_reward']:.2f}")
            print(f"   Duration: {new_training['duration']:.1f}s")
            print()
            
            # Compare with last run
            print("📈 COMPARISON WITH LAST RUN:")
            print("-" * 80)
            
            avg_diff = new_training['avg_reward'] - last_training['avg_reward']
            final_diff = new_training['final_reward'] - last_training['final_reward']
            
            avg_pct = (avg_diff / abs(last_training['avg_reward'])) * 100 if last_training['avg_reward'] != 0 else 0
            final_pct = (final_diff / abs(last_training['final_reward'])) * 100 if last_training['final_reward'] != 0 else 0
            
            print(f"   Avg Reward:   {avg_diff:+.2f} ({avg_pct:+.1f}%)")
            print(f"   Final Reward: {final_diff:+.2f} ({final_pct:+.1f}%)")
            print()
            
            if final_diff > 0:
                print("   🚀 IMPROVEMENT! AI is learning!")
            elif final_diff < 0:
                print("   📉 Decrease (exploration phase)")
            else:
                print("   ➡️  Stable")
            
            print()
            print("=" * 80)
            print(f"Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print("=" * 80)
        else:
            print("\n⚠️  Training may have been delayed or skipped")
            print("    Check backend logs for details")
    else:
        print(f"   ⚠️  Should have run {int(-time_until)} seconds ago")
        print(f"   Checking if training is running now...")
        
        # Check logs for recent activity
        result = subprocess.run(
            ["docker", "logs", "quantum_backend", "--tail", "50"],
            capture_output=True,
            text=True
        )
        
        if 'Training' in result.stdout and 'RUN_ID' in result.stdout:
            print("   ✅ Training activity detected in recent logs!")
        else:
            print("   ⚠️  No recent training activity")
else:
    print("⚠️  No previous training runs found in logs")
    print("   The daemon may have just started")
    print("   Next training will occur within 30 minutes")

print()
print("=" * 80)
print("💡 TIP: Run this script again to monitor the next training run!")
print("=" * 80)
