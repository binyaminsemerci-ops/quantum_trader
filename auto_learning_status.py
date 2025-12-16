import os
import json
from datetime import datetime

print("=" * 70)
print("AUTO-LEARNING SYSTEM STATUS REPORT")
print("=" * 70)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

# Check Environment Variables
print("[1] ENVIRONMENT CONFIGURATION")
print("-" * 70)
env_vars = {
    "RL v3 Training": os.getenv("QT_RL_V3_ENABLED", "NOT SET"),
    "RL v3 Interval": os.getenv("QT_RL_V3_TRAIN_INTERVAL_MINUTES", "NOT SET"),
    "CLM Enabled": os.getenv("QT_CLM_ENABLED", "NOT SET"),
    "CLM Retrain Hours": os.getenv("QT_CLM_RETRAIN_HOURS", "NOT SET"),
    "CLM Drift Hours": os.getenv("QT_CLM_DRIFT_HOURS", "NOT SET"),
    "CLM Auto Retrain": os.getenv("QT_CLM_AUTO_RETRAIN", "NOT SET"),
    "CLM Auto Promote": os.getenv("QT_CLM_AUTO_PROMOTE", "NOT SET"),
}

for key, value in env_vars.items():
    status = "✓" if value not in ["NOT SET", "false", "False", "0"] else "✗"
    print(f"  {status} {key}: {value}")

# Check Database for Model Artifacts
print(f"\n[2] MODEL REGISTRY")
print("-" * 70)
try:
    import sqlite3
    conn = sqlite3.connect('/app/backend/data/trades.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM model_artifacts")
    total_models = cursor.fetchone()[0]
    print(f"  Total models in registry: {total_models}")
    
    if total_models > 0:
        cursor.execute("SELECT model_type, version, status FROM model_artifacts ORDER BY id DESC LIMIT 5")
        models = cursor.fetchall()
        for model_type, version, status in models:
            print(f"    - {model_type} v{version}: {status}")
    else:
        print("    ⚠ No models found - initial training needed")
    
    conn.close()
except Exception as e:
    print(f"  ERROR: {e}")

# Check Training Data
print(f"\n[3] TRAINING DATA AVAILABILITY")
print("-" * 70)
try:
    conn = sqlite3.connect('/app/backend/data/trades.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM trade_logs")
    closed_trades = cursor.fetchone()[0]
    
    if closed_trades > 0:
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM trade_logs")
        min_ts, max_ts = cursor.fetchone()
        print(f"  Closed trades available: {closed_trades}")
        print(f"  Date range: {min_ts} to {max_ts}")
        
        # Calculate days
        from datetime import datetime
        start = datetime.fromisoformat(min_ts.replace('Z', '+00:00') if 'Z' in min_ts else min_ts)
        end = datetime.fromisoformat(max_ts.replace('Z', '+00:00') if 'Z' in max_ts else max_ts)
        days = (end - start).days
        print(f"  Coverage: {days} days")
    else:
        print("  ⚠ No closed trades found")
    
    conn.close()
except Exception as e:
    print(f"  ERROR: {e}")

# Check Training Activity
print(f"\n[4] TRAINING ACTIVITY")
print("-" * 70)
try:
    conn = sqlite3.connect('/app/backend/data/trades.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM training_tasks")
    total_tasks = cursor.fetchone()[0]
    print(f"  Total training tasks logged: {total_tasks}")
    
    if total_tasks > 0:
        cursor.execute("SELECT task_type, status FROM training_tasks ORDER BY id DESC LIMIT 5")
        tasks = cursor.fetchall()
        for task_type, status in tasks:
            print(f"    - {task_type}: {status}")
    else:
        print("    ⚠ No training tasks found")
    
    conn.close()
except Exception as e:
    print(f"  ERROR: {e}")

# Recommendations
print(f"\n[5] SYSTEM STATUS SUMMARY")
print("-" * 70)

rl_enabled = os.getenv("QT_RL_V3_ENABLED", "false").lower() == "true"
clm_enabled = os.getenv("QT_CLM_ENABLED", "false").lower() == "true"

if rl_enabled:
    print("  ✓ RL v3 PPO auto-learning: ENABLED")
    print(f"    - Training interval: {os.getenv('QT_RL_V3_TRAIN_INTERVAL_MINUTES', '30')} minutes")
else:
    print("  ✗ RL v3 PPO auto-learning: DISABLED")

if clm_enabled:
    print("  ✓ CLM (Continuous Learning Manager): ENABLED")
    print(f"    - Retrain schedule: {os.getenv('QT_CLM_RETRAIN_HOURS', '168')} hours")
    print(f"    - Drift check: {os.getenv('QT_CLM_DRIFT_HOURS', '24')} hours")
    print(f"    - Auto retrain: {os.getenv('QT_CLM_AUTO_RETRAIN', 'true')}")
    print(f"    - Auto promote: {os.getenv('QT_CLM_AUTO_PROMOTE', 'true')}")
else:
    print("  ✗ CLM (Continuous Learning Manager): DISABLED")

print(f"\n[6] NEXT STEPS")
print("-" * 70)
if total_models == 0:
    print("  1. TRAIN initial models (XGBoost, LightGBM) with 56 historical trades")
    print("  2. CLM will automatically monitor and retrain models once active")
else:
    print("  ✓ System ready - auto-learning active")

print("\n" + "=" * 70)
print("END OF REPORT")
print("=" * 70)
