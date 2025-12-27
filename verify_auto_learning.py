"""
Verify Auto-Learning System Status

Checks:
1. RL v3 PPO training (every 30 minutes)
2. CLM continuous learning (periodic checks)
3. Model registry status
4. Recent training activity
"""

import sys
sys.path.insert(0, '/app')

from backend.database import SessionLocal
from sqlalchemy import text
from datetime import datetime, timedelta

db = SessionLocal()

print("\n" + "=" * 70)
print("AUTO-LEARNING SYSTEM STATUS REPORT")
print("=" * 70)
print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

# 1. Check RL v3 Status from PolicyStore
print("[1] RL v3 PPO TRAINING")
print("-" * 70)
try:
    result = db.execute(text("SELECT * FROM policy_store WHERE key LIKE '%rl_v3%'"))
    rl_policies = result.fetchall()
    
    if rl_policies:
        print("Status: ACTIVE")
        for policy in rl_policies:
            key = policy[1]  # key column
            value = policy[2]  # value column
            print(f"  {key}: {value}")
    else:
        print("Status: NOT CONFIGURED")
except Exception as e:
    print(f"ERROR: {e}")

# 2. Check CLM Configuration
print("\n[2] CLM (CONTINUOUS LEARNING MANAGER)")
print("-" * 70)
try:
    result = db.execute(text("SELECT * FROM policy_store WHERE key LIKE '%clm%'"))
    clm_policies = result.fetchall()
    
    if clm_policies:
        print("Status: CONFIGURED")
        for policy in clm_policies:
            key = policy[1]
            value = policy[2]
            print(f"  {key}: {value}")
    else:
        print("Status: NOT CONFIGURED (using defaults)")
except Exception as e:
    print(f"ERROR: {e}")

# 3. Check Model Registry
print("\n[3] MODEL REGISTRY")
print("-" * 70)
try:
    result = db.execute(text("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
               SUM(CASE WHEN status='shadow' THEN 1 ELSE 0 END) as shadow,
               SUM(CASE WHEN status='training' THEN 1 ELSE 0 END) as training
        FROM model_artifacts
    """))
    stats = result.fetchone()
    
    print(f"Total models: {stats[0]}")
    print(f"  Active: {stats[1]}")
    print(f"  Shadow: {stats[2]}")
    print(f"  Training: {stats[3]}")
    
    # Show recent models
    result = db.execute(text("""
        SELECT model_type, version, status, created_at
        FROM model_artifacts
        ORDER BY created_at DESC
        LIMIT 5
    """))
    recent = result.fetchall()
    
    if recent:
        print("\nRecent models:")
        for model in recent:
            print(f"  {model[0]} v{model[1]} - {model[2]} ({model[3]})")
    
except Exception as e:
    print(f"ERROR: {e}")

# 4. Check Training Tasks
print("\n[4] TRAINING ACTIVITY")
print("-" * 70)
try:
    # Check recent training runs
    result = db.execute(text("""
        SELECT COUNT(*) as total,
               MAX(timestamp) as last_run
        FROM training_tasks
        WHERE timestamp >= datetime('now', '-7 days')
    """))
    stats = result.fetchone()
    
    print(f"Training runs (last 7 days): {stats[0]}")
    print(f"Last training: {stats[1] if stats[1] else 'Never'}")
    
    # Show recent training tasks
    result = db.execute(text("""
        SELECT task_type, model_type, status, timestamp
        FROM training_tasks
        ORDER BY timestamp DESC
        LIMIT 5
    """))
    recent = result.fetchall()
    
    if recent:
        print("\nRecent training tasks:")
        for task in recent:
            print(f"  {task[0]} - {task[1]}: {task[2]} ({task[3]})")
    
except Exception as e:
    print(f"ERROR: {e}")

# 5. Check Trade Data Availability
print("\n[5] TRAINING DATA AVAILABILITY")
print("-" * 70)
try:
    result = db.execute(text("""
        SELECT COUNT(*) as total,
               MIN(timestamp) as oldest,
               MAX(timestamp) as newest
        FROM trade_logs
        WHERE status='CLOSED' AND exit_price IS NOT NULL
    """))
    stats = result.fetchone()
    
    print(f"Closed trades available: {stats[0]}")
    if stats[0] > 0:
        print(f"  Date range: {stats[1]} to {stats[2]}")
        
        # Calculate days of data
        if stats[1] and stats[2]:
            from datetime import datetime
            oldest = datetime.fromisoformat(stats[1])
            newest = datetime.fromisoformat(stats[2])
            days = (newest - oldest).days
            print(f"  Coverage: {days} days")
    
except Exception as e:
    print(f"ERROR: {e}")

# 6. System Recommendations
print("\n[6] SYSTEM RECOMMENDATIONS")
print("-" * 70)

recommendations = []

# Check if RL v3 is active
try:
    result = db.execute(text("SELECT value FROM policy_store WHERE key='rl_v3.training.enabled'"))
    rl_enabled = result.scalar()
    if not rl_enabled or rl_enabled.lower() != 'true':
        recommendations.append("ENABLE RL v3 training in PolicyStore")
except:
    recommendations.append("CONFIGURE RL v3 in PolicyStore")

# Check trade data
try:
    result = db.execute(text("SELECT COUNT(*) FROM trade_logs WHERE status='CLOSED'"))
    trade_count = result.scalar()
    if trade_count < 50:
        recommendations.append(f"ACCUMULATE more trade data (have {trade_count}, need 50+)")
except:
    pass

# Check model availability
try:
    result = db.execute(text("SELECT COUNT(*) FROM model_artifacts WHERE status='active'"))
    active_count = result.scalar()
    if active_count == 0:
        recommendations.append("TRAIN initial models (no active models found)")
except:
    pass

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("No actions needed - System is operational!")

print("\n" + "=" * 70)
print("END OF REPORT")
print("=" * 70 + "\n")

db.close()
