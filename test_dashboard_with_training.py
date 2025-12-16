"""
Test dashboard API med training metrics.
"""

from fastapi.testclient import TestClient
from backend.main import app
from backend.domains.learning.rl_v3.metrics_v3 import RLv3MetricsStore

# Clear metrics
metrics = RLv3MetricsStore.instance()
metrics.clear()

# Add mock training runs
for i in range(5):
    metrics.record_training_run({
        "timestamp": f"2025-12-02T16:00:{i:02d}Z",
        "episodes": 3,
        "duration_seconds": 15.5 + i,
        "success": True,
        "error": None,
        "avg_reward": 10.0 + i * 2,
        "final_reward": 12.0 + i * 2,
        "avg_policy_loss": 0.05 - i * 0.01,
        "avg_value_loss": 0.03 - i * 0.005
    })

# Add mock decisions
for i in range(3):
    metrics.record_decision({
        "symbol": "BTCUSDT",
        "timestamp": f"2025-12-02T16:10:{i:02d}Z",
        "action": i % 6,
        "confidence": 0.8 + i * 0.05,
        "value": 5.0 + i,
        "trace_id": f"trace-{i}",
        "shadow_mode": True
    })

# Test dashboard endpoint
client = TestClient(app)

print("📊 Testing GET /api/v1/rl-v3/dashboard/full...")
response = client.get("/api/v1/rl-v3/dashboard/full")

print(f"Status code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    
    print("\n✅ Dashboard response received!")
    print(f"\n📈 Decision Summary:")
    print(f"   Total decisions: {data['summary']['total_decisions']}")
    print(f"   Recent decisions: {len(data['recent_decisions'])}")
    
    print(f"\n🎓 Training Summary:")
    print(f"   Total runs: {data['training_summary']['total_runs']}")
    print(f"   Success: {data['training_summary']['success_count']}")
    print(f"   Failures: {data['training_summary']['failure_count']}")
    print(f"   Avg reward: {data['training_summary']['avg_reward']}")
    print(f"   Avg duration: {data['training_summary']['avg_duration_seconds']}s")
    print(f"   Last run at: {data['training_summary']['last_run_at']}")
    
    print(f"\n📋 Recent Training Runs:")
    for run in data['recent_training_runs'][:3]:
        print(f"   - {run['timestamp']}: {run['episodes']} episodes, "
              f"reward={run['avg_reward']:.2f}, duration={run['duration_seconds']:.1f}s, "
              f"success={run['success']}")
    
    if data['training_summary']['total_runs'] >= 5:
        print("\n✅ All tests passed! Dashboard includes training metrics.")
    else:
        print(f"\n⚠️ Expected at least 5 training runs, got {data['training_summary']['total_runs']}")
else:
    print(f"\n❌ Request failed: {response.text}")
