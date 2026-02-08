#!/usr/bin/env python3
"""Test Meta-Agent V2 Learning Cadence Integration"""
import sys
sys.path.insert(0, "/home/qt/quantum_trader")

from ai_engine.agents.meta_agent_v2 import MetaAgentV2

# Initialize Meta-Agent V2
print("Initializing Meta-Agent V2...")
meta = MetaAgentV2()

# Get statistics (should include learning readiness)
stats = meta.get_statistics()

print("\n=== META-AGENT V2 STATISTICS ===")
print(f"Total predictions: {stats.get('total_predictions')}")
print(f"Meta overrides: {stats.get('meta_overrides')}")
print(f"Model ready: {stats.get('model_ready')}")

print("\n=== LEARNING READINESS CONTEXT ===")
learning = stats.get("learning_readiness", {})
ready_status = learning.get("ready")
print(f"Ready: {ready_status}")
print(f"Reason: {learning.get('reason')}")
print(f"Allowed actions: {learning.get('allowed_actions')}")
print(f"Last checked: {learning.get('last_checked')}")

if ready_status is False:
    print("\n⏸️  Learning NOT READY (expected - need more trades)")
elif ready_status is True:
    print("\n🟢 Learning READY!") 
else:
    print("\n⚠️  Learning readiness unknown (API may be unavailable)")

print("\n✅ Integration test complete!")
