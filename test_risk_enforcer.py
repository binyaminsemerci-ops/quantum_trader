#!/usr/bin/env python3
"""Test Risk Policy Enforcer on VPS"""

import sys
sys.path.insert(0, '/opt/quantum')

from microservices.risk_policy_enforcer import create_enforcer, SystemState, log_risk_metrics

print("Testing Risk Policy Enforcer...")
print("=" * 60)

# Create enforcer
enforcer = create_enforcer('redis://localhost:6379')
print("✓ Enforcer created")

# Test LAYER 0 (infrastructure)
print("\nLAYER 0 Test: Infrastructure check")
metrics = enforcer.compute_system_state()
log_risk_metrics(metrics)

# Test LAYER 1 (leverage cap)
print("\nLAYER 1 Test: Leverage cap")
ok, clamped = enforcer.check_leverage(5.0)
print(f"Leverage 5.0: {'PASS' if ok else 'CLAMPED'} -> {clamped}")
ok, clamped = enforcer.check_leverage(15.0)
print(f"Leverage 15.0: {'PASS' if ok else 'CLAMPED'} -> {clamped}")

# Test LAYER 2 (symbol whitelist)
print("\nLAYER 2 Test: Symbol whitelist")
allowed, state, reason = enforcer.allow_trade('BTCUSDT', 5.0)
print(f"BTCUSDT: {'ALLOWED' if allowed else 'BLOCKED'} ({state.value})")
allowed, state, reason = enforcer.allow_trade('DOGEUSDT', 5.0)
print(f"DOGEUSDT: {'ALLOWED' if allowed else 'BLOCKED'} ({state.value}): {reason}")

# Test full execution gate
print("\nFull Execution Gate Test:")
allowed, state, reason = enforcer.allow_trade(
    symbol='BTCUSDT',
    requested_leverage=8.0,
    volatility=0.015,
    spread_bps=5.0
)
print(f"Result: {'✓ ALLOWED' if allowed else '✗ BLOCKED'}")
print(f"State: {state.value}")
if reason:
    print(f"Reason: {reason}")

print("\n" + "=" * 60)
print("Risk Enforcer Test Complete")
