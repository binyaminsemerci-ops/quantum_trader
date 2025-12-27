#!/usr/bin/env python3
"""Check LIVE MODE Step 2 activation status."""

from backend.services.orchestrator_config import OrchestratorIntegrationConfig

# Check config
config = OrchestratorIntegrationConfig.create_live_mode_gradual()

print("=" * 60)
print("[TARGET] LIVE MODE STEP 2 ACTIVATION STATUS")
print("=" * 60)
print(f"Mode: {config.mode.value}")
print(f"Orchestrator enabled: {config.enable_orchestrator}")
print()
print("FEATURE FLAGS:")
print(f"  [OK] Step 1 - Signal Filter:        {config.use_for_signal_filter}")
print(f"  [OK] Step 1 - Confidence Threshold: {config.use_for_confidence_threshold}")
print(f"  {'[OK]' if config.use_for_risk_sizing else '❌'} Step 2 - Risk Sizing:         {config.use_for_risk_sizing}")
print(f"  ⏳ Step 3 - Position Limits:      {config.use_for_position_limits}")
print(f"  ⏳ Step 4 - Trading Gate:         {config.use_for_trading_gate}")
print(f"  ⏳ Step 5 - Exit Mode:            {config.use_for_exit_mode}")
print()

if config.use_for_risk_sizing:
    print("🎉 SUCCESS! LIVE MODE Step 2 is ACTIVE")
    print("Risk scaling will be applied based on OrchestratorPolicy")
else:
    print("[WARNING] WARNING! Step 2 is still INACTIVE")
    print("Need to set use_for_risk_sizing=True")
