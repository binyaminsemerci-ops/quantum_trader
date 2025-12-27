"""
Quick Status Check: Orchestrator LIVE Mode Step 1

This script checks if the Orchestrator is active and enforcing signal filtering.
"""

import docker
import re
from datetime import datetime

def check_orchestrator_status():
    """Check Orchestrator LIVE mode status from backend logs."""
    
    print("=" * 70)
    print("[RED_CIRCLE] ORCHESTRATOR LIVE MODE - STEP 1 STATUS CHECK")
    print("=" * 70)
    print()
    
    try:
        client = docker.from_env()
        container = client.containers.get("quantum_backend")
        
        # Get logs
        logs = container.logs(tail=500).decode('utf-8', errors='ignore')
        
        # Check for LIVE mode initialization
        live_init = re.search(r'Orchestrator LIVE enforcing: (.+?)["\n]', logs)
        if live_init:
            print("[OK] LIVE MODE ACTIVE")
            print(f"   Enforcing: {live_init.group(1)}")
        else:
            print("❌ LIVE MODE NOT DETECTED")
            print("   Expected: 'Orchestrator LIVE enforcing: signal_filter, confidence'")
        
        print()
        
        # Check for recent policy enforcement
        policy_enforced = re.findall(r'LIVE MODE - Policy ENFORCED: (.+?)["\n]', logs)
        if policy_enforced:
            print(f"[OK] Policy enforced {len(policy_enforced)} times")
            print(f"   Latest: {policy_enforced[-1]}")
        else:
            print("⏳ No policy enforcement logged yet (may be in cooldown)")
        
        print()
        
        # Check for blocked signals
        blocked_signals = re.findall(r'BLOCKED by policy: (.+?)["\n]', logs)
        if blocked_signals:
            print(f"[BLOCKED] {len(blocked_signals)} signals blocked by policy")
            for i, block in enumerate(blocked_signals[-3:], 1):  # Show last 3
                print(f"   {i}. {block}")
        else:
            print("[OK] No signals blocked yet (policy allowing all)")
        
        print()
        
        # Check for policy confidence adjustments
        conf_changes = re.findall(r'Using policy confidence: (\d+\.\d+) \(was (\d+\.\d+)\)', logs)
        if conf_changes:
            latest = conf_changes[-1]
            print(f"[CHART] Confidence threshold: {latest[0]} (was {latest[1]})")
            print(f"   Policy adjusted {len(conf_changes)} times")
        else:
            print("[CHART] No confidence adjustments logged yet")
        
        print()
        
        # Check for recent trading cycles
        cycles = re.findall(r'_check_and_execute\(\) started', logs)
        if cycles:
            print(f"🔄 Trading cycles active: {len(cycles)} cycles in last 500 log lines")
        
        print()
        print("=" * 70)
        print(f"Status checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
    except docker.errors.NotFound:
        print("❌ Container 'quantum_backend' not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_orchestrator_status()
