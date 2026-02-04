#!/usr/bin/env python3
"""
Dry-run test: Trigger _get_effective_allowlist() to see truth logging.
"""
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from microservices.intent_bridge.main import IntentBridge

print("=== DRY RUN: Effective Allowlist ===")
print("")

bridge = IntentBridge()

print("")
print("Calling _get_effective_allowlist()...")
print("")

allowlist = bridge._get_effective_allowlist()

print("")
print(f"Result: {len(allowlist)} symbols")
print(f"Sample: {sorted(list(allowlist))[:10]}")
print("")
print("Check journalctl logs for ALLOWLIST_EFFECTIVE and TESTNET_INTERSECTION")
