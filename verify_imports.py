#!/usr/bin/env python3
"""Verify sys.path and import locations for debugging."""

import sys
import os

print("=" * 60)
print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NOT SET"))
print("=" * 60)
print("\nSYS.PATH entries with quantum_trader or /mnt/c:")
for i, p in enumerate(sys.path):
    if "quantum_trader" in p.lower() or p.startswith("/mnt/c"):
        print(f"  [{i}] {p}")

print("\n" + "=" * 60)
print("IMPORT LOCATION VERIFICATION:")
print("=" * 60)

try:
    sys.path.insert(0, "/home/belen/quantum_trader")
    import backend.core.health_contract
    from backend.core.health_contract import ServiceHealth
    import inspect
    
    print("health_contract.__file__:", backend.core.health_contract.__file__)
    print("ServiceHealth from:", inspect.getfile(ServiceHealth))
    print("hasattr(ServiceHealth, 'create'):", hasattr(ServiceHealth, "create"))
    
    if hasattr(ServiceHealth, "create"):
        print("create() is callable:", callable(getattr(ServiceHealth, "create")))
        print("create() signature:", inspect.signature(ServiceHealth.create))
    else:
        print("WARNING: ServiceHealth.create() NOT FOUND!")
        print("Available methods:", [m for m in dir(ServiceHealth) if not m.startswith('_')])
        
except Exception as e:
    print(f"ERROR importing: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
