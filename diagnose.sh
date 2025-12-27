#!/bin/bash
cd ~/quantum_trader
source .venv/bin/activate

echo "=== WHERE AM I ==="
pwd
echo "PYTHONPATH=$PYTHONPATH"
echo "VIRTUAL_ENV=$VIRTUAL_ENV"

echo
echo "=== SYS.PATH (only quantum_trader + /mnt/c) ==="
python - <<'PY'
import sys, os
print("PYTHONPATH env:", os.environ.get("PYTHONPATH"))
for i,p in enumerate(sys.path):
    if "quantum_trader" in p.lower() or p.startswith("/mnt/c/"):
        print(f"[{i}] {p}")
PY

echo
echo "=== WHICH health_contract & ServiceHealth AM I USING ==="
python - <<'PY'
import sys, inspect, os
sys.path.insert(0, "/home/belen/quantum_trader")
import backend.core.health_contract as hc
from backend.core.health_contract import ServiceHealth
print("health_contract.__file__:", hc.__file__)
print("ServiceHealth file:", inspect.getfile(ServiceHealth))
print("ServiceHealth has create:", hasattr(ServiceHealth, "create"))
print("ServiceHealth module:", ServiceHealth.__module__)
PY
