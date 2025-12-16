#!/bin/bash
cd ~/quantum_trader
source .venv/bin/activate

echo "=== STEG 1A: sys.path audit ==="
python3 -c "
import sys
print('sys.path entries med quantum_trader eller /mnt/c:')
for p in sys.path:
    if 'quantum_trader' in p or '/mnt/c' in p:
        print('  ', p)
"

echo ""
echo "=== STEG 1B: Hvilken fil importeres ==="
python3 -c "
import sys
sys.path.insert(0, '.')
from backend.core.health_contract import ServiceHealth
import inspect

print('backend.core.health_contract.__file__:')
print('  ', __import__('backend.core.health_contract').__file__)
print('ServiceHealth class:', ServiceHealth)
print('ServiceHealth.__module__:', ServiceHealth.__module__)
print('ServiceHealth file:', inspect.getfile(ServiceHealth))
print('Has create():', hasattr(ServiceHealth, 'create'))
"

echo ""
echo "=== STEG 1C: Runtime context test ==="
python3 -c "
import sys
sys.path.insert(0, '.')

from backend.core.health_contract import ServiceHealth as HealthContractSH
try:
    from microservices.ai_engine.models import ServiceHealth as ModelsSH
    print('⚠️  models.ServiceHealth importert:', ModelsSH)
    print('Has create():', hasattr(ModelsSH, 'create'))
except ImportError as e:
    print('✓ models.ServiceHealth IKKE importert (forventet etter fix)')

print()
print('health_contract.ServiceHealth:', HealthContractSH)
print('Has create():', hasattr(HealthContractSH, 'create'))
"

echo ""
echo "=== STEG 1D: Sjekk service.py imports ==="
echo "ServiceHealth imports i service.py:"
grep -n "ServiceHealth" ~/quantum_trader/microservices/ai_engine/service.py | head -10

echo ""
echo "=== Environment ==="
pwd
echo "PYTHONPATH: ${PYTHONPATH:-<not set>}"
