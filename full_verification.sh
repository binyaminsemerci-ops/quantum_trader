#!/bin/bash
# Full verification of ServiceHealth import fix

cd ~/quantum_trader
source .venv/bin/activate

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  SERVICEHEALTH IMPORT VERIFICATION                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "🔍 STEG 1A: sys.path audit"
echo "─────────────────────────────────────────────────────────"
python3 -c "
import sys
entries = [p for p in sys.path if 'quantum_trader' in p or '/mnt/c' in p]
if entries:
    print('Entries med quantum_trader eller /mnt/c:')
    for p in entries:
        print(f'  {p}')
else:
    print('✓ Ingen /mnt/c entries i sys.path')
"
echo ""

echo "🔍 STEG 1B: Hvilken ServiceHealth importeres"
echo "─────────────────────────────────────────────────────────"
python3 -c "
import sys
sys.path.insert(0, '.')
from backend.core.health_contract import ServiceHealth
import inspect

print(f'File: {inspect.getfile(ServiceHealth)}')
print(f'Module: {ServiceHealth.__module__}')
print(f'Has create(): {hasattr(ServiceHealth, \"create\")}')
if hasattr(ServiceHealth, 'create'):
    print('✓ ServiceHealth.create() TILGJENGELIG')
else:
    print('✗ ServiceHealth.create() MANGLER')
"
echo ""

echo "🔍 STEG 1C: Import collision test"
echo "─────────────────────────────────────────────────────────"
python3 -c "
import sys
sys.path.insert(0, '.')

# Test import fra health_contract
from backend.core.health_contract import ServiceHealth as HealthSH
print(f'health_contract.ServiceHealth: {HealthSH}')
print(f'  Has create(): {hasattr(HealthSH, \"create\")}')

# Test om models har ServiceHealth
import microservices.ai_engine.models as models
if hasattr(models, 'ServiceHealth'):
    ModelsSH = models.ServiceHealth
    print(f'models.ServiceHealth: {ModelsSH}')
    print(f'  Has create(): {hasattr(ModelsSH, \"create\")}')
    if HealthSH is ModelsSH:
        print('⚠️  SAMME KLASSE - ingen collision')
    else:
        print('✗ FORSKJELLIGE KLASSER - collision detektert!')
else:
    print('✓ models.ServiceHealth IKKE eksportert (forventet etter fix)')
"
echo ""

echo "🔍 STEG 1D: service.py imports"
echo "─────────────────────────────────────────────────────────"
echo "ServiceHealth mentions i service.py:"
grep -n "ServiceHealth" ~/quantum_trader/microservices/ai_engine/service.py | while read line; do
    echo "  $line"
done
echo ""

echo "📊 OPPSUMMERING"
echo "═════════════════════════════════════════════════════════"
if grep -q "# NOTE: ServiceHealth removed" ~/quantum_trader/microservices/ai_engine/service.py; then
    echo "✓ Fix er implementert i service.py"
else
    echo "✗ Fix mangler i service.py"
fi

python3 -c "
import sys
sys.path.insert(0, '.')
from backend.core.health_contract import ServiceHealth
if hasattr(ServiceHealth, 'create'):
    print('✓ ServiceHealth.create() fungerer')
else:
    print('✗ ServiceHealth.create() mangler')
" 2>/dev/null

echo ""
echo "Current directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH:-<not set>}"
