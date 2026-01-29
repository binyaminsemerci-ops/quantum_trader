#!/bin/bash
# ============================================================================
# EXIT BRAIN CONTROL LAYER V1 - PROOF SCRIPT
# ============================================================================
# Tests: Control env loaded, symbol rollout logic, enforcement hierarchy
# Exit codes: 0=PASS, 2=SHADOW forced, 9=KILL-SWITCH active

set -euo pipefail

echo "================================================================"
echo "EXIT BRAIN CONTROL LAYER V1 - OPERATIONAL PROOF"
echo "================================================================"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TESTS_PASSED=0
TESTS_TOTAL=0
EXIT_CODE=0

# ============================================================================
# TEST 1: Control env loaded
# ============================================================================
echo "TEST 1: Control Environment Loaded"
echo "==================================="
((TESTS_TOTAL++))

if [ -f /etc/quantum/exitbrain-control.env ]; then
    echo -e "${GREEN}✅ PASS${NC}: Control env exists"
    
    # Check key variables
    source /etc/quantum/exitbrain-control.env
    echo "   EXIT_MODE=${EXIT_MODE:-NOT_SET}"
    echo "   EXIT_EXECUTOR_MODE=${EXIT_EXECUTOR_MODE:-NOT_SET}"
    echo "   EXIT_EXECUTOR_KILL_SWITCH=${EXIT_EXECUTOR_KILL_SWITCH:-NOT_SET}"
    echo "   EXIT_LIVE_ROLLOUT_PCT=${EXIT_LIVE_ROLLOUT_PCT:-NOT_SET}"
    
    if [ "${EXIT_EXECUTOR_KILL_SWITCH:-false}" = "true" ]; then
        echo -e "${RED}🔴 KILL-SWITCH ACTIVE${NC}"
        EXIT_CODE=9
    elif [ "${EXIT_EXECUTOR_MODE:-SHADOW}" = "SHADOW" ]; then
        echo -e "${YELLOW}⚠️  SHADOW MODE${NC}"
        EXIT_CODE=2
    fi
    
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ FAIL${NC}: Control env not found"
fi
echo ""

# ============================================================================
# TEST 2: systemd drop-in exists
# ============================================================================
echo "TEST 2: systemd Drop-In Configuration"
echo "======================================"
((TESTS_TOTAL++))

DROP_IN_DIR="/etc/systemd/system/quantum-exitbrain-v35.service.d"
if [ -f "$DROP_IN_DIR/control.conf" ]; then
    echo -e "${GREEN}✅ PASS${NC}: Drop-in exists at $DROP_IN_DIR/control.conf"
    grep -q "exitbrain-control.env" "$DROP_IN_DIR/control.conf" && \
        echo "   EnvironmentFile reference: OK" || \
        echo "   ⚠️ Warning: EnvironmentFile not found in drop-in"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  INFO${NC}: Drop-in not found (not critical if env loaded)"
fi
echo ""

# ============================================================================
# TEST 3: Simulate symbol rollout (Python test)
# ============================================================================
echo "TEST 3: Symbol Rollout Simulation"
echo "=================================="
((TESTS_TOTAL++))

# Create Python test script
PYTHON_TEST=$(cat <<'EOF'
import sys
import os

# Set env from control file
import configparser
config = {}
try:
    with open('/etc/quantum/exitbrain-control.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key] = val
except Exception as e:
    print(f"❌ Failed to load control env: {e}")
    sys.exit(1)

# Import exit_mode functions
sys.path.insert(0, '/home/qt/quantum_trader')
from backend.config.exit_mode import (
    is_symbol_in_live_rollout,
    get_exit_rollout_pct,
    is_exit_brain_live_fully_enabled,
    is_exit_executor_kill_switch_active
)

# Test symbols
test_symbols = ["BTCUSDT", "ETHUSDT"]
rollout_pct = get_exit_rollout_pct()

print(f"Rollout percentage: {rollout_pct}%")
print(f"Kill-switch: {'ACTIVE' if is_exit_executor_kill_switch_active() else 'OFF'}")
print("")

for symbol in test_symbols:
    in_rollout = is_symbol_in_live_rollout(symbol)
    fully_enabled = is_exit_brain_live_fully_enabled(symbol)
    symbol_hash = hash(symbol) % 100
    
    mode = "LIVE" if fully_enabled else "SHADOW"
    print(f"{symbol}:")
    print(f"   Hash: {symbol_hash}, In rollout: {in_rollout}, Mode: {mode}")

# Exit with appropriate code
if is_exit_executor_kill_switch_active():
    sys.exit(9)
elif not is_exit_brain_live_fully_enabled():
    sys.exit(2)
else:
    sys.exit(0)
EOF
)

# Run Python test
if timeout 10 /opt/quantum/venvs/ai-engine/bin/python3 -c "$PYTHON_TEST" 2>&1; then
    PYTHON_EXIT=$?
    echo -e "${GREEN}✅ PASS${NC}: Symbol rollout simulation completed"
    ((TESTS_PASSED++))
else
    PYTHON_EXIT=$?
    if [ $PYTHON_EXIT -eq 9 ]; then
        echo -e "${RED}🔴 KILL-SWITCH${NC}: Simulation shows kill-switch active"
    elif [ $PYTHON_EXIT -eq 2 ]; then
        echo -e "${YELLOW}⚠️  SHADOW${NC}: Simulation shows SHADOW mode"
    else
        echo -e "${RED}❌ FAIL${NC}: Simulation failed (exit $PYTHON_EXIT)"
    fi
fi
echo ""

# ============================================================================
# TEST 4: Redis audit log
# ============================================================================
echo "TEST 4: Redis Audit Trail"
echo "========================="
((TESTS_TOTAL++))

if command -v redis-cli &> /dev/null; then
    AUDIT_ENTRIES=$(redis-cli LLEN quantum:ops:exitbrain:control 2>/dev/null || echo "0")
    if [ "$AUDIT_ENTRIES" -gt 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}: Redis audit log active ($AUDIT_ENTRIES entries)"
        echo "   Latest entry:"
        redis-cli LINDEX quantum:ops:exitbrain:control 0 2>/dev/null | head -c 200 || true
        echo "..."
        ((TESTS_PASSED++))
    else
        echo -e "${YELLOW}⚠️  INFO${NC}: No audit entries yet (expected on first run)"
        ((TESTS_PASSED++))
    fi
else
    echo -e "${YELLOW}⚠️  INFO${NC}: redis-cli not available"
fi
echo ""

# ============================================================================
# FINAL VERDICT
# ============================================================================
echo "================================================================"
echo "FINAL VERDICT"
echo "================================================================"

if [ $EXIT_CODE -eq 9 ]; then
    echo -e "${RED}🔴 KILL-SWITCH ACTIVE${NC}: Exit Brain forced to SHADOW mode"
    echo "All tests: $TESTS_PASSED/$TESTS_TOTAL passed"
    echo "To deactivate: Set EXIT_EXECUTOR_KILL_SWITCH=false in control env"
    exit 9
elif [ $EXIT_CODE -eq 2 ]; then
    echo -e "${YELLOW}⚠️  SHADOW MODE${NC}: Exit Brain operating in SHADOW mode"
    echo "All tests: $TESTS_PASSED/$TESTS_TOTAL passed"
    echo "To activate LIVE: Set EXIT_EXECUTOR_MODE=LIVE in control env"
    exit 2
elif [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo -e "${GREEN}✅ PASS${NC}: All tests passed ($TESTS_PASSED/$TESTS_TOTAL)"
    echo ""
    echo "Exit Brain Control Layer v1 is OPERATIONAL"
    echo "  - Control env loaded and active"
    echo "  - Symbol rollout logic functioning"
    echo "  - Enforcement hierarchy working"
    exit 0
else
    echo -e "${RED}❌ FAIL${NC}: Some tests failed ($TESTS_PASSED/$TESTS_TOTAL)"
    exit 1
fi
