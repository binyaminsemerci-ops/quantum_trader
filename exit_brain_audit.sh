#!/bin/bash
set -euo pipefail

echo "=== EXIT BRAIN V3.5 AUDIT ==="
echo "Timestamp: $(date -u)"
echo "Git commit: $(cd /home/qt/quantum_trader && git rev-parse --short HEAD)"
echo ""

echo "=== STEP 0: DISCOVERY ==="
echo "1) Active services (exit/apply/execution):"
systemctl list-units --type=service --all | grep -E "exit|v35|position|apply|execution" | head -80 || echo "No matches in active"
echo ""

echo "2) Unit files (exit/v35/position):"
systemctl list-unit-files | grep -E "exit|v35|position" | head -80 || echo "No matches in unit files"
echo ""

echo "3) Exit Brain code paths:"
find /home/qt/quantum_trader/microservices -maxdepth 2 -type d | grep -iE "exit|v35" || echo "No exit/v35 directories"
echo ""

echo "4) Stream/Redis patterns in code:"
grep -r "quantum:stream:.*exit" /home/qt/quantum_trader/microservices --include="*.py" | head -30 || echo "No exit stream patterns"
echo ""

echo "5) Position state patterns:"
grep -r "quantum.*position" /home/qt/quantum_trader/microservices --include="*.py" | grep -E "XADD|XREAD|HSET|HGET|redis" | head -30 || echo "No position patterns"
echo ""

echo "=== STEP 1: SERVICE HEALTH ==="
echo "Checking quantum-exit-brain-v35.service..."
systemctl is-active quantum-exit-brain-v35.service || echo "Service not active"
systemctl status quantum-exit-brain-v35.service --no-pager -l | head -60 || echo "Service not found"
echo ""

echo "Recent logs (last 100 lines):"
journalctl -u quantum-exit-brain-v35.service -n 100 --no-pager | tail -80 || echo "No logs found"
echo ""

echo "=== STEP 2: REDIS EVIDENCE ==="
redis-cli PING
echo ""

echo "Position keys:"
redis-cli --scan --pattern "quantum:*position*" | head -30
echo ""

echo "Exit streams:"
redis-cli --scan --pattern "quantum:stream:*exit*" | head -30
echo ""

echo "Exit intent stream check:"
STREAM_EXIT="quantum:stream:exit.intent"
redis-cli EXISTS "$STREAM_EXIT"
redis-cli XLEN "$STREAM_EXIT" 2>/dev/null || echo "Stream not found"
redis-cli XREVRANGE "$STREAM_EXIT" + - COUNT 3 || echo "Cannot read stream"
echo ""

echo "=== STEP 3: APPLY/EXEC EVIDENCE ==="
systemctl is-active quantum-apply-layer quantum-execution || true
echo ""

echo "Apply layer exit handling (last 200 lines):"
journalctl -u quantum-apply-layer -n 200 --no-pager | grep -iE "exit|close|reduce|intent" | tail -40 || echo "No exit-related logs"
echo ""

echo "Execution exit events:"
journalctl -u quantum-execution -n 200 --no-pager | grep -iE "exit|close|reduce|order" | tail -40 || echo "No execution exit logs"
echo ""

echo "Trade intent stream:"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 3 | head -60 || echo "No trade.intent stream"
echo ""

echo "Execution result stream:"
redis-cli XREVRANGE quantum:stream:execution.result + - COUNT 3 | head -60 || echo "No execution.result stream"
echo ""

echo "=== AUDIT DISCOVERY COMPLETE ==="
