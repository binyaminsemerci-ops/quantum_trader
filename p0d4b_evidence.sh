#!/bin/bash
set -euo pipefail
TS=$(date -u +%Y%m%d_%H%M%S)
R=/tmp/p0d4b_evidence_${TS}.txt
echo "=== P0.D.4b EVIDENCE ONLY ===" | tee $R
echo "Time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a $R
echo "Host: $(hostname)" | tee -a $R
echo | tee -a $R

echo "=== 1) STREAM DISCOVERY (execution*) ===" | tee -a $R
redis-cli --scan --pattern "quantum:stream:*execution*" | sort | tee -a $R
echo | tee -a $R

echo "=== 2) CORE STREAM SNAPSHOT ===" | tee -a $R
for k in quantum:stream:trade.intent quantum:stream:trade.execution.res quantum:stream:execution.result quantum:stream:execution.results quantum:stream:execution.res quantum:stream:execution quantum:stream:execution.events quantum:stream:execution.out; do
  echo "--- $k" | tee -a $R
  redis-cli XLEN $k 2>&1 | tee -a $R
  redis-cli XINFO STREAM $k 2>&1 | grep -E "(length|last-generated-id|first-entry|last-entry)" | tee -a $R || true
  echo | tee -a $R
done

echo "=== 3) GROUP METRICS (trade.intent execution group) ===" | tee -a $R
STREAM=quantum:stream:trade.intent
GROUP=quantum:group:execution:trade.intent
redis-cli XINFO GROUPS $STREAM 2>&1 | tee -a $R
echo | tee -a $R
redis-cli XINFO CONSUMERS $STREAM $GROUP 2>&1 | head -200 | tee -a $R
echo | tee -a $R

echo "=== 4) EXECUTION + BRIDGE LOG EVIDENCE (last 2h) ===" | tee -a $R
journalctl -u quantum-execution.service --since "2 hours ago" --no-pager 2>/dev/null | grep -iE "(error|exception|traceback|publish|xadd|execution\.result|execution\.res|result stream|redis)" | tail -200 | tee -a $R || true
echo | tee -a $R
journalctl -u quantum-execution-result-bridge.service --since "2 hours ago" --no-pager 2>/dev/null | grep -iE "(error|exception|traceback|publish|xadd|trade\.execution\.res|bridge|redis)" | tail -200 | tee -a $R || true

echo
echo "REPORT: $R"
tail -80 $R
