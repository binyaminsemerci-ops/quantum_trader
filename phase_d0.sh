#!/bin/bash
set -euo pipefail
DIR=/tmp/phase_d_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DIR"/{before,after,proof,backup,patches,logs}

# Record services status
systemctl is-active quantum-ai-engine quantum-ai-strategy-router quantum-execution > "$DIR/before/services_status.txt" || true

# Unit files
systemctl cat quantum-ai-strategy-router > "$DIR/before/router.unit.txt" || true
systemctl cat quantum-execution > "$DIR/before/execution.unit.txt" || true

# Redis streams
/usr/bin/redis-cli XLEN quantum:stream:ai.decision.made > "$DIR/before/stream_ai_decision.txt" || true
/usr/bin/redis-cli XLEN quantum:stream:trade.intent > "$DIR/before/stream_trade_intent.txt" || true
/usr/bin/redis-cli XLEN quantum:stream:execution.result > "$DIR/before/stream_execution_result.txt" || true
/usr/bin/redis-cli XINFO GROUPS quantum:stream:trade.intent > "$DIR/before/xinfo_groups_trade_intent.txt" || true
/usr/bin/redis-cli XPENDING quantum:stream:trade.intent quantum:group:execution:trade.intent - + 50 > "$DIR/before/xpending_trade_intent.txt" || true

# Logs
tail -200 /var/log/quantum/ai-strategy-router.log > "$DIR/before/router.log" || true
tail -200 /var/log/quantum/execution.log > "$DIR/before/execution.log" || true

# Evidence summary stub
{
  echo "DIR=$DIR"
  echo -n "Services status: "
  paste -sd' ' "$DIR/before/services_status.txt" 2>/dev/null
  echo "XLEN ai.decision: $(cat $DIR/before/stream_ai_decision.txt 2>/dev/null)"
  echo "XLEN trade.intent: $(cat $DIR/before/stream_trade_intent.txt 2>/dev/null)"
  echo "XLEN execution.result: $(cat $DIR/before/stream_execution_result.txt 2>/dev/null)"
} > "$DIR/before/summary.txt"

echo "$DIR"