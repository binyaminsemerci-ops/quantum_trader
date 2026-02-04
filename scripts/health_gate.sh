#!/bin/bash
###############################################################################
# Quantum Trader - Production Health Gate
###############################################################################
# Purpose: Automated 5-minute health monitoring with fail-closed gates
# Author: Quantum Trader Team
# Date: 2026-02-02
#
# Deploy as systemd timer:
#   systemctl enable --now quantum-health-gate.timer
#
# Or run as cron:
#   */5 * * * * /root/quantum_trader/scripts/health_gate.sh
###############################################################################

set -euo pipefail

# Config
REDIS_CLI="redis-cli"
JOURNALCTL="journalctl"
WINDOW_MINUTES=5
CRITICAL_ALERT_FILE="/tmp/quantum_health_gate_alert"
LOG_FILE="/var/log/quantum/health_gate.log"

# Timestamp
NOW=$(date '+%Y-%m-%d %H:%M:%S')

# Logging helper
log() {
    echo "[$NOW] $1" | tee -a "$LOG_FILE"
}

log "============================================="
log "🩺 QUANTUM HEALTH GATE - $WINDOW_MINUTES MIN CHECK"
log "============================================="

# Gate A: Trading Alive? (FILLED orders in last N minutes)
log "📊 Gate A: Trading Activity"
FILLED_COUNT=$($JOURNALCTL -u quantum-intent-executor --since "$WINDOW_MINUTES minutes ago" --no-pager 2>/dev/null | grep -c "executed=True" || echo "0")
log "   FILLED orders (last ${WINDOW_MINUTES}m): $FILLED_COUNT"

if [ "$FILLED_COUNT" -eq 0 ]; then
    log "   ⚠️  WARNING: No executions in last ${WINDOW_MINUTES} minutes"
    log "   (Market quiet OR Governor blocking all entries)"
    # NOT fail-closed: Market can be quiet, Governor can be conservative
else
    log "   ✅ PASS: Trading active ($FILLED_COUNT executions)"
fi

# Gate B: Snapshot Coverage (expect >= allowlist count)
log ""
log "📊 Gate B: Snapshot Coverage"
SNAPSHOT_COUNT=$($REDIS_CLI --scan --pattern "quantum:position:snapshot:*" 2>/dev/null | wc -l || echo "0")
log "   Snapshots available: $SNAPSHOT_COUNT"

# Get P3.3 allowlist count from env
P33_ALLOWLIST=$($REDIS_CLI GET "quantum:cfg:p33_allowlist_count" 2>/dev/null || echo "11")
log "   Expected (P3.3 allowlist): $P33_ALLOWLIST"

if [ "$SNAPSHOT_COUNT" -lt "$P33_ALLOWLIST" ]; then
    log "   ❌ FAIL-CLOSED: Snapshot coverage regression detected!"
    log "   Action: Check P3.3 Position State Brain logs"
    touch "$CRITICAL_ALERT_FILE"
    echo "SNAPSHOT_COVERAGE_FAIL" > "$CRITICAL_ALERT_FILE"
else
    log "   ✅ PASS: Full snapshot coverage ($SNAPSHOT_COUNT >= $P33_ALLOWLIST)"
fi

# Gate C: WAVESUSDT Regression Check (should be 0)
log ""
log "📊 Gate C: WAVESUSDT Regression"
WAVES_COUNT=$($JOURNALCTL -u quantum-intent-executor --since "$WINDOW_MINUTES minutes ago" --no-pager 2>/dev/null | grep -c "WAVESUSDT" || echo "0")
log "   WAVESUSDT attempts (last ${WINDOW_MINUTES}m): $WAVES_COUNT"

if [ "$WAVES_COUNT" -gt 0 ]; then
    log "   ❌ FAIL-CLOSED: WAVESUSDT regression detected!"
    log "   Action: Check allowlist configs (intent-bridge, intent-executor, position-state-brain)"
    touch "$CRITICAL_ALERT_FILE"
    echo "WAVESUSDT_REGRESSION" > "$CRITICAL_ALERT_FILE"
else
    log "   ✅ PASS: No WAVESUSDT spam"
fi

# Gate D: Stream Freshness (apply.plan and apply.result should be active)
log ""
log "📊 Gate D: Stream Freshness"

PLAN_LENGTH=$($REDIS_CLI XINFO STREAM quantum:stream:apply.plan 2>/dev/null | grep -A1 "^length$" | tail -1 || echo "0")
RESULT_LENGTH=$($REDIS_CLI XINFO STREAM quantum:stream:apply.result 2>/dev/null | grep -A1 "^length$" | tail -1 || echo "0")

log "   apply.plan stream length: $PLAN_LENGTH"
log "   apply.result stream length: $RESULT_LENGTH"

if [ "$PLAN_LENGTH" -eq 0 ] || [ "$RESULT_LENGTH" -eq 0 ]; then
    log "   ⚠️  WARNING: Stream may be empty (check if first boot)"
else
    log "   ✅ PASS: Streams flowing"
fi

# Gate E: Ledger Lag Check (ZECUSDT example)
log ""
log "📊 Gate E: Ledger Lag Status"

# Check if any position exists in ledger
LEDGER_KEYS=$($REDIS_CLI --scan --pattern "quantum:position:ledger:*" 2>/dev/null | grep -v "::" | wc -l || echo "0")
log "   Ledger entries: $LEDGER_KEYS"

if [ "$LEDGER_KEYS" -eq 0 ]; then
    log "   ⚠️  INFO: No ledger entries (may be first trades or lag)"
    log "   Check: quantum:position:snapshot:* for exchange truth"
else
    log "   ✅ INFO: Ledger active ($LEDGER_KEYS symbols)"
fi

# Gate F: Service Health
log ""
log "📊 Gate F: Core Services"

for service in quantum-intent-executor quantum-intent-bridge quantum-position-state-brain quantum-ai-engine; do
    STATUS=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
    if [ "$STATUS" = "active" ]; then
        log "   ✅ $service: $STATUS"
    else
        log "   ❌ FAIL-CLOSED: $service is $STATUS"
        touch "$CRITICAL_ALERT_FILE"
        echo "SERVICE_DOWN_$service" > "$CRITICAL_ALERT_FILE"
    fi
done

# Final Status
log ""
if [ -f "$CRITICAL_ALERT_FILE" ]; then
    ALERT_REASON=$(cat "$CRITICAL_ALERT_FILE")
    log "❌ HEALTH GATE: FAIL-CLOSED"
    log "   Reason: $ALERT_REASON"
    log "   Alert file: $CRITICAL_ALERT_FILE"
    log "   Action: Investigate immediately"
else
    log "✅ HEALTH GATE: PASS"
    log "   All critical gates passed"
fi

log "============================================="
log ""

# Exit code (0 = pass, 1 = fail)
if [ -f "$CRITICAL_ALERT_FILE" ]; then
    exit 1
else
    exit 0
fi
