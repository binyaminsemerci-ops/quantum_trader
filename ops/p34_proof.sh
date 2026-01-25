#!/usr/bin/env bash
set -euo pipefail

# P3.4 Reconcile Engine - Proof Script
# Verifies P3.4 is working correctly

COLOR_BLUE='\033[0;34m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_YELLOW='\033[1;33m'
COLOR_RESET='\033[0m'

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[✓]${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_RED}[✗]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[!]${COLOR_RESET} $1"
}

FAILED=0

echo "════════════════════════════════════════════════════════════════"
echo "  P3.4 Reconcile Engine - Proof Tests"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test 1: Service Active
log_info "Test 1: Service running..."
if systemctl is-active --quiet quantum-reconcile-engine; then
    log_success "P3.4 service is active (PID=$(systemctl show -p MainPID --value quantum-reconcile-engine))"
else
    log_error "P3.4 service NOT running"
    systemctl status quantum-reconcile-engine --no-pager | head -15
    FAILED=1
fi

# Test 2: Prometheus metrics
log_info "Test 2: Prometheus metrics endpoint..."
METRICS=$(curl -s http://localhost:8046/metrics || echo "FAILED")
if echo "$METRICS" | grep -q "p34_reconcile_drift_total"; then
    log_success "Metrics endpoint responding (found p34_reconcile_drift_total)"
    
    # Show sample metrics
    echo "$METRICS" | grep "^p34_" | head -10
else
    log_error "Metrics endpoint not responding or no P3.4 metrics"
    FAILED=1
fi

# Test 3: Recent logs
log_info "Test 3: Recent service logs..."
LOGS=$(journalctl -u quantum-reconcile-engine -n 5 --no-pager)
if echo "$LOGS" | grep -q "P3.4"; then
    log_success "Service logging correctly"
    echo "$LOGS"
else
    log_warn "No recent P3.4 logs found"
fi

# Test 4: Redis state keys
log_info "Test 4: Redis reconcile state..."
STATE_KEYS=$(redis-cli KEYS "quantum:reconcile:state:*" 2>/dev/null || echo "")
if [[ -n "$STATE_KEYS" ]]; then
    log_success "Found reconcile state keys:"
    echo "$STATE_KEYS"
    
    # Show state for BTCUSDT if exists
    if redis-cli EXISTS quantum:reconcile:state:BTCUSDT | grep -q "1"; then
        log_info "BTCUSDT reconcile state:"
        redis-cli HGETALL quantum:reconcile:state:BTCUSDT
    fi
else
    log_warn "No state keys yet (normal if just started)"
fi

# Test 5: Hold keys
log_info "Test 5: Check for active holds..."
HOLD_KEYS=$(redis-cli KEYS "quantum:reconcile:hold:*" 2>/dev/null || echo "")
if [[ -n "$HOLD_KEYS" ]]; then
    log_warn "Found active holds (P3.3 will block execution):"
    echo "$HOLD_KEYS"
    
    for key in $HOLD_KEYS; do
        TTL=$(redis-cli TTL "$key")
        echo "  $key (TTL: ${TTL}s)"
    done
else
    log_success "No active holds (normal operation)"
fi

# Test 6: Event stream
log_info "Test 6: Reconcile event stream..."
EVENT_COUNT=$(redis-cli XLEN quantum:stream:reconcile.events 2>/dev/null || echo "0")
if [[ "$EVENT_COUNT" -gt 0 ]]; then
    log_success "Event stream has $EVENT_COUNT events"
    
    log_info "Last 3 events:"
    redis-cli XREVRANGE quantum:stream:reconcile.events + - COUNT 3
else
    log_warn "No events yet (wait for first reconciliation loop)"
fi

# Test 7: P3.3 integration
log_info "Test 7: P3.3 hold check integration..."
if systemctl is-active --quiet quantum-position-state-brain; then
    log_success "P3.3 service is running"
    
    # Check if P3.3 has recent logs mentioning hold
    P33_LOGS=$(journalctl -u quantum-position-state-brain -n 20 --no-pager || echo "")
    if echo "$P33_LOGS" | grep -q "reconcile_hold"; then
        log_info "P3.3 is checking reconcile holds"
    else
        log_info "P3.3 running (no hold activity in recent logs)"
    fi
else
    log_error "P3.3 service NOT running"
    FAILED=1
fi

# Test 8: Simulate mismatch (optional test)
log_info "Test 8: Simulate position mismatch (test hold behavior)..."
echo ""
echo "To manually test hold behavior:"
echo "  1. Create fake mismatch:"
echo "     redis-cli HSET quantum:position:ledger:BTCUSDT ledger_amt 9999"
echo ""
echo "  2. Wait 1-2 seconds, then check hold:"
echo "     redis-cli GET quantum:reconcile:hold:BTCUSDT"
echo ""
echo "  3. Check P3.3 denies with 'reconcile_hold_active':"
echo "     journalctl -u quantum-apply-layer -n 5"
echo ""
echo "  4. Restore correct value:"
echo "     redis-cli HSET quantum:position:ledger:BTCUSDT ledger_amt <correct_amt>"
echo ""

# Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    log_success "All proof tests passed"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "✅ P3.4 Reconcile Engine is operational"
    echo ""
    echo "Monitor live:"
    echo "  journalctl -u quantum-reconcile-engine -f"
    echo ""
    echo "Check metrics:"
    echo "  curl http://localhost:8046/metrics | grep p34_"
    echo ""
    echo "Check reconcile state:"
    echo "  redis-cli HGETALL quantum:reconcile:state:BTCUSDT"
    echo ""
    exit 0
else
    log_error "Some tests failed"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "❌ P3.4 Reconcile Engine has issues"
    echo ""
    echo "Debug:"
    echo "  systemctl status quantum-reconcile-engine"
    echo "  journalctl -u quantum-reconcile-engine -n 50"
    echo ""
    exit 1
fi
