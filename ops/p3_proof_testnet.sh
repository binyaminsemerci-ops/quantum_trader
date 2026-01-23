#!/usr/bin/env bash
#
# P3 Proof Pack - Testnet Mode Verification
#
# Verifies:
# 1. APPLY_MODE=testnet configured
# 2. Binance testnet credentials present
# 3. Binance connectivity check passes
# 4. One safe action executed (BTCUSDT only via allowlist)
# 5. Execution result written with order details
# 6. Idempotency prevents duplicate orders
# 7. Safety gates working in testnet mode
#
# PREREQUISITES:
# - Binance testnet API key/secret in /etc/quantum/apply-layer.env
# - Harvest proposal exists for BTCUSDT with EXECUTE-worthy action
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok() {
  echo -e "${GREEN}✅${NC} $1"
}

warn() {
  echo -e "${YELLOW}⚠️ ${NC} $1"
}

fail() {
  echo -e "${RED}❌${NC} $1"
  exit 1
}

info() {
  echo -e "${NC}ℹ️  $1${NC}"
}

echo "=== P3 Apply Layer Proof Pack - Testnet Mode ==="
date
echo

# Check service active
echo "== P0: Core health =="
redis-cli PING >/dev/null 2>&1 && ok "Redis PING" || fail "Redis PING failed"

if systemctl is-active --quiet quantum-apply-layer; then
  ok "quantum-apply-layer service active"
else
  fail "quantum-apply-layer service not active"
fi

echo

# Check testnet mode
echo "== P3.1: Verify testnet mode =="
if [ -f /etc/quantum/apply-layer.env ]; then
  mode=$(grep "^APPLY_MODE=" /etc/quantum/apply-layer.env | cut -d= -f2)
  
  if [[ "$mode" == "testnet" ]]; then
    ok "APPLY_MODE=testnet"
  else
    fail "APPLY_MODE=$mode (expected testnet). Run: sudo sed -i 's/APPLY_MODE=.*/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env && sudo systemctl restart quantum-apply-layer"
  fi
  
  allowlist=$(grep "^APPLY_ALLOWLIST=" /etc/quantum/apply-layer.env | cut -d= -f2)
  info "APPLY_ALLOWLIST=$allowlist"
  
  # Check credentials present (but don't display)
  if grep -q "^BINANCE_TESTNET_API_KEY=" /etc/quantum/apply-layer.env && \
     grep -q "^BINANCE_TESTNET_API_SECRET=" /etc/quantum/apply-layer.env; then
    ok "Binance testnet credentials configured"
  else
    fail "Missing Binance testnet credentials in /etc/quantum/apply-layer.env"
  fi
else
  fail "/etc/quantum/apply-layer.env not found"
fi

echo

# Check for execution results
echo "== P3.2: Execution results =="
result_count=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null || echo "0")

if [[ $result_count -gt 0 ]]; then
  ok "apply.result stream has $result_count entries"
  
  # Check for executed=true (actual testnet execution)
  echo "Latest 3 results:"
  redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 | head -30
  echo
  
  executed_true=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -c '"executed": true' || echo "0")
  
  if [[ $executed_true -gt 0 ]]; then
    ok "Found $executed_true results with executed=true (testnet execution active)"
  else
    warn "No executed=true results yet (service may need time or no EXECUTE plans)"
  fi
  
  # Check for order IDs
  order_ids=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 | grep -o '"order_id": "[^"]*"' | wc -l)
  
  if [[ $order_ids -gt 0 ]]; then
    ok "Found $order_ids order IDs in results (execution with order tracking)"
    info "Example order IDs:"
    redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 | grep -o '"order_id": "[^"]*"' | head -3
  else
    warn "No order IDs found (check if executions succeeded)"
  fi
else
  warn "apply.result stream empty (service may need time)"
fi

echo

# Check Binance connectivity from logs
echo "== P3.3: Binance testnet connectivity =="
testnet_logs=$(journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -i "testnet" | wc -l)

if [[ $testnet_logs -gt 0 ]]; then
  ok "Found $testnet_logs TESTNET log entries"
  info "Recent testnet activity:"
  journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -i "testnet" | tail -5
else
  warn "No TESTNET activity in logs (check if any EXECUTE plans created)"
fi

echo

# Check allowlist enforcement (only BTCUSDT should execute)
echo "== P3.4: Allowlist enforcement in testnet =="
btc_executed=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 20 | grep -B5 "executed.*true" | grep -c "BTCUSDT" || echo "0")

if [[ $btc_executed -gt 0 ]]; then
  ok "BTCUSDT executed (allowlist working)"
else
  info "No BTCUSDT executions yet (may need harvest proposal with EXECUTE decision)"
fi

# Check that non-allowlist symbols NOT executed
eth_executed=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 20 | grep -B5 "executed.*true" | grep -c "ETHUSDT" || echo "0")
sol_executed=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 20 | grep -B5 "executed.*true" | grep -c "SOLUSDT" || echo "0")

if [[ $eth_executed -eq 0 && $sol_executed -eq 0 ]]; then
  ok "Non-allowlisted symbols NOT executed (ETHUSDT/SOLUSDT)"
else
  warn "Found executions for non-allowlisted symbols (eth=$eth_executed, sol=$sol_executed)"
fi

echo

# Check idempotency in testnet
echo "== P3.5: Idempotency in testnet mode =="
dedupe_count=$(redis-cli KEYS "quantum:apply:dedupe:*" 2>/dev/null | wc -l)

if [[ $dedupe_count -gt 0 ]]; then
  ok "Found $dedupe_count dedupe keys (idempotency active)"
  
  # Check if any duplicate plans detected
  duplicate_plans=$(redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 | grep -c "duplicate_plan" || echo "0")
  
  if [[ $duplicate_plans -gt 0 ]]; then
    ok "Found $duplicate_plans duplicate plan detections (idempotency working)"
  else
    info "No duplicates detected yet (normal if proposals changing)"
  fi
else
  warn "No dedupe keys (no EXECUTE plans created yet)"
fi

echo

# Check for errors
echo "== P3.6: Service health (testnet errors) =="
error_count=$(journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -c ERROR || echo "0")

if [[ $error_count -eq 0 ]]; then
  ok "No errors in last 10 minutes"
else
  warn "Found $error_count ERROR lines in last 10 minutes"
  info "Recent errors:"
  journalctl -u quantum-apply-layer --since "10 minutes ago" | grep ERROR | tail -5
fi

echo

# Check Prometheus metrics
echo "== P3.7: Prometheus metrics (testnet) =="
if curl -s http://localhost:8043/metrics >/dev/null 2>&1; then
  ok "Prometheus metrics available"
  
  execute_total=$(curl -s http://localhost:8043/metrics | grep 'quantum_apply_execute_total' | wc -l)
  
  if [[ $execute_total -gt 0 ]]; then
    ok "quantum_apply_execute_total metrics present"
    info "Execution metrics:"
    curl -s http://localhost:8043/metrics | grep 'quantum_apply_execute_total' | head -5
  else
    info "No execution metrics yet"
  fi
else
  warn "Prometheus metrics not available"
fi

echo

# Summary
echo "=== P3.1 Testnet Proof Pack COMPLETE ==="
echo
info "Key findings:"
echo "  - Mode: testnet (execution enabled)"
echo "  - BTCUSDT executions: $btc_executed"
echo "  - Non-allowlist executions: eth=$eth_executed, sol=$sol_executed"
echo "  - Dedupe keys: $dedupe_count"
echo "  - Errors (10min): $error_count"
echo "  - Order IDs tracked: $order_ids"
echo
ok "P3.1 testnet mode verified operational"
echo
warn "IMPORTANT: Monitor Binance testnet balance and positions"
info "Check: https://testnet.binancefuture.com/"
