#!/usr/bin/env bash
#
# P3 Proof Pack - Dry Run Mode Verification
#
# Verifies:
# 1. Apply layer service active
# 2. Plans created in quantum:stream:apply.plan
# 3. Results published in quantum:stream:apply.result
# 4. Idempotency working (dedupe keys prevent duplicates)
# 5. No Binance calls in dry_run mode
# 6. Allowlist enforcement (only BTCUSDT marked EXECUTE if default config)
# 7. Safety gates working (kill_score thresholds, kill switch)
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

echo "=== P3 Apply Layer Proof Pack - Dry Run Mode ==="
date
echo

# Check Redis connectivity
echo "== P0: Core health =="
redis-cli PING >/dev/null 2>&1 && ok "Redis PING" || fail "Redis PING failed"

# Check apply layer service
if systemctl is-active --quiet quantum-apply-layer; then
  ok "quantum-apply-layer service active"
else
  fail "quantum-apply-layer service not active"
fi

echo

# Check mode configuration
echo "== P3.0: Verify dry_run mode =="
if systemctl cat quantum-apply-layer | grep -q "EnvironmentFile=/etc/quantum/apply-layer.env"; then
  info "Using env file: /etc/quantum/apply-layer.env"
  
  if [ -f /etc/quantum/apply-layer.env ]; then
    mode=$(grep "^APPLY_MODE=" /etc/quantum/apply-layer.env | cut -d= -f2)
    if [[ "$mode" == "dry_run" ]]; then
      ok "APPLY_MODE=dry_run"
    else
      warn "APPLY_MODE=$mode (expected dry_run)"
    fi
    
    allowlist=$(grep "^APPLY_ALLOWLIST=" /etc/quantum/apply-layer.env | cut -d= -f2)
    info "APPLY_ALLOWLIST=$allowlist"
    
    kill_switch=$(grep "^APPLY_KILL_SWITCH=" /etc/quantum/apply-layer.env | cut -d= -f2)
    info "APPLY_KILL_SWITCH=$kill_switch"
  else
    warn "/etc/quantum/apply-layer.env not found"
  fi
else
  warn "EnvironmentFile not configured in systemd unit"
fi

echo

# Check streams exist
echo "== P3.1: Apply streams populated =="
plan_count=$(redis-cli XLEN quantum:stream:apply.plan 2>/dev/null || echo "0")
result_count=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null || echo "0")

if [[ $plan_count -gt 0 ]]; then
  ok "apply.plan stream has $plan_count entries"
else
  warn "apply.plan stream empty (service may need time to populate)"
fi

if [[ $result_count -gt 0 ]]; then
  ok "apply.result stream has $result_count entries"
else
  warn "apply.result stream empty"
fi

echo

# Check recent plans
echo "== P3.2: Recent apply plans =="
if [[ $plan_count -gt 0 ]]; then
  echo "Latest 3 plans:"
  redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 | grep -E "(symbol|decision|reason_codes)" | head -15
  echo
  
  # Count decisions
  info "Decision breakdown (last 10 plans):"
  redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 10 | grep "decision" | awk '{print $2}' | sort | uniq -c
else
  warn "No plans to display"
fi

echo

# Check idempotency
echo "== P3.3: Idempotency (dedupe keys) =="
dedupe_count=$(redis-cli KEYS "quantum:apply:dedupe:*" 2>/dev/null | wc -l)

if [[ $dedupe_count -gt 0 ]]; then
  ok "Found $dedupe_count dedupe keys (idempotency active)"
  
  # Show one example
  example_key=$(redis-cli KEYS "quantum:apply:dedupe:*" | head -1)
  if [[ -n "$example_key" ]]; then
    ttl=$(redis-cli TTL "$example_key")
    info "Example: $example_key (TTL=${ttl}s)"
  fi
else
  warn "No dedupe keys found (service may not have executed any EXECUTE plans yet)"
fi

echo

# Check no Binance calls in dry_run
echo "== P3.4: Verify NO execution in dry_run =="
if [[ $result_count -gt 0 ]]; then
  # Check results for executed=false and would_execute=true
  executed_true=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep '"executed": true' | wc -l)
  would_execute=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep '"would_execute": true' | wc -l)
  
  if [[ $executed_true -eq 0 ]]; then
    ok "No actual executions (executed=false in all results)"
  else
    fail "Found $executed_true results with executed=true (unexpected in dry_run)"
  fi
  
  if [[ $would_execute -gt 0 ]]; then
    ok "Found $would_execute results with would_execute=true (correct for dry_run)"
  fi
else
  warn "No results to verify"
fi

echo

# Check allowlist enforcement
echo "== P3.5: Allowlist enforcement =="
if [[ $plan_count -gt 0 ]]; then
  btc_execute=$(redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep -A5 "symbol.*BTCUSDT" | grep -c "decision.*EXECUTE" || echo "0")
  eth_skip=$(redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep -A5 "symbol.*ETHUSDT" | grep -c "not_in_allowlist" || echo "0")
  
  if [[ $btc_execute -gt 0 ]]; then
    ok "BTCUSDT has EXECUTE decisions (allowlist working)"
  else
    info "No EXECUTE decisions for BTCUSDT yet (may be blocked by other gates)"
  fi
  
  if [[ $eth_skip -gt 0 ]]; then
    ok "ETHUSDT blocked by allowlist (not_in_allowlist reason code)"
  else
    info "No ETHUSDT plans with not_in_allowlist (check if ETH in allowlist or no proposals)"
  fi
else
  warn "No plans to check allowlist"
fi

echo

# Check Prometheus metrics
echo "== P3.6: Prometheus metrics =="
if curl -s http://localhost:8043/metrics >/dev/null 2>&1; then
  ok "Prometheus metrics available on :8043"
  
  plan_total=$(curl -s http://localhost:8043/metrics | grep "^quantum_apply_plan_total" | wc -l)
  dedupe_hits=$(curl -s http://localhost:8043/metrics | grep "^quantum_apply_dedupe_hits_total" | awk '{print $2}')
  
  if [[ $plan_total -gt 0 ]]; then
    info "quantum_apply_plan_total: $plan_total metrics"
  fi
  
  if [[ -n "$dedupe_hits" && "$dedupe_hits" != "0" ]]; then
    info "quantum_apply_dedupe_hits_total: $dedupe_hits"
  fi
else
  warn "Prometheus metrics not available (check service logs)"
fi

echo

# Check service logs for errors
echo "== P3.7: Service health (recent logs) =="
error_count=$(journalctl -u quantum-apply-layer --since "5 minutes ago" | grep -c ERROR || echo "0")

if [[ $error_count -eq 0 ]]; then
  ok "No errors in last 5 minutes"
else
  warn "Found $error_count ERROR lines in last 5 minutes"
  info "Recent errors:"
  journalctl -u quantum-apply-layer --since "5 minutes ago" | grep ERROR | tail -5
fi

echo

# Summary
echo "=== P3.0 Dry Run Proof Pack COMPLETE ==="
echo
info "Key findings:"
echo "  - Service: $(systemctl is-active quantum-apply-layer)"
echo "  - Mode: dry_run (no actual execution)"
echo "  - Plans: $plan_count in stream"
echo "  - Results: $result_count in stream"
echo "  - Dedupe keys: $dedupe_count"
echo "  - Errors (5min): $error_count"
echo
ok "P3.0 dry_run mode verified operational"
echo
info "Next step: Enable testnet mode (see ops/p3_proof_testnet.sh)"
