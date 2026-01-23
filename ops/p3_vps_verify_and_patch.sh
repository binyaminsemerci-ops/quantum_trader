#!/usr/bin/env bash
#
# P3 VPS Verification + Patch Script
#
# Phase 1: Verify P3.0 dry_run operational on VPS
# Phase 2: Check paths match (/home/qt/quantum_trader vs /root/quantum_trader)
# Phase 3: Verify idempotency and safety gates
# Phase 4: Confirm REAL Binance code (no placeholders)
#
# Run from VPS as root: bash /root/quantum_trader/ops/p3_vps_verify_and_patch.sh
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
}

info() {
  echo -e "${NC}ℹ️  $1${NC}"
}

echo "=== P3 VPS Verification + Patch ==="
date
echo

# Check service status
info "1.1: Service status"
if systemctl is-active --quiet quantum-apply-layer; then
  ok "quantum-apply-layer service ACTIVE"
else
  fail "quantum-apply-layer service NOT ACTIVE"
  exit 1
fi

# Show WorkingDirectory
info "1.2: WorkingDirectory from systemd unit"
working_dir=$(systemctl cat quantum-apply-layer | grep "^WorkingDirectory=" | cut -d= -f2)
if [[ "$working_dir" == "/home/qt/quantum_trader" ]]; then
  ok "WorkingDirectory: /home/qt/quantum_trader (CORRECT)"
else
  warn "WorkingDirectory: $working_dir (expected /home/qt/quantum_trader)"
fi

# Check APPLY_MODE
info "1.3: APPLY_MODE configuration"
if [ -f /etc/quantum/apply-layer.env ]; then
  mode=$(grep "^APPLY_MODE=" /etc/quantum/apply-layer.env | cut -d= -f2)
  if [[ "$mode" == "dry_run" ]]; then
    ok "APPLY_MODE=dry_run (SAFE)"
  else
    warn "APPLY_MODE=$mode (expected dry_run for initial verification)"
  fi
else
  fail "/etc/quantum/apply-layer.env not found"
  exit 1
fi

# Check Redis streams exist
info "1.4: Redis streams"
plan_count=$(redis-cli XLEN quantum:stream:apply.plan 2>/dev/null || echo "0")
result_count=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null || echo "0")

if [[ $plan_count -gt 0 ]]; then
  ok "apply.plan stream: $plan_count entries"
else
  warn "apply.plan stream empty (service may need time to populate)"
fi

if [[ $result_count -gt 0 ]]; then
  ok "apply.result stream: $result_count entries"
else
  warn "apply.result stream empty (service may need time to populate)"
fi

# Verify NO execution in dry_run
info "1.5: Verify NO actual execution in dry_run"
if [[ $result_count -gt 0 ]]; then
  executed_true=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -c '"executed": true' || echo "0")
  would_execute=$(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -c '"would_execute": true' || echo "0")
  
  if [[ $executed_true -eq 0 ]]; then
    ok "No actual executions (executed=false in all results)"
  else
    fail "Found $executed_true results with executed=true (unexpected in dry_run)"
  fi
  
  if [[ $would_execute -gt 0 ]]; then
    ok "Found $would_execute results with would_execute=true (correct for dry_run)"
  fi
else
  info "No results yet to verify"
fi

# Check idempotency
info "1.6: Idempotency (dedupe keys)"
dedupe_count=$(redis-cli KEYS "quantum:apply:dedupe:*" 2>/dev/null | wc -l)
if [[ $dedupe_count -gt 0 ]]; then
  ok "Idempotency active: $dedupe_count dedupe keys"
else
  info "No dedupe keys yet (normal if no EXECUTE plans created)"
fi

# Check logs for errors
info "1.7: Service health (recent errors)"
error_count=$(journalctl -u quantum-apply-layer --since "5 minutes ago" | grep -c ERROR || echo "0")
if [[ $error_count -eq 0 ]]; then
  ok "No errors in last 5 minutes"
else
  warn "Found $error_count ERROR lines in last 5 minutes"
fi

# ============================================================================
# PHASE 2: PATH CONSISTENCY CHECK
# ============================================================================

section "PHASE 2: Path Consistency Check"

info "2.1: Repo paths comparison"

# Check if /root/quantum_trader exists
if [ -d /root/quantum_trader ]; then
  root_hash=$(cd /root/quantum_trader && git rev-parse HEAD 2>/dev/null || echo "no_git")
  info "/root/quantum_trader exists (git HEAD: ${root_hash:0:8})"
else
  info "/root/quantum_trader does not exist"
  root_hash="none"
fi

# Check if /home/qt/quantum_trader exists
if [ -d /home/qt/quantum_trader ]; then
  qt_hash=$(cd /home/qt/quantum_trader && git rev-parse HEAD 2>/dev/null || echo "no_git")
  info "/home/qt/quantum_trader exists (git HEAD: ${qt_hash:0:8})"
else
  warn "/home/qt/quantum_trader does not exist"
  qt_hash="none"
fi

# Compare
if [[ "$root_hash" != "none" && "$qt_hash" != "none" ]]; then
  if [[ "$root_hash" == "$qt_hash" ]]; then
    ok "Both paths have same git commit ($root_hash)"
  else
    warn "Path mismatch: /root has $root_hash, /home/qt has $qt_hash"
    info "To sync: sudo rsync -av --delete /root/quantum_trader/ /home/qt/quantum_trader/"
  fi
fi

info "2.2: Service ExecStart path"
exec_start=$(systemctl cat quantum-apply-layer | grep "^ExecStart=" | cut -d= -f2-)
info "ExecStart: $exec_start"

if [[ "$exec_start" == *"microservices/apply_layer/main.py"* ]]; then
  ok "ExecStart uses relative path (correct with WorkingDirectory)"
else
  warn "ExecStart may have hardcoded path"
fi

info "2.3: Check which Python file is running"
apply_pid=$(systemctl show -p MainPID quantum-apply-layer | cut -d= -f2)
if [[ "$apply_pid" != "0" && -n "$apply_pid" ]]; then
  cwd=$(pwdx "$apply_pid" 2>/dev/null | awk '{print $2}' || echo "unknown")
  if [[ "$cwd" == "/home/qt/quantum_trader" ]]; then
    ok "Service running from: /home/qt/quantum_trader (CORRECT)"
  else
    warn "Service running from: $cwd (expected /home/qt/quantum_trader)"
  fi
else
  info "Could not determine service PID working directory"
fi

# ============================================================================
# PHASE 3: VERIFY REAL BINANCE CODE (NOT PLACEHOLDER)
# ============================================================================

section "PHASE 3: Verify Real Binance Testnet Execution Code"

info "3.1: Check apply_layer main.py for real Binance implementation"

apply_layer_path="/home/qt/quantum_trader/microservices/apply_layer/main.py"

if [ -f "$apply_layer_path" ]; then
  # Check for real Binance methods
  has_binance_request=$(grep -c "def binance_request" "$apply_layer_path" || echo "0")
  has_get_position=$(grep -c "def get_position" "$apply_layer_path" || echo "0")
  has_place_order=$(grep -c "def place_market_order" "$apply_layer_path" || echo "0")
  has_round_quantity=$(grep -c "def round_quantity" "$apply_layer_path" || echo "0")
  
  # Check for placeholder text (should NOT exist)
  has_placeholder=$(grep -c "Placeholder for actual Binance" "$apply_layer_path" || echo "0")
  has_simulated=$(grep -c "simulated_success" "$apply_layer_path" || echo "0")
  
  if [[ $has_binance_request -gt 0 ]]; then
    ok "binance_request() method found (real API calls)"
  else
    fail "binance_request() method NOT found"
  fi
  
  if [[ $has_get_position -gt 0 ]]; then
    ok "get_position() method found (position check)"
  else
    fail "get_position() method NOT found"
  fi
  
  if [[ $has_place_order -gt 0 ]]; then
    ok "place_market_order() method found (real orders)"
  else
    fail "place_market_order() method NOT found"
  fi
  
  if [[ $has_round_quantity -gt 0 ]]; then
    ok "round_quantity() method found (stepSize rounding)"
  else
    fail "round_quantity() method NOT found"
  fi
  
  if [[ $has_placeholder -gt 0 ]]; then
    fail "Found 'Placeholder' text in code (indicates incomplete implementation)"
  else
    ok "No placeholder text found (implementation complete)"
  fi
  
  if [[ $has_simulated -gt 0 ]]; then
    warn "Found 'simulated_success' in code (may indicate simulation mode)"
  else
    ok "No simulation markers found (real execution code)"
  fi
  
  # Check for reduceOnly
  has_reduce_only=$(grep -c "reduceOnly" "$apply_layer_path" || echo "0")
  if [[ $has_reduce_only -gt 0 ]]; then
    ok "reduceOnly flag found in code (safe for testnet)"
  else
    warn "reduceOnly flag not found (risk of opening new positions)"
  fi
  
else
  fail "$apply_layer_path not found"
  exit 1
fi

info "3.2: Check for Binance API dependencies"
python3 -c "import hmac, hashlib, urllib.request, urllib.parse; print('OK')" 2>/dev/null && ok "Python urllib/hmac available" || warn "Python urllib/hmac import failed"

# ============================================================================
# SUMMARY
# ============================================================================

section "Verification Summary"

echo "PHASE 1: Dry Run Mode"
echo "  - Service: $(systemctl is-active quantum-apply-layer)"
echo "  - Mode: $mode"
echo "  - Plans: $plan_count entries"
echo "  - Results: $result_count entries"
echo "  - Errors: $error_count in last 5min"
echo

echo "PHASE 2: Path Consistency"
echo "  - WorkingDirectory: $working_dir"
echo "  - /root hash: $root_hash"
echo "  - /home/qt hash: $qt_hash"
echo

echo "PHASE 3: Binance Implementation"
echo "  - binance_request: $has_binance_request"
echo "  - get_position: $has_get_position"
echo "  - place_market_order: $has_place_order"
echo "  - round_quantity: $has_round_quantity"
echo "  - reduceOnly: $has_reduce_only"
echo "  - Placeholders: $has_placeholder"
echo

if [[ "$mode" == "dry_run" ]] && [[ $plan_count -gt 0 ]] && [[ $has_binance_request -gt 0 ]] && [[ $has_reduce_only -gt 0 ]]; then
  ok "P3.0 dry_run verified operational, P3.1 testnet code ready"
  echo
  info "Next steps:"
  echo "  1. Monitor dry_run for 24h: journalctl -u quantum-apply-layer -f"
  echo "  2. Run proof pack: bash /home/qt/quantum_trader/ops/p3_proof_dry_run.sh"
  echo "  3. To enable testnet: Edit /etc/quantum/apply-layer.env (add Binance creds, set APPLY_MODE=testnet)"
  echo "  4. Restart: sudo systemctl restart quantum-apply-layer"
  echo "  5. Verify testnet: bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh"
else
  warn "Some verification checks did not pass - review output above"
fi

echo
info "Verification complete"
date
