# Quantum Trader - Operational Proof Scripts Ledger

This ledger tracks all binary proof scripts used to validate system components.

## Policy Autonomy Proofs

### PolicyStore (Fail-Closed Semantics)
- **Script:** `scripts/proof_policy_fail_closed.sh`
- **Purpose:** Validates that PolicyStore enforces fail-closed semantics (no hardcoded fallbacks)
- **Tests:** 10 tests covering policy loading, validation, and fail-closed behavior
- **Status:** ✅ DEPLOYED (10/10 PASS on VPS)
- **Commit:** 19f84364f

### Exit Ownership Gate
- **Script:** `scripts/proof_exit_owner_gate.sh`
- **Purpose:** Validates that only exitbrain_v3_5 can place reduceOnly=true orders
- **Tests:** 6 tests covering exit ownership enforcement in apply_layer
- **Status:** ✅ DEPLOYED (6/6 PASS on VPS)
- **Commit:** 3510e0b20

## Policy Automation Proofs

### Policy Refresh Automation
- **Script:** `scripts/proof_policy_refresh.sh`
- **Purpose:** Validates automated policy refresh (30min timer, atomic validation, audit trail)
- **Tests:** 10 tests covering:
  - Refresh script existence and executability
  - Systemd service and timer configuration
  - Policy field validation (version, hash, valid_until)
  - Expiry time checks
  - Audit trail integration in PolicyStore
  - Fail-open semantics
- **Status:** 🔄 CREATED (pending deployment)
- **Dependencies:**
  - `scripts/policy_refresh.sh`
  - `deploy/systemd/quantum-policy-refresh.service`
  - `deploy/systemd/quantum-policy-refresh.timer`
  - `lib/policy_store.py` (audit trail)

### Exit Owner Monitoring
- **Script:** `scripts/proof_exit_owner_watch.sh`
- **Purpose:** Validates exit owner violation monitoring (5min checks, alerting)
- **Tests:** 10 tests covering:
  - Watch script existence and executability
  - Systemd service and timer configuration
  - DENY_NOT_EXIT_OWNER detection
  - Alert publishing to quantum:stream:alerts
  - Required alert fields (alert_type, deny_count, window, timestamp)
  - Monitoring correct service (quantum-apply-layer)
- **Status:** 🔄 CREATED (pending deployment)
- **Dependencies:**
  - `scripts/exit_owner_watch.sh`
  - `deploy/systemd/quantum-exit-owner-watch.service`
  - `deploy/systemd/quantum-exit-owner-watch.timer`

## Deployment Instructions

### Policy Refresh (30min automation)
```bash
# 1. Copy systemd files
sudo cp deploy/systemd/quantum-policy-refresh.service /etc/systemd/system/
sudo cp deploy/systemd/quantum-policy-refresh.timer /etc/systemd/system/
sudo systemctl daemon-reload

# 2. Enable and start timer
sudo systemctl enable --now quantum-policy-refresh.timer

# 3. Verify timer active
systemctl list-timers | grep quantum-policy-refresh

# 4. Run proof
bash scripts/proof_policy_refresh.sh
```

### Exit Owner Monitoring (5min automation)
```bash
# 1. Copy systemd files
sudo cp deploy/systemd/quantum-exit-owner-watch.service /etc/systemd/system/
sudo cp deploy/systemd/quantum-exit-owner-watch.timer /etc/systemd/system/
sudo systemctl daemon-reload

# 2. Enable and start timer
sudo systemctl enable --now quantum-exit-owner-watch.timer

# 3. Verify timer active
systemctl list-timers | grep quantum-exit-owner-watch

# 4. Run proof
bash scripts/proof_exit_owner_watch.sh
```

## Monitoring

### Policy Refresh Logs
```bash
# Watch timer triggers
journalctl -u quantum-policy-refresh.timer -f

# Check refresh service logs
journalctl -u quantum-policy-refresh.service -n 50 | grep POLICY-REFRESH

# Verify audit trail
redis-cli XREAD COUNT 10 STREAMS quantum:stream:policy.audit 0
```

### Exit Owner Watch Logs
```bash
# Watch timer triggers
journalctl -u quantum-exit-owner-watch.timer -f

# Check watch service logs
journalctl -u quantum-exit-owner-watch.service -n 50 | grep EXIT-OWNER-WATCH

# Verify alerts (if any)
redis-cli XREAD COUNT 5 STREAMS quantum:stream:alerts 0
```

## Success Criteria

### Policy Refresh
- ✅ Timer runs every 30 minutes
- ✅ Policy always fresh (valid_until > now + 30min)
- ✅ Audit trail contains 48+ entries per day
- ✅ No POLICY_STALE errors

### Exit Owner Monitoring
- ✅ Timer runs every 5 minutes
- ✅ DENY events detected within 5 minutes
- ✅ Alerts written to quantum:stream:alerts
- ✅ No unauthorized exit attempts succeed
