# P2.6 Portfolio Gate - VPS Validation Checklist

**Commit:** 2343184e  
**Component:** P2.6 Portfolio Gate + THREE-permit Apply Layer integration  
**Date:** 2026-01-27

---

## PRE-DEPLOYMENT: 5 Critical Validation Points

### A) Stream Name Must Match Exactly

**Check harvest proposal stream exists and has data:**

```bash
redis-cli EXISTS quantum:stream:harvest.proposal
# Expected: 1

redis-cli XLEN quantum:stream:harvest.proposal
# Expected: >0 (if P2.5 is running)

redis-cli XINFO STREAM quantum:stream:harvest.proposal | head -20
# Should show consumer groups, entries, etc.
```

**If XLEN=0 for long periods:**
- P2.6 has nothing to consume
- Check P2.5 Harvest Proposal Publisher is running:
  ```bash
  systemctl is-active quantum-harvest-proposal
  journalctl -u quantum-harvest-proposal -n 50 --no-pager
  ```

---

### B) Apply Layer Must Check Same Permit Key P2.6 Writes

**P2.6 writes:** `quantum:permit:p26:{plan_id}`  
**Apply Layer must check:** `quantum:permit:p26:{plan_id}` in Lua script

**Verify Apply Layer code:**

```bash
grep -n "permit:p26" /home/qt/quantum_trader/microservices/apply_layer/main.py | head
# Should show:
#   - local p26_key = ... "quantum:permit:p26"...
#   - p26 = redis.call("GET", p26_key)
#   - etc.
```

**Critical:** Key format must be EXACTLY `quantum:permit:p26:{plan_id}`, not `p2.6`, not `p26_permit`, etc.

---

### C) TTL Window: Permit TTL vs Pipeline Latency

**Config:** TTL=60s (default)

**Risk:** If pipeline is slow/backlogged, permit can expire before Apply Layer consumes it.

**Monitor permit TTLs:**

```bash
# List current P2.6 permits
redis-cli --scan --pattern 'quantum:permit:p26:*' | head -5

# Pick a permit key and check remaining TTL
redis-cli TTL quantum:permit:p26:<plan_id>
# Should be between 0-60

# If TTL frequently <10s when Apply Layer tries to consume → increase TTL
```

**If permits expire too fast:**
- Increase `P26_PERMIT_TTL` in `/etc/quantum/portfolio-gate.env`
- Restart: `systemctl restart quantum-portfolio-gate`

---

### D) Fail-Closed Can Become "Fail-Always"

**Risk:** If `quantum:position:snapshot:*` is missing/stale, P2.6 will fail-closed → no permits → Apply Layer blocks ALL execution.

**Check position snapshots exist and are fresh:**

```bash
# List position snapshots
redis-cli --scan --pattern 'quantum:position:snapshot:*'
# Expected: BTCUSDT, ETHUSDT, SOLUSDT (for default allowlist)

# Check timestamp for staleness
redis-cli HGET quantum:position:snapshot:BTCUSDT ts_epoch
# Compare with current time:
date +%s
# Age should be <300s (5 minutes)
```

**If snapshots missing:**
- Check P3.3 Position State Brain is running:
  ```bash
  systemctl is-active quantum-position-state-brain
  journalctl -u quantum-position-state-brain -n 50 --no-pager
  ```

**If snapshots stale:**
- Check P3.3 is receiving exchange updates
- Check P3.3 logs for errors

**Observability:** New metric `p26_symbols_with_snapshot` shows how many symbols have valid data.

---

### E) Cooldown Key Can Be Too Aggressive

**Risk:** If cooldown triggers too often, P2.6 downgrades everything → portfolio never de-risks properly.

**Monitor cooldown activity:**

```bash
# List active cooldowns
redis-cli --scan --pattern 'quantum:p26:cooldown:*'
# Should be 0-3 keys typically

# Check TTL on cooldown
redis-cli TTL quantum:p26:cooldown:BTCUSDT
# Should be between 0-60s
```

**If cooldown is always active for all symbols:**
- Check `P26_COOLDOWN_SEC` (default 60s)
- Review `p26_actions_downgraded_total{reason="cooldown"}` metric
- Consider reducing cooldown or adjusting allowlist

---

## DEPLOYMENT: VPS Installation

**Run as root:**

```bash
cd /root/quantum_trader
git pull origin main
bash ops/p26_deploy_and_proof.sh
```

**Expected output:**
- [1/10] Git pull ✓
- [2/10] Rsync ✓
- [3/10] Config install ✓
- [4/10] Systemd install ✓
- [5/10] Service start ✓ (quantum-portfolio-gate ACTIVE)
- [6/10] Metrics respond ✓
- [7/10] Apply Layer restart ✓
- [8/10] Stream connectivity ✓
- [9/10] P2.6 permits check
- [10/10] Proof document saved: `/home/qt/quantum_trader/docs/P2_6_VPS_PROOF.txt`

---

## POST-DEPLOYMENT: 6 Critical Proofs

### 1. Service Health

```bash
systemctl is-active quantum-portfolio-gate quantum-apply-layer
# Both should return: active
```

**If quantum-portfolio-gate not active:**
```bash
systemctl status quantum-portfolio-gate --no-pager
journalctl -u quantum-portfolio-gate -n 50 --no-pager
```

**If quantum-apply-layer not active:**
```bash
systemctl status quantum-apply-layer --no-pager
journalctl -u quantum-apply-layer -n 50 --no-pager
```

---

### 2. Metrics Responding

```bash
curl -s http://127.0.0.1:8047/metrics | grep -E "p26_(stress|heat|stream_reads_total|permit_issued_total|fail_closed_total|symbols_with_snapshot|total_abs_notional)" | head -50
```

**Expected metrics:**
- `p26_stress` (gauge 0-1)
- `p26_heat` (gauge 0-1)
- `p26_stream_reads_total` (counter, should increment)
- `p26_permit_issued_total` (counter, may be low if mostly HOLD)
- `p26_fail_closed_total` (counter, should be 0 or low)
- `p26_symbols_with_snapshot` (gauge, should be 1-3)
- `p26_total_abs_notional` (gauge, should match portfolio size)

**If metrics not responding:**
- Check port 8047 not blocked
- Check service logs for startup errors

---

### 3. Portfolio Gate Stream Populated

```bash
# Check stream exists and has entries
redis-cli XINFO STREAM quantum:stream:portfolio.gate | head -25

# Read last 3 decisions
redis-cli XREVRANGE quantum:stream:portfolio.gate + - COUNT 3
```

**Expected:** Entries with fields:
- `plan_id`
- `symbol`
- `action_proposed`
- `final_action`
- `gate_reason`
- `stress`
- `heat`

**If stream empty:**
- Check harvest.proposal stream has data (see 4 below)
- Check P2.6 logs for consumption errors

---

### 4. Harvest Proposal Stream Check

```bash
# Verify upstream is producing
redis-cli XINFO STREAM quantum:stream:harvest.proposal | head -25

# Read last 3 proposals
redis-cli XREVRANGE quantum:stream:harvest.proposal + - COUNT 3
```

**Expected:** Entries with:
- `plan_id`
- `symbol`
- `action_proposed` or `harvest_action`
- `kill_score`

**If harvest.proposal empty:**
- P2.5 Harvest Proposal Publisher not running or no positions to harvest
- Check: `systemctl is-active quantum-harvest-proposal`

---

### 5. Service Logs Healthy

```bash
journalctl -u quantum-portfolio-gate --since "5 minutes ago" -n 120 --no-pager
```

**Look for:**
- ✅ "P2.6 Portfolio Gate Started"
- ✅ "Consumer group 'p26_portfolio_gate' created" or "already exists"
- ✅ "Processing: BTCUSDT PARTIAL_50 (ks=0.xyz)"
- ✅ "Portfolio metrics: heat=0.xyz, conc=0.xyz, stress=0.xyz"
- ✅ "Permit issued: plan_id" (if not HOLD)

**Red flags:**
- ❌ Repeated "No valid snapshots"
- ❌ "Error reading stream" (repeated)
- ❌ Python exceptions/tracebacks

---

### 6. Critical Functional Test: End-to-End Permit Flow

**Goal:** Verify at least ONE plan gets P2.6 permit issued.

**Step 1: Find a recent harvest proposal**

```bash
redis-cli XREVRANGE quantum:stream:harvest.proposal + - COUNT 1
```

**Copy the `plan_id` from output.**

**Step 2: Check if P2.6 issued permit for that plan_id**

```bash
redis-cli EXISTS quantum:permit:p26:<plan_id>
# Expected: 1 (if final_action != HOLD)

redis-cli TTL quantum:permit:p26:<plan_id>
# Expected: 0-60 (seconds remaining)
```

**Step 3: Check portfolio.gate stream for decision**

```bash
redis-cli XREVRANGE quantum:stream:portfolio.gate + - COUNT 5 | grep -A20 "<plan_id>"
```

**Expected fields:**
- `final_action` should be set (HOLD, PARTIAL_25, PARTIAL_50, PARTIAL_75, or FULL_CLOSE_PROPOSED)
- `gate_reason` explains why (pass, forbid_full_close_portfolio_cold, riskoff_accelerate, etc.)

**If EXISTS=0 but P2.6 logs show "final_action != HOLD":**
- **BUG:** Permit write failed
- Check P2.6 logs for Redis errors
- Check Redis connection

**If EXISTS=1 but Apply Layer still blocks:**
- Check Apply Layer is looking for correct key: `quantum:permit:p26:{plan_id}`
- Check Lua script in apply_layer/main.py includes p26_key

---

## DRY-RUN / DISARMED EXPECTATIONS

**In dry_run mode (default):**

| Metric | Expected Behavior |
|--------|-------------------|
| `p26_stream_reads_total` | ✅ Should increment (P2.6 consuming) |
| `p26_plans_seen_total` | ✅ Should increment (proposals processed) |
| `p26_permit_issued_total` | ⚠️  May be LOW (HOLD is common in dry-run) |
| `p26_fail_closed_total` | ✅ Should be 0 or very low |
| Portfolio.gate stream | ✅ Should have entries |
| Apply Layer execution | ⚠️  Still mostly SKIP (dry-run), but now requires P2.6 permit |

**Key point:** Even in dry-run, P2.6 should:
- Consume proposals ✅
- Compute portfolio metrics ✅
- Write decisions to portfolio.gate ✅
- Issue permits (when final_action != HOLD) ✅

Apply Layer will still SKIP execution (dry-run mode), but it MUST check for P2.6 permit first.

---

## ROLLBACK PROCEDURE

**If P2.6 causes issues, roll back immediately:**

```bash
# Stop P2.6 service
systemctl stop quantum-portfolio-gate
systemctl disable quantum-portfolio-gate

# Revert code
cd /root/quantum_trader
git revert 2343184e
git push origin main

# Sync to working directory
rsync -av --delete --exclude='.git' /root/quantum_trader/ /home/qt/quantum_trader/
chown -R qt:qt /home/qt/quantum_trader

# Restart Apply Layer (reverts to TWO-permit mode)
systemctl restart quantum-apply-layer

# Verify rollback
systemctl is-active quantum-apply-layer
journalctl -u quantum-apply-layer -n 50 --no-pager | grep -i permit
```

**Recovery time:** ~2 minutes

**After rollback:**
- Apply Layer should go back to requiring only TWO permits (Governor + P3.3)
- Execution should resume (if it was blocked by missing P2.6 permit)

---

## DEBUGGING TIPS

### "No permits issued"

**Symptom:** `p26_permit_issued_total` stays at 0

**Check:**
1. Are harvest proposals arriving? (see Proof #4)
2. Are all decisions HOLD? (check portfolio.gate stream)
3. Is portfolio state missing? (check `p26_fail_closed_total{reason="portfolio_state"}`)

**Fix:**
- If no harvest proposals → check P2.5 is running
- If all HOLD → check allowlist matches active symbols
- If portfolio state missing → check P3.3 Position State Brain

---

### "Permits expiring before consumption"

**Symptom:** Apply Layer logs "missing_p26" but P2.6 issued permit earlier

**Check:**
```bash
# Monitor permit creation vs consumption time
journalctl -u quantum-portfolio-gate -f | grep "Permit issued"
journalctl -u quantum-apply-layer -f | grep "p26\|permit"
```

**Fix:**
- Increase `P26_PERMIT_TTL` from 60s to 120s or 300s
- Reduce Apply Layer `APPLY_POLL_SEC` to consume faster

---

### "Portfolio always cold/hot"

**Symptom:** Stress stuck at 0.0 or 1.0

**Check:**
```bash
curl -s http://127.0.0.1:8047/metrics | grep p26_
# Look at: p26_heat, p26_concentration, p26_stress, p26_total_abs_notional
```

**Debug:**
- If `p26_total_abs_notional` near 0 → no positions, stress=0 (correct)
- If `p26_symbols_with_snapshot` = 0 → no snapshot data (fail-closed)
- If stress always high → check thresholds (H_MIN, H_MAX, stress coefficients)

**Fix:**
- Adjust `P26_H_MIN`, `P26_H_MAX` in `/etc/quantum/portfolio-gate.env`
- Restart: `systemctl restart quantum-portfolio-gate`

---

## SUCCESS CRITERIA CHECKLIST

- [ ] Service `quantum-portfolio-gate` is `active`
- [ ] Service `quantum-apply-layer` is `active` (restarted with new code)
- [ ] Metrics respond on port 8047
- [ ] `p26_stream_reads_total` incrementing
- [ ] `p26_symbols_with_snapshot` shows 1-3 symbols
- [ ] `p26_total_abs_notional` matches expected portfolio size
- [ ] `portfolio.gate` stream has entries
- [ ] At least one P2.6 permit issued (if any non-HOLD decisions)
- [ ] Apply Layer logs mention "p26" or "missing_p26" (permit check active)
- [ ] No Python exceptions in P2.6 logs
- [ ] `p26_fail_closed_total` is 0 or very low

**If all checked:** P2.6 is operational ✅

**Next steps:**
1. Monitor for 1-2 hours
2. Check Grafana for P2.6 metrics
3. Observe policy interventions (anti-panic, risk-off)
4. Gradually expand allowlist if needed

---

## CONTACTS & REFERENCES

- **Commit:** 2343184e
- **Proof document:** `/home/qt/quantum_trader/docs/P2_6_VPS_PROOF.txt`
- **Metrics:** http://localhost:8047/metrics
- **Logs:** `journalctl -u quantum-portfolio-gate -f`
- **Config:** `/etc/quantum/portfolio-gate.env`
- **Service:** `/etc/systemd/system/quantum-portfolio-gate.service`
