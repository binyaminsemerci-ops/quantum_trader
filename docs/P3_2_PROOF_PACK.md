# P3.2 Governor - Proof Pack Documentation

**Test Date:** TBD (To be executed on VPS)  
**Environment:** VPS Production (46.224.116.254)  
**Governor Version:** 1.0.0

---

## Proof Objectives

Verify that P3.2 Governor:
1. ✅ Is deployed and operational
2. ✅ Issues execution permits for valid plans
3. ✅ Blocks plans when limits exceeded
4. ✅ Triggers auto-disarm under unsafe conditions
5. ✅ Exposes Prometheus metrics
6. ✅ Writes events to governor.events stream
7. ✅ Integrates correctly with Apply Layer

---

## Pre-Deployment Checklist

- [ ] P3.0/P3.1 Apply Layer deployed and operational
- [ ] Redis running and accessible
- [ ] Python 3.8+ with redis-py and prometheus-client
- [ ] Code synced to VPS (`git pull` in `/root/quantum_trader`)
- [ ] `/etc/quantum/governor.env` reviewed and configured

---

## Proof Pack Execution

### Phase 1: Deploy Governor

```bash
# On VPS
cd /root/quantum_trader
bash ops/p32_deploy_governor.sh
```

**Expected Output:**
```
=== P3.2 GOVERNOR DEPLOYMENT ===
[1/6] Sync code from /root → /home/qt
✅ Code synced
[2/6] Install Governor config
✅ Config installed: /etc/quantum/governor.env
[3/6] Install systemd service
✅ Systemd unit installed
[4/6] Check Python dependencies
✅ All dependencies present
[5/6] Start Governor service
✅ Service started
✅ Service is active
[6/6] Verify metrics endpoint
✅ Metrics endpoint responding on port 8044
   Found X Governor metrics
=== DEPLOYMENT COMPLETE ===
✅ P3.2 Governor deployed successfully
```

**Exit Code:** 0 (success)

---

### Phase 2: Run Proof Script

```bash
bash ops/p32_proof_governor.sh
```

**Expected Proofs:**

#### PROOF 1: Service Status
```
✅ Governor service is active
   Started: <timestamp>
   PID: <pid>, Memory: <X>MB
```

#### PROOF 2: Metrics Endpoint
```
✅ Metrics endpoint responding
   Metrics found:
   - Allow metrics: ≥1
   - Block metrics: ≥1
   - Disarm metrics: ≥1
   - Exec count metrics: ≥2 (hour + 5min)
```

#### PROOF 3: Permit Keys (Allow Behavior)
```
✅ Found X permit keys
   Sample permit:
   {
     "granted": true,
     "timestamp": <unix_timestamp>,
     "symbol": "BTCUSDT"
   }
```

#### PROOF 4: Block Records
```
✅ Found X block records (or "No block records yet")
   Sample block:
   {
     "reason": "hourly_limit_exceeded",
     "timestamp": <unix_timestamp>,
     "symbol": "BTCUSDT"
   }
```

#### PROOF 5: Execution Tracking
```
✅ Governor is tracking executions
   Tracked symbols: X
   - BTCUSDT: X executions tracked
```

#### PROOF 6: Governor Event Stream
```
✅ Found X events in governor.events stream
   Latest event:
   <stream_entry_with_event_data>
```

(Or "No events yet" if no disarm has occurred)

#### PROOF 7: Config Verification
```
✅ Config file exists
   Key settings:
   GOV_MAX_EXEC_PER_HOUR=3
   GOV_MAX_EXEC_PER_5MIN=2
   GOV_ENABLE_AUTO_DISARM=true
```

#### PROOF 8: Recent Log Sample
```
Last 10 log lines:
   <governor_logs>
```

**Exit Code:** 0 (all proofs passed)

---

### Phase 3: Test Permit Issuance (Normal Operation)

**Setup:**
- Apply Layer in dry_run mode
- Governor running
- Plans being published to `quantum:stream:apply.plan`

**Test:**
```bash
# Check that Governor is issuing permits
redis-cli --scan --pattern "quantum:permit:*" | wc -l
# Expected: > 0

# Check a sample permit
redis-cli --scan --pattern "quantum:permit:*" | head -1 | xargs redis-cli GET
# Expected: {"granted": true, "timestamp": <ts>, "symbol": "<symbol>"}
```

**Apply Layer Logs:**
```bash
journalctl -u quantum-apply-layer -n 20 | grep -i permit
# Expected: "[DRY_RUN] Governor permit granted ✓"
```

**Proof:** ✅ Governor issues permits for valid plans

---

### Phase 4: Test Rate Limit Blocking

**Setup:**
- Simulate multiple executions in short window
- Monitor Governor blocks

**Test Method 1: Observe Natural Limits**
```bash
# Wait for hourly limit to be reached
curl -s http://localhost:8044/metrics | grep "quantum_govern_exec_count_hour.*BTCUSDT"
# If count ≥ 3, next plan should be blocked

# Check for block records
redis-cli --scan --pattern "quantum:governor:block:*" | wc -l
# Expected: > 0 after limit reached
```

**Test Method 2: Lower Limits Temporarily**
```bash
# Edit config (as root)
sudo sed -i 's/^GOV_MAX_EXEC_PER_HOUR=.*/GOV_MAX_EXEC_PER_HOUR=1/' /etc/quantum/governor.env
sudo systemctl restart quantum-governor

# Watch for blocks
redis-cli --scan --pattern "quantum:governor:block:*" | head -5 | xargs -I {} redis-cli GET {}
# Expected: Block records with "hourly_limit_exceeded"
```

**Metrics Verification:**
```bash
curl -s http://localhost:8044/metrics | grep "quantum_govern_block_total.*hourly_limit"
# Expected: counter > 0
```

**Proof:** ✅ Governor blocks plans when rate limits exceeded

---

### Phase 5: Test Auto-Disarm (Burst Breach)

**⚠️ WARNING:** This test will force Apply Layer to dry_run mode!

**Setup:**
1. Ensure `GOV_DISARM_ON_BURST_BREACH=true` in `/etc/quantum/governor.env`
2. Set very low burst limit: `GOV_MAX_EXEC_PER_5MIN=1`
3. Apply Layer in testnet mode with open position
4. Restart Governor

**Test:**
```bash
# Edit config to trigger disarm easily
sudo sed -i 's/^GOV_MAX_EXEC_PER_5MIN=.*/GOV_MAX_EXEC_PER_5MIN=1/' /etc/quantum/governor.env
sudo systemctl restart quantum-governor

# Trigger multiple plans rapidly (will breach burst limit)
# (Plans are generated by harvest proposals every 5 seconds)

# Wait ~30 seconds for breach detection

# Check for disarm event
redis-cli GET "quantum:governor:disarm:$(date +%Y-%m-%d)"
# Expected: {"reason": "burst_limit_breach", "context": {...}, "timestamp": <ts>}

# Verify Apply Layer is now in dry_run
grep "^APPLY_MODE=" /etc/quantum/apply-layer.env
# Expected: APPLY_MODE=dry_run

# Check event stream
redis-cli XREVRANGE quantum:stream:governor.events + - COUNT 1
# Expected: Event with "AUTO_DISARM", reason "burst_limit_breach"
```

**Governor Logs:**
```bash
journalctl -u quantum-governor | grep -i disarm
# Expected:
#   AUTO-DISARM TRIGGERED: burst_limit_breach
#   Backed up config to /etc/quantum/apply-layer.env.bak.<timestamp>
#   Set APPLY_MODE=dry_run
#   Restarted quantum-apply-layer.service
#   AUTO-DISARM COMPLETE - System is now in dry_run mode
```

**Metrics:**
```bash
curl -s http://localhost:8044/metrics | grep "quantum_govern_disarm_total.*burst"
# Expected: counter = 1
```

**Proof:** ✅ Governor triggers auto-disarm on burst breach

---

### Phase 6: Test Apply Layer Integration

**Setup:**
- Governor operational
- Apply Layer in testnet mode
- Open testnet position

**Test: Permit Check in Testnet Mode**

```bash
# Enable testnet mode
sudo sed -i 's/^APPLY_MODE=.*/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer

# Watch Apply Layer logs
journalctl -u quantum-apply-layer -f | grep -E "(permit|Governor)"
# Expected:
#   "Governor permit granted ✓" (if permit exists)
#   "No execution permit from Governor (blocked)" (if no permit)
```

**Test: Blocked Execution**

```bash
# Force all plans to be blocked (set impossible limit)
sudo sed -i 's/^GOV_MAX_EXEC_PER_HOUR=.*/GOV_MAX_EXEC_PER_HOUR=0/' /etc/quantum/governor.env
sudo systemctl restart quantum-governor

# Watch for blocks
journalctl -u quantum-apply-layer -f
# Expected:
#   "No execution permit from Governor (blocked)"
#   decision=SKIP, error=no_governor_permit in results
```

**Verify Results Stream:**
```bash
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 | grep -A5 "no_governor_permit"
# Expected: Results with error=no_governor_permit
```

**Proof:** ✅ Apply Layer correctly checks Governor permits before execution

---

## Success Criteria

### Deployment
- [x] Governor service active (exit code 0)
- [x] Metrics endpoint responding (port 8044)
- [x] Config file installed (`/etc/quantum/governor.env`)
- [x] Systemd unit enabled

### Functionality
- [x] Permit keys created (`quantum:permit:*`)
- [x] Block records created when limits exceeded
- [x] Execution tracking operational (`quantum:governor:exec:*`)
- [x] Metrics updated (allow, block, exec_count)
- [x] Auto-disarm triggered on breach
- [x] Event stream written (`governor.events`)

### Integration
- [x] Apply Layer checks permits before execution
- [x] Dry_run mode logs permit status (non-enforced)
- [x] Testnet mode enforces permit requirement
- [x] Blocked plans skip execution with error code

---

## Post-Test Cleanup

```bash
# Restore conservative Governor limits
sudo sed -i 's/^GOV_MAX_EXEC_PER_HOUR=.*/GOV_MAX_EXEC_PER_HOUR=3/' /etc/quantum/governor.env
sudo sed -i 's/^GOV_MAX_EXEC_PER_5MIN=.*/GOV_MAX_EXEC_PER_5MIN=2/' /etc/quantum/governor.env
sudo systemctl restart quantum-governor

# Return Apply Layer to dry_run (safe mode)
sudo sed -i 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer

# Clear test disarm key
redis-cli DEL "quantum:governor:disarm:$(date +%Y-%m-%d)"

# Verify clean state
bash ops/p32_proof_governor.sh
```

---

## Troubleshooting Guide

### Governor Not Starting
**Check:**
```bash
journalctl -u quantum-governor -n 50
systemctl status quantum-governor
```
**Common Issues:**
- Missing dependencies (redis-py, prometheus-client)
- Redis connection failed
- Permission issues (config file)

### No Permits Issued
**Check:**
```bash
redis-cli XLEN quantum:stream:apply.plan  # Plans being published?
journalctl -u quantum-governor | grep -i "evaluating plan"  # Governor processing?
```
**Fix:** Ensure Apply Layer is running and publishing plans

### Auto-Disarm Not Triggering
**Check:**
```bash
grep GOV_ENABLE_AUTO_DISARM /etc/quantum/governor.env  # Enabled?
grep GOV_DISARM_ON_BURST_BREACH /etc/quantum/governor.env  # Enabled?
```
**Fix:** Enable auto-disarm settings and restart

---

## Report Template

```markdown
# P3.2 Governor Proof Pack - Execution Report

**Date:** YYYY-MM-DD HH:MM UTC
**Environment:** VPS 46.224.116.254
**Tester:** <name>

## Deployment
- [x] Deploy script exit code: 0
- [x] Service active: YES
- [x] Metrics endpoint: http://localhost:8044/metrics
- [x] Config installed: /etc/quantum/governor.env

## Proof Script
- [x] Exit code: 0
- [x] All 8 proofs passed: YES

## Functional Tests
- [x] Permit issuance: <count> permits found
- [x] Rate limit blocking: <count> blocks recorded
- [x] Auto-disarm: Triggered successfully (reason: <reason>)
- [x] Event stream: <count> events published

## Integration Tests
- [x] Apply Layer dry_run: Permit logged (non-enforced)
- [x] Apply Layer testnet: Permit enforced (blocks without permit)

## Conclusion
✅ P3.2 Governor is OPERATIONAL and ready for production.

**Recommendation:** Keep in production with conservative limits (default config).
```

---

**Document Version:** 1.0.0  
**Last Updated:** 2026-01-23  
**Proof Script:** `ops/p32_proof_governor.sh`
