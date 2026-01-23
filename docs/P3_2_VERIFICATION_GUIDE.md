# P3.2 Governor - Verification & Operations Guide

**Status:** Production Ready (Hardened)  
**Date:** 2026-01-24  
**VPS:** 46.224.116.254

---

## 0️⃣ Quick Verification (3 Commands)

Run as root on VPS to verify deployment:

```bash
# 1) Apply Layer mode
grep -E '^APPLY_MODE=' /etc/quantum/apply-layer.env

# 2) Governor running + metrics up
systemctl is-active quantum-governor && curl -s http://127.0.0.1:8044/metrics | head -20

# 3) Proof files exist
ls -la /home/qt/quantum_trader/docs/ | grep -E 'P3_DRY_RUN_RELOCK_PROOF|P3_2_VPS_PROOF'
```

**Expected Output:**
- ✅ `APPLY_MODE=dry_run`
- ✅ Governor: `active`
- ✅ Metrics: Prometheus output with `quantum_govern_*` metrics
- ✅ Proof files: Both `P3_2_VPS_PROOF.txt` and `P3_DRY_RUN_RELOCK_PROOF.txt`

---

## 1️⃣ Switch to TESTNET Mode (Controlled)

**⚠️ WARNING:** This enables REAL order execution on Binance testnet!

### Option A: Interactive Script (Recommended)

```bash
cd /root/quantum_trader
git pull
bash ops/p3_switch_to_testnet.sh
```

- Requires typing `YES` to confirm
- Backs up config automatically
- Restarts Apply Layer
- Verifies service status

### Option B: Manual Commands

```bash
# Backup config
cp /etc/quantum/apply-layer.env /etc/quantum/apply-layer.env.bak.$(date +%s)

# Switch mode
sed -i 's/^APPLY_MODE=.*/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env

# Restart
systemctl restart quantum-apply-layer

# Verify
systemctl is-active quantum-apply-layer
grep ^APPLY_MODE= /etc/quantum/apply-layer.env
```

### Verify Testnet Behavior

```bash
bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh
```

**Expected Results:**
- ✅ If Governor blocks (no permit) → Apply Layer returns `decision=BLOCKED`, `error=missing_permit_or_redis`
- ✅ If Governor allows (permit issued) → Apply Layer executes with `executed=True`, `reduceOnly=True`
- ✅ Only BTCUSDT processes (allowlist)
- ✅ Real `orderId` in result

---

## 2️⃣ Release Gate Checklist (10-Point Verification)

Run before/after any mode change or deployment:

```bash
cd /root/quantum_trader
git pull
bash ops/p32_release_gate_checklist.sh
```

### What It Checks

1. **Service Status** - Both Governor and Apply Layer active
2. **Apply Layer Mode** - dry_run (safe) or testnet (live)
3. **Governor Metrics** - Endpoint responding, counters working
4. **Governor Events** - Latest auto-disarm events
5. **Recent Apply Results** - Last 3 executions
6. **Permit Enforcement** - Fail-closed blocking detected
7. **Execution Verification** - Real orders executed (testnet only)
8. **ReduceOnly Flag** - Present in logs (testnet only)
9. **Burst Limit Detection** - Limits enforcing correctly
10. **Auto-Disarm Status** - System armed or disarmed

### Manual Checklist Commands

```bash
# 1) Services active
systemctl is-active quantum-governor quantum-apply-layer

# 2) Mode check
grep ^APPLY_MODE= /etc/quantum/apply-layer.env

# 3) Metrics
curl -s http://127.0.0.1:8044/metrics | grep quantum_govern | head -20

# 4) Governor events
redis-cli XREVRANGE quantum:stream:governor.events + - COUNT 1

# 5) Recent results
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3

# 6) Permit blocking (fail-closed)
redis-cli XREVRANGE quantum:stream:apply.result + - | grep -c "missing_permit_or_redis"

# 7) Executed orders (testnet only)
redis-cli XREVRANGE quantum:stream:apply.result + - | grep '"executed": true' | head -5

# 8) ReduceOnly logs (testnet only)
journalctl -u quantum-apply-layer --since "5 minutes ago" | grep reduceOnly | tail -10

# 9) Burst limit blocks
curl -s http://127.0.0.1:8044/metrics | grep 'quantum_govern_block_total.*burst_limit'

# 10) Disarm status
redis-cli GET "quantum:governor:disarm:$(date -u +%Y-%m-%d)"
```

---

## 3️⃣ Expected Behaviors

### Dry_Run Mode
- ✅ Governor evaluates plans, issues permits
- ✅ Apply Layer logs permit status but doesn't execute
- ✅ No real orders sent to Binance
- ✅ `executed=False`, `would_execute=True` in results

### Testnet Mode - Permit Missing
- ✅ Governor blocks (daily limit, burst, kill score, etc.)
- ✅ No permit issued to Redis
- ✅ Apply Layer checks permit → NOT FOUND
- ✅ Apply Layer returns `decision=BLOCKED`, `error=missing_permit_or_redis`
- ✅ No order sent to Binance
- ✅ Logged: "No execution permit from Governor (blocked)"

### Testnet Mode - Permit Granted
- ✅ Governor evaluates plan → PASS all limits
- ✅ Governor fetches real `positionAmt` from Binance API
- ✅ Governor fetches real `markPrice` from Binance API
- ✅ Governor computes `close_qty` based on action type
- ✅ Governor computes `notional = close_qty × markPrice`
- ✅ Governor issues permit with 60s TTL: `quantum:permit:<plan_id>`
- ✅ Permit contains `computed_qty` and `computed_notional`
- ✅ Apply Layer validates permit → FOUND + granted=True
- ✅ Apply Layer DELETEs permit (atomic consumption)
- ✅ Apply Layer sends reduceOnly order to Binance
- ✅ Order executed, `orderId` returned
- ✅ Result: `executed=True`, `reduce_only=True`

### Burst Limit Triggered
- ✅ 3rd execution in 5min window → Governor BLOCKS
- ✅ `quantum_govern_block_total{reason="burst_limit_exceeded"}` increments
- ✅ If `GOV_DISARM_ON_BURST_BREACH=true` → Auto-disarm triggered
- ✅ Governor edits `/etc/quantum/apply-layer.env` → `APPLY_MODE=dry_run`
- ✅ Governor restarts `quantum-apply-layer.service`
- ✅ Event written to `quantum:stream:governor.events`
- ✅ `quantum_govern_disarm_total{reason="burst_limit_breach"}` increments
- ✅ System returns to safe mode

---

## 4️⃣ Current VPS Status (2026-01-24)

```
✅ APPLY_MODE=dry_run (safe)
✅ Governor: active (hardened with real Binance data)
✅ Apply Layer: active (fail-closed enforcement)
✅ Metrics: http://127.0.0.1:8044/metrics (responding)
✅ Binance Credentials: Configured in /etc/quantum/governor.env
✅ Auto-Disarm: Triggered once (burst limit breach)
```

### Metrics Snapshot

```
quantum_govern_allow_total{symbol="BTCUSDT"} 2.0
quantum_govern_block_total{reason="burst_limit_exceeded",symbol="BTCUSDT"} 2.0
quantum_govern_disarm_total{reason="burst_limit_breach"} 1.0
quantum_govern_exec_count_hour{symbol="BTCUSDT"} 2.0
quantum_govern_exec_count_5min{symbol="BTCUSDT"} 2.0
```

**Analysis:**
- 2 plans allowed
- 2 plans blocked (burst limit)
- 1 auto-disarm event (system protected itself)
- Governor is operational and enforcing limits correctly

---

## 5️⃣ Troubleshooting

### Governor Not Starting

```bash
# Check logs
journalctl -u quantum-governor -n 50 --no-pager

# Common issues:
# - Redis not running: systemctl start redis
# - Port 8044 in use: netstat -tlnp | grep 8044
# - Config missing: check /etc/quantum/governor.env
```

### Apply Layer Blocks Everything (Testnet)

```bash
# Check permit keys
redis-cli KEYS "quantum:permit:*"

# If no permits:
# - Governor may be blocking due to limits
# - Check Governor logs: journalctl -u quantum-governor -f
# - Check metrics for block reasons

# If permits exist but Apply Layer still blocks:
# - Check Apply Layer logs: journalctl -u quantum-apply-layer -f
# - Verify Redis connection: redis-cli PING
```

### Disarm Not Resetting

```bash
# Check disarm key
redis-cli GET "quantum:governor:disarm:$(date -u +%Y-%m-%d)"

# If disarmed and you want to reset:
# WARNING: Only do this if you understand why it disarmed
redis-cli DEL "quantum:governor:disarm:$(date -u +%Y-%m-%d)"

# Then restart Governor
systemctl restart quantum-governor
```

### Metrics Not Showing Data

```bash
# Test endpoint
curl http://127.0.0.1:8044/metrics

# If nothing:
# - Governor may not have processed any plans yet
# - Check if plans are being published: redis-cli XLEN quantum:stream:apply.plan
# - Check Governor is consuming: journalctl -u quantum-governor | grep "Evaluating plan"
```

---

## 6️⃣ Safe Rollback to Dry_Run

If you need to immediately stop testnet execution:

```bash
# Emergency stop
sed -i 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/' /etc/quantum/apply-layer.env
systemctl restart quantum-apply-layer

# Verify
grep ^APPLY_MODE= /etc/quantum/apply-layer.env
systemctl is-active quantum-apply-layer
```

---

## 7️⃣ Files Reference

### Configuration
- `/etc/quantum/apply-layer.env` - Apply Layer config (APPLY_MODE)
- `/etc/quantum/governor.env` - Governor config (limits, Binance creds)
- `/etc/quantum/testnet.env` - Binance testnet credentials

### Services
- `/etc/systemd/system/quantum-governor.service` - Governor systemd unit
- `/etc/systemd/system/quantum-apply-layer.service` - Apply Layer systemd unit

### Runtime
- `/home/qt/quantum_trader/` - Runtime directory (code synced here)
- `/root/quantum_trader/` - Git working directory

### Proof Files
- `/home/qt/quantum_trader/docs/P3_2_VPS_PROOF.txt` - Governor deployment proof
- `/home/qt/quantum_trader/docs/P3_DRY_RUN_RELOCK_PROOF.txt` - Dry_run mode proof

### Scripts
- `ops/p32_vps_deploy_and_proof.sh` - Full deployment (run from /root/quantum_trader)
- `ops/p32_release_gate_checklist.sh` - 10-point verification
- `ops/p3_switch_to_testnet.sh` - Safe testnet mode activation
- `ops/p3_proof_testnet.sh` - Testnet mode verification

---

## 8️⃣ Quick Commands Cheatsheet

```bash
# Deploy/update from GitHub
cd /root/quantum_trader && git pull && bash ops/p32_vps_deploy_and_proof.sh

# Run release gate checklist
cd /root/quantum_trader && bash ops/p32_release_gate_checklist.sh

# Switch to testnet (interactive)
cd /root/quantum_trader && bash ops/p3_switch_to_testnet.sh

# Emergency dry_run
sed -i 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/' /etc/quantum/apply-layer.env && systemctl restart quantum-apply-layer

# View Governor logs (live)
journalctl -u quantum-governor -f

# View Apply Layer logs (live)
journalctl -u quantum-apply-layer -f

# Check Governor metrics
curl http://127.0.0.1:8044/metrics | grep quantum_govern

# Check latest permit
redis-cli --scan --pattern "quantum:permit:*" | head -1 | xargs redis-cli GET | jq .

# Check latest result
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 1

# Check disarm status
redis-cli GET "quantum:governor:disarm:$(date -u +%Y-%m-%d)"
```

---

**Documentation Version:** 1.0  
**Last Updated:** 2026-01-24  
**Status:** Production Ready ✅
