# P3.3 Position State Brain - Implementation Complete

**Date**: 2026-01-23  
**Status**: ✅ READY FOR VPS DEPLOYMENT  
**Commit**: b2efc23b

---

## What Was Built

P3.3 Position State Brain adds a critical safety layer between P3.2 Governor and Apply Layer execution. It performs a **3-way sanity check** reconciling:

1. **Exchange truth** (Binance testnet position API)
2. **Redis truth** (harvest/risk proposals in streams)
3. **Internal ledger** (shadow ledger from executed orders)

### Key Features

✅ **Real-time Exchange Monitoring**
- Polls Binance `/fapi/v2/positionRisk` every 5s
- Caches in `quantum:position:snapshot:<symbol>`
- Tracks: positionAmt, side, entryPrice, markPrice, leverage

✅ **Shadow Ledger Tracking**
- Listens to `quantum:apply.result` stream
- Updates `quantum:position:ledger:<symbol>` on executed=True
- Tracks: last_known_amt, last_order_id, last_order_time

✅ **6-Point Sanity Checks**
1. Stale snapshot (>10s) → DENY
2. No position → DENY
3. Side mismatch → DENY (reconcile_required)
4. Qty mismatch >1% → DENY (reconcile_required)
5. Cooldown <15s → DENY (in_flight protection)
6. Safe close qty computation (clamped + rounded)

✅ **Permit Issuance**
- Issues `quantum:permit:p33:<plan_id>` (60s TTL)
- Allow permits include `safe_close_qty`
- Deny permits include `reason` code

✅ **Apply Layer Integration**
- Testnet mode requires **BOTH** permits:
  * P3.2 Governor permit (limits OK)
  * P3.3 Position permit (state OK)
- Fail-closed: missing either → BLOCKED
- Uses P3.3's `safe_close_qty` (no duplicate computation)

✅ **Metrics & Monitoring**
- Port 8045 (separate from Governor 8044)
- Prometheus metrics: snapshots, permits, evaluations
- Systemd service with auto-restart

---

## Files Created/Modified

### Core Service
- `microservices/position_state_brain/main.py` (600+ lines)
  * BinanceTestnetClient (API integration)
  * PositionStateBrain (sanity check logic)
  * Redis streams processing
  * Metrics collection

### Configuration
- `deployment/config/position-state-brain.env`
  * Poll interval: 5s
  * Stale threshold: 10s
  * Cooldown: 15s
  * Qty tolerance: 1%
  * Permit TTL: 60s

### Systemd
- `deployment/systemd/quantum-position-state-brain.service`
  * WorkingDirectory: /home/qt/quantum_trader
  * Metrics port: 8045
  * Auto-restart on failure

### Integration
- `microservices/apply_layer/main.py` (MODIFIED)
  * Added P3.3 permit check after Governor permit
  * Replaced close_qty computation with P3.3's safe_close_qty
  * Fail-closed error handling

### Operations
- `ops/p33_deploy_and_proof.sh`
  * One-command deployment to VPS
  * Idempotent (safe to re-run)
  * Runs proof pack automatically

- `ops/p33_proof.sh`
  * 8-point verification script
  * Checks service, metrics, snapshots, ledger, permits
  * Validates Apply Layer integration

### Documentation
- `docs/P3_3_POSITION_STATE_BRAIN.md`
  * Architecture diagrams
  * Redis key schemas
  * 6 sanity rules explained
  * Configuration reference
  * Metrics reference
  * Operational guide
  * Troubleshooting

---

## Redis Key Schema

### Exchange Snapshots
```
Key: quantum:position:snapshot:<symbol>
Type: HASH
TTL: 3600s
Fields:
  - position_amt
  - side (LONG|SHORT)
  - entry_price
  - mark_price
  - leverage
  - unrealized_pnl
  - ts_epoch (CRITICAL: must be < 10s old)
  - ts_iso
```

### Internal Ledger
```
Key: quantum:position:ledger:<symbol>
Type: HASH
TTL: 86400s
Fields:
  - last_known_amt
  - last_known_side
  - last_order_id
  - last_order_time_epoch
  - last_update_ts
```

### P3.3 Permits
```
Key: quantum:permit:p33:<plan_id>
Type: STRING (JSON)
TTL: 60s

ALLOW:
{
  "allow": true,
  "safe_close_qty": 0.025,
  "exchange_position_amt": 0.038,
  "ledger_position_amt": 0.038,
  "snapshot_age_sec": 3.2
}

DENY:
{
  "allow": false,
  "reason": "stale_exchange_state",
  "details": {...}
}
```

---

## Deployment Instructions

### Automated (Recommended)

**VIA WSL SSH**:
```powershell
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'bash /home/qt/quantum_trader/ops/p33_deploy_and_proof.sh'
```

This will:
1. Pull latest code from GitHub
2. Rsync to /home/qt/quantum_trader
3. Install config to /etc/quantum/
4. Copy Binance credentials from testnet.env
5. Install systemd service
6. Start P3.3 service
7. Restart Apply Layer (to pick up P3.3 integration)
8. Run proof script
9. Save proof to docs/P3_3_VPS_PROOF.txt

### Manual

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Pull code
cd /root/quantum_trader
git pull origin main

# Sync to working dir
rsync -av --delete --exclude='.git' /root/quantum_trader/ /home/qt/quantum_trader/

# Install config
cp /home/qt/quantum_trader/deployment/config/position-state-brain.env /etc/quantum/

# Copy Binance credentials
grep "^BINANCE_API_KEY=" /etc/quantum/testnet.env >> /etc/quantum/position-state-brain.env
grep "^BINANCE_API_SECRET=" /etc/quantum/testnet.env >> /etc/quantum/position-state-brain.env

# Install systemd service
cp /home/qt/quantum_trader/deployment/systemd/quantum-position-state-brain.service /etc/systemd/system/
systemctl daemon-reload

# Start P3.3
systemctl enable --now quantum-position-state-brain

# Restart Apply Layer
systemctl restart quantum-apply-layer

# Verify
systemctl status quantum-position-state-brain
systemctl status quantum-apply-layer

# Check metrics
curl -s http://localhost:8045/metrics | grep "^p33_"

# Run proof
bash /home/qt/quantum_trader/ops/p33_proof.sh > /home/qt/quantum_trader/docs/P3_3_VPS_PROOF.txt
```

---

## Verification Checklist

After deployment, verify:

- [ ] P3.3 service active: `systemctl is-active quantum-position-state-brain`
- [ ] Apply Layer active: `systemctl is-active quantum-apply-layer`
- [ ] Metrics responding: `curl http://localhost:8045/metrics`
- [ ] Snapshots updating: Check `ts_epoch` in Redis < 10s
- [ ] Ledger initialized: Check `quantum:position:ledger:*` exists
- [ ] Permits being issued: Check `quantum:permit:p33:*` keys
- [ ] Apply Layer logs show P3.3 checks: `journalctl -u quantum-apply-layer | grep "P3.3"`
- [ ] No errors in logs: `journalctl -u quantum-position-state-brain -n 50`

---

## Expected Behavior

### Normal Flow
1. P3.3 updates exchange snapshot every 5s
2. Harvest/Risk publishes EXECUTE plan to quantum:apply.plan
3. P3.2 Governor evaluates limits → issues Governor permit
4. P3.3 evaluates exchange state → issues P3.3 permit with safe_close_qty
5. Apply Layer checks BOTH permits → executes with safe_close_qty
6. P3.3 updates ledger from apply.result

### Blocked Scenarios

**Stale Snapshot**:
```
Snapshot age: 15s (> 10s threshold)
P3.3: DENY (stale_exchange_state)
Apply Layer: BLOCKED (missing_or_denied_p33_permit)
```

**No Position**:
```
Exchange: positionAmt = 0.0
P3.3: DENY (no_position)
Apply Layer: BLOCKED
```

**Cooldown**:
```
Last order: 8s ago (< 15s cooldown)
P3.3: DENY (cooldown_in_flight)
Apply Layer: BLOCKED
```

**Qty Mismatch**:
```
Exchange: 0.050 BTC
Ledger: 0.045 BTC (10% diff)
P3.3: DENY (reconcile_required_qty_mismatch)
Apply Layer: BLOCKED
```

---

## Monitoring

### Health Checks

```bash
# Service status
systemctl status quantum-position-state-brain

# Recent logs
journalctl -u quantum-position-state-brain --since "5 minutes ago"

# Metrics
curl -s http://localhost:8045/metrics | grep -E "^p33_(snapshot|permit|evaluation)"

# Redis keys
redis-cli KEYS "quantum:position:*"
redis-cli KEYS "quantum:permit:p33:*"
```

### Key Metrics

- `p33_snapshot_total{symbol, status}` - Snapshot updates (should increment every 5s)
- `p33_permit_total{symbol, decision}` - Permits issued (allow/deny counts)
- `p33_snapshot_age_sec{symbol}` - Current snapshot age (should be < 10s)
- `p33_position_amt{symbol}` - Current exchange position
- `p33_ledger_amt{symbol}` - Current ledger position

### Alerts

**Critical**:
- P3.3 service not active
- Snapshot age > 10s consistently
- All permits denied for > 5 minutes

**Warning**:
- Frequent qty mismatches (reconciliation needed)
- High cooldown rejection rate (may need tuning)

---

## Rollback Plan

If P3.3 causes issues:

```bash
# Stop P3.3 service
systemctl stop quantum-position-state-brain

# Revert Apply Layer (remove P3.3 checks)
cd /root/quantum_trader
git revert b2efc23b
git push origin main

# Redeploy Apply Layer
rsync -av /root/quantum_trader/microservices/apply_layer/ /home/qt/quantum_trader/microservices/apply_layer/
systemctl restart quantum-apply-layer

# Verify Governor-only mode
journalctl -u quantum-apply-layer --since "1 minute ago" | grep -E "(EXECUTE|BLOCKED)"
```

---

## Next Steps

1. **Deploy to VPS** (use automated script)
2. **Monitor for 24 hours** (watch metrics, logs)
3. **Validate sanity checks** (inject test scenarios)
4. **Tune thresholds** (if needed based on production behavior)
5. **Document edge cases** (any unexpected DENY reasons)

---

## Success Criteria

P3.3 is successful if:

✅ Service runs stable for 24+ hours  
✅ Snapshots stay fresh (< 10s age)  
✅ Permits issued for valid plans  
✅ Invalid plans blocked (stale, mismatch, cooldown)  
✅ Apply Layer integrates cleanly  
✅ No false positives (valid plans denied)  
✅ No false negatives (invalid plans allowed)  
✅ Metrics show healthy operation  

---

## Contact

Questions or issues? Check:
- Documentation: `docs/P3_3_POSITION_STATE_BRAIN.md`
- Logs: `journalctl -u quantum-position-state-brain`
- Metrics: `http://localhost:8045/metrics`
- Proof: `docs/P3_3_VPS_PROOF.txt` (after deployment)

---

**Implementation Complete** ✅  
**Ready for VPS Deployment** 🚀
