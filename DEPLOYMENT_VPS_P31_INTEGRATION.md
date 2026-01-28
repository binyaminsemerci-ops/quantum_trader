# VPS Deployment - P3.1 Integration Steps 1 & 2

## Quick Start (5 commands)

Run these on your local machine to deploy and test on VPS:

```bash
# 1. Pull latest code on VPS
ssh root@VPS 'cd /home/qt/quantum_trader && git pull origin main'

# 2. Deploy Step 1 (Allocation Target Shadow)
ssh root@VPS 'cp /home/qt/quantum_trader/deploy/allocation-target.env /etc/quantum/ && \
  cp /home/qt/quantum_trader/deploy/systemd/quantum-allocation-target.service /etc/systemd/system/ && \
  systemctl daemon-reload && systemctl enable quantum-allocation-target && systemctl start quantum-allocation-target'

# 3. Deploy Step 2 (Governor with P3.1)
ssh root@VPS 'cp /home/qt/quantum_trader/deployment/config/governor.env /etc/quantum/governor.env && \
  systemctl restart quantum-governor'

# 4. Test Step 1 (on VPS)
ssh root@VPS 'cd /home/qt/quantum_trader && bash scripts/proof_p31_step1_allocation_shadow.sh'

# 5. Test Step 2 (on VPS)
ssh root@VPS 'cd /home/qt/quantum_trader && bash scripts/proof_p31_step2_governor_downsize.sh'
```

## Verification

After deployment, verify both services are running:

```bash
# Check services
ssh root@VPS 'systemctl status quantum-allocation-target quantum-governor'

# Check metrics
ssh root@VPS 'curl -s http://localhost:8065/metrics | grep p29_shadow | head -5'
ssh root@VPS 'curl -s http://localhost:8044/metrics | grep p32_eff | head -5'

# Check logs
ssh root@VPS 'journalctl -u quantum-allocation-target -n 20 --no-pager'
ssh root@VPS 'journalctl -u quantum-governor -n 20 --no-pager'
```

## Expected Results

### Step 1 Proof
```
✅ PASS: High efficiency (0.9, conf=0.9) → multiplier > 1.0, proposed > base
✅ PASS: Low efficiency (0.3, conf=0.9) → multiplier < 1.0, proposed < base
✅ PASS: Missing efficiency → multiplier = 1.0, proposed = base, reason=missing_eff
✅ PASS: Low confidence (conf=0.5 < MIN_CONF=0.65) → multiplier = 1.0, reason=low_conf
✅ PASS: Stale efficiency (>600s) → multiplier = 1.0, reason=stale_eff
✅ PASS: Service is active
✅ PASS: No errors in recent logs
✅ SUMMARY: PASS (all tests passed)
```

### Step 2 Proof
```
✅ PASS: Low efficiency (0.2, conf=0.9) → eff_action=DOWNSIZE, eff_factor ∈ [0.25, 1.0)
✅ PASS: High efficiency (0.8, conf=0.9) → eff_action=NONE, eff_factor=1.0
✅ PASS: Missing efficiency → eff_action=NONE, eff_reason=missing_eff
✅ PASS: Low confidence (conf=0.3 < MIN_CONF=0.65) → eff_action=NONE, reason=low_conf
✅ PASS: P3.1 metrics registered (p32_eff_apply_total)
✅ PASS: Governor service is active
✅ SUMMARY: PASS (all tests passed)
```

## Files Changed

### Step 1: Allocation Target Shadow
- `microservices/allocation_target/main.py` (447 lines) - New service
- `deploy/allocation-target.env` - Configuration
- `deploy/systemd/quantum-allocation-target.service` - Systemd unit
- `scripts/proof_p31_step1_allocation_shadow.sh` - Proof script (251 lines)
- `scripts/proof_p31_step1_inject_efficiency.py` - Test helper

### Step 2: Governor Downsize
- `microservices/governor/main.py` - Modified with P3.1 integration
  - Added `_read_p31_efficiency()` method
  - Enhanced `_issue_permit()` with efficiency fields
  - Added P3.1 metrics
- `deployment/config/governor.env` - Updated with P3.1 parameters
- `scripts/proof_p31_step2_governor_downsize.sh` - Proof script (206 lines)
- `scripts/proof_p31_step2_inject_plan.py` - Test helper

### Documentation
- `docs/P3.1_STEP1_STEP2_INTEGRATION.md` - Complete integration guide

## Rollback (if needed)

```bash
# Revert to previous governor
ssh root@VPS 'git -C /home/qt/quantum_trader log --oneline -10'
ssh root@VPS 'cd /home/qt/quantum_trader && git revert c39bcd431 && git push origin main'
ssh root@VPS 'systemctl restart quantum-governor'

# Stop Allocation service (if problems)
ssh root@VPS 'systemctl stop quantum-allocation-target'
```

---

**Deployment Date**: 2026-01-28  
**Status**: Ready for VPS ✅  
**Risk Level**: MEDIUM (shadow mode, no hard blocks)
