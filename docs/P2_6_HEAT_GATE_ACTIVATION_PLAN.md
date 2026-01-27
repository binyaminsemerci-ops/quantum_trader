# P2.6 Portfolio Heat Gate — Activation Plan

**Current Status**: SHADOW MODE (deployed, logging, not affecting trading)  
**Target**: ENFORCE MODE (actively downgrading aggressive exits)  
**Date**: 2026-01-27

---

## 🎯 Current State (Verified 2026-01-27 06:00 UTC)

### ✅ Heat Gate Service
- **Status**: `active (running)` since 2026-01-26 23:59:50 UTC
- **PID**: 1295373
- **Memory**: 18.3M / 512M max
- **Mode**: `P26_MODE=shadow`
- **Port**: 8056 (Prometheus metrics)

### 📊 Metrics Snapshot
```
p26_heat_value: 0.8825 (HOT bucket)
p26_bucket{state="HOT"}: 1.0
p26_actions_downgraded_total: 0 (no proposals processed yet)
```

### 🔄 Stream Status
- **Consumer Group**: `p26_heat_gate` registered on `quantum:stream:harvest.proposal`
- **Output Stream**: `quantum:stream:harvest.calibrated` (2 entries)
- **Apply Layer**: Reading from `quantum:harvest:proposal:{symbol}` **hash keys** (NOT stream)

### ⚠️ Critical Discovery
**Apply Layer does NOT consume streams** — it reads hash keys:
- Current: `quantum:harvest:proposal:{symbol}` (hash)
- Heat Gate writes to: `quantum:stream:harvest.calibrated` (stream)
- **MISMATCH**: Apply Layer won't see Heat Gate output without patching

---

## 🚧 Activation Blockers

### 🔴 BLOCKER #1: Apply Layer Integration
**Issue**: Apply Layer reads `quantum:harvest:proposal:{symbol}` hash keys, but Heat Gate publishes to `quantum:stream:harvest.calibrated` stream.

**Required Changes**:
1. **Option A: Heat Gate writes to hash keys** (simpler, less disruptive)
   - Modify `portfolio_heat_gate/main.py` to write calibrated proposals to:
     - `quantum:harvest:proposal:{symbol}` (overwrite original) OR
     - `quantum:harvest:calibrated:{symbol}` (new key, requires Apply Layer patch)
   
2. **Option B: Apply Layer reads from stream** (architecturally cleaner)
   - Modify `apply_layer/main.py` to consume `quantum:stream:harvest.calibrated` stream
   - Requires consumer group setup, stream processing logic
   - Higher risk, more testing required

**Recommendation**: **Option A** — Heat Gate overwrites hash keys when in enforce mode.

### 🟡 BLOCKER #2: Shadow Data Collection
**Issue**: No harvest proposals processed yet (`p26_actions_downgraded_total: 0`).

**Required**:
- 24-48 hours of shadow-mode logs showing:
  - FULL_CLOSE → PARTIAL_75/50/25 downgrades
  - Heat bucket distribution (COLD/WARM/HOT)
  - PnL impact analysis (would downgrades have helped?)

**Current**: Only 6 hours runtime, no proposals logged.

### 🟢 READY: Infrastructure
- ✅ Service deployed and stable
- ✅ Metrics publishing correctly
- ✅ Heat calculation working (0.8825 HOT)
- ✅ Consumer group registered
- ✅ Systemd service file configured

---

## 📋 Activation Steps (When Ready)

### Phase 1: Code Patching (Option A)
**Files to Modify**:
1. `microservices/portfolio_heat_gate/main.py`:
   ```python
   # Current: only writes to stream
   self.redis.xadd("quantum:stream:harvest.calibrated", data)
   
   # NEW: also write to hash key when enforce mode
   if self.config.mode == "enforce":
       calibrated_key = f"quantum:harvest:calibrated:{symbol}"
       self.redis.hset(calibrated_key, mapping=calibrated_data)
       self.redis.expire(calibrated_key, 300)  # 5min TTL
   ```

2. `microservices/apply_layer/main.py`:
   ```python
   # Current: reads quantum:harvest:proposal:{symbol}
   def get_harvest_proposal(self, symbol: str):
       key = f"quantum:harvest:proposal:{symbol}"
       
   # NEW: read calibrated if exists, fallback to proposal
   def get_harvest_proposal(self, symbol: str):
       # Check for heat gate calibrated proposal first
       calibrated_key = f"quantum:harvest:calibrated:{symbol}"
       data = self.redis.hgetall(calibrated_key)
       if data:
           return data  # Use heat-calibrated proposal
       
       # Fallback to original proposal
       key = f"quantum:harvest:proposal:{symbol}"
       return self.redis.hgetall(key)
   ```

### Phase 2: Shadow Data Analysis
**Required Evidence**:
1. Extract 24h of SHADOW-COMPARE logs:
   ```bash
   journalctl -u quantum-portfolio-heat-gate --since "24 hours ago" | grep SHADOW-COMPARE
   ```

2. Analyze:
   - **Downgrade frequency**: How often FULL_CLOSE → PARTIAL?
   - **Heat distribution**: % time in COLD/WARM/HOT buckets
   - **False positives**: Did downgrades prevent good exits?
   - **Drawdown reduction**: Would heat gate have limited losses?

3. Decision criteria:
   - ✅ Proceed if: >30% downgrades in COLD/WARM, no obvious false positives
   - ❌ Hold if: <10% downgrades, or clear false positives

### Phase 3: Deployment
**Deploy Script** (`ops/p26_heat_activate.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🚀 P2.6 Heat Gate Activation — Shadow → Enforce"

# 1. Sync patched files to VPS
rsync -av microservices/portfolio_heat_gate/main.py root@46.224.116.254:/home/qt/quantum_trader/microservices/portfolio_heat_gate/
rsync -av microservices/apply_layer/main.py root@46.224.116.254:/home/qt/quantum_trader/microservices/apply_layer/

# 2. Update config to enforce mode
ssh root@46.224.116.254 "sed -i 's/P26_MODE=shadow/P26_MODE=enforce/' /etc/quantum/portfolio-heat-gate.env"

# 3. Restart services
ssh root@46.224.116.254 "systemctl restart quantum-portfolio-heat-gate"
ssh root@46.224.116.254 "systemctl restart quantum-apply-layer"

# 4. Verify
sleep 5
ssh root@46.224.116.254 "systemctl is-active quantum-portfolio-heat-gate"
ssh root@46.224.116.254 "systemctl is-active quantum-apply-layer"
ssh root@46.224.116.254 "curl -s http://localhost:8056/metrics | grep p26_mode"

echo "✅ P2.6 Heat Gate activated in ENFORCE mode"
```

### Phase 4: Validation
**Post-Deployment Checks** (5-10 minutes after activation):
1. **Metrics**:
   ```bash
   curl http://localhost:8056/metrics | grep p26_
   # Expect: p26_actions_downgraded_total > 0
   ```

2. **Logs**:
   ```bash
   journalctl -u quantum-portfolio-heat-gate -f
   # Look for: "✅ DOWNGRADE: FULL_CLOSE → PARTIAL_75 (heat=0.45 WARM)"
   ```

3. **Apply Layer**:
   ```bash
   journalctl -u quantum-apply-layer -n 20
   # Look for: Apply Layer reading from quantum:harvest:calibrated:{symbol}
   ```

4. **Trading Impact**:
   - Monitor position changes
   - Verify partial closes execute correctly
   - Check for unexpected HOLD decisions

### Phase 5: Rollback (If Needed)
**Rollback Script** (`ops/p26_heat_rollback.sh`):
```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🔄 P2.6 Heat Gate Rollback — Enforce → Shadow"

# 1. Revert config to shadow mode
ssh root@46.224.116.254 "sed -i 's/P26_MODE=enforce/P26_MODE=shadow/' /etc/quantum/portfolio-heat-gate.env"

# 2. Restart Heat Gate (revert to logging only)
ssh root@46.224.116.254 "systemctl restart quantum-portfolio-heat-gate"

# 3. Verify Apply Layer reverts to original proposals
ssh root@46.224.116.254 "journalctl -u quantum-apply-layer -n 10"

echo "✅ Rolled back to shadow mode"
echo "   Apply Layer now reads quantum:harvest:proposal:{symbol} only"
```

**Rollback Triggers**:
- Apply Layer errors reading calibrated proposals
- Unexpected HOLD decisions blocking all exits
- Heat Gate crashes or restarts frequently
- Trading PnL significantly worse than pre-activation

---

## 📊 Success Metrics

**Week 1 (Validation)**:
- [ ] Heat Gate uptime > 99.5%
- [ ] Downgrade rate 10-40% of proposals
- [ ] No unintended position locks (false positives)
- [ ] Max drawdown reduction vs previous week

**Week 2 (Optimization)**:
- [ ] Fine-tune heat thresholds (COLD/WARM/HOT)
- [ ] Adjust downgrade rules (FULL_CLOSE → which PARTIAL?)
- [ ] Monitor correlation with P2.7 cluster stress

**Month 1 (Production)**:
- [ ] Sharpe ratio improvement vs pre-heat-gate
- [ ] Max drawdown < 15% (vs historical 20-25%)
- [ ] Win rate maintained (heat gate shouldn't prevent good exits)

---

## 🛡️ Risk Assessment

### Risk Class
**SERVICE_RESTART** (requires Apply Layer + Heat Gate restart)

### Blast Radius
- **Directly Affected**: P2.5 Harvest → P2.6 Heat Gate → P3.1 Apply Layer flow
- **Indirect**: All position management (exits, sizing, rebalancing)
- **NOT Affected**: Entry signals, AI predictions, execution layer

### Rollback Strategy
- **Fast Rollback** (<2 min): Config change to shadow mode, restart Heat Gate
- **Full Rollback** (<10 min): Git revert patches, rsync original files, restart services
- **Emergency**: Set `quantum:kill=1` to halt all trading, investigate

### Fail-Safe Mechanisms
- **Fail-Open**: If Heat Gate crashes, Apply Layer reads original proposals (no downgrade)
- **TTL Safety**: Calibrated hash keys expire after 5min → stale downgrades auto-cleared
- **Metrics Alerting**: Monitor `p26_actions_downgraded_total` rate (alert if >80% downgraded)

---

## 📝 Operations Ledger Entry (Post-Activation)

**When activated, create entry**:
```bash
python3 ops/ops_ledger_append.py \
  --operation "P2.6 Heat Gate Activation" \
  --objective "Activate heat-based exit calibration (shadow→enforce)" \
  --risk_class SERVICE_RESTART \
  --blast_radius "Harvest-to-Apply flow (P2.5→P2.6→P3.1)" \
  --allowed_paths microservices/portfolio_heat_gate/ microservices/apply_layer/ \
  --allowed_services quantum-portfolio-heat-gate quantum-apply-layer \
  --changes_summary "Patched Heat Gate to write hash keys, Apply Layer to read calibrated proposals, switched P26_MODE=enforce" \
  --proof_path "/home/qt/P2_6_HEAT_ACTIVATION_PROOF.txt" \
  --metrics_urls "http://127.0.0.1:8056/metrics" \
  --metrics_grep "p26_heat|p26_bucket|p26_downgraded" \
  --redis_cmds "HGETALL quantum:harvest:calibrated:BTCUSDT" \
  --notes "24h shadow data analyzed, downgrade rate validated, no false positives observed"
```

---

## 🚦 Current Recommendation

**STATUS**: ⏸️ **HOLD ACTIVATION**

**Reasoning**:
1. ❌ **Blocker #1 Active**: Apply Layer integration required (hash key patch)
2. ❌ **Blocker #2 Active**: No shadow data collected (0 proposals processed)
3. ✅ Infrastructure ready
4. ✅ Service stable

**Next Actions**:
1. **Immediate**: Implement hash key write in Heat Gate (Option A patch)
2. **Wait 24-48h**: Collect shadow-mode logs
3. **Analyze**: Review SHADOW-COMPARE entries for downgrade patterns
4. **Decision Gate**: Proceed if data validates heat-based calibration
5. **Deploy**: Execute activation plan with rollback script ready

**Timeline Estimate**: 2-3 days minimum before activation (1 day code, 2 days shadow data)

---

**Document Created**: 2026-01-27 06:15 UTC  
**Status**: Draft — awaiting shadow data collection
