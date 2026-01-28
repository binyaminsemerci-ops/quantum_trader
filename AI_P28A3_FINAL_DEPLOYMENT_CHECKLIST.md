# P2.8A.3 - Final Deployment Checklist

**Date**: 2026-01-29  
**Status**: ✅ PRODUCTION-READY (Full Sign-Off)  
**Commit**: 43e936d65

---

## ✅ Sign-Off Complete

**All critical issues fixed**:
- ✅ Inflight leaks (3 paths fixed)
- ✅ Init race (dedicated lock)
- ✅ Reason propagation (timeout/redis_error in metrics)
- ✅ Prefix consistency (no defaults)
- ✅ Stop conditions (deterministic)
- ✅ Metrics cardinality (5 reason values)

**Production hygiene approved** ✅

---

## Pre-Deploy: Environment Configuration

**File**: `/etc/quantum/apply-layer.env`

**Add these ENV variables** (idempotent):
```bash
grep -q "^P28_LATE_OBS_ENABLED=" /etc/quantum/apply-layer.env || cat >> /etc/quantum/apply-layer.env <<'EOF'

# P2.8A.3 Late Observer
P28_LATE_OBS_ENABLED=true
P28_LATE_OBS_MAX_WAIT_MS=2000
P28_LATE_OBS_POLL_MS=100
P28_LATE_OBS_MAX_WORKERS=4
P28_LATE_OBS_MAX_INFLIGHT=200
EOF
```

**Verify**:
```bash
grep P28_LATE /etc/quantum/apply-layer.env
```

**Expected Output**:
```
P28_LATE_OBS_ENABLED=true
P28_LATE_OBS_MAX_WAIT_MS=2000
P28_LATE_OBS_POLL_MS=100
P28_LATE_OBS_MAX_WORKERS=4
P28_LATE_OBS_MAX_INFLIGHT=200
```

---

## Deploy Steps

### 1. Pull Latest Code
```bash
cd /home/qt/quantum_trader
git pull origin main

# Verify commit
git log --oneline -3 | grep -E "p28a3|late observer"
```

**Expected**: Commit 43e936d65 present (init race + reason propagation)

### 2. Restart Apply Service
```bash
sudo systemctl restart quantum-apply-layer.service
sleep 2
sudo systemctl is-active quantum-apply-layer.service
```

**Expected**: `active`

### 3. Check Logs
```bash
journalctl -u quantum-apply-layer -n 30 --no-pager | grep -E "P2.8A|Late Observer"
```

**Expected Log Lines**:
```
P2.8A Heat Observer: True
P2.8A.3 Late Observer: True (wait=2000ms, poll=100ms, workers=4, max_inflight=200)
P2.8A.3: Late observer pool created (max_workers=4)
```

---

## ✅ Post-Deploy Verification (3 Minutes)

**Wait 5-10 minutes** after deploy for events to flow, then run these three never-lie commands:

### Verification 1: Confirm publish_plan_post Events Appear

```bash
redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 600 | \
awk '
tolower($0)=="obs_point"{getline; op=$0}
tolower($0)=="heat_found"{getline; hf=$0; c[op,hf]++}
END{for (k in c) print k, c[k]}' | sort
```

**Expected Output** (after 5-10 min):
```
create_apply_plan 0 480
create_apply_plan 1 20
publish_plan_post 0 40
publish_plan_post 1 60
```

**Key Indicators**:
- ✅ `publish_plan_post 1 > 0` → Late observer is working!
- ✅ `publish_plan_post 0` exists → Timeout/timing issues are real (expected)
- ✅ `create_apply_plan 0` still high → Early observer timing problem (expected)

**Coverage Calculation**:
- Late observer: 60/(60+40) = 60% coverage (expected 50-90%)
- Early observer: 20/(20+480) = 4% coverage (expected low)

### Verification 2: Confirm timeout/redis_error in Metrics

```bash
curl -s http://localhost:8043/metrics | \
grep -E '^p28_heat_reason_total\{obs_point="publish_plan_post",reason="(timeout|redis_error)"\}'
```

**Expected Output**:
```
p28_heat_reason_total{obs_point="publish_plan_post",reason="timeout"} 35
p28_heat_reason_total{obs_point="publish_plan_post",reason="redis_error"} 0
```

**Key Indicators**:
- ✅ `timeout > 0` → Normal (key didn't appear within 2s)
- ✅ `redis_error = 0` → Good (no infra issues)
- ⚠️ `redis_error > 0` → Infra problem (redis connection issues)

**If timeout rate is high** (>50%), consider:
- Increase `P28_LATE_OBS_MAX_WAIT_MS` to 3000-5000ms
- Check HeatBridge consumer lag

### Verification 3: Confirm Dropped Counter Exists

```bash
curl -s http://localhost:8043/metrics | grep -E '^p28_late_obs_dropped_total(\{| )'
```

**Expected Output**:
```
p28_late_obs_dropped_total 0
```

**Key Indicators**:
- ✅ Counter exists → Saturation monitoring active (tune with data, not gut feel)
- ✅ `= 0` → No saturation (max_inflight=200 is sufficient)
- ⚠️ `> 0` → Saturation detected, increase `P28_LATE_OBS_MAX_INFLIGHT` or `MAX_WORKERS`

**If counter is missing** → Metrics not working (check prometheus_client)

---

## Success Criteria

**P2.8A.3 deployment is successful if**:

1. ✅ Apply service logs show "P2.8A.3 Late Observer: True"
2. ✅ `publish_plan_post` obs_point appears in stream
3. ✅ `publish_plan_post` with `heat_found=1` > 0 (late observer hits)
4. ✅ `timeout` reason appears in metrics (not masked as `missing`)
5. ✅ `redis_error` = 0 (or very low if infra issues)

**Coverage expectation**:
- Before P2.8A.3: 5-10% (early observer only)
- After P2.8A.3: 50-90% (late observer majority)

---

## Troubleshooting

### Issue: No publish_plan_post events

**Check logs**:
```bash
journalctl -u quantum-apply-layer -n 100 --no-pager | grep "Late observer"
```

**Possible causes**:
- ENV not loaded: Verify `grep P28_LATE /etc/quantum/apply-layer.env`
- Service not restarted: `sudo systemctl restart quantum-apply-layer.service`
- Code not updated: `git log --oneline -3`

### Issue: High redis_error rate

**Check Redis connection**:
```bash
redis-cli PING
redis-cli INFO stats | grep -E "total_connections_received|rejected_connections"
```

**Possible causes**:
- Redis max connections hit
- Network issues between Apply and Redis
- Redis server overloaded

### Issue: High timeout rate (>50%)

**Check HeatBridge lag**:
```bash
redis-cli XINFO GROUPS quantum:stream:harvest.heat.decision | grep -E "name:|lag:"
```

**Tuning**:
```bash
# Increase wait time
sed -i 's/^P28_LATE_OBS_MAX_WAIT_MS=.*/P28_LATE_OBS_MAX_WAIT_MS=3000/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service
```

---

## Rollback Plan

**If issues occur**:
```bash
# Disable late observer
sed -i 's/^P28_LATE_OBS_ENABLED=.*/P28_LATE_OBS_ENABLED=false/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service

# Verify rollback
journalctl -u quantum-apply-layer -n 10 --no-pager | grep "P2.8A.3"
# Should show: P2.8A.3 Late Observer: False
```

**Hotfix still active**:
- HeatGate reads `quantum:stream:apply.plan` (6.2% coverage preserved)
- Early observer continues working

---

## Grafana Dashboard Updates (Post-Deploy)

**Add panels for late observer**:

### Panel 1: Late Observer Saturation (drops/min)
```promql
60 * rate(p28_late_obs_dropped_total[5m])
```

**Tuning Guide**:
- If `> 0` sustained → Increase `P28_LATE_OBS_MAX_INFLIGHT` or `MAX_WORKERS`
- Or reduce `POLL_MS`/`MAX_WAIT_MS` to free up slots faster

### Panel 2: Late Observer Outcome Split (publish_plan_post)
```promql
sum by (heat_found) (rate(p28_observed_total{obs_point="publish_plan_post"}[5m]))
```

### Panel 3: Coverage Split by obs_point
```promql
sum by (obs_point, heat_found) (rate(p28_observed_total[5m]))
```

### Panel 4: Reason Distribution (Late Observer)
```promql
sum by (reason) (rate(p28_heat_reason_total{obs_point="publish_plan_post"}[5m]))
```

### Alert 1: Saturation Sustained

**Trigger**: System dropping tasks continuously (sustained rate, not single drops)
```promql
sum(rate(p28_late_obs_dropped_total[10m])) > 0.01
```

**Threshold**: `> 0.01` avoids flapping from single drops (use `> 0.1` for high volume)

**Action**: Increase `P28_LATE_OBS_MAX_INFLIGHT` or `MAX_WORKERS`

### Alert 2: Late Observer Timing Too Short

**Trigger**: Timeout dominates (>50% of observations) with sufficient volume
```promql
(
  sum(rate(p28_heat_reason_total{obs_point="publish_plan_post",reason="timeout"}[10m]))
>
  sum(rate(p28_observed_total{obs_point="publish_plan_post"}[10m])) * 0.5
)
and
(
  sum(rate(p28_observed_total{obs_point="publish_plan_post"}[10m])) > 0.05
)
```

**Guard**: `> 0.05/s` (~3 events/min) prevents noise during low traffic

**Action**: Increase `MAX_WAIT_MS` or check HeatBridge latency

### Alert 3: Redis Error (Infra vs Timing)

**Trigger**: Redis connection/capacity issues detected
```promql
sum(rate(p28_heat_reason_total{reason="redis_error"}[5m])) > 0
```

**Purpose**: Distinguish infrastructure problems (Redis sick) from timing problems (wait too short)

**Action**: Check Redis connection/capacity

---

## Alert Response Playbook (On-Call Safe)

**When Alert 1 fires (Saturation Sustained)**:
```bash
# Step 1: Increase MAX_INFLIGHT first (cheapest fix)
sed -i 's/^P28_LATE_OBS_MAX_INFLIGHT=.*/P28_LATE_OBS_MAX_INFLIGHT=400/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service

# Step 2: If still saturated, increase MAX_WORKERS (more CPU/threads)
sed -i 's/^P28_LATE_OBS_MAX_WORKERS=.*/P28_LATE_OBS_MAX_WORKERS=8/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service
```

**Tuning order**: MAX_INFLIGHT first (cheap), then MAX_WORKERS (more resources)

**When Alert 2 fires (Timing Too Short)**:
```bash
# Check drops first
curl -s http://localhost:8043/metrics | grep p28_late_obs_dropped_total

# If drops ≈ 0 → increase wait time (not workers)
sed -i 's/^P28_LATE_OBS_MAX_WAIT_MS=.*/P28_LATE_OBS_MAX_WAIT_MS=3000/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service

# If still timing out, try 4000ms
sed -i 's/^P28_LATE_OBS_MAX_WAIT_MS=.*/P28_LATE_OBS_MAX_WAIT_MS=4000/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service
```

**Critical**: Only tune MAX_WAIT_MS if drops ≈ 0 (typical: 2000 → 3000 → 4000)

**When Alert 3 fires (Redis Error)**:
```bash
# DO NOT tune P2.8A.3 config - fix Redis directly

# Check Redis health
redis-cli PING
redis-cli INFO stats | grep -E "total_connections_received|rejected_connections"
redis-cli INFO clients | grep connected_clients

# Check Redis latency
redis-cli --latency-history

# Check connection limits
redis-cli CONFIG GET maxclients
```

**Critical**: redis_error = infrastructure issue, NOT config tuning

---

## Expected Results

**Before P2.8A.3**:
- Total coverage: 5-10%
- Only `create_apply_plan` obs_point
- Mostly `missing` reason

**After P2.8A.3**:
- Total coverage: 50-90%
- Two obs_points: `create_apply_plan` + `publish_plan_post`
- Late observer reasons: `ok`, `timeout`, `redis_error` (visible in metrics)
- Grafana shows timing problems (timeout) vs infra problems (redis_error)

---

## ✅ Production-Ready Confirmation

**All critical bugs fixed** ✅  
**All races eliminated** ✅  
**Full observability** ✅  
**Never-lie verification commands** ✅  
**Rollback plan ready** ✅

**Safe to deploy** 🚀

---

**Deploy Time**: ~5 minutes  
**Verification Time**: ~2 minutes (after 5-10 min warm-up)  
**Expected Coverage Increase**: 5-10% → 50-90%
