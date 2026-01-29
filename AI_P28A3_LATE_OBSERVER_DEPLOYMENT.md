# P2.8A.3 Late Observer - Deployment Guide

**Date**: 2026-01-28  
**Issue**: Heat coverage at 6.2% due to timing problem (observer runs before HeatBridge writes by_plan keys)  
**Solution**: Post-publish delayed observation (polls for key up to 2s, non-blocking)  
**Expected Result**: Heat coverage increases from 6.2% → 50-90%+

---

## Executive Summary

**Problem**: Observer at `create_apply_plan` runs BEFORE `publish_plan`, so HeatBridge hasn't written the by_plan key yet. Result: 0% heat coverage at that observation point.

**P2.8A.3 Solution**: Add a second observer that runs AFTER `publish_plan`, polls for by_plan key for up to 2s (giving HeatBridge time to write), then emits observed event.

**Key Properties**:
- ✅ Non-blocking (background thread)
- ✅ Fail-open (never crashes Apply)
- ✅ Zero execution impact
- ✅ New observation point: `obs_point="publish_plan_post"`
- ✅ Separate dedupe key per obs_point

---

## Architecture

### Timeline Comparison

**BEFORE P2.8A.3** (6.2% coverage):
```
T+0ms:  Apply creates plan
T+1ms:  Observer checks by_plan key → NOT FOUND (heat_found=0) ❌
T+2ms:  Apply publishes to apply.plan stream
T+500ms: HeatGate reads apply.plan → writes heat.decision
T+501ms: HeatBridge reads heat.decision → writes by_plan key ✅
        (but observer already checked and missed it!)
```

**AFTER P2.8A.3** (expected 50-90% coverage):
```
T+0ms:  Apply creates plan
T+1ms:  Observer checks by_plan key → NOT FOUND (heat_found=0)
T+2ms:  Apply publishes to apply.plan stream
        → Late observer spawned in background thread
T+100ms: Late observer polls by_plan key → NOT FOUND YET
T+200ms: Late observer polls by_plan key → NOT FOUND YET
T+500ms: HeatGate/HeatBridge writes by_plan key ✅
T+600ms: Late observer polls by_plan key → FOUND! (heat_found=1) ✅
        Late observer emits event with obs_point="publish_plan_post"
```

### Code Changes

**1. heat_observer.py** (new function):
```python
def observe_late_async(
    redis_client, plan_id, symbol,
    max_wait_ms=2000, poll_ms=100,
    obs_point="publish_plan_post", ...
):
    """
    Polls for by_plan key for up to max_wait_ms.
    Runs in background thread (non-blocking).
    Emits observed event when done.
    """
```

**2. main.py** (ENV config):
```python
# P2.8A.3 Late Observer config
self.p28_late_enabled = heat_observer.is_enabled(os.getenv("P28_LATE_OBS_ENABLED", "true"))
self.p28_late_max_wait_ms = int(os.getenv("P28_LATE_OBS_MAX_WAIT_MS", "2000"))
self.p28_late_poll_ms = int(os.getenv("P28_LATE_OBS_POLL_MS", "100"))
```

**3. main.py publish_plan()** (hook):
```python
if self.p28_late_enabled and heat_observer:
    heat_observer.observe_late_async(
        redis_client=self.redis,
        plan_id=plan.plan_id,
        symbol=plan.symbol,
        obs_point="publish_plan_post",
        ...
    )
```

---

## Deployment Steps

### Pre-Deploy: Run Golden Verification (REQUIRED)

**Before deploying P2.8A.3**, verify pipeline is working:

See **AI_P28A_GOLDEN_VERIFICATION.md** for complete 15-second verification guide.

Quick verification:
```bash
# 1. Verify pipeline flowing (15 seconds)
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'NR==3{print $1; exit}')
echo "Testing plan: $PLAN_ID"

# A) Check heat.decision produced
redis-cli --raw XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep -m1 "$PLAN_ID" && echo "✅ heat.decision OK" || echo "❌ heat.decision MISSING"

# B) Check by_plan key exists
redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID" | grep -q 1 && echo "✅ by_plan key OK" || echo "❌ by_plan key MISSING"

# Expected: A ✅ + B 1 = pipeline working
# If MISSING → Fix pipeline first (see AI_P28A_GOLDEN_VERIFICATION.md)
```

### Step 1: Update Code on VPS

```bash
cd /home/qt/quantum_trader && git pull origin main

# Verify P2.8A.3 commit present
git log --oneline -3 | grep -E "p28a3|late observer"
```

### Step 2: Add ENV Configuration (Production-Safe)

**Copy/paste this entire block** (idempotent, safe to re-run):

```bash
grep -q "^P28_LATE_OBS_ENABLED=" /etc/quantum/apply-layer.env || cat >> /etc/quantum/apply-layer.env <<'EOF'

# P2.8A.3 Late Observer
P28_LATE_OBS_ENABLED=true
P28_LATE_OBS_MAX_WAIT_MS=2000
P28_LATE_OBS_POLL_MS=100
P28_LATE_OBS_MAX_WORKERS=4
P28_LATE_OBS_MAX_INFLIGHT=200
EOF

# Verify
grep P28_LATE /etc/quantum/apply-layer.env
```

**Expected output**:
```
P28_LATE_OBS_ENABLED=true
P28_LATE_OBS_MAX_WAIT_MS=2000
P28_LATE_OBS_POLL_MS=100
P28_LATE_OBS_MAX_WORKERS=4
P28_LATE_OBS_MAX_INFLIGHT=200
```

### Step 3: Restart Apply Service

```bash
sudo systemctl restart quantum-apply-layer.service
sleep 2
sudo systemctl is-active quantum-apply-layer.service

# Check logs for P2.8A.3 confirmation
journalctl -u quantum-apply-layer -n 30 --no-pager | grep -E "P2.8A|Late Observer"
```

**Expected log lines**:
```
P2.8A Heat Observer: True
P2.8A.3 Late Observer: True (wait=2000ms, poll=100ms, workers=4, max_inflight=200)
P2.8A.3: Late observer pool created (max_workers=4)
```

### Step 4: Verify Late Observer is Producing Events

**Wait 3-5 minutes** for events to flow, then run:

#### Verification Command 1: Confirm publish_plan_post obs_point exists

```bash
redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 200 | awk '
  tolower($0)=="obs_point"{getline; op=$0}
  tolower($0)=="heat_found"{getline; hf=$0; c[op,hf]++}
  END{
    for(k in c) print k, c[k]
  }' | sort
```

**Expected output** (you MUST see `publish_plan_post` lines):
```
create_apply_plan 0 150
create_apply_plan 1 10
publish_plan_post 0 20
publish_plan_post 1 140
```

**Analysis**: `publish_plan_post 1 140` means late observer found heat 140 times → ~87% coverage!

#### Verification Command 2: Confirm HeatGate still reads apply.plan

```bash
grep -E '^HEAT_STREAM_IN=' /etc/quantum/heat-gate.env
```

**Expected output**: `HEAT_STREAM_IN=quantum:stream:apply.plan`

**Critical**: If this shows `harvest.proposal`, hotfix is missing! Don't proceed.

### Step 5: Monitor Metrics (Production Truth)

```bash
curl -s http://localhost:8043/metrics | grep -E '^p28_observed_total|^p28_heat_reason_total' | grep publish_plan_post
```

**Expected** (numbers should be INCREASING over time):
```
p28_observed_total{obs_point="publish_plan_post",heat_found="0"} 20
p28_observed_total{obs_point="publish_plan_post",heat_found="1"} 140
p28_heat_reason_total{obs_point="publish_plan_post",reason="ok"} 140
p28_heat_reason_total{obs_point="publish_plan_post",reason="timeout"} 15
p28_heat_reason_total{obs_point="publish_plan_post",reason="missing"} 5
```

**Reason enums** (low cardinality, production-safe):
- `ok`: Key found, data read successfully
- `missing`: Key not found after max_wait_ms
- `timeout`: Key not found (alias for missing in late observer)
- `redis_error`: Redis connection issues (after 2 retries)
- `parse_error`: Data corruption (rare)

---

## Verification Checklist

After deployment, verify:

- [ ] **Golden verification passed** (apply.plan → heat.decision → by_plan key)
- [ ] Apply service restarted successfully (`is-active` shows active)
- [ ] Logs show: `P2.8A.3 Late Observer: True (wait=2000ms, poll=100ms, workers=4, max_inflight=200)`
- [ ] New obs_point exists: `publish_plan_post` (see Verification Command 1)
- [ ] Late observer shows heat_found=1 events (expect 50-90% of publish_plan_post events)
- [ ] HeatGate still reads apply.plan (Verification Command 2)
- [ ] No errors in Apply logs related to late observer
- [ ] Metrics show two observation points (create_apply_plan + publish_plan_post)

---

## Rollback Plan

If issues occur:

```bash
# Disable late observer
sed -i 's/^P28_LATE_OBS_ENABLED=.*/P28_LATE_OBS_ENABLED=false/' /etc/quantum/apply-layer.env

# Restart
systemctl restart quantum-apply-layer

# Or restore backup
cp /etc/quantum/apply-layer.env.bak.p28a3_* /etc/quantum/apply-layer.env
systemctl restart quantum-apply-layer
```

---

## Expected Results

### Coverage Improvement

**Before** (create_apply_plan only):
- heat_found=0: ~94% (observer runs too early)
- heat_found=1: ~6% (lucky timing when HeatBridge is fast)

**After** (publish_plan_post):
- heat_found=0: ~10-50% (missing_inputs or race)
- heat_found=1: ~50-90% (HeatBridge had time to write)

### Why Not 100%?

Late observer still has limits:
- `max_wait_ms=2000` (2s timeout)
- If HeatGate/HeatBridge is slow (>2s), still misses
- If Redis is under load, HeatBridge might delay writes
- If heat decisions are `unknown/missing_inputs`, by_plan key still written but may expire quickly

**But**: 50-90% is MUCH better than 6%!

### Performance Impact

**Zero blocking**: Late observer runs in daemon thread
**Minimal overhead**: 
- 1 EXISTS check per 100ms for up to 2s
- Max 20 Redis EXISTS calls per plan
- Total: ~1-2ms Redis time per plan (background)

Apply processing continues immediately after publish - no blocking!

---

## Grafana Updates

### New Panels

**Panel 3: Heat Coverage by Observation Point**
```promql
sum by (obs_point, heat_found) (
  increase(p28_observed_total[15m])
)
```

**Panel 4: Late Observer Success Rate**
```promql
sum(increase(p28_observed_total{obs_point="publish_plan_post",heat_found="1"}[15m]))
/
sum(increase(p28_observed_total{obs_point="publish_plan_post"}[15m]))
```

Expected: 0.5 to 0.9 (50-90%)

---

## Troubleshooting

### Late observer not starting

**Symptom**: No events with `obs_point="publish_plan_post"`

**Check**:
```bash
# ENV enabled?
grep P28_LATE_OBS_ENABLED /etc/quantum/apply-layer.env

# Apply logs?
journalctl -u quantum-apply-layer -n 100 | grep -i "late"
```

**Fix**: Ensure `P28_LATE_OBS_ENABLED=true` and restart

### Late observer always heat_found=0

**Symptom**: All late observer events have `heat_found=0`

**Check**:
```bash
# Is HeatBridge writing by_plan keys?
redis-cli KEYS "quantum:harvest:heat:by_plan:*" | wc -l

# HeatBridge service status?
systemctl status quantum-heat-bridge

# HeatBridge logs?
journalctl -u quantum-heat-bridge -n 50
```

**Fix**: Check HeatBridge is running and writing keys

### Late observer thread errors

**Symptom**: Apply logs show late observer errors

**Check**:
```bash
journalctl -u quantum-apply-layer | grep "Late observer"
```

**Fix**: Review errors, may need to adjust max_wait_ms or handle edge cases

---

## Next Steps After Deployment

1. **Monitor for 24h**: Track coverage improvement
2. **Tune timeouts**: If coverage still low, increase `P28_LATE_OBS_MAX_WAIT_MS` to 3000 or 5000
3. **Alert on regression**: If late observer coverage drops below 40%, investigate
4. **Update docs**: Add late observer to P2.8A architecture diagram

---

## Success Criteria

✅ **Late observer active**: Events with `obs_point="publish_plan_post"` exist  
✅ **Coverage improved**: heat_found=1 rate increases from 6% → 50-90%  
✅ **No blocking**: Apply logs show no timing impact  
✅ **No errors**: No late observer errors in Apply logs  
✅ **Metrics clean**: Two obs_points visible in Grafana  

---

**Deployed by**: AI Agent  
**Verified on**: VPS quantumtrader-prod-1 (46.224.116.254)  
**Backup**: /etc/quantum/apply-layer.env.bak.p28a3_YYYYMMDD_HHMMSS
