# P2.8A.3 Pre-Deploy Verification - Never-Lie Commands

**Date**: 2026-01-28  
**Purpose**: Final production checks before deploying P2.8A.3 late observer  
**Time Required**: 2 minutes  

---

## Critical: Run These on VPS Before Deploying

### A) Verify Wiring Still Correct (Hotfix Intact)

```bash
grep -E '^HEAT_STREAM_IN=' /etc/quantum/heat-gate.env
```

**Expected Output**:
```
HEAT_STREAM_IN=quantum:stream:apply.plan
```

**If shows `harvest.proposal`**: ⚠️ STOP - Hotfix regressed! Fix before deploying P2.8A.3.

---

### B) Verify Prefix Consistency (No Code Divergence)

```bash
# Check Apply config
grep P28_HEAT_LOOKUP_PREFIX /etc/quantum/apply-layer.env

# Sample existing keys
redis-cli --raw SCAN 0 MATCH "quantum:harvest:heat:by_plan:*" COUNT 1000 \
  | sed '1d' | head -3
```

**Expected**:
```
P28_HEAT_LOOKUP_PREFIX=quantum:harvest:heat:by_plan:

# Keys should all start with quantum:harvest:heat:by_plan:
quantum:harvest:heat:by_plan:123abc...
quantum:harvest:heat:by_plan:456def...
quantum:harvest:heat:by_plan:789ghi...
```

**If no P28_HEAT_LOOKUP_PREFIX in config**: Uses default `quantum:harvest:heat:by_plan:` (OK)  
**If keys show `quantum:heat:bridge:by_plan:`**: ⚠️ STOP - Wrong prefix in production!

---

### C) Baseline Coverage Before P2.8A.3

```bash
redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 400 | \
awk '
tolower($0)=="obs_point"{getline; op=$0}
tolower($0)=="heat_found"{getline; hf=$0; c[op,hf]++}
END{
  for (k in c) print k, c[k]
}' | sort
```

**Expected Output** (BEFORE P2.8A.3):
```
create_apply_plan 0 380
create_apply_plan 1 20
```

**Key Points**:
- Only `create_apply_plan` obs_point exists (early observer)
- No `publish_plan_post` yet (late observer not deployed)
- Coverage: ~5% (20/400)

**After P2.8A.3 deploys**, you should see:
```
create_apply_plan 0 150
create_apply_plan 1 10
publish_plan_post 0 20
publish_plan_post 1 140
```
(Late observer: 140/160 = 87.5% coverage)

---

### D) Port Sanity Check

```bash
curl -s -o /dev/null -w "apply_metrics 8043 => %{http_code}\n" http://localhost:8043/metrics
curl -s -o /dev/null -w "heatbridge_metrics 8070 => %{http_code}\n" http://localhost:8070/metrics
```

**Expected**:
```
apply_metrics 8043 => 200
heatbridge_metrics 8070 => 200
```

**If 000 or timeout**: Service is down - fix before deploying P2.8A.3.

---

### E) Redis Connection Health

```bash
redis-cli PING
redis-cli INFO server | grep uptime_in_seconds
```

**Expected**:
```
PONG
uptime_in_seconds:1234567
```

---

### F) Apply Service Health

```bash
systemctl is-active quantum-apply-layer
journalctl -u quantum-apply-layer -n 5 --no-pager | tail -3
```

**Expected**:
```
active

(Recent log lines showing normal operation)
```

**If inactive or error logs**: Fix Apply service first.

---

## Pre-Deploy Checklist

Run all commands above and verify:

- [ ] **Wiring correct**: HeatGate reads `quantum:stream:apply.plan`
- [ ] **Prefix consistent**: All keys use `quantum:harvest:heat:by_plan:`
- [ ] **Baseline coverage**: Only `create_apply_plan` obs_point exists (5-10%)
- [ ] **Ports responding**: All services return HTTP 200
- [ ] **Redis healthy**: PONG response
- [ ] **Apply service active**: systemctl shows active

**If all checks pass**: ✅ Safe to deploy P2.8A.3  
**If any check fails**: ⚠️ Fix issue before deploying

---

## Post-Deploy Verification (5 Minutes After)

### Verify publish_plan_post Appears

```bash
# Wait 5 minutes for events to flow
sleep 300

redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 400 | \
awk '
tolower($0)=="obs_point"{getline; op=$0}
tolower($0)=="heat_found"{getline; hf=$0; c[op,hf]++}
END{
  for (k in c) print k, c[k]
}' | sort
```

**Expected**: You MUST see `publish_plan_post` lines appear:
```
create_apply_plan 0 150
create_apply_plan 1 10
publish_plan_post 0 20       # NEW!
publish_plan_post 1 140      # NEW!
```

**If no `publish_plan_post`**: P2.8A.3 not running - check Apply logs:
```bash
journalctl -u quantum-apply-layer -n 50 --no-pager | grep -E "P2.8A|Late Observer"
```

Expected log lines:
```
P2.8A.3 Late Observer: True (wait=2000ms, poll=100ms, workers=4, max_inflight=200)
P2.8A.3: Late observer pool created (max_workers=4)
```

---

## Rollback Command (If Issues)

```bash
# Disable late observer
sed -i 's/^P28_LATE_OBS_ENABLED=.*/P28_LATE_OBS_ENABLED=false/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer.service
```

---

## Why These Commands Never Lie

1. **`grep -E '^HEAT_STREAM_IN='`**: Reads actual config file (not memory/cache)
2. **awk obs_point grouping**: Parses actual stream data (not metrics that could lag)
3. **curl with `%{http_code}`**: Tests actual HTTP response (not just port open)
4. **SCAN not KEYS**: Production-safe (doesn't block Redis)
5. **Proper if/then/else**: No `cmd ; echo ✅ ; echo ❌` (which prints both!)

**These commands test actual production state, not assumptions.** ✅

---

**Ready to deploy**: See `AI_P28A3_LATE_OBSERVER_DEPLOYMENT.md` for deployment steps.
