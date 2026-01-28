# P2.8A Golden Verification - Production Truth Tests

**Purpose**: Prove entire pipeline works in 15 seconds, without false negatives.

---

## ⚠️ Common Pitfalls (Anti-False-Negative Checklist)

**Read these first to avoid false negatives**:

1. **Prefix Always**: `quantum:harvest:heat:by_plan:` (NOT `quantum:heat:bridge:by_plan:`)
2. **Never KEYS**: Use SCAN (KEYS blocks Redis in production)
3. **redis-cli EXISTS**: Returns stdout `0` or `1`, NOT exit code
4. **Grep fields vs values**: Use `--raw` or `awk` to parse stream correctly
5. **COUNT window**: Must be large enough (2000-5000 for heat.decision)

**Golden rules**:
- ✅ Use `if [ "$EXISTS" = "1" ]` (test stdout value)
- ✅ Use `grep -q` in if-condition (proper branching)
- ❌ Never `cmd ; echo ✅ ; echo ❌` (prints both!)
- ❌ Never `redis-cli KEYS` (use SCAN loop)

---

## Hard-Coded Truths (Never Change These)

- ✅ Prefix: `quantum:harvest:heat:by_plan:<plan_id>`
- ✅ Search window: `COUNT 5000` (not 100)
- ✅ Apply metrics: `http://localhost:8043/metrics`
- ✅ HeatBridge metrics: `http://localhost:8070/metrics`
- ✅ HeatGate metrics: `http://localhost:8068/metrics`

---

## 1. Golden Pipeline Test (15 Seconds) - Production-Safe

## 1. Golden Pipeline Test (15 Seconds) - Production-Safe

**Tests entire chain**: apply.plan → heat.decision → by_plan key

```bash
set -euo pipefail

PID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 \
  | awk 'tolower($0)=="plan_id"{getline; print; exit}')

echo "PLAN_ID=$PID"
echo ""

echo "A) by_plan exists?"
EX=$(redis-cli --raw EXISTS "quantum:harvest:heat:by_plan:$PID")
if [ "$EX" = "1" ]; then
  echo "✅ by_plan exists"
else
  echo "❌ by_plan missing"
fi

echo ""
echo "B) by_plan TTL (if exists):"
redis-cli --raw TTL "quantum:harvest:heat:by_plan:$PID" || true

echo ""
echo "C) heat.decision contains plan_id (sample last 2000):"
if redis-cli --raw XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 2000 | grep -q "$PID"; then
  echo "✅ FOUND in heat.decision window"
else
  echo "❌ NOT FOUND in last 2000 heat.decision events"
  echo "   (Increase COUNT to 5000 if needed)"
fi

echo ""
echo "D) by_plan content (first 20 lines):"
redis-cli --raw HGETALL "quantum:harvest:heat:by_plan:$PID" | head -20
```

### Interpretation:

| Result | Meaning | Action |
|--------|---------|--------|
| A ✅ + B 1 | **Pipeline OK** | HeatGate + HeatBridge working |
| A ✅ + B 0 | **HeatBridge lag/TTL issue** | Check consumer group, increase TTL |
| A ❌ | **HeatGate not processing** | Check input stream config, or plan_id too fresh |

---

## 2. Port Sanity Check - Production Truth

**Correct ports** (never trust memory, always verify):

```bash
echo "=== Port Sanity Check ==="
curl -s -o /dev/null -w "apply_metrics 8043 => %{http_code}\n" http://localhost:8043/metrics
curl -s -o /dev/null -w "heatgate_metrics 8068 => %{http_code}\n" http://localhost:8068/metrics
curl -s -o /dev/null -w "heatgate_health 8069 => %{http_code}\n" http://localhost:8069/health
curl -s -o /dev/null -w "heatbridge_metrics 8070 => %{http_code}\n" http://localhost:8070/metrics
curl -s -o /dev/null -w "heatbridge_health 8071 => %{http_code}\n" http://localhost:8071/health
```

**Expected Output**:
```
apply_metrics 8043 => 200
heatgate_metrics 8068 => 200
heatgate_health 8069 => 200
heatbridge_metrics 8070 => 200
heatbridge_health 8071 => 200
```

**Service → Port Mapping** (permanent truth):
| Service | Metrics Port | Health Port |
|---------|-------------|-------------|
| Apply | 8043 | N/A |
| HeatGate | 8068 | 8069 |
| HeatBridge | 8070 | 8071 |

---

## 3. "Why Don't I See Keys?" - Debug Checklist

### Cause A: Wrong Prefix (FIXED)

**Wrong** ❌: `quantum:heat:bridge:by_plan:<plan_id>`  
**Right** ✅: `quantum:harvest:heat:by_plan:<plan_id>`

**Verify** (safe SCAN, never KEYS):
```bash
# Count matching keys (production-safe)
redis-cli --raw SCAN 0 MATCH "quantum:harvest:heat:by_plan:*" COUNT 5000 \
  | sed '1d' | wc -l
# Should be > 0

# Sample 5 keys
redis-cli --raw SCAN 0 MATCH "quantum:harvest:heat:by_plan:*" COUNT 5000 \
  | sed '1d' | head -5
```

### Cause B: Keys Expired (TTL)

**Check TTL on existing key** (safe SCAN):
```bash
K=$(redis-cli --raw SCAN 0 MATCH "quantum:harvest:heat:by_plan:*" COUNT 1000 \
  | sed '1d' | head -1)
echo "Sample key: $K"
redis-cli TTL "$K"
# Should be > 0 (e.g., 1500-1800 for 30min TTL)
```

**If TTL is low**, check HeatBridge config:
```bash
cat /etc/quantum/heat-bridge.env | grep TTL
# Expected: P27_TTL_PLAN_SEC=1800 (30 min) or higher
```

### Cause C: Consumer Group Lag

**Check consumer group**:
```bash
redis-cli XINFO GROUPS quantum:stream:harvest.heat.decision
redis-cli XINFO CONSUMERS quantum:stream:harvest.heat.decision heat_bridge 2>/dev/null || echo "No consumer info"
```

**Expected**: `lag: 0` or small number

---

## 4. Timing Test - Prove P2.8A.3 is Needed

**Purpose**: Measure delay between publish → by_plan key exists

```bash
set -euo pipefail

PID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 \
  | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "PLAN_ID=$PID"
echo ""

for i in 0 1 2 3 4 5; do
  ex=$(redis-cli --raw EXISTS "quantum:harvest:heat:by_plan:$PID")
  echo "t=${i}s exists=$ex"
  [ "$ex" = "1" ] && break
  sleep 1
done
```

---

## 5. Common Grep/Pipe Mistakes (Fixed)

### ❌ WRONG (prints both lines):
```bash
cmd | grep -q "$PID" ; echo "✅" ; echo "❌"
```

### ✅ CORRECT (proper branching):
```bash
if cmd | grep -q "$PID"; then
  echo "✅ FOUND"
else
  echo "❌ NOT FOUND"
fi
```

---

## 6. Pre-Deploy Checklist

Before deploying P2.8A.3, verify:

- [ ] Golden pipeline test shows: A ✅ + B 1
- [ ] Port sanity check: all 200
- [ ] Timing test: key appears after 1-3s
- [ ] Wiring correct: HeatGate reads apply.plan
- [ ] TTL reasonable: 1800s+ (30min+)
- [ ] Consumer group lag: 0 or small

**If all pass**: P2.8A.3 ready to deploy ✅

---

## Quick Copy/Paste: Full Diagnostic (Production-Safe)

```bash
#!/bin/bash
set -euo pipefail

echo "=== P2.8A Golden Verification ==="
echo ""

# 1. Golden pipeline test
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 \
  | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "1. Latest plan_id: $PLAN_ID"

if redis-cli --raw XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 2000 | grep -q "$PLAN_ID"; then
  echo "   ✅ In heat.decision"
else
  echo "   ❌ Not in heat.decision (increase COUNT to 5000 if needed)"
fi

EXISTS=$(redis-cli --raw EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID")
if [ "$EXISTS" = "1" ]; then
  echo "   ✅ by_plan key exists"
  TTL=$(redis-cli TTL "quantum:harvest:heat:by_plan:$PLAN_ID")
  echo "   TTL: ${TTL}s"
else
  echo "   ❌ by_plan key missing"
fi

echo ""
echo "2. Port check:"
curl -s -o /dev/null -w "apply_metrics 8043 => %{http_code}\n" http://localhost:8043/metrics
curl -s -o /dev/null -w "heatgate_metrics 8068 => %{http_code}\n" http://localhost:8068/metrics
curl -s -o /dev/null -w "heatgate_health 8069 => %{http_code}\n" http://localhost:8069/health
curl -s -o /dev/null -w "heatbridge_metrics 8070 => %{http_code}\n" http://localhost:8070/metrics
curl -s -o /dev/null -w "heatbridge_health 8071 => %{http_code}\n" http://localhost:8071/health

echo ""
echo "3. Wiring check:"
echo -n "   HeatGate input: "
grep HEAT_STREAM_IN /etc/quantum/heat-gate.env | cut -d= -f2

echo ""
echo "4. TTL config:"
grep TTL /etc/quantum/heat-bridge.env

echo ""
echo "5. Consumer lag:"
redis-cli XINFO GROUPS quantum:stream:harvest.heat.decision | grep -E "name:|lag:" | paste -d' ' - -

echo ""
echo "6. Sample keys (SCAN safe):"
redis-cli --raw SCAN 0 MATCH "quantum:harvest:heat:by_plan:*" COUNT 1000 \
  | sed '1d' | head -3

echo ""
echo "=== Diagnostic Complete ==="
```

---

## Permanent Truths (Lock These In!)

### Redis Keys
```
quantum:harvest:heat:by_plan:<plan_id>           ✅ ALWAYS this prefix
quantum:harvest:heat:latest:<symbol>             ✅ Latest heat per symbol
quantum:harvest:heat:latest_plan_id:<symbol>     ✅ Latest plan_id per symbol
```

### Streams
```
quantum:stream:apply.plan                        ✅ Apply output
quantum:stream:harvest.heat.decision             ✅ HeatGate output
quantum:stream:apply.heat.observed               ✅ Observer output
```

### Search Windows
```
XREVRANGE ... COUNT 5000                         ✅ For heat.decision
XREVRANGE ... COUNT 2000                         ✅ For apply.plan
XREVRANGE ... COUNT 800                          ✅ For observed (coverage analysis)
```

### Ports (Production)
```
Apply:      8043 (metrics)
HeatGate:   8068 (metrics), 8069 (health)
HeatBridge: 8070 (metrics), 8071 (health)
```

---

**These truths never change. Copy/paste with confidence!** ✅
