# P2.8A Golden Verification - Production Truth Tests

**Purpose**: Prove entire pipeline works in 15 seconds, without false negatives.

**Hard-Coded Truths** (never change these):
- ✅ Prefix: `quantum:harvest:heat:by_plan:<plan_id>`
- ✅ Search window: `COUNT 5000` (not 100)
- ✅ Apply metrics: `http://localhost:8043/metrics`
- ✅ HeatBridge metrics: `http://localhost:8070/metrics`
- ✅ HeatGate metrics: `http://localhost:8068/metrics`

---

## 1. Golden Pipeline Test (15 Seconds)

**Tests entire chain**: apply.plan → heat.decision → by_plan key

```bash
PLAN_ID=$(
  redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 \
  | awk 'tolower($0)=="plan_id"{getline; print; exit}'
)

echo "PLAN_ID=$PLAN_ID"
echo ""

echo "A) HeatGate decision exists? (search last 5000)"
if redis-cli --raw XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep -q "$PLAN_ID"; then
  echo "✅ heat.decision contains plan_id"
else
  echo "❌ missing in heat.decision window"
fi

echo ""
echo "B) HeatBridge by_plan key?"
EXISTS=$(redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID")
echo "EXISTS=$EXISTS"
if [ "$EXISTS" = "1" ]; then
  echo "✅ by_plan key exists"
else
  echo "❌ by_plan key missing"
fi

echo ""
echo "C) by_plan content (first 40 lines)"
redis-cli HGETALL "quantum:harvest:heat:by_plan:$PLAN_ID" | head -40
```

### Interpretation:

| Result | Meaning | Action |
|--------|---------|--------|
| A ✅ + B 1 | **Pipeline OK** | HeatGate + HeatBridge working |
| A ✅ + B 0 | **HeatBridge lag/TTL issue** | Check consumer group, increase TTL |
| A ❌ | **HeatGate not processing** | Check input stream config, or plan_id too fresh |

---

## 2. Port Sanity Check (Freeze This!)

**Correct ports** (never trust memory, always verify):

```bash
echo "=== Port Sanity Check ==="
for url in \
  http://localhost:8068/metrics \
  http://localhost:8069/health \
  http://localhost:8070/metrics \
  http://localhost:8071/health \
  http://localhost:8043/metrics
do
  echo -n "$url => "
  curl -s -o /dev/null -w "%{http_code}\n" "$url"
done
```

**Expected**:
```
http://localhost:8068/metrics => 200  (HeatGate metrics)
http://localhost:8069/health => 200   (HeatGate health)
http://localhost:8070/metrics => 200  (HeatBridge metrics)
http://localhost:8071/health => 200   (HeatBridge health)
http://localhost:8043/metrics => 200  (Apply metrics)
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

**Verify**:
```bash
redis-cli KEYS "quantum:harvest:heat:by_plan:*" | wc -l
# Should be > 0
```

### Cause B: Keys Expired (TTL)

**Check TTL on existing key**:
```bash
K=$(redis-cli --raw KEYS "quantum:harvest:heat:by_plan:*" | head -1)
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

## 4. Timing Test (Prove P2.8A.3 is Needed)

**Purpose**: Measure delay between publish → by_plan key exists

```bash
PLAN_ID=$(
  redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 \
  | awk 'tolower($0)=="plan_id"{getline; print; exit}'
)

echo "PLAN_ID=$PLAN_ID"
echo "Watching for by_plan key to appear..."
echo ""

for s in 0 1 2 3 4 5; do
  sleep 1
  EXISTS=$(redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID")
  echo "t+${s}s exists=$EXISTS"
  if [ "$EXISTS" = "1" ]; then
    echo ""
    echo "✅ Key appeared after ${s} seconds"
    echo "This proves P2.8A.3 late observer is needed!"
    break
  fi
done
```

**If key goes from 0 → 1 after 1-3 seconds**:
- ✅ **P2.8A.3 is 100% correct solution**
- Early observer (t+0s) will always miss
- Late observer (wait 2s) will catch it

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

## Quick Copy/Paste: Full Diagnostic

```bash
#!/bin/bash
echo "=== P2.8A Golden Verification ==="
echo ""

# 1. Golden pipeline test
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "1. Latest plan_id: $PLAN_ID"

if redis-cli --raw XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep -q "$PLAN_ID"; then
  echo "   ✅ In heat.decision"
else
  echo "   ❌ Not in heat.decision"
fi

EXISTS=$(redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID")
if [ "$EXISTS" = "1" ]; then
  echo "   ✅ by_plan key exists"
  TTL=$(redis-cli TTL "quantum:harvest:heat:by_plan:$PLAN_ID")
  echo "   TTL: ${TTL}s"
else
  echo "   ❌ by_plan key missing"
fi

echo ""
echo "2. Port check:"
for port in 8043 8068 8069 8070 8071; do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/metrics 2>/dev/null || curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health 2>/dev/null)
  echo "   Port $port: $CODE"
done

echo ""
echo "3. Wiring check:"
echo -n "   HeatGate input: "
cat /etc/quantum/heat-gate.env | grep HEAT_STREAM_IN | cut -d= -f2

echo ""
echo "4. TTL config:"
cat /etc/quantum/heat-bridge.env | grep TTL

echo ""
echo "5. Consumer lag:"
redis-cli XINFO GROUPS quantum:stream:harvest.heat.decision | grep -E "name:|lag:" | paste -d' ' - -

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
