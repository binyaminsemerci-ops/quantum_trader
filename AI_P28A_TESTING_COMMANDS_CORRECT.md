# P2.8A Testing - RIKTIGE Kommandoer

**VIKTIG**: Bruk disse kommandoene for testing. Tidligere tester brukte feil prefix!

---

## Riktig Prefix (KRITISK!)

**RIKTIG** ✅:
```bash
quantum:harvest:heat:by_plan:{plan_id}
```

**FEIL** ❌:
```bash
quantum:heat:bridge:by_plan:{plan_id}  # Finnes IKKE!
```

---

## Robust Test-Kommandoer (Copy/Paste)

### 1. Verifiser Plan ID Matching

```bash
echo "Latest apply.plan plan_id:"
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "PLAN_ID=$PLAN_ID"

echo ""
echo "Latest observed plan_id:"
OBS_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "OBS_ID=$OBS_ID"

echo ""
echo "Compare (checking if observed plan exists in apply.plan stream):"
if redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 2000 | grep -q "$OBS_ID"; then
  echo "✅ FOUND IN apply.plan (IDs match)"
else
  echo "❌ NOT FOUND"
fi
```

### 2. Sjekk by_plan Key Eksistens

```bash
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')

echo "by_plan key exists?"
redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN_ID"

echo ""
echo "by_plan key content:"
redis-cli HGETALL "quantum:harvest:heat:by_plan:$PLAN_ID"

echo ""
echo "by_plan key TTL:"
redis-cli TTL "quantum:harvest:heat:by_plan:$PLAN_ID"
```

### 3. Søk i heat.decision (Bred Søk)

```bash
PLAN_ID=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')

echo "Search heat.decision (last 5000 events):"
redis-cli XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep -c "$PLAN_ID"

echo ""
echo "If found, show event:"
redis-cli XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep "$PLAN_ID" -B 5 -A 20 | head -30
```

### 4. Liste Aktive by_plan Keys

```bash
echo "Active by_plan keys:"
redis-cli KEYS "quantum:harvest:heat:by_plan:*"

echo ""
echo "Count:"
redis-cli KEYS "quantum:harvest:heat:by_plan:*" | wc -l

echo ""
echo "Sample key content:"
FIRST_KEY=$(redis-cli KEYS "quantum:harvest:heat:by_plan:*" | head -1)
if [ -n "$FIRST_KEY" ]; then
    redis-cli HGETALL "$FIRST_KEY"
    echo ""
    redis-cli TTL "$FIRST_KEY"
fi
```

### 5. Riktige Metrics Endpoints

```bash
echo "=== Verify Service Names & Ports ==="
echo ""
echo "Services:"
systemctl list-units --type=service | grep -E "quantum-(apply|heat)"

echo ""
echo "=== Metrics ==="
echo ""
echo "Apply metrics (port 8043):"
curl -s http://localhost:8043/metrics | grep -E "^p28_(observed_total|heat_reason_total)"

echo ""
echo "HeatBridge metrics (port 8070):"
curl -s http://localhost:8070/metrics | grep -E "^p27_"

echo ""
echo "HeatGate metrics (port 8068):"
curl -s http://localhost:8068/metrics | grep -E "^p26_"
```

### 6. Heat Coverage Analyse (Siste 15 Min)

```bash
echo "=== Heat Coverage (Last 15 Minutes) ==="
echo ""

CUTOFF=$(($(date +%s) - 900))

redis-cli XRANGE quantum:stream:apply.heat.observed - + | awk -v cutoff="$CUTOFF" '
  /^[0-9]+-[0-9]+$/ { 
    ts = substr($1, 1, 10) 
  }
  /^obs_point$/ { 
    getline
    obs_point = $0
  }
  /^heat_found$/ { 
    getline
    if (ts >= cutoff) {
      count[obs_point "_" $0]++
      total[obs_point]++
    }
  }
  END { 
    for (key in count) {
      print key ": " count[key]
    }
    print ""
    for (key in total) {
      heat1 = count[key "_1"]
      heat0 = count[key "_0"]
      if (total[key] > 0) {
        pct = (heat1 / total[key]) * 100
        printf "%s coverage: %.1f%% (%d/%d)\n", key, pct, heat1, total[key]
      }
    }
  }'
```

### 7. Full Diagnostic (All Streams)

```bash
echo "=== Full P2.8A Diagnostic ==="
echo ""

echo "1. Stream Lengths:"
echo "   apply.plan: $(redis-cli XLEN quantum:stream:apply.plan)"
echo "   heat.decision: $(redis-cli XLEN quantum:stream:harvest.heat.decision)"
echo "   apply.heat.observed: $(redis-cli XLEN quantum:stream:apply.heat.observed)"

echo ""
echo "2. by_plan Keys Count:"
redis-cli KEYS "quantum:harvest:heat:by_plan:*" | wc -l

echo ""
echo "3. Latest Plan IDs:"
APPLY_PLAN=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
OBSERVED_PLAN=$(redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "   apply.plan:  $APPLY_PLAN"
echo "   observed:    $OBSERVED_PLAN"
[ "$APPLY_PLAN" = "$OBSERVED_PLAN" ] && echo "   Status: MATCH ✅" || echo "   Status: MISMATCH ❌"

echo ""
echo "4. by_plan Key for Latest Plan:"
redis-cli EXISTS "quantum:harvest:heat:by_plan:$APPLY_PLAN"
redis-cli TTL "quantum:harvest:heat:by_plan:$APPLY_PLAN"

echo ""
echo "5. Service Status:"
systemctl is-active quantum-apply-layer
systemctl is-active quantum-heat-bridge
systemctl is-active quantum-heat-gate

echo ""
echo "6. Latest Observer Results:"
redis-cli XREVRANGE quantum:stream:apply.heat.observed + - COUNT 3 | grep -E "obs_point|heat_found|heat_reason" -A 1 | head -20
```

---

## Vanlige Feil

### Feil #1: Feil by_plan Prefix

**Symptom**: `EXISTS` returnerer 0 selv om heat_found=1

**Årsak**: Bruker `quantum:heat:bridge:by_plan:` i stedet for `quantum:harvest:heat:by_plan:`

**Fix**: Bruk riktig prefix (se kommandoer over)

### Feil #2: For Lite Data i Søk

**Symptom**: Finner ikke plan_id i heat.decision

**Årsak**: Søker bare i 100 events (for lite ved høy throughput)

**Fix**: Bruk `COUNT 5000` i stedet for `COUNT 100`

### Feil #3: Feil Metrics Port

**Symptom**: Metrics viser ingen data

**Årsak**: Sjekker feil port (8002 i stedet for 8043)

**Fix**: 
- Apply: port 8043
- HeatBridge: port 8070
- HeatGate: port 8068

### Feil #4: Feil Service Navn

**Symptom**: `systemctl status` finner ikke service

**Årsak**: Søker etter `quantum-apply` i stedet for `quantum-apply-layer`

**Fix**: Bruk `systemctl list-units --type=service | grep quantum` for å se alle

---

## Quick Verification (1 Kommando)

```bash
# Kopiér-lim hele blokken:
PLAN=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}'); \
echo "Latest plan_id: $PLAN"; \
echo "by_plan EXISTS: $(redis-cli EXISTS quantum:harvest:heat:by_plan:$PLAN)"; \
echo "by_plan TTL: $(redis-cli TTL quantum:harvest:heat:by_plan:$PLAN)"; \
echo "In heat.decision: $(redis-cli XREVRANGE quantum:stream:harvest.heat.decision + - COUNT 5000 | grep -c $PLAN)"; \
echo "Metrics (p28): $(curl -s http://localhost:8043/metrics | grep p28_observed_total | wc -l) lines"
```

**Forventet output**:
```
Latest plan_id: abc123def456
by_plan EXISTS: 1
by_plan TTL: 1500
In heat.decision: 1
Metrics (p28): 4 lines
```

---

## Timing Test (Fresh Plan)

```bash
echo "Creating fresh plan flow..."
echo "Wait 3 seconds for full pipeline..."
sleep 3

PLAN=$(redis-cli --raw XREVRANGE quantum:stream:apply.plan + - COUNT 1 | awk 'tolower($0)=="plan_id"{getline; print; exit}')
echo "Fresh plan_id: $PLAN"

for i in 0 1 2 3; do
    echo ""
    echo "Check $i (after ${i}s):"
    EXISTS=$(redis-cli EXISTS "quantum:harvest:heat:by_plan:$PLAN")
    echo "  EXISTS: $EXISTS"
    if [ "$EXISTS" = "1" ]; then
        echo "  FOUND at ${i}s! ✅"
        redis-cli HGETALL "quantum:harvest:heat:by_plan:$PLAN" | head -10
        break
    fi
    [ $i -lt 3 ] && sleep 1
done
```

---

## P2.8A.3 Verification (Late Observer)

### A) Confirm publish_plan_post Appearing

```bash
redis-cli XREVRANGE quantum:stream:apply.heat.observed + - COUNT 200 | grep -c "publish_plan_post"
# Should be > 0 if late observer is running
```

### B) Coverage Split by obs_point

```bash
redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 800 | awk '
  tolower($0)=="obs_point"{getline; op=$0}
  tolower($0)=="heat_found"{getline; hf=$0; c[op,hf]++}
  END{
    for (k in c) print k, c[k]
  }' | sort
```

**Expected**:
```
create_apply_plan 0 150  (early, still mostly 0)
create_apply_plan 1 10
publish_plan_post 0 20
publish_plan_post 1 140  (late, mostly 1!)
```

### C) Metrics Confirmation

```bash
curl -s http://localhost:8043/metrics | grep -E '^p28_observed_total|^p28_heat_reason_total' | grep publish_plan_post
```

**Expected**: `p28_observed_total{obs_point="publish_plan_post",heat_found="1"}` increasing

### D) Verify Wiring Still Correct

```bash
# HeatGate input
cat /etc/quantum/heat-gate.env | grep HEAT_STREAM_IN
# Expected: HEAT_STREAM_IN=quantum:stream:apply.plan

# HeatBridge config
cat /etc/quantum/heat-bridge.env | grep -E 'P27_STREAM_IN|TTL'
# Expected: P27_STREAM_IN=quantum:stream:harvest.heat.decision, TTL=1800+

# Consumer group exists
redis-cli XINFO GROUPS quantum:stream:harvest.heat.decision
# Expected: heat_bridge group exists
```

---

**HUSK**: Alltid bruk `quantum:harvest:heat:by_plan:` som prefix!
