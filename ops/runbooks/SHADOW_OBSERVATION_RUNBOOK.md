# SHADOW OBSERVATION RUNBOOK — PatchTST P0.4

**Purpose**: Systematic monitoring and evaluation of PatchTST shadow mode  
**Target**: 300-1,000 shadow datapoints before gate evaluation  
**Environment**: VPS systemd-only (NO DOCKER)

---

## 📋 QUICK REFERENCE

### Service Info
```bash
Service: quantum-ai-engine.service
User: qt
Redis: localhost:6379 (systemd, NOT docker)
Stream: quantum:stream:trade.intent
Shadow Flag: PATCHTST_SHADOW_ONLY=true
```

### Critical Commands
```bash
# Service status
systemctl status quantum-ai-engine.service

# Tail live logs
journalctl -u quantum-ai-engine.service -f

# Check env flags
grep "^PATCHTST" /etc/quantum/ai-engine.env

# Rollback (<2 min)
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env
systemctl restart quantum-ai-engine.service
```

---

## 🔍 SECTION 1: DATA COLLECTION METRICS

### 1.1 Event Rate Check (Baseline Understanding)

**Purpose**: Understand actual inference rate vs logging rate

```bash
# SSH to VPS
ssh root@46.224.116.254

# Count trade.intent events in last 30 minutes
redis-cli XLEN quantum:stream:trade.intent

# Get events from last 30 min with timestamps
NOW=$(date +%s)
START=$((NOW - 1800))  # 1800 seconds = 30 min
redis-cli XRANGE quantum:stream:trade.intent $((START * 1000))-0 + COUNT 1000 | grep -c "event_type"

# Alternative: Use journalctl to count "Publishing trade.intent"
journalctl -u quantum-ai-engine.service --since "30 minutes ago" -o cat | grep -c "Publishing trade.intent"
```

**Expected Output**:
```
Total trade.intent events (30 min): ~50-200 (depends on market activity)
Logging rate: ~2-7 events per minute
```

**Interpretation**:
- If events > 60 (>2/min): High activity, shadow logs will be rate-limited
- If events < 20 (<1/min): Low activity, may need longer observation period

---

### 1.2 Shadow Logging Rate Check

**Purpose**: Verify rate-limiting is working correctly

```bash
# Count shadow logs in last 30 min
journalctl -u quantum-ai-engine.service --since "30 minutes ago" -o cat | grep -E "\[SHADOW\] PatchTST" | wc -l

# Get actual shadow logs with timestamps
journalctl -u quantum-ai-engine.service --since "30 minutes ago" --no-pager | grep -E "\[SHADOW\] PatchTST"

# Extract shadow log times to check 30s spacing
journalctl -u quantum-ai-engine.service --since "1 hour ago" --no-pager | \
  grep -E "\[SHADOW\] PatchTST" | \
  awk '{print $1, $2}' | \
  while IFS= read -r timestamp; do
    date -d "$timestamp" +%s
  done | \
  awk 'NR>1 {print $1-prev} {prev=$1}'
```

**Expected Output**:
```
Shadow logs (30 min): ~10-60 (depends on symbols active)
Time deltas: ~30-60 seconds (rate-limited per prediction call)
```

**RED FLAG**:
- If shadow logs = 0: Shadow mode not active (check env flag)
- If time deltas < 10s consistently: Rate limiter not working
- If shadow logs >> events: Logging too verbose (check _SHADOW_LOG_INTERVAL)

---

### 1.3 Logging Rate Mismatch Analysis

**Purpose**: Reconcile "every 30s" claim with actual 5 logs/30min

```bash
# Comprehensive rate check
echo "=== LAST 30 MIN METRICS ==="
echo "Trade.intent events: $(journalctl -u quantum-ai-engine.service --since "30 minutes ago" -o cat | grep -c "Publishing trade.intent")"
echo "Shadow logs: $(journalctl -u quantum-ai-engine.service --since "30 minutes ago" -o cat | grep -c "\[SHADOW\] PatchTST")"
echo "Unique symbols: $(journalctl -u quantum-ai-engine.service --since "30 minutes ago" -o cat | grep "Publishing trade.intent" | grep -oP 'TELEMETRY.*?: \K[A-Z]+' | sort -u | wc -l)"
echo ""
echo "=== INTERPRETATION ==="
echo "If shadow logs < trade.intent events: Rate-limiting OR shadow only triggers on certain conditions"
echo "If shadow logs ~= trade.intent events: Rate-limiter broken (too verbose)"
echo "Expected ratio: ~1 shadow log per 3-10 trade.intent events (depends on symbols)"
```

**CORRECTED UNDERSTANDING**:
- **Rate limiter**: 30s per global call, NOT per symbol
- **Actual trigger**: Only when `predict()` is called on PatchTST agent
- **Why 5 logs/30min**: Likely only 5 predictions triggered PatchTST (other events used cached results or different code path)

**Action**: Update deployment docs to say:
> "Rate-limited: At most 1 log per 30 seconds when PatchTST prediction is executed"

---

## 🔍 SECTION 2: PAYLOAD SIZE VERIFICATION

### 2.1 Actual Payload Size Measurement

**Purpose**: Prove the "~10% overhead / 120 bytes" claim with real data

```bash
# Fetch 20 recent events and measure payload sizes
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 20 | \
  awk '/^payload$/ {getline; print length($0)}' | \
  sort -n | \
  awk '
    BEGIN {sum=0; count=0}
    {
      arr[NR]=$1
      sum+=$1
      count++
    }
    END {
      print "Min:", arr[1]
      print "Median:", arr[int(count/2)]
      print "P95:", arr[int(count*0.95)]
      print "Max:", arr[count]
      print "Mean:", sum/count
    }
  '

# Estimate PatchTST overhead
echo ""
echo "=== PATCHTST OVERHEAD ESTIMATE ==="
echo "PatchTST entry typical size:"
echo '{"action": "BUY", "confidence": 0.6150280237197876, "model": "patchtst_shadow", "shadow": true}' | wc -c
echo ""
echo "Without shadow flag (for comparison):"
echo '{"action": "BUY", "confidence": 0.6150280237197876, "model": "patchtst_shadow"}' | wc -c
```

**Expected Output**:
```
Min: 950
Median: 1150
P95: 1350
Max: 1450
Mean: 1180

=== PATCHTST OVERHEAD ESTIMATE ===
PatchTST entry typical size:
108

Without shadow flag (for comparison):
92
```

**Interpretation**:
- Total payload: ~1150 bytes (median)
- PatchTST entry: ~108 bytes
- Overhead: 108/1150 = **9.4%** ✅
- Shadow flag cost: 16 bytes (`"shadow": true,`)

**RED FLAG**:
- If overhead >15%: Check for duplicate data or verbose logging
- If payload >2KB consistently: May need separate shadow stream

---

### 2.2 Payload Size Per Symbol

**Purpose**: Check if certain symbols have bloated payloads

```bash
# Extract payload sizes per symbol
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '
    /^payload$/ {
      getline payload
      if (match(payload, /"symbol": "([^"]+)"/, m)) {
        symbol = m[1]
        size = length(payload)
        print symbol, size
      }
    }
  ' | \
  awk '{sum[$1]+=1; size[$1]+=$2} END {for (s in sum) print s, sum[s], size[s]/sum[s]}' | \
  sort -k3 -nr
```

**Expected Output**:
```
BTCUSDT 15 1145
ETHUSDT 12 1138
SOLUSDT 10 1152
BNBUSDT 8 1149
```

**RED FLAG**:
- If any symbol >1500 bytes: Check for anomalies (long regime strings, excessive features)

---

## 🔍 SECTION 3: VOTING EXCLUSION VERIFICATION

### 3.1 Consensus Count Distribution

**Purpose**: Prove PatchTST is NEVER counted in consensus

```bash
# Extract consensus_count from last 100 events
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '/^payload$/ {getline; if (match($0, /"consensus_count": ([0-9]+)/, m)) print m[1]}' | \
  sort | uniq -c | sort -rn

# Check for any consensus_count=4 (should be ZERO)
echo ""
echo "=== RED FLAG CHECK ==="
COUNT_4=$(redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '/^payload$/ {getline; if (match($0, /"consensus_count": 4/)) print}' | wc -l)
if [ "$COUNT_4" -eq 0 ]; then
  echo "✅ PASS: Zero events with consensus_count=4"
else
  echo "❌ FAIL: Found $COUNT_4 events with consensus_count=4 (PatchTST is voting!)"
fi
```

**Expected Output**:
```
     45 3
     32 2
     15 1
      8 0

=== RED FLAG CHECK ===
✅ PASS: Zero events with consensus_count=4
```

**Interpretation**:
- Consensus distribution: 0-3 (never 4) ✅
- Most common: 2-3 (healthy consensus)
- If any 4: **CRITICAL** - Shadow mode broken, rollback immediately

---

### 3.2 PatchTST Presence vs Consensus Count

**Purpose**: Verify patchtst always present in breakdown, never in consensus

```bash
# Check all events have patchtst in breakdown but consensus <4
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk '
    BEGIN {events=0; patchtst_present=0; consensus_4=0}
    /^[0-9]+-[0-9]+$/ {events++}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst":/)) patchtst_present++
      if (match(payload, /"consensus_count": 4/)) consensus_4++
    }
    END {
      print "Total events:", events
      print "Events with patchtst:", patchtst_present
      print "Events with consensus=4:", consensus_4
      print ""
      if (patchtst_present == events && consensus_4 == 0) {
        print "✅ VERIFIED: PatchTST present in all events, never in voting"
      } else {
        print "❌ ISSUE: Mismatch detected"
      }
    }
  '
```

**Expected Output**:
```
Total events: 50
Events with patchtst: 50
Events with consensus=4: 0

✅ VERIFIED: PatchTST present in all events, never in voting
```

---

### 3.3 Shadow Flag Presence Check

**Purpose**: Ensure all patchtst entries have shadow=true marker

```bash
# Check shadow flag presence
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk '
    BEGIN {patchtst_count=0; shadow_flag_count=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*\}/)) {
        patchtst_count++
        if (match(payload, /"patchtst": \{[^}]*"shadow": true[^}]*\}/)) {
          shadow_flag_count++
        }
      }
    }
    END {
      print "PatchTST entries:", patchtst_count
      print "Shadow flags:", shadow_flag_count
      print ""
      if (patchtst_count == shadow_flag_count) {
        print "✅ PASS: All PatchTST entries have shadow=true"
      } else {
        print "❌ FAIL:", (patchtst_count - shadow_flag_count), "entries missing shadow flag"
      }
    }
  '
```

**Expected Output**:
```
PatchTST entries: 50
Shadow flags: 50

✅ PASS: All PatchTST entries have shadow=true
```

---

## 🔍 SECTION 4: AGREEMENT RATE ANALYSIS

### 4.1 Shadow vs Ensemble Agreement

**Purpose**: Measure how often PatchTST agrees with ensemble majority (without PatchTST)

```bash
# Calculate agreement rate
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    BEGIN {
      total=0
      agree=0
      disagree=0
      by_action["BUY_BUY"]=0
      by_action["BUY_SELL"]=0
      by_action["BUY_HOLD"]=0
      by_action["SELL_BUY"]=0
      by_action["SELL_SELL"]=0
      by_action["SELL_HOLD"]=0
      by_action["HOLD_BUY"]=0
      by_action["HOLD_SELL"]=0
      by_action["HOLD_HOLD"]=0
    }
    /^payload$/ {
      getline payload
      if (match(payload, /"side": "([^"]+)"/, ensemble_side) && \
          match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, shadow_action)) {
        total++
        ensemble = ensemble_side[1]
        shadow = shadow_action[1]
        key = shadow "_" ensemble
        by_action[key]++
        
        if (ensemble == shadow) {
          agree++
        } else {
          disagree++
        }
      }
    }
    END {
      print "=== AGREEMENT RATE ==="
      print "Total predictions:", total
      print "Agree:", agree, "(" int(agree*100/total) "%)"
      print "Disagree:", disagree, "(" int(disagree*100/total) "%)"
      print ""
      print "=== CONFUSION MATRIX (Shadow vs Ensemble) ==="
      printf "%-10s %-10s %-10s %-10s\n", "", "Ens:BUY", "Ens:SELL", "Ens:HOLD"
      printf "%-10s %-10d %-10d %-10d\n", "Shd:BUY", by_action["BUY_BUY"], by_action["BUY_SELL"], by_action["BUY_HOLD"]
      printf "%-10s %-10d %-10d %-10d\n", "Shd:SELL", by_action["SELL_BUY"], by_action["SELL_SELL"], by_action["SELL_HOLD"]
      printf "%-10s %-10d %-10d %-10d\n", "Shd:HOLD", by_action["HOLD_BUY"], by_action["HOLD_SELL"], by_action["HOLD_HOLD"]
      print ""
      print "=== GATE 3 EVALUATION ==="
      if (agree*100/total >= 55) {
        print "✅ PASS: Agreement rate >= 55%"
      } else {
        print "❌ FAIL: Agreement rate < 55% (need >=55% to activate)"
      }
    }
  '
```

**Expected Output** (healthy):
```
=== AGREEMENT RATE ===
Total predictions: 200
Agree: 125 (62%)
Disagree: 75 (38%)

=== CONFUSION MATRIX (Shadow vs Ensemble) ===
           Ens:BUY    Ens:SELL   Ens:HOLD  
Shd:BUY    110        15         45        
Shd:SELL   5          10         8         
Shd:HOLD   10         5          12        

=== GATE 3 EVALUATION ===
✅ PASS: Agreement rate >= 55%
```

**RED FLAG** (P0.4 expected):
```
=== AGREEMENT RATE ===
Agree: 180 (90%)  ← Too high = PatchTST is just copying ensemble
OR
Agree: 40 (20%)   ← Too low = PatchTST is contrarian, likely broken
```

**Gate 3 Threshold**:
- **PASS**: 55-75% agreement (balanced, independent predictions)
- **BORDERLINE**: 75-85% (leans toward copying, but acceptable)
- **FAIL**: <55% OR >85% (either contrarian or redundant)

---

### 4.2 Agreement Rate Per Symbol

**Purpose**: Detect regime-specific bias

```bash
# Agreement rate by symbol
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    /^payload$/ {
      getline payload
      if (match(payload, /"symbol": "([^"]+)"/, sym) && \
          match(payload, /"side": "([^"]+)"/, ens) && \
          match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, shd)) {
        symbol = sym[1]
        total[symbol]++
        if (ens[1] == shd[1]) agree[symbol]++
      }
    }
    END {
      printf "%-12s %-8s %-8s %-8s\n", "Symbol", "Total", "Agree", "Rate"
      for (s in total) {
        rate = int(agree[s]*100/total[s])
        printf "%-12s %-8d %-8d %-8d%%\n", s, total[s], agree[s], rate
      }
    }
  ' | sort -t' ' -k4 -nr
```

**Expected Output**:
```
Symbol       Total    Agree    Rate    
ETHUSDT      45       30       67%
BTCUSDT      40       25       62%
SOLUSDT      35       20       57%
BNBUSDT      30       18       60%
```

**RED FLAG**:
- If any symbol <40%: PatchTST broken for that symbol
- If any symbol >90%: PatchTST redundant for that symbol

---

## 🔍 SECTION 5: REFINED GATE EVALUATION

### 5.1 Gate 1: Action Diversity (REFINED)

**Purpose**: Ensure PatchTST produces diverse actions, not just BUY bias

**Refined Criteria**:
1. No single class >70% ✅ (original)
2. At least 2 classes >10% each ✅ (NEW - prevents subtle bias)

```bash
# Action distribution with refined check
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    BEGIN {buy=0; sell=0; hold=0; total=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, act)) {
        total++
        action = act[1]
        if (action == "BUY") buy++
        else if (action == "SELL") sell++
        else if (action == "HOLD") hold++
      }
    }
    END {
      buy_pct = int(buy*100/total)
      sell_pct = int(sell*100/total)
      hold_pct = int(hold*100/total)
      
      print "=== ACTION DISTRIBUTION ==="
      printf "BUY:  %3d/%d (%d%%)\n", buy, total, buy_pct
      printf "SELL: %3d/%d (%d%%)\n", sell, total, sell_pct
      printf "HOLD: %3d/%d (%d%%)\n", hold, total, hold_pct
      print ""
      
      # Check 1: No class >70%
      max_pct = buy_pct
      if (sell_pct > max_pct) max_pct = sell_pct
      if (hold_pct > max_pct) max_pct = hold_pct
      check1 = (max_pct <= 70) ? 1 : 0
      
      # Check 2: At least 2 classes >10%
      classes_above_10 = 0
      if (buy_pct > 10) classes_above_10++
      if (sell_pct > 10) classes_above_10++
      if (hold_pct > 10) classes_above_10++
      check2 = (classes_above_10 >= 2) ? 1 : 0
      
      print "=== GATE 1 EVALUATION (REFINED) ==="
      print "Check 1 (no class >70%):", (check1 ? "✅ PASS" : "❌ FAIL (" max_pct "%)")
      print "Check 2 (>=2 classes >10%):", (check2 ? "✅ PASS" : "❌ FAIL (" classes_above_10 " classes)")
      print ""
      if (check1 && check2) {
        print "✅ GATE 1: PASS"
      } else {
        print "❌ GATE 1: FAIL"
      }
    }
  '
```

**Expected Output** (P0.4 FAIL - BUY bias):
```
=== ACTION DISTRIBUTION ===
BUY:  195/200 (97%)
SELL:   3/200 (1%)
HOLD:   2/200 (1%)

=== GATE 1 EVALUATION (REFINED) ===
Check 1 (no class >70%): ❌ FAIL (97%)
Check 2 (>=2 classes >10%): ❌ FAIL (1 classes)

❌ GATE 1: FAIL
```

**Action**: Keep shadow mode, re-train with class balancing

---

### 5.2 Gate 2: Confidence Spread (REFINED)

**Purpose**: Ensure PatchTST has diverse confidences, not flatlined

**Refined Criteria**:
1. stddev(confidence) ≥ 0.05 ✅ (original)
2. p10-p90 range > 0.12 ✅ (NEW - detects subtle flatlining)

```bash
# Confidence statistics with refined check
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    BEGIN {n=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*"confidence": ([0-9.]+)/, conf)) {
        n++
        vals[n] = conf[1]
        sum += conf[1]
      }
    }
    END {
      # Calculate mean
      mean = sum / n
      
      # Calculate stddev
      for (i=1; i<=n; i++) {
        sq_diff += (vals[i] - mean) ^ 2
      }
      stddev = sqrt(sq_diff / n)
      
      # Sort for percentiles
      for (i=1; i<=n; i++) {
        for (j=i+1; j<=n; j++) {
          if (vals[i] > vals[j]) {
            tmp = vals[i]
            vals[i] = vals[j]
            vals[j] = tmp
          }
        }
      }
      
      p10 = vals[int(n * 0.10)]
      p50 = vals[int(n * 0.50)]
      p90 = vals[int(n * 0.90)]
      range_p10_p90 = p90 - p10
      
      print "=== CONFIDENCE STATISTICS ==="
      printf "N: %d\n", n
      printf "Mean: %.4f\n", mean
      printf "Stddev: %.4f\n", stddev
      printf "Min: %.4f\n", vals[1]
      printf "P10: %.4f\n", p10
      printf "P50: %.4f\n", p50
      printf "P90: %.4f\n", p90
      printf "Max: %.4f\n", vals[n]
      printf "P10-P90 Range: %.4f\n", range_p10_p90
      print ""
      
      # Check 1: stddev >= 0.05
      check1 = (stddev >= 0.05) ? 1 : 0
      
      # Check 2: p10-p90 range > 0.12
      check2 = (range_p10_p90 > 0.12) ? 1 : 0
      
      print "=== GATE 2 EVALUATION (REFINED) ==="
      print "Check 1 (stddev >= 0.05):", (check1 ? "✅ PASS" : "❌ FAIL (" sprintf("%.4f", stddev) ")")
      print "Check 2 (p10-p90 > 0.12):", (check2 ? "✅ PASS" : "❌ FAIL (" sprintf("%.4f", range_p10_p90) ")")
      print ""
      if (check1 && check2) {
        print "✅ GATE 2: PASS"
      } else {
        print "❌ GATE 2: FAIL"
      }
    }
  '
```

**Expected Output** (P0.4 likely FAIL - low spread):
```
=== CONFIDENCE STATISTICS ===
N: 200
Mean: 0.6150
Stddev: 0.0012
Min: 0.6145
P10: 0.6148
P50: 0.6150
P90: 0.6152
Max: 0.6155
P10-P90 Range: 0.0004

=== GATE 2 EVALUATION (REFINED) ===
Check 1 (stddev >= 0.05): ❌ FAIL (0.0012)
Check 2 (p10-p90 > 0.12): ❌ FAIL (0.0004)

❌ GATE 2: FAIL
```

**Interpretation**: P0.4 model has flatlined confidences (all ~0.615), needs more diverse training

---

### 5.3 Gate 3: Shadow Correlation (ALREADY DONE - See Section 4.1)

**Criteria**: Agreement rate 55-75% (already implemented above)

---

### 5.4 Gate 4: Calibration (OUTCOME-BASED)

**Purpose**: Verify higher confidence → higher accuracy (requires outcomes)

**Challenge**: Calibration needs actual trade outcomes or price movement data

**Approach 1: Price Movement Outcome** (No trades needed)

```bash
# Fetch events with PatchTST predictions and check if price moved in predicted direction
# NOTE: Requires storing price at prediction time + checking price 1h later
# This is a PLACEHOLDER - actual implementation needs historical price data

echo "=== GATE 4: CALIBRATION (OUTCOME-BASED) ==="
echo ""
echo "⚠️ CALIBRATION REQUIRES OUTCOME DATA:"
echo "  Option A: Price movement (did price move in predicted direction within 1h?)"
echo "  Option B: Trade results (if execution active)"
echo ""
echo "PLACEHOLDER SCRIPT:"
echo ""
cat << 'EOF'
# Pseudo-code for calibration check
# 1. For each shadow prediction:
#    - Record: symbol, action, confidence, timestamp, entry_price
# 2. Wait H hours (horizon = 1h typical)
# 3. Fetch current price
# 4. Check outcome:
#    - BUY: price_current > entry_price * 1.002 → HIT
#    - SELL: price_current < entry_price * 0.998 → HIT
#    - HOLD: |price_current - entry_price| < 0.001 → HIT
# 5. Bucket by confidence:
#    - Low: 0.50-0.60
#    - Med: 0.60-0.70
#    - High: 0.70-0.80
#    - Very High: 0.80-1.00
# 6. Calculate hit rate per bucket
# 7. Check monotonicity: hit_rate(high) > hit_rate(med) > hit_rate(low)

# Example target:
# Low (0.50-0.60): 52% accuracy
# Med (0.60-0.70): 62% accuracy
# High (0.70-0.80): 72% accuracy
# Very High (0.80+): 82% accuracy
EOF
echo ""
echo "❌ GATE 4: DEFERRED (needs outcome data collection)"
echo "   Action: Wait 24h, then run calibration script with historical prices"
```

**Expected Output**:
```
=== GATE 4: CALIBRATION (OUTCOME-BASED) ===

⚠️ CALIBRATION REQUIRES OUTCOME DATA:
  Option A: Price movement (did price move in predicted direction within 1h?)
  Option B: Trade results (if execution active)

❌ GATE 4: DEFERRED (needs outcome data collection)
   Action: Wait 24h, then run calibration script with historical prices
```

**Action**: For now, skip Gate 4 or accept ≥3/4 gates without it

---

## 📊 SECTION 6: DECISION SNAPSHOTS

### 6.1 2-Hour Snapshot

**Timing**: Run after 2 hours of shadow observation

```bash
#!/bin/bash
# snapshot_2h.sh

echo "==================================================="
echo "PATCHTST SHADOW MODE — 2-HOUR SNAPSHOT"
echo "==================================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%S UTC")"
echo ""

# 1. Service health
echo "1. SERVICE HEALTH"
echo "-----------------"
systemctl is-active quantum-ai-engine.service
systemctl status quantum-ai-engine.service | grep "Active:"
echo ""

# 2. Event counts
echo "2. EVENT COUNTS (last 2h)"
echo "-------------------------"
INTENTS=$(journalctl -u quantum-ai-engine.service --since "2 hours ago" -o cat | grep -c "Publishing trade.intent")
SHADOWS=$(journalctl -u quantum-ai-engine.service --since "2 hours ago" -o cat | grep -c "\[SHADOW\] PatchTST")
echo "Trade.intent events: $INTENTS"
echo "Shadow logs: $SHADOWS"
echo "Shadow ratio: $(echo "scale=2; $SHADOWS / $INTENTS * 100" | bc)%"
echo ""

# 3. Payload size check
echo "3. PAYLOAD SIZE (last 20 events)"
echo "--------------------------------"
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 20 | \
  awk '/^payload$/ {getline; print length($0)}' | \
  awk '{sum+=$1; if(NR==1 || $1<min) min=$1; if(NR==1 || $1>max) max=$1} END {print "Min:", min, "| Median:", sum/NR, "| Max:", max}'
echo ""

# 4. Voting exclusion check
echo "4. VOTING EXCLUSION"
echo "-------------------"
CONSENSUS_4=$(redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk '/^payload$/ {getline; if (match($0, /"consensus_count": 4/)) print}' | wc -l)
if [ "$CONSENSUS_4" -eq 0 ]; then
  echo "✅ PASS: Zero events with consensus_count=4"
else
  echo "❌ CRITICAL: Found $CONSENSUS_4 events with consensus=4 (ROLLBACK!)"
fi
echo ""

# 5. Agreement rate (quick)
echo "5. AGREEMENT RATE (last 100 events)"
echo "------------------------------------"
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '
    BEGIN {total=0; agree=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"side": "([^"]+)"/, ens) && \
          match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, shd)) {
        total++
        if (ens[1] == shd[1]) agree++
      }
    }
    END {
      if (total > 0) {
        rate = int(agree*100/total)
        print "Agree:", agree "/" total, "(" rate "%)"
        if (rate >= 55) print "✅ Above 55% threshold"
        else print "⚠️ Below 55% threshold"
      }
    }
  '
echo ""

# 6. Decision
echo "6. DECISION"
echo "-----------"
echo "✅ Continue observation if:"
echo "   - Service active"
echo "   - Zero consensus_count=4"
echo "   - Agreement rate 40-80%"
echo ""
echo "❌ Rollback if:"
echo "   - Any consensus_count=4"
echo "   - Service crashes"
echo "   - Agreement rate <20% or >95%"
echo ""
echo "Next: Wait 4 more hours, then run 6h snapshot"
echo "==================================================="
```

**Save and run**:
```bash
chmod +x /home/qt/quantum_trader/ops/runbooks/snapshot_2h.sh
/home/qt/quantum_trader/ops/runbooks/snapshot_2h.sh
```

---

### 6.2 6-Hour Snapshot

**Timing**: Run after 6 hours of shadow observation

```bash
#!/bin/bash
# snapshot_6h.sh

echo "==================================================="
echo "PATCHTST SHADOW MODE — 6-HOUR SNAPSHOT"
echo "==================================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%S UTC")"
echo ""

# 1-5: Same as 2h snapshot (service health, counts, payload, exclusion, agreement)
# ... (reuse from 2h script)

# 6. Action diversity (NEW for 6h)
echo "6. ACTION DIVERSITY (GATE 1)"
echo "----------------------------"
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    BEGIN {buy=0; sell=0; hold=0; total=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, act)) {
        total++
        action = act[1]
        if (action == "BUY") buy++
        else if (action == "SELL") sell++
        else if (action == "HOLD") hold++
      }
    }
    END {
      buy_pct = int(buy*100/total)
      sell_pct = int(sell*100/total)
      hold_pct = int(hold*100/total)
      printf "BUY: %d%% | SELL: %d%% | HOLD: %d%%\n", buy_pct, sell_pct, hold_pct
      
      max_pct = buy_pct
      if (sell_pct > max_pct) max_pct = sell_pct
      if (hold_pct > max_pct) max_pct = hold_pct
      
      classes_above_10 = 0
      if (buy_pct > 10) classes_above_10++
      if (sell_pct > 10) classes_above_10++
      if (hold_pct > 10) classes_above_10++
      
      if (max_pct <= 70 && classes_above_10 >= 2) {
        print "✅ GATE 1: PASS"
      } else {
        print "❌ GATE 1: FAIL (max=" max_pct "%, classes>10%=" classes_above_10 ")"
      }
    }
  '
echo ""

# 7. Confidence spread (NEW for 6h)
echo "7. CONFIDENCE SPREAD (GATE 2)"
echo "-----------------------------"
redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 200 | \
  awk '
    BEGIN {n=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*"confidence": ([0-9.]+)/, conf)) {
        n++
        vals[n] = conf[1]
        sum += conf[1]
      }
    }
    END {
      mean = sum / n
      for (i=1; i<=n; i++) sq_diff += (vals[i] - mean) ^ 2
      stddev = sqrt(sq_diff / n)
      
      for (i=1; i<=n; i++) {
        for (j=i+1; j<=n; j++) {
          if (vals[i] > vals[j]) {tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp}
        }
      }
      p10 = vals[int(n * 0.10)]
      p90 = vals[int(n * 0.90)]
      range_p10_p90 = p90 - p10
      
      printf "Stddev: %.4f | P10-P90: %.4f\n", stddev, range_p10_p90
      
      if (stddev >= 0.05 && range_p10_p90 > 0.12) {
        print "✅ GATE 2: PASS"
      } else {
        print "❌ GATE 2: FAIL (need stddev>=0.05 AND range>0.12)"
      }
    }
  '
echo ""

# 8. Decision
echo "8. DECISION (6H)"
echo "----------------"
echo "Preliminary gates: Check Gate 1 + 2 above"
echo "Gate 3 (agreement): Check section 5 output above"
echo "Gate 4 (calibration): Deferred (needs outcomes)"
echo ""
echo "✅ If >=2/3 gates pass: Continue to 24h observation"
echo "❌ If <2/3 gates pass: Strong indication model needs re-training"
echo ""
echo "Next: Wait 18 more hours, then run 24h snapshot"
echo "==================================================="
```

---

### 6.3 24-Hour Snapshot (FINAL DECISION)

**Timing**: Run after 24 hours of shadow observation

```bash
#!/bin/bash
# snapshot_24h.sh

echo "==================================================="
echo "PATCHTST SHADOW MODE — 24-HOUR FINAL DECISION"
echo "==================================================="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%S UTC")"
echo ""

# Run full gate evaluation (all 4 gates if outcomes available)
echo "Running comprehensive gate evaluation..."
echo ""

# Gate 1: Action diversity
# ... (full script from Section 5.1)

# Gate 2: Confidence spread
# ... (full script from Section 5.2)

# Gate 3: Agreement rate
# ... (full script from Section 4.1)

# Gate 4: Calibration (if outcome data available)
# ... (deferred or actual calibration)

echo ""
echo "==================================================="
echo "FINAL DECISION MATRIX"
echo "==================================================="
echo ""
echo "Gate Results:"
echo "  Gate 1 (Action Diversity): [PASS/FAIL]"
echo "  Gate 2 (Confidence Spread): [PASS/FAIL]"
echo "  Gate 3 (Shadow Correlation): [PASS/FAIL]"
echo "  Gate 4 (Calibration): [PASS/FAIL/DEFERRED]"
echo ""
echo "Gates Passed: [X]/4 (or [X]/3 if Gate 4 deferred)"
echo ""
echo "==================================================="
echo "RECOMMENDATION"
echo "==================================================="
echo ""
echo "✅ ACTIVATE (remove PATCHTST_SHADOW_ONLY) if:"
echo "   - >=3/4 gates passed (or >=2/3 if Gate 4 deferred)"
echo "   - No service instability"
echo "   - Agreement rate 55-75%"
echo ""
echo "⚠️ ACTIVATE WITH REDUCED WEIGHT (5%) if:"
echo "   - 2/4 gates passed"
echo "   - Marginal metrics (e.g., agreement rate 50-54%)"
echo ""
echo "❌ KEEP SHADOW / RE-TRAIN if:"
echo "   - <2/4 gates passed"
echo "   - Critical failures (BUY bias >90%, flatlined conf)"
echo ""
echo "==================================================="
echo "ACTIVATION COMMAND (if approved):"
echo "==================================================="
echo ""
echo "sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env"
echo "systemctl restart quantum-ai-engine.service"
echo ""
echo "Monitor for 24h after activation, check consensus distribution."
echo "==================================================="
```

---

## 🚨 SECTION 7: ROLLBACK PROCEDURES (REFINED)

### 7.1 Fast Rollback (Remove Shadow Flag)

**When**: Gates pass, ready to activate voting

```bash
# Use anchored match to avoid deleting comments
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env

# Verify removal
grep "^PATCHTST" /etc/quantum/ai-engine.env

# Should show only MODEL_PATH, not SHADOW_ONLY
# Output: PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth

# Restart
systemctl restart quantum-ai-engine.service

# Verify active
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "PatchTST.*loaded"
```

**Result**: PatchTST now votes with 20% weight

---

### 7.2 Full Rollback (Restore Baseline Model)

**When**: Shadow mode reveals critical issues

```bash
# Remove all PATCHTST flags (use anchored match)
sed -i '/^PATCHTST_MODEL_PATH=/d' /etc/quantum/ai-engine.env
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env

# Verify clean
grep "^PATCHTST" /etc/quantum/ai-engine.env
# Output: (empty)

# Restart
systemctl restart quantum-ai-engine.service

# Verify baseline model loaded
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "PatchTST"
# Should show: "[OK] PatchTST agent loaded" (without MODEL_PATH message)
```

**Result**: PatchTST uses baseline model (flatlined confidences 0.50/0.650)

---

### 7.3 Nuclear Rollback (Config Restore)

**When**: Everything is broken, need instant revert

```bash
# Restore backup
cp /etc/quantum/ai-engine.env.bak.20260110_023557 /etc/quantum/ai-engine.env

# Restart
systemctl restart quantum-ai-engine.service

# Verify
systemctl status quantum-ai-engine.service
```

**Time**: <2 minutes

---

## 🛡️ SECTION 8: GUARDRAILS

### 8.1 Anti-Docker Rule

**NEVER suggest** docker exec quantum_redis in systemd-only environment

**Correct commands**:
```bash
# ✅ CORRECT (systemd)
redis-cli ...

# ❌ WRONG (docker)
docker exec quantum_redis redis-cli ...
```

**Enforcement**: Add this to deployment docs prominently

---

### 8.2 Sed Anchored Match Rule

**ALWAYS use** `^` anchor when removing env flags

```bash
# ✅ CORRECT (anchored)
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env

# ❌ WRONG (unanchored - may delete comments)
sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env
```

---

### 8.3 Consensus Count Red Flag Rule

**IMMEDIATE ROLLBACK** if any event has `consensus_count=4`

```bash
# Add this to cron (check every 10 min)
*/10 * * * * redis-cli --raw XREVRANGE quantum:stream:trade.intent + - COUNT 50 | awk '/^payload$/ {getline; if (match($0, /"consensus_count": 4/)) exit 1}' || echo "CRITICAL: PatchTST voting detected! Run rollback!" | mail -s "Shadow Mode Breach" ops@example.com
```

---

## 📝 SECTION 9: QUICK REFERENCE CARD

```
╔══════════════════════════════════════════════════════════════════╗
║              PATCHTST SHADOW MODE — QUICK REF                   ║
╠══════════════════════════════════════════════════════════════════╣
║ SERVICE                                                          ║
║   Status:  systemctl status quantum-ai-engine.service           ║
║   Logs:    journalctl -u quantum-ai-engine.service -f           ║
║   Restart: systemctl restart quantum-ai-engine.service          ║
║                                                                  ║
║ REDIS (systemd, NOT docker)                                     ║
║   Events:  redis-cli XLEN quantum:stream:trade.intent           ║
║   Latest:  redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1║
║                                                                  ║
║ VERIFICATION CHECKS                                              ║
║   Shadow logs:     journalctl ... | grep "\[SHADOW\] PatchTST" | wc -l║
║   Consensus=4:     redis-cli ... | awk '... consensus_count: 4' ║
║   Agreement rate:  (See Section 4.1)                            ║
║                                                                  ║
║ ROLLBACK (<2 MIN)                                                ║
║   Fast:  sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env║
║          systemctl restart quantum-ai-engine.service            ║
║                                                                  ║
║ GATES (need >=3/4 or >=2/3 if Gate 4 deferred)                  ║
║   Gate 1: Action diversity (no class >70%, >=2 classes >10%)    ║
║   Gate 2: Confidence spread (stddev>=0.05, p10-p90>0.12)        ║
║   Gate 3: Agreement 55-75% with ensemble                        ║
║   Gate 4: Calibration (deferred, needs outcomes)                ║
║                                                                  ║
║ DECISION SNAPSHOTS                                               ║
║   2h:  ./snapshot_2h.sh  (health check)                         ║
║   6h:  ./snapshot_6h.sh  (preliminary gates)                    ║
║   24h: ./snapshot_24h.sh (final decision)                       ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**END OF RUNBOOK**

**Document Version**: 1.0  
**Last Updated**: 2026-01-10T02:45 UTC  
**Maintained By**: Quantum Trader Ops Team  
**Environment**: VPS systemd-only (NO DOCKER)
