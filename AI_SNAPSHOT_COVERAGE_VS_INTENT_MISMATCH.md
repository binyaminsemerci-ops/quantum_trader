# Snapshot Coverage vs AI Intent Mismatch - Analysis

**Date**: 2026-02-02  
**Status**: 🔴 ROOT CAUSE IDENTIFIED  
**Priority**: CRITICAL (blocks all AI trading)

---

## 🎯 Executive Summary

**Snapshot expansion worked perfectly** (2 → 33 snapshots), but we're snapshotting the **wrong 33 symbols**.

- ✅ `no_exchange_snapshot` dropped from 56.7% → 0% (SOLVED)
- ❌ AI intents blocked: symbols not in snapshot set
- ❌ No EXECUTE plans reaching execution (0 in last 5 min)

**Root Cause**: Top-K selection uses alphabetical sort → snapshots `1000BONKUSDT`, `0GUSDT` instead of AI's targets (`GALAUSDT`, `WAVESUSDT`, `MKRUSDT`)

---

## 📊 GO/NO-GO Checklist Results

### ✅ 1. P3.5 Gate Distribution (PASS)

```
Total Decisions: 14 (last 5 min)

kill_score_warning_risk_increase    8  (57.1%)
symbol_not_in_allowlist:WAVESUSDT   4  (28.6%)
none                                2  (14.3%)
```

**Result**: `no_exchange_snapshot` **ELIMINATED** from top gates! Snapshot fix worked.

### ✅ 2. Symbols Missing Snapshots (PASS - but wrong symbols)

```
P3.3_SNAPSHOT_MISS: 0 (last 20 minutes)
```

**Result**: Zero snapshot misses = candidates cover all P3.3 evaluations.  
**But**: Evaluations are for alphabetically-first symbols, not AI's intent symbols.

### 🔴 3. EXECUTE Plans & Executions (FAIL)

```
EXECUTE plans (last 5 min): 0
executed=true (last 5 min): 0

P3.5 decision counts:
  BLOCKED: 12
  EXECUTE: 2  (found in counts but not in stream - likely old)
```

**Result**: No trading activity. AI intents blocked before reaching execution.

### ✅ 4. P3.5 Reason Distribution (INFORMATIVE)

```
Top reasons (5 min):
  kill_score_warning_risk_increase: 8  (Governor blocking exits)
  symbol_not_in_allowlist:WAVESUSDT: 4  (testnet issue)
  none: 2  (success)
```

**Result**: Real gates visible now (not infrastructure issues).

### 🔴 5. Tradable Symbols vs Intent Symbols (ROOT CAUSE)

**AI Intent Symbols** (last 5 messages, 3 min ago):
- GALAUSDT (SELL, 0.539 confidence)
- WAVESUSDT (BUY, 0.510 confidence) ← blocked by allowlist
- MKRUSDT (SELL, 0.534 confidence)
- VETUSDT (SELL, 0.543 confidence)
- HBARUSDT (SELL, 0.554 confidence)

**Symbols WITH Snapshots** (33 total):
```
0GUSDT, 1000000BOBUSDT, 1000000MOGUSDT, 1000BONKUSDC, 1000BONKUSDT,
1000CATUSDT, 1000CHEEMSUSDT, 1000FLOKIUSDT, 1000LUNCUSDT, 1000PEPEUSDC,
1000PEPEUSDT, 1000RATSUSDT, 1000SATSUSDT, 1000SHIBUSDC, 1000SHIBUSDT,
1000WHYUSDT, 1000XECUSDT, 1000XUSDT, 1INCHUSDT, 1MBABYDOGEUSDT,
42USDT, 4USDT, A2ZUSDT, AAVEUSDC, AAVEUSDT, ACEUSDT, ACHUSDT,
ACTUSDT, ACUUSDT, ACXUSDT, BTCUSDT, ETHUSDT, TRXUSDT
```

**Overlap**: Only BTCUSDT, ETHUSDT, TRXUSDT (3 out of 33)

**Result**: 
- ❌ GALAUSDT has snapshot? **NO** (G comes after ACXUSDT alphabetically)
- ❌ MKRUSDT has snapshot? **NO** (M comes after TRXUSDT)
- ❌ VETUSDT has snapshot? **NO** (V not in top-33)
- ❌ HBARUSDT has snapshot? **NO** (H not in top-33)
- ⚠️ WAVESUSDT? In P3.3 env allowlist but not in testnet universe

---

## 🔍 Technical Analysis

### Current Snapshot Logic

**File**: `microservices/position_state_brain/main.py`  
**Method**: `_build_snapshot_candidates()`

```python
# 3) Top-K from universe (formula-based)
top_k = max(
    self.config.MIN_TOP_K,
    int(self.config.TARGET_CYCLE_SEC / (self.ewma_latency_ms / 1000))
)

# Get top-K symbols alphabetically (deterministic)
sorted_allowlist = sorted(list(self.allowlist))  # ← PROBLEM: alphabetical
for symbol in sorted_allowlist[:top_k]:
    candidates.add(symbol)
```

**Problem**: Alphabetical sort prioritizes:
- `0GUSDT` (rank 1)
- `1000BONKUSDT` (rank 5)
- `42USDT` (rank 21)

Over:
- `GALAUSDT` (rank ~200)
- `MKRUSDT` (rank ~350)
- `VETUSDT` (rank ~500)

### Universe vs Environment Allowlist

**P3.3 Environment** (`/etc/quantum/position-state-brain.env`):
```bash
P33_ALLOWLIST=BTCUSDT,ETHUSDT,WAVESUSDT,TRXUSDT,MANAUSDT,ZECUSDT,GALAUSDT,FILUSDT,CRVUSDT,DOTUSDT,APTUSDT,SEIUSDT
```
12 symbols (curated for testnet)

**Runtime Allowlist** (from Universe Service):
```
source=universe, count=566
```
566 symbols (full production universe)

**Behavior**: Universe Service overrides environment allowlist via 3-tier fallback:
1. `quantum:cfg:universe:active` (566 symbols) ✅ USED
2. `quantum:cfg:universe:last_ok` (fallback)
3. Environment variable (final fallback)

**Impact**: P3.3 snapshots random 33 symbols from 566, ignoring the curated 12-symbol testnet list.

---

## 🎯 Solution Options

### Option A: Use Environment Allowlist (FAST, LOW RISK)

**Change**: Disable Universe Service for P3.3 testnet

**Implementation**:
```bash
# /etc/quantum/position-state-brain.env
P33_ALLOWLIST_SOURCE=env  # ← Change from "universe" to "env"
```

**Result**:
- P3.3 snapshots only 12 curated symbols (GALAUSDT, ZECUSDT, FILUSDT, etc.)
- All AI intents covered (AI likely targets these symbols)
- Cycle time: <1 second (12 symbols vs 33)

**Pros**:
- ✅ Immediate fix (1 line change + restart)
- ✅ Testnet-appropriate (curated symbols)
- ✅ AI intents align with allowlist

**Cons**:
- ⚠️ Loses production-readiness (Universe Service unused)
- ⚠️ Manual maintenance (add symbols to env file)

### Option B: Volume-Based Top-K (CORRECT, SLOWER)

**Change**: Replace alphabetical sort with volume/activity ranking

**Implementation**:
```python
# Load 24h volume from Universe metadata
universe_data = json.loads(self.redis.get("quantum:cfg:universe:active"))
volumes = {s['symbol']: s.get('volume_24h', 0) for s in universe_data.get('symbols', [])}

# Sort by volume descending
sorted_allowlist = sorted(
    list(self.allowlist), 
    key=lambda s: volumes.get(s, 0), 
    reverse=True
)
```

**Result**:
- Snapshots high-volume symbols (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
- AI likely targets liquid markets
- Dynamic ranking (adapts to market conditions)

**Pros**:
- ✅ Intelligent ranking (volume ≈ tradability)
- ✅ Production-ready
- ✅ Uses Universe Service metadata

**Cons**:
- ⚠️ Requires Universe Service schema change (add volume_24h)
- ⚠️ Implementation time: 30-60 minutes

### Option C: Publish Tradable Symbols Set (USER'S SUGGESTION)

**Change**: P3.3 publishes `quantum:cfg:tradable_symbols` for AI Engine to consume

**Implementation**:
```python
# In P3.3 after snapshot cycle:
tradable = list(candidates)  # Symbols with snapshots
self.redis.sadd("quantum:cfg:tradable_symbols", *tradable)
self.redis.expire("quantum:cfg:tradable_symbols", 3600)
```

**Result**:
- AI Engine filters intents to only tradable symbols
- No wasted evaluations on unsnapshot symbols
- Feedback loop: P3.3 ← AI ← P3.3

**Pros**:
- ✅ Closes the loop (AI knows what's tradable)
- ✅ Reduces noise (no intents for dead symbols)
- ✅ Works with any ranking strategy

**Cons**:
- ⚠️ Requires AI Engine modification
- ⚠️ Doesn't solve ranking problem (still alphabetical)

### Option D: Intent-Based Ranking (HYBRID - RECOMMENDED)

**Change**: Prioritize symbols from recent `trade.intent` stream in top-K

**Implementation**:
```python
# 3) Top-K: prioritize recent intents, backfill with volume/alphabetical
intent_symbols = {symbol for symbol in recent_intent_symbols if symbol in self.allowlist}
remaining_k = max(0, top_k - len(intent_symbols))

if remaining_k > 0:
    # Backfill with volume-ranked or alphabetical
    backfill = [s for s in sorted_allowlist if s not in intent_symbols][:remaining_k]
    candidates.update(backfill)
```

**Result**:
- Snapshots exactly what AI is trying to trade (reactive)
- Backfills with stable symbols (proactive)
- No wasted snapshots on symbols AI doesn't want

**Pros**:
- ✅ Aligns with AI behavior (intent-driven)
- ✅ Simple implementation (already have recent_intents)
- ✅ Self-correcting (AI changes → snapshots follow)

**Cons**:
- ⚠️ Cold start problem (no intents → alphabetical)
- ⚠️ May miss opportunities (AI doesn't know what's possible)

---

## 📌 Recommended Action Plan

### Immediate (Tonight - 5 minutes)

**Option A**: Switch to environment allowlist

```bash
# VPS
echo 'P33_ALLOWLIST_SOURCE=env' >> /etc/quantum/position-state-brain.env
systemctl restart quantum-position-state-brain

# Verify
journalctl -u quantum-position-state-brain --since "1 minute ago" | grep "allowlist:"
# Expected: source=env, count=12

# Wait 2 minutes for snapshot cycle
redis-cli --scan --pattern "quantum:position:snapshot:*" | grep -E "(GALA|ZEC|FIL|MANA|CRV)"
# Expected: all 12 env symbols present
```

**Impact**:
- ✅ GALAUSDT, ZECUSDT, FILUSDT intents will have snapshots
- ✅ Entries can start flowing (if Governor permits)
- ✅ Testnet-appropriate symbol set

### Short-Term (This Week - 1 hour)

**Option D**: Implement intent-based ranking

```python
# Priority 1: Recent intent symbols (30 min window)
intent_symbols = set(recent_intent_symbols) & self.allowlist
candidates.update(intent_symbols)

# Priority 2: Open positions
candidates.update(open_positions)

# Priority 3: Backfill to MIN_TOP_K
remaining = self.config.MIN_TOP_K - len(candidates)
if remaining > 0:
    backfill = [s for s in sorted(self.allowlist) if s not in candidates][:remaining]
    candidates.update(backfill)
```

**Impact**:
- ✅ Snapshots follow AI's actual trading targets
- ✅ No manual allowlist maintenance
- ✅ Works in production with full universe

### Medium-Term (Next Sprint - 4 hours)

**Option C**: Publish tradable symbols set

```python
# P3.3: After snapshot cycle
tradable_set = list(candidates)
self.redis.delete("quantum:cfg:tradable_symbols")
self.redis.sadd("quantum:cfg:tradable_symbols", *tradable_set)
self.redis.expire("quantum:cfg:tradable_symbols", 300)  # 5 min TTL

# AI Engine: Filter intents
tradable = redis.smembers("quantum:cfg:tradable_symbols")
if tradable and symbol not in tradable:
    logger.debug(f"{symbol}: Skip intent (not in tradable set)")
    continue
```

**Impact**:
- ✅ AI stops wasting cycles on untradable symbols
- ✅ Feedback loop established
- ✅ System self-optimizes

### Long-Term (Production - 8 hours)

**Option B**: Volume-based ranking + Universe schema enhancement

**Steps**:
1. Add `volume_24h` to Universe Service schema
2. Update Universe writer to populate volume from Binance API
3. Modify P3.3 to use volume ranking
4. Add config: `P33_SNAPSHOT_RANKING=volume|intent|alphabetical`

**Impact**:
- ✅ Production-ready ranking (volume = liquidity)
- ✅ Configurable strategy per environment
- ✅ Aligns with real-world trading priorities

---

## 🎯 Immediate Next Command (DO THIS NOW)

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'echo "" && echo "P33_ALLOWLIST_SOURCE=env" >> /etc/quantum/position-state-brain.env && systemctl restart quantum-position-state-brain && sleep 5 && echo "=== Verifying ===" && journalctl -u quantum-position-state-brain --since "10 seconds ago" | grep -E "(BUILD_TAG|allowlist:)" && echo "" && echo "Wait 2 minutes for snapshot cycle, then check:" && echo "redis-cli --scan --pattern \"quantum:position:snapshot:*\" | grep -E \"(GALA|ZEC|FIL|MANA|CRV)\""'
```

**Expected**:
- `source=env, count=12`
- After 2 min: GALAUSDT, ZECUSDT, FILUSDT snapshots present
- AI intents start flowing to P3.3 → P3.5 → Intent Executor

---

## 📊 Success Metrics (After Fix)

**Immediate (5 minutes)**:
- ✅ Snapshot count: 12 (down from 33, but **correct** 12)
- ✅ GALAUSDT snapshot exists
- ✅ ZECUSDT snapshot exists
- ✅ No `P3.3_SNAPSHOT_MISS` for env symbols

**Short-Term (15 minutes)**:
- ✅ P3.3 ALLOW decisions for env symbols
- ✅ P3.5 EXECUTE decisions (if Governor permits)
- ✅ `executed=true` in apply.result (if sizing/exchange OK)

**Medium-Term (1 hour)**:
- ✅ Open positions appear (if market conditions + Governor permit)
- ✅ Gate distribution stabilizes (kill_score, cooldown, capital limits)
- ✅ No infrastructure gates (no_exchange_snapshot eliminated)

---

## 🔗 Related Issues

1. **WAVESUSDT allowlist** (28.6% of denials)
   - Symbol in P3.3 env allowlist
   - Not in testnet universe (Binance testnet limitation)
   - **Fix**: Remove from P3.3_ALLOWLIST or add to testnet if available

2. **kill_score_warning_risk_increase** (57.1% of denials)
   - Governor blocking exit evaluations (testnet mode behavior)
   - Not a bug - working as designed
   - **Action**: Monitor, may need Governor tuning for testnet

3. **AI fallback-trend-following model** (all recent intents)
   - Confidence: 0.51-0.55 (marginal)
   - Using fallback strategy (not production models)
   - **Action**: Verify AI Engine model loading

---

## 📝 Key Learnings

### 1. Deterministic != Correct

**Mistake**: "Alphabetically sorted for determinism"  
**Reality**: Deterministic but useless ranking

**Lesson**: Ranking strategy must align with business logic (tradability, volume, intent frequency)

### 2. Integration Testing Required

**Mistake**: Tested snapshot expansion in isolation  
**Reality**: Snapshots exist but for wrong symbols

**Lesson**: End-to-end test: AI intent → snapshot → permit → execution

### 3. Environment vs Service Config

**Mistake**: Assumed environment allowlist used in production  
**Reality**: Universe Service overrides with 566 symbols

**Lesson**: Document config precedence clearly (3-tier fallback not obvious)

### 4. Monitoring Gaps

**Current**: Snapshot count, cycle time, EWMA latency  
**Missing**: Snapshot coverage rate (% of intents with snapshots)

**Fix**: Add metric:
```python
self.metric_snapshot_coverage = Gauge(
    'p33_snapshot_coverage_pct',
    'Percentage of intent symbols with snapshots'
)
```

---

## 🏆 Status

**Snapshot Expansion**: ✅ WORKING  
**Snapshot Coverage**: 🔴 WRONG SYMBOLS  
**Entry Flow**: 🔴 BLOCKED (zero coverage of AI intents)

**Next Step**: Run immediate fix command above to switch to env allowlist.

---

**Analysis Time**: 2026-02-02 00:10:00  
**Priority**: P0 (blocks all trading)  
**Owner**: Requires config change + restart  
**ETA**: 5 minutes
