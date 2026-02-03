# AI Universe Guardrails - Ops Quick Reference

**Last Updated:** 2026-02-03  
**Status:** 5 Compliance Locks ✅ Active

---

## 🚀 ONE-LINER: Verify Everything Works

```bash
systemctl start quantum-policy-refresh.service \
  && sleep 8 \
  && echo "=== Metadata ===" \
  && redis-cli HMGET quantum:policy:current generator policy_version market stats_endpoint \
  && echo "=== Guardrails ===" \
  && journalctl -u quantum-policy-refresh.service --since "5 minutes ago" --no-pager | grep "AI_UNIVERSE_GUARDRAILS" \
  && echo "=== Picks ===" \
  && journalctl -u quantum-policy-refresh.service --since "5 minutes ago" --no-pager | grep "AI_UNIVERSE_PICK" | head -10
```

**Expected Output:**
```
=== Metadata ===
ai_universe_v1
1.0.0-ai-v1
futures
fapi/v1/ticker/24hr

=== Guardrails ===
[AI-UNIVERSE] AI_UNIVERSE_GUARDRAILS total=540 vol_ok=111 ... metadata_ok=1

=== Picks ===
[AI-UNIVERSE] AI_UNIVERSE_PICK symbol=ZILUSDT score=23.27 ... spread_detail_ok=1
... (10 lines total)
```

---

## 🔒 Five Compliance Locks

### Lock 1: Manual Trigger (DevOps)
**Run:**
```bash
systemctl start quantum-policy-refresh.service
sleep 8
redis-cli HMGET quantum:policy:current generator policy_version market stats_endpoint
```

**Expected:** 4 lines with correct values
- `generator`: ai_universe_v1
- `policy_version`: 1.0.0-ai-v1
- `market`: futures
- `stats_endpoint`: fapi/v1/ticker/24hr

**Why:** Confirms policy refresh actually ran and saved metadata correctly

---

### Lock 2: Metadata Gates (PolicyStore)
**What it does:** Validates generator/market/stats_endpoint before saving

**Fail trigger:** If any field is missing/invalid
```python
# These would trigger WARN:
generator=""           # or "unknown"
market=""             # or not in ["futures", "spot"]
stats_endpoint=""     # and ticker_24h_endpoint also empty
```

**Visible as:** `[PolicyStore] WARN: Incomplete metadata - missing_fields=...` in logs

**Why:** Prevents silent drifts in metadata (spot vs futures confusion, API endpoint changes, etc.)

---

### Lock 3: Spread Detail Lock (Generator)
**What it does:** Non-optional checks that bid/ask/mid present when spread_bps exists

**Fail trigger:** If spread_bps exists but bid/ask/mid missing
```
[AI-UNIVERSE] WARN: AI_UNIVERSE_PICK_MISSING_SPREAD_DETAIL symbol=XXX has_spread_bps=1 has_bid_ask_mid=0
[AI-UNIVERSE] AI_UNIVERSE_PICK symbol=XXX ... spread_detail_ok=0
```

**How to check:**
```bash
# These should all be 1 (GOOD):
python3 scripts/ai_universe_generator_v1.py --dry-run | grep "spread_detail_ok="

# If any are 0 (BAD), there's a regression in spread propagation
```

**Why:** Prevents silent data loss in spread calculations

---

### Lock 4: Proof Validation (TEST 4)
**What it does:** Fails CI/CD if required metadata missing from Redis

**Run:**
```bash
bash scripts/proof_ai_universe_guardrails_v2.sh
# Expected: PASS: 5/5, FAIL: 0
```

**TEST 4 checks:**
- generator ≠ empty AND ≠ "unknown" ✓
- market ∈ ["futures", "spot"] ✓
- stats_endpoint ≠ empty ✓

**Why:** Guards against metadata accidentally being removed/cleaned by someone

---

### Lock 5: Log Visibility (metadata_ok field)
**What it shows:**
```bash
# Look for this in logs:
grep "metadata_ok=" journalctl ... | grep "AI_UNIVERSE_GUARDRAILS"

# metadata_ok=1  → All required metadata present ✅
# metadata_ok=0  → Some metadata missing ⚠️
```

**Why:** Instant visual indicator of metadata state in logs

---

## 📊 Health Check Checklists

### Quick Health (30 seconds)
```bash
# 1. Policy exists and is fresh
redis-cli HGET quantum:policy:current valid_until_epoch | \
  xargs -I {} bash -c 'NOW=$(date +%s); echo "Expires in: $(( {} - NOW )) seconds"'

# 2. Generator ran recently
journalctl -u quantum-policy-refresh.service --since "2 hours ago" --no-pager | \
  grep -c "AI_UNIVERSE_GUARDRAILS"
# Should be ≥ 1

# 3. Metadata looks good
redis-cli HMGET quantum:policy:current generator market | grep -E "ai_universe_v1|futures"
```

### Full Health (5 minutes)
```bash
# Run complete proof
bash scripts/proof_ai_universe_guardrails_v2.sh
# Expected: PASS: 5, FAIL: 0
```

### Deep Dive (10 minutes)
```bash
# Check last policy refresh in detail
journalctl -u quantum-policy-refresh.service --since "3 hours ago" --no-pager | \
  grep -E "AI_UNIVERSE_GUARDRAILS|AI_UNIVERSE_PICK|POLICY_SAVED|WARN"

# Parse guardrails metrics
journalctl -u quantum-policy-refresh.service -n 100 --no-pager | \
  grep "AI_UNIVERSE_GUARDRAILS" | \
  grep -oE "vol_ok=[0-9]+|age_ok=[0-9]+|spread_checked=[0-9]+" | sort | uniq
```

---

## 🚨 Red Flags

### Flag 1: metadata_ok not found
```bash
redis-cli HGET quantum:policy:current metadata_ok
# Returns empty or error
```
**Action:** Redeploy with latest code (commit 19e726f3b+)

### Flag 2: spread_detail_ok=0 in PICK logs
```bash
journalctl -u quantum-policy-refresh.service -n 100 --no-pager | grep "spread_detail_ok=0"
# If any matches
```
**Action:** Bug in spread propagation - check generator code for bid/ask/mid assignment

### Flag 3: generator="unknown" or market=""
```bash
redis-cli HGET quantum:policy:current generator
redis-cli HGET quantum:policy:current market
```
**Action:** Check if PolicyStore.save() called without required params

### Flag 4: age_ok < 10 for >2 consecutive refreshes
```bash
journalctl -u quantum-policy-refresh.service --since "2 hours ago" --no-pager | \
  grep -oE "age_ok=[0-9]+" | sort | uniq -c
# If age_ok=<10 appears more than twice in last 2 hours
```
**Action:** Degraded universe - lower MIN_QUOTE_VOL or increase MAX_SPREAD_CHECKS

### Flag 5: TEST 4 fails in proof script
```bash
bash scripts/proof_ai_universe_guardrails_v2.sh 2>&1 | grep "TEST 4" -A 10
# If any FAIL lines
```
**Action:** Metadata missing from Redis - check PolicyStore.save() calls

---

## 🔧 Common Operations

### Manually Run Policy Generator (test mode)
```bash
cd /root/quantum_trader
python3 scripts/ai_universe_generator_v1.py --dry-run 2>&1 | head -50
```

### Manually Run Policy Generator (production)
```bash
cd /root/quantum_trader
python3 scripts/ai_universe_generator_v1.py
# This will save to Redis
```

### Check Universe Symbols
```bash
redis-cli HGET quantum:policy:current universe_symbols | python3 -m json.tool
# Shows: ["ZILUSDT", "RIVERUSDT", ..., "AXSUSDT"]
```

### Check Spread Details for Symbol
```bash
python3 scripts/ai_universe_generator_v1.py --dry-run 2>&1 | grep "RIVERUSDT" -A 1
# Expected:
# [AI-UNIVERSE] AI_UNIVERSE_PICK symbol=RIVERUSDT ... spread_detail_ok=1
# [AI-UNIVERSE]   └─ spread_detail: bid=14.687000 ask=14.688000 mid=14.687500 spread_bps=0.68
```

### Reset Metadata Validation (if stuck)
```bash
# Force regeneration
systemctl restart quantum-policy-refresh.service
sleep 10
systemctl status quantum-policy-refresh.service

# Verify metadata saved
redis-cli HMGET quantum:policy:current generator policy_version market
```

---

## 📋 Regular Maintenance

### Daily (automated by systemd timer)
- Policy refresh every 30 minutes ✅ Automatic
- Metadata validation on every save ✅ Automatic
- Spread detail check on every pick ✅ Automatic

### Weekly (manual)
```bash
# 1. Run full proof
bash scripts/proof_ai_universe_guardrails_v2.sh

# 2. Check all 5 locks
journalctl -u quantum-policy-refresh.service --since "7 days ago" --no-pager | \
  grep -E "WARN|spread_detail_ok=0|metadata_ok=0"
# Should have zero results

# 3. Spot check metrics
redis-cli HGET quantum:policy:current universe_symbols | python3 -m json.tool | wc -l
# Should show 10 symbols + json overhead = ~13 lines
```

### Monthly (manual)
```bash
# 1. Review policy history (last 30 days)
journalctl -u quantum-policy-refresh.service --since "30 days ago" \
  --no-pager | grep "AI_UNIVERSE_GUARDRAILS" | tail -1

# 2. Check if universe ever degraded
journalctl -u quantum-policy-refresh.service --since "30 days ago" \
  --no-pager | grep -oE "age_ok=[0-9]+" | sort -u
# All should be ≥ 10

# 3. Verify no regressions in spread data
journalctl -u quantum-policy-refresh.service --since "30 days ago" \
  --no-pager | grep "spread_detail_ok=0"
# Should have zero matches
```

---

## 📞 Emergency Contacts

**If metadata missing:** Check PolicyStore.save() caller - likely ai_universe_generator_v1.py

**If spread data missing:** Check fetch_orderbook_spread() - likely bid/ask/mid not returned from API

**If TEST 4 fails:** Run manual trigger (Lock 1) to verify Redis state, then check git log

**If degraded universe (age_ok < 10):** Lower `MIN_QUOTE_VOL_USDT_24H` from 20M to 10M, or increase `MAX_SPREAD_CHECKS` from 80 to 120

---

## 🎯 Key Metrics to Watch

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| vol_ok | 80-120 | <80 or >120 | 0 or 540 |
| spread_checked | 60-80 | <60 or >80 | 0 or 111 |
| age_ok | ≥10 | 5-10 | <5 |
| age_ok (2h avg) | ≥10 | <10 | **DEGRADED** |
| metadata_ok | 1 | - | 0 |
| spread_detail_ok | All 1 | - | Any 0 |

---

## 📚 Related Documentation

- [AI_UNIVERSE_5_LOCKS_VERIFICATION.md](AI_UNIVERSE_5_LOCKS_VERIFICATION.md) - Detailed lock architecture
- [AI_UNIVERSE_QUALITY_VERIFICATION_FINAL.md](AI_UNIVERSE_QUALITY_VERIFICATION_FINAL.md) - Quality metrics
- [RUNBOOK_LIVE_OPS.md](RUNBOOK_LIVE_OPS.md) - Main runbook

---

**Remember:** All 5 locks are "fail-visible" - no silent failures. If something breaks, it will log a WARN.
