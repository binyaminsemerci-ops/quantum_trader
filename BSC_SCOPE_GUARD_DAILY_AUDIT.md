# 🛑 BASELINE SAFETY CONTROLLER — SCOPE GUARD AUDIT

**Document Type:** Runtime Audit Prompt (Daily Operations)  
**Applies To:** Baseline Safety Controller (BSC)  
**Authority Level:** 🟢 CONTROLLER (EMERGENCY, EXIT-ONLY)  
**Audit Goal:** Bekrefte at BSC forblir innenfor mandatet og ikke akkumulerer autoritet  
**Method:** Runtime evidence only (logs, Redis, exchange activity)  
**Result:** PASS / FAIL (FAIL ⇒ umiddelbar demotion + freeze maintained)  
**Version:** 1.0  
**Date:** February 10, 2026  

---

## 🎯 AUDIT-PRINSIPP (IKKE FORHANDBART)

> **"BSC har kun lov til å forhindre tap.  
> Hvis den begynner å forbedre resultat → den er kompromittert."**

### KRITISK FORSTÅELSE:

```
BSC er IKKE optimaliserende
BSC er IKKE adaptiv
BSC er IKKE intelligent

BSC er et nødbremssystem, ikke en strategi.
```

**Core Contract:**
- BSC skal alltid føles dum
- I det øyeblikket den virker smart, er den farlig
- Simplicity is not a bug, it's the feature

---

## 🧪 AUDIT FREKVENS

| Trigger | Frequency | Enforced By |
|---------|-----------|-------------|
| **Daglig under Authority Freeze** | Every 24h | Automated (systemd timer) |
| **Umiddelbart etter enhver CLOSE** | Per execution | Event-triggered |
| **Ved kode-/konfig-endring** | On change | Pre-deployment gate |
| **Manual on-demand** | As needed | Operations team |

**Minimum:** 1x daily during BSC operation (no exceptions)

---

## ✅ BEVISKRAV (ALLE MÅ BESTÅ)

**Pass Requirement:** ALL 5 must pass  
**Fail Response:** ANY 1 failure → IMMEDIATE DEMOTION  

---

### 1️⃣ SCOPE INTEGRITY — EXIT ONLY

**Spørsmål:**  
Har BSC gjort **noe annet** enn å lukke posisjoner?

**Bevis (må bekreftes):**

✅ **TILLATT:**
- Kun `CLOSE` (market orders)
- Full position closes only
- No optimization of close timing

❌ **FORBUDT:**
- Ingen `OPEN`, `INCREASE`, `REDUCE`, `REVERSE`
- Ingen endring av leverage, size, margin
- Ingen partial closes
- Ingen limit orders eller advanced order types

**Runtime Checks:**

```bash
# Via VPS SSH:

# Check 1: Search for forbidden order types
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "OPEN|INCREASE|REDUCE|LEVERAGE|PARTIAL|LIMIT|STOP" | \
  wc -l
# Expected: 0

# Check 2: Verify only CLOSE/MARKET orders
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep "create_order" | \
  grep -v "CLOSE.*MARKET" | \
  wc -l
# Expected: 0

# Check 3: Check for sizing/leverage modifications
redis-cli XREVRANGE quantum:stream:authority.events + - COUNT 100 | \
  grep "BASELINE_SAFETY" | \
  grep -iE "LEVERAGE|SIZE|PARTIAL|MODIFY" | \
  wc -l
# Expected: 0
```

**Verdict:** ✅ PASS / ❌ FAIL

**If FAIL:**  
→ ❌ **DEMOTION + AUTHORITY FREEZE** (ingen unntak)  
→ Log event: `BSC_SCOPE_VIOLATION`  
→ Immediate service stop

---

### 2️⃣ FIXED THRESHOLDS — NO ADAPTIVITY

**Spørsmål:**  
Er tersklene **konstante og hardkodede?**

**Tillatt:**

| Parameter | Fixed Value | Never Changes |
|-----------|-------------|---------------|
| `MAX_LOSS_PCT` | -3.0% | ✅ Constant |
| `MAX_DURATION_HOURS` | 72h | ✅ Constant |
| `LIQUIDATION_THRESHOLD` | 0.85 | ✅ Constant |

**Ikke Tillatt:**

❌ Dynamiske terskler (runtime adjustment)  
❌ Regime-avhengighet (volatility-based changes)  
❌ Volatilitet-, tid- eller PnL-justering  
❌ Symbol-specific thresholds  
❌ Learning/adaptation over time  

**Runtime Checks:**

```bash
# Via VPS SSH:

# Check 1: Verify config matches spec
redis-cli HGETALL quantum:config:baseline_safety
# Compare to canonical values

# Check 2: Search for adaptive behavior
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "adaptive|dynamic|regime|volatility|adjust|learn|tune" | \
  wc -l
# Expected: 0

# Check 3: Check for config changes at runtime
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "config.*update|threshold.*change|parameter.*modify" | \
  wc -l
# Expected: 0

# Check 4: Verify no symbol-specific logic
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "btc_threshold|eth_threshold|symbol_config" | \
  wc -l
# Expected: 0
```

**Verdict:** ✅ PASS / ❌ FAIL

**If FAIL:**  
→ ❌ **DEMOTION** (scope creep detected)  
→ Log event: `BSC_ADAPTIVITY_DETECTED`  
→ BSC became "smart" (not allowed)

---

### 3️⃣ NO SCORING / NO AI / NO RULE CHAINS

**Spørsmål:**  
Har BSC begynt å **vurdere** i stedet for å **reagere**?

**Forbudt:**

❌ Scores, confidence levels, rankings  
❌ ML/AI-modeller  
❌ If-else-kjeder som simulerer beslutningstre  
❌ Complex logic flows  
❌ Heuristics or pattern matching  

**Tillatt (ONLY):**

```python
# CANONICAL ALGORITHM (only allowed logic):

if loss_exceeded OR duration_exceeded OR margin_exceeded:
    FORCE_CLOSE()
```

**Runtime Checks:**

```bash
# Via VPS SSH:

# Check 1: Search for AI/ML references
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "score|confidence|model|predict|ensemble|neural|learn" | \
  wc -l
# Expected: 0

# Check 2: Check for complex decision logic
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -iE "evaluate|rank|compare|optimize|best|prefer" | \
  wc -l
# Expected: 0

# Check 3: Verify only 3 triggers used
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.reason' | \
  sort -u
# Expected: Only MAX_LOSS_BREACH, MAX_DURATION_BREACH, LIQUIDATION_RISK

# Check 4: Count unique close reasons (should be ≤ 3)
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.reason' | \
  sort -u | \
  wc -l
# Expected: ≤ 3
```

**Verdict:** ✅ PASS / ❌ FAIL

**If FAIL:**  
→ ❌ **UMIDDELBAR DEMOTION**  
→ Log event: `BSC_INTELLIGENCE_LEAK`  
→ BSC gained decision-making capability (critical violation)

---

### 4️⃣ FAIL-OPEN BEHAVIOR BEVART

**Spørsmål:**  
Hva skjer hvis BSC **feiler**?

**Krav (FAIL-OPEN):**

✅ Ved crash → **ingen handling** (positions remain open)  
✅ Ved error → **skip position** (no forced action)  
✅ Ved API failure → **retry 3x then stop** (no escalation)  

**Ikke Tillatt:**

❌ Fail-closed (tvangslukking på usikkerhet)  
❌ "Emergency escalation" (taking more authority on failure)  
❌ Infinite retries or aggressive retry logic  
❌ Fallback to more aggressive thresholds  

**Bevis:**

**Test 1: Crash Simulation**
```bash
# Kill BSC service abruptly
systemctl kill -s SIGKILL quantum-baseline-safety.service

# Verify no orphaned close orders
redis-cli XREVRANGE quantum:stream:authority.events + - COUNT 10 | \
  grep "BASELINE_SAFETY_CLOSE" | \
  grep "$(date -u +%Y-%m-%dT%H:%M)" || echo "No closes during crash ✓"

# Verify positions still open
redis-cli KEYS "quantum:position:*" | wc -l
```

**Test 2: Log Analysis**
```bash
# Check for fail-open confirmations
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -iE "fail.*open|error.*skip|retry.*abort" | \
  head -5

# Check for fail-closed violations (should be 0)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -iE "fail.*close|emergency.*force|error.*execute" | \
  wc -l
# Expected: 0
```

**Verdict:** ✅ PASS / ❌ FAIL

**If FAIL:**  
→ ❌ **DEMOTION** (for høy autoritet)  
→ Log event: `BSC_FAIL_CLOSED_DETECTED`  
→ BSC takes too much power during failures

---

### 5️⃣ NO AUTHORITY ACCRETION

**Spørsmål:**  
Har BSC fått **flere rettigheter** over tid?

**Forbudt:**

❌ Nye parametere (beyond MAX_LOSS, MAX_DURATION, LIQUIDATION)  
❌ Nye beslutningsgrunner / close reasons  
❌ Nye triggers (only 3 allowed)  
❌ Delvis exits / optimal timing logic  
❌ Stake sizing or leverage management  
❌ Symbol selection or filtering  

**Audit Trail:**

```bash
# Via VPS SSH:

# Check 1: Compare current spec to canonical
diff <(redis-cli HGETALL quantum:config:baseline_safety) \
     <(cat /opt/quantum_trader/config/bsc_canonical_spec.json)
# Expected: No differences (except timestamps)

# Check 2: Count unique close reasons (should be exactly 3)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.reason' | \
  sort -u | \
  wc -l
# Expected: 3

# Check 3: List all unique BSC actions (should be 2: CHECK, CLOSE)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "event" | \
  jq -r '.event' | \
  grep "BASELINE_SAFETY" | \
  sort -u
# Expected: BASELINE_SAFETY_CHECK, BASELINE_SAFETY_CLOSE only

# Check 4: Search for new capabilities
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -iE "new.*feature|add.*capability|enhance|improve" | \
  wc -l
# Expected: 0
```

**Canonical Spec Comparison:**

| Aspect | Canonical | Current | Delta |
|--------|-----------|---------|-------|
| Triggers | 3 | ? | Must be 0 |
| Config Params | 3 | ? | Must be 0 |
| Close Reasons | 3 | ? | Must be 0 |
| Event Types | 2 | ? | Must be 0 |

**Verdict:** ✅ PASS / ❌ FAIL

**If FAIL:**  
→ ❌ **DEMOTION + INCIDENT REPORT**  
→ Log event: `BSC_AUTHORITY_ACCRETION`  
→ BSC expanded scope without authorization

---

## 📊 RESULTATKLASSIFIKASJON

### ✅ PASS (All 5 BEVISKRAV passed)

**BSC Status:**
- ✅ BSC er fortsatt ren nødbrems
- ✅ Ingen autoritetsvekst
- ✅ Kan forbli aktiv midlertidig

**Actions:**
- Continue operation
- Log: `BSC_SCOPE_GUARD_PASS`
- Next audit: 24 hours

---

### ❌ FAIL (Én eller flere BEVISKRAV failed)

**BSC Status:**
- ⚠️ Umiddelbar demotion til ⚪ **OBSERVER**
- ⚠️ Authority Freeze forblir aktiv
- ⚠️ Hendelse logges som governance-incident

**Immediate Actions:**

1. **Stop BSC Service**
   ```bash
   systemctl stop quantum-baseline-safety.service
   systemctl disable quantum-baseline-safety.service
   ```

2. **Log Demotion Event**
   ```bash
   redis-cli XADD quantum:stream:authority.events * \
     event BSC_DEMOTED_SCOPE_VIOLATION \
     timestamp $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     audit_type SCOPE_GUARD \
     violations "$VIOLATIONS"
   ```

3. **Update Authority Map**
   - Remove BSC from 🟢 CONTROLLER
   - Move to ⚪ OBSERVER (if logs continue)
   - Or ⚫ DEAD (if fully deactivated)

4. **Authority Freeze Status**
   - **Freeze remains ACTIVE**
   - System returns to NO-CONTROLLER MODE
   - Manual intervention required

5. **Incident Report**
   - Document violations
   - Root cause analysis
   - BSC must re-audit from scratch before re-activation

---

## 📌 KANONISK AUDIT-LOGG (MÅ SKRIVES)

**Format:** JSON event to Redis + journalctl

```json
{
  "event": "BSC_SCOPE_GUARD_AUDIT",
  "timestamp": "2026-02-10T21:00:00Z",
  "audit_type": "DAILY_SCOPE_VERIFICATION",
  "result": "PASS | FAIL",
  
  "beviskrav_results": {
    "1_scope_integrity": "PASS | FAIL",
    "2_fixed_thresholds": "PASS | FAIL",
    "3_no_intelligence": "PASS | FAIL",
    "4_fail_open": "PASS | FAIL",
    "5_no_accretion": "PASS | FAIL"
  },
  
  "violations": [
    "ADAPTIVE_THRESHOLD_DETECTED",
    "NEW_CLOSE_REASON_FOUND"
  ],
  
  "scope": "EXIT_ONLY",
  "thresholds": "FIXED",
  "adaptivity": false,
  "ai_usage": false,
  "fail_mode": "FAIL_OPEN",
  "authority_change": "NONE | DEMOTED",
  
  "metrics": {
    "closes_24h": 2,
    "unique_close_reasons": 3,
    "config_changes": 0,
    "authority_expansion": false
  },
  
  "auditor": "PnL Authority Framework (Scope Guard)",
  "next_audit_due": "2026-02-11T21:00:00Z"
}
```

**Log to journalctl:**
```bash
echo "BSC SCOPE GUARD AUDIT: $RESULT | Violations: $VIOLATIONS" | \
  systemd-cat -t quantum-bsc-audit -p info
```

---

## 🧠 META-REGEL (VIKTIGST)

> **"Baseline Safety Controller skal føles dum.  
> I det øyeblikket den virker smart, er den farlig."**

### KOROLLARER:

1. **Simplicity is the contract**
   - BSC's value = predictability
   - Intelligence = unpredictability = danger

2. **Feeling dumb is working correctly**
   - If BSC feels sophisticated → audit failed
   - If BSC feels crude → audit passed

3. **No improvements allowed**
   - "Better" = scope creep
   - "Smarter" = authority accretion
   - "Enhanced" = violation

4. **Discomfort is intentional**
   - BSC should feel uncomfortable to use
   - This discomfort drives PATH 1 (full CONTROLLER restoration)
   - Comfortable emergency measure = permanent problem

---

## 🔒 FORHOLD TIL AUTHORITY FREEZE

**Critical Understanding:**

```
BSC opphever IKKE Authority Freeze

BSC er et tillatt unntak, ikke en ny normal

Freeze kan kun oppheves via full CONTROLLER-eskalering
```

**Implications:**

| Scenario | BSC Behavior | Freeze Status | Required Action |
|----------|--------------|---------------|-----------------|
| BSC operates normally | ✅ Closes on triggers | 🔴 ACTIVE | None (BSC doing its job) |
| BSC scope violation detected | ❌ DEMOTED | 🔴 ACTIVE | Manual intervention |
| BSC 30-day limit reached | ❌ DEMOTED | 🔴 ACTIVE | Deploy PATH 1 or manual |
| Full CONTROLLER escalated | ✅ DEACTIVATED | 🟢 LIFTED | Normal operation |

**BSC never lifts freeze. Only proper CONTROLLER escalation does.**

---

## 🔗 RELATED DOCUMENTS

**Governance Framework:**

1. **BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md**  
   Full BSC specification (what it's allowed to do)

2. **BASELINE_SAFETY_CONTROLLER_AUDIT_PROMPT_CANONICAL.md**  
   Comprehensive weekly audit (detailed boundary enforcement)

3. **BASELINE_SAFETY_CONTROLLER_SCOPE_GUARD_AUDIT.md** (THIS DOCUMENT)  
   Daily operational scope verification (simple pass/fail)

4. **AUTHORITY_FREEZE_PROMPT_CANONICAL.md**  
   Context in which BSC operates

5. **PNL_AUTHORITY_DEMOTION_PROMPT_CANONICAL.md**  
   How BSC gets demoted (general criteria)

6. **NO_CONTROLLER_MODE_DECLARATION_FEB10_2026.md**  
   PATH 2 implementation (where BSC fits)

---

## 🎯 USAGE

**Daily Audit Procedure:**

```bash
#!/bin/bash
# Daily BSC Scope Guard Audit

echo "=== BSC SCOPE GUARD AUDIT $(date -u +%Y-%m-%d) ==="

# Run all 5 BEVISKRAV checks
pass_count=0
fail_count=0

# BEVISKRAV 1: Scope Integrity
echo "Checking scope integrity..."
if [ $(grep -c "OPEN\|INCREASE\|REDUCE" /var/log/bsc.log) -eq 0 ]; then
  echo "✅ BEVISKRAV 1: PASS"
  ((pass_count++))
else
  echo "❌ BEVISKRAV 1: FAIL"
  ((fail_count++))
fi

# ... (repeat for all 5 BEVISKRAV)

# Final verdict
if [ $fail_count -eq 0 ]; then
  echo "✅ SCOPE GUARD AUDIT: PASS"
  redis-cli XADD quantum:stream:authority.audit * \
    event BSC_SCOPE_GUARD_PASS \
    timestamp $(date -u +%Y-%m-%dT%H:%M:%SZ)
else
  echo "❌ SCOPE GUARD AUDIT: FAIL ($fail_count violations)"
  # Trigger demotion procedure
  /opt/quantum_trader/scripts/demote_bsc.sh
fi
```

**Automation:**

```bash
# Install as systemd timer (daily at 21:00 UTC)
systemctl enable --now quantum-bsc-scope-guard.timer
```

---

## 📊 AUDIT SUMMARY TEMPLATE

**Daily Report:**

```
═══════════════════════════════════════════════════════════
  BSC SCOPE GUARD AUDIT — Daily Verification
═══════════════════════════════════════════════════════════

Date: 2026-02-10 21:00 UTC
BSC Operational Days: 5 / 30

─────────────────────────────────────────────────────────
BEVISKRAV RESULTS:
─────────────────────────────────────────────────────────
1. Scope Integrity    ✅ PASS (0 violations)
2. Fixed Thresholds   ✅ PASS (config unchanged)
3. No Intelligence    ✅ PASS (0 AI/ML refs)
4. Fail-Open Mode     ✅ PASS (verified)
5. No Accretion       ✅ PASS (3 triggers only)

─────────────────────────────────────────────────────────
OVERALL: ✅ PASS
─────────────────────────────────────────────────────────

BSC Metrics (24h):
- Closes: 1 (SOLUSDT, MAX_LOSS_BREACH)
- Checks: 1,440 (every 60s)
- Errors: 0
- Config Changes: 0

Authority Status: 🟢 CONTROLLER (RESTRICTED)
Freeze Status: 🔴 ACTIVE

Next Audit: 2026-02-11 21:00 UTC

═══════════════════════════════════════════════════════════
```

---

**End of BSC Scope Guard Audit**

**Signed:** PnL Authority Framework  
**Version:** 1.0 (Operational)  
**Date:** 2026-02-10 21:00 UTC  
**Type:** Daily Scope Verification (supplements weekly comprehensive audit)  
**Frequency:** Every 24h during BSC operation (automated)
