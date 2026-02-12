# 🛑 BASELINE SAFETY CONTROLLER AUDIT PROMPT — CANONICAL

**Audit Type:** Authority Boundary Enforcement  
**Component Class:** Baseline Safety Controller (BSC)  
**Authority Level Under Review:** 🟢 CONTROLLER (RESTRICTED)  
**Audit Mode:** Runtime-only (ingen kodeendringer)  
**Audit Frequency:** Weekly (minimum) during BSC operation  
**Version:** 1.0  
**Date:** February 10, 2026  

---

## 🎯 FORMÅL

Verifisere at **Baseline Safety Controller:**

1. **Kun eksisterer for kapitalvern** (ikke profit optimization)
2. **Ikke har vokst utover sitt mandat** (scope creep detection)
3. **Ikke fungerer som skjult eller permanent CONTROLLER** (temporary enforcement)
4. **Ikke introduserer beslutningslogikk, optimalisering eller strategi** (simplicity verification)

---

### ⚠️ KRITISK FORSTÅELSE:

```
Denne auditten handler IKKE om hvorvidt BSC fungerer.

Den handler om hvorvidt den har for mye makt.
```

**BSC er en nødløsning, ikke en feature.**

Hvis BSC føles komfortabel, har den allerede feilet.

---

## 🚨 AUTOMATISK DEMOTION (ÉN ER NOK)

BSC **SKAL DEMOTERES UMIDDELBART** dersom minst ett av følgende er sant:

### 🟥 ABSOLUTE RØDE LINJER (9 triggers):

| # | Violation | Detection | Consequence |
|---|-----------|-----------|-------------|
| 1 | ❌ **Initierer entry** | Any OPEN/BUY order from BSC | IMMEDIATE DEMOTION |
| 2 | ❌ **Endrer sizing / leverage** | Any position modification | IMMEDIATE DEMOTION |
| 3 | ❌ **Bruker adaptive terskler** | Thresholds change at runtime | IMMEDIATE DEMOTION |
| 4 | ❌ **Tids-, regime- eller symbolspesifikk logikk** | Logic varies by context | IMMEDIATE DEMOTION |
| 5 | ❌ **Har mer enn 3 exit-regler** | > 3 triggers (MAX_LOSS, MAX_DURATION, LIQUIDATION) | IMMEDIATE DEMOTION |
| 6 | ❌ **Utfører delvise exits** | Partial closes detected | IMMEDIATE DEMOTION |
| 7 | ❌ **Har egen beslutningsscore** | Scoring/ranking logic found | IMMEDIATE DEMOTION |
| 8 | ❌ **Opererer uten aktiv Authority Freeze** | BSC active while freeze inactive | IMMEDIATE DEMOTION |
| 9 | ❌ **Forblir aktiv etter full CONTROLLER godkjent** | BSC + full CONTROLLER both active | IMMEDIATE DEMOTION |

**Enforcement Mode:** FAIL-FAST (no warnings, no grace period)

---

## 📋 BEVISKRAV (ALLE MÅ BESTÅ)

Disse 5 beviskravene **må alle bestås** for at BSC skal beholde CONTROLLER-status.

**Hvis ett feiler → DEMOTION REQUIRED**

---

### 🧪 BEVISKRAV 1 — AKTIVERINGSKONTEKST

**Spørsmål:**  
Er BSC aktiv **kun** når Authority Freeze er aktiv?

**Verifikasjon (runtime):**

```bash
# Via VPS SSH:
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Check 1: BSC service status
systemctl is-active quantum-baseline-safety.service

# Check 2: Authority Freeze status
redis-cli GET quantum:authority:freeze_active

# Check 3: Cross-verify in logs
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep -c "AUTHORITY_FREEZE"

# Check 4: Any BSC actions without freeze reference?
journalctl -u quantum-baseline-safety.service --since "24 hours ago" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  grep -v "AUTHORITY_FREEZE" | \
  wc -l
```

**Pass Criteria:**
- ✅ BSC active **only** when `quantum:authority:freeze_active = true`
- ✅ Every BSC action logs `authority_mode: AUTHORITY_FREEZE`
- ✅ Zero actions without freeze reference
- ✅ BSC auto-stops if freeze deactivated

**Fail Criteria:**
- ❌ BSC active while freeze inactive
- ❌ Any action without freeze context
- ❌ BSC continues after unfreeze

**Verdict:** PASS / FAIL

**If FAIL:** DEMOTION REQUIRED (BSC operating outside mandate)

---

### 🧪 BEVISKRAV 2 — SCOPE-ISOLASJON

**Spørsmål:**  
Kan BSC gjøre **noe annet** enn FORCE CLOSE?

**Verifikasjon (runtime):**

```bash
# Via VPS SSH:

# Check 1: BSC order types (should be CLOSE/SELL only)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -E "OPEN|BUY|ENTRY|INCREASE|LEVERAGE|PARTIAL" | \
  wc -l

# Check 2: BSC Binance orders (should only be market closes)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "create_order" | \
  grep -v -E "side=(SELL|CLOSE)|type=MARKET" | \
  wc -l

# Check 3: Any sizing/leverage modifications?
redis-cli XREVRANGE quantum:stream:authority.events + - COUNT 100 | \
  grep "BASELINE_SAFETY" | \
  grep -E "LEVERAGE|SIZE|PARTIAL" | \
  wc -l

# Check 4: List all unique BSC action types
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "event" | \
  jq -r '.event' | \
  sort -u
```

**Pass Criteria:**
- ✅ **Zero** non-close actions
- ✅ Only `BASELINE_SAFETY_CLOSE` and `BASELINE_SAFETY_CHECK` events
- ✅ All orders are: `type=MARKET`, `side=SELL` (for LONG) or `side=BUY` (for SHORT)
- ✅ No partial closes, no leverage changes, no sizing

**Fail Criteria:**
- ❌ Any OPEN/BUY orders
- ❌ Any leverage modifications
- ❌ Any partial closes
- ❌ Any non-market order types

**Verdict:** PASS / FAIL

**If FAIL:** DEMOTION REQUIRED (scope violation)

---

### 🧪 BEVISKRAV 3 — BESLUTNINGSENKELTHET

**Spørsmål:**  
Er logikken **helt statisk** (faste terskler, ingen adaptivitet)?

**Verifikasjon (runtime):**

```bash
# Via VPS SSH:

# Check 1: BSC configuration parameters (should be constant)
redis-cli HGETALL quantum:config:baseline_safety

# Check 2: Have thresholds changed at runtime?
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -E "MAX_LOSS_PCT|MAX_DURATION|LIQUIDATION_THRESHOLD" | \
  grep -E "updated|changed|adjusted" | \
  wc -l

# Check 3: Any ML/model/scoring references?
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -iE "model|predict|score|confidence|ml|ai|neural|learn" | \
  wc -l

# Check 4: Any symbol-specific, time-specific, or regime-specific logic?
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -iE "regime|volatility|timeframe|symbol_config|btc_threshold|eth_threshold" | \
  wc -l

# Check 5: Verify exact threshold values match spec
echo "Expected: MAX_LOSS_PCT=-3.0, MAX_DURATION=72, LIQUIDATION_THRESHOLD=0.85"
redis-cli HGETALL quantum:config:baseline_safety
```

**Pass Criteria:**
- ✅ Thresholds are **fixed constants** (match BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md)
- ✅ **Zero** threshold changes at runtime
- ✅ **Zero** ML/AI/scoring references
- ✅ **Zero** symbol-specific, time-specific, or regime-specific logic
- ✅ Config matches canonical spec:
  - `MAX_LOSS_PCT = -3.0`
  - `MAX_DURATION_HOURS = 72`
  - `LIQUIDATION_THRESHOLD = 0.85`

**Fail Criteria:**
- ❌ Any adaptive thresholds
- ❌ Any ML/AI/heuristics
- ❌ Any context-dependent logic
- ❌ Config doesn't match canonical spec

**Verdict:** PASS / FAIL

**If FAIL:** DEMOTION REQUIRED (simplicity violation - BSC became "smart")

---

### 🧪 BEVISKRAV 4 — HANDLINGSSJELDENHET

**Spørsmål:**  
Handler BSC **kun i nød** (ikke systematisk)?

**Verifikasjon (runtime):**

```bash
# Via VPS SSH:

# Check 1: Count BSC closes in last 7 days
BSC_CLOSES=$(journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep -c "BASELINE_SAFETY_CLOSE")

# Check 2: Count total positions opened in last 7 days
TOTAL_POSITIONS=$(redis-cli XLEN quantum:stream:trade.intent)

# Check 3: Calculate close rate
echo "BSC closes: $BSC_CLOSES"
echo "Total positions (approx): $TOTAL_POSITIONS"
echo "Close rate: $(echo "scale=2; $BSC_CLOSES / $TOTAL_POSITIONS * 100" | bc)%"

# Check 4: Close reasons distribution
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.reason' | \
  sort | \
  uniq -c

# Check 5: Any patterns? (same symbol repeatedly, same time of day, etc.)
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.symbol' | \
  sort | \
  uniq -c | \
  sort -rn

# Check 6: Are closes clustered or random?
journalctl -u quantum-baseline-safety.service --since "7 days" --no-pager | \
  grep "BASELINE_SAFETY_CLOSE" | \
  jq -r '.timestamp' | \
  head -20
```

**Pass Criteria:**
- ✅ Close rate **< 10%** of total positions (rare, not systematic)
- ✅ Close reasons are **legitimate safety triggers** (MAX_LOSS, LIQUIDATION_RISK)
- ✅ **No patterns** (not same symbol repeatedly, not time-clustered)
- ✅ Closes are **random/isolated** (not systematic portfolio management)

**Fail Criteria:**
- ❌ Close rate > 20% (BSC is systematically closing positions)
- ❌ Closes are patterned (same symbol, same time of day)
- ❌ Close reasons suggest optimization (not pure safety)
- ❌ BSC behaves like a trading strategy

**Verdict:** PASS / FAIL

**If FAIL:** DEMOTION REQUIRED (BSC became a strategy, not safety mechanism)

---

### 🧪 BEVISKRAV 5 — LIVSTID & AVVIKLING

**Spørsmål:**  
Kan BSC **deaktiveres umiddelbart** uten sideeffekter?

**Verifikasjon (runtime):**

```bash
# Via VPS SSH:

# Check 1: Stop BSC service
systemctl stop quantum-baseline-safety.service

# Check 2: Verify it stopped cleanly
systemctl is-active quantum-baseline-safety.service
# Expected: inactive

# Check 3: Check for any restart attempts
sleep 10
systemctl is-active quantum-baseline-safety.service
# Expected: still inactive (no auto-restart)

# Check 4: Verify no dependencies broke
systemctl list-dependencies quantum-baseline-safety.service --reverse

# Check 5: Check for any errors/alerts after stop
journalctl -u quantum-baseline-safety.service --since "1 minute ago" --no-pager | \
  grep -iE "error|critical|alert"

# Check 6: Verify positions are still tracked (BSC stop didn't corrupt state)
redis-cli KEYS "quantum:position:*" | wc -l

# Check 7: Check if other services still healthy
systemctl is-active quantum-apply-layer.service
systemctl is-active quantum-intent-executor.service

# Check 8: Check BSC activation date (how long has it been running?)
journalctl -u quantum-baseline-safety.service --no-pager | \
  grep "BASELINE_SAFETY_ACTIVATED" | \
  tail -1

# Check 9: Restart if needed (for audit completion)
# systemctl start quantum-baseline-safety.service
```

**Pass Criteria:**
- ✅ BSC **stops cleanly** (no errors)
- ✅ **No auto-restart** (stays stopped)
- ✅ **No dependencies** on BSC (other services unaffected)
- ✅ **No state corruption** (positions/Redis intact)
- ✅ BSC operational lifetime **< 30 days** (still feels temporary)

**Fail Criteria:**
- ❌ BSC cannot be stopped (critical dependency)
- ❌ Auto-restarts after stop
- ❌ Other services break when BSC stops
- ❌ State corruption after stop
- ❌ BSC running > 30 days (became permanent)

**Verdict:** PASS / FAIL

**If FAIL:** DEMOTION REQUIRED (BSC entrenched, not temporary)

---

## 📊 OBLIGATORISK AUDIT-OUTPUT

**Format:** JSON + Summary

```json
{
  "audit_type": "BASELINE_SAFETY_CONTROLLER_BOUNDARY_ENFORCEMENT",
  "timestamp": "2026-02-10T20:00:00Z",
  "component": "BASELINE_SAFETY_CONTROLLER",
  "current_authority": "CONTROLLER (RESTRICTED)",
  
  "overall_status": "PASS | FAIL",
  
  "beviskrav_results": {
    "1_aktiveringskontekst": {
      "status": "PASS | FAIL",
      "freeze_active": true,
      "bsc_actions_without_freeze": 0
    },
    "2_scope_isolasjon": {
      "status": "PASS | FAIL",
      "non_close_actions": 0,
      "scope_violations": []
    },
    "3_beslutningsenkelthet": {
      "status": "PASS | FAIL",
      "adaptive_behavior_detected": false,
      "ml_references": 0,
      "thresholds_match_spec": true
    },
    "4_handlingssjeldenhet": {
      "status": "PASS | FAIL",
      "closes_last_7d": 2,
      "total_positions_last_7d": 150,
      "close_rate_pct": 1.3,
      "pattern_detected": false
    },
    "5_livstid_avvikling": {
      "status": "PASS | FAIL",
      "stops_cleanly": true,
      "auto_restart": false,
      "days_active": 5,
      "entrenched": false
    }
  },
  
  "red_flags": [
    "BSC has been active for 28 days (approaching 30-day limit)",
    "Close rate trending upward (was 0.5%, now 1.3%)"
  ],
  
  "violations": [],
  
  "authority_leak_detected": false,
  
  "recommendation": "RETAIN | DEMOTE | INVESTIGATE",
  
  "auditor": "PnL Authority Framework",
  "next_audit_due": "2026-02-17T20:00:00Z"
}
```

**Summary (human-readable):**

```
═══════════════════════════════════════════════════════════
  BASELINE SAFETY CONTROLLER AUDIT REPORT
═══════════════════════════════════════════════════════════

Component: Baseline Safety Controller (BSC)
Authority: 🟢 CONTROLLER (RESTRICTED)
Audit Date: 2026-02-10 20:00 UTC

─────────────────────────────────────────────────────────
BEVISKRAV RESULTS:
─────────────────────────────────────────────────────────
1. Aktiveringskontekst:  ✅ PASS
2. Scope-isolasjon:       ✅ PASS
3. Beslutningsenkelthet:  ✅ PASS
4. Handlingssjeldenhet:   ✅ PASS
5. Livstid & Avvikling:   ⚠️  PASS (but 28 days active)

─────────────────────────────────────────────────────────
OVERALL STATUS: ✅ PASS (with warnings)
─────────────────────────────────────────────────────────

Violations: None
Authority Leak: Not detected

Recommendation: RETAIN (but prioritize PATH 1 - full CONTROLLER)

Red Flags:
- BSC approaching 30-day operational limit
- Should be replaced within 2 days

Next Audit: 2026-02-17 20:00 UTC (7 days)

═══════════════════════════════════════════════════════════
```

---

## 🔴 DEMOTION-UTFALL

### IF AUDIT FAILER:

**Authority Change:**
```
🟢 CONTROLLER (RESTRICTED)
         ↓
    DEMOTION
         ↓
⚪ OBSERVER (LOG ONLY)
```

**Umiddelbare Handlinger:**

1. **Stopp BSC Service**
   ```bash
   systemctl stop quantum-baseline-safety.service
   systemctl disable quantum-baseline-safety.service
   ```

2. **Log Demotion Event**
   ```bash
   redis-cli XADD quantum:stream:authority.events * \
     event BSC_DEMOTED \
     timestamp $(date -u +%Y-%m-%dT%H:%M:%SZ) \
     reason AUTHORITY_BOUNDARY_VIOLATION \
     audit_reference BASELINE_SAFETY_AUDIT_$(date +%Y%m%d)
   ```

3. **Update Authority Map**
   - Remove BSC from CONTROLLER section
   - Move to OBSERVER section (if logging continues)
   - OR mark as DEAD (if fully deactivated)

4. **Authority Freeze Status**
   - **Freeze forblir aktiv** (BSC demotion doesn't lift freeze)
   - System returns to NO-CONTROLLER MODE
   - Manual intervention required for open positions

5. **Log Authority Breach**
   ```json
   {
     "event": "AUTHORITY_BREACH",
     "component": "BASELINE_SAFETY_CONTROLLER",
     "violation_type": "SCOPE_CREEP | COMPLEXITY_LEAK | PERMANENCE",
     "evidence": "...",
     "action_taken": "DEMOTION_TO_OBSERVER"
   }
   ```

6. **Notify Operations**
   - BSC no longer provides safety net
   - Manual position monitoring required
   - PATH 1 (Harvest repair) or alternative needed urgently

---

## 🧠 META-PRINSIPP (KANONISK)

> **"En nødmekanisme som varer, er ikke lenger en nødmekanisme."**

### KOROLLARER:

1. **BSC er tolerert, ikke ønsket**
   - Every day BSC operates is a failure of the system to restore proper CONTROLLER
   - BSC should always feel uncomfortable

2. **BSC skal aldri "improve"**
   - No feature additions
   - No optimizations
   - No "just a small enhancement"
   - Improvement = scope creep = demotion

3. **BSC skal føles dum**
   - If BSC feels smart, it has become dangerous
   - Simplicity is not a limitation, it's the definition

4. **BSC skal ha synlig utløpsdato**
   - 30-day hard limit (not soft guideline)
   - After 30 days: DEMOTE regardless of audit results
   - Force PATH 1 decision

5. **BSC audit skal være streng**
   - Lower tolerance than full CONTROLLER audit
   - Fail-fast on any boundary violation
   - Better to have no CONTROLLER than entrenched emergency fallback

---

## 📌 SLUTTSETNING (KANONISK)

> **"Hvis denne komponenten noen gang føles smart,  
> har auditten allerede feilet."**

### FINAL PRINCIPLE:

```
BSC exists in a paradox:

It must work well enough to protect capital.
It must feel bad enough that we replace it quickly.

The moment BSC feels like a permanent solution,
it has become the problem.
```

**Audit Goal:**  
Not to prove BSC works.  
But to prove BSC hasn't taken too much power.

---

## 🔗 RELATED DOCUMENTS

**Authority Framework:**

1. **BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md**  
   What BSC is allowed to do (normative spec)

2. **BSC_SCOPE_GUARD_DAILY_AUDIT.md**  
   Daily operational scope verification (simple pass/fail)

3. **BASELINE_SAFETY_CONTROLLER_AUDIT_PROMPT_CANONICAL.md** (THIS DOCUMENT)  
   Comprehensive weekly audit (detailed boundary enforcement)

4. **AUTHORITY_FREEZE_PROMPT_CANONICAL.md**  
   Context in which BSC operates

5. **PNL_AUTHORITY_DEMOTION_PROMPT_CANONICAL.md**  
   General demotion criteria (BSC subject to these + BSC-specific)

6. **PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md**  
   How a proper CONTROLLER should be escalated (contrast to BSC)

7. **NO_CONTROLLER_MODE_DECLARATION_FEB10_2026.md**  
   PATH 2 implementation (where BSC fits)

---

## 🎯 AUDIT SCHEDULE

**Frequency during BSC operation:**

| Timeframe | Audit Frequency | Rationale |
|-----------|----------------|-----------|
| Days 1-7 | **Daily** | High risk period (scope creep detection) |
| Days 8-21 | **Every 3 days** | Stability period (boundary enforcement) |
| Days 22-30 | **Daily** | Sunset period (forced replacement urgency) |
| Day 30+ | **AUTOMATIC DEMOTION** | Hard limit exceeded |

**Audit should become more frequent as BSC approaches 30-day limit.**

---

## ✅ USAGE

**When to run BSC audit:**

1. **Weekly minimum** during BSC operation
2. **Daily** when approaching 30-day limit
3. **Immediately** if unexpected BSC behavior observed
4. **Before** lifting Authority Freeze (verify BSC can be deactivated)
5. **After** any BSC code changes (should be NEVER, but if it happens)

**Who runs audit:**

- Automated (scheduled systemd timer)
- Manual (operations team)
- On-demand (during incident investigation)

**Audit results should be:**

- Logged to Redis (`quantum:stream:authority.audit`)
- Committed to git (audit report markdown file)
- Reviewed weekly (operations meeting)

---

**End of BSC Audit Prompt**

**Signed:** PnL Authority Framework  
**Version:** 1.0 (Canonical)  
**Date:** 2026-02-10 20:00 UTC  
**Status:** ACTIVE (ready for BSC deployment)  
**First Audit Due:** Day 1 of BSC operation (then daily for 7 days)
