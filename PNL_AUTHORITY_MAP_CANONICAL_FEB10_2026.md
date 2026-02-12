# 🧾 PnL AUTHORITY MAP — FINAL, CANONICAL

**System**: quantum_trader  
**Scope**: Realized PnL i produksjon  
**Kilde**: Runtime-bevis (systemd, Redis, logs, execution-paths)  
**Audit Period**: February 8-10, 2026  
**Last Updated**: February 10, 2026 22:45 UTC (BSC Technical Verification)  
**Gyldighet**: Nåværende produksjonstilstand  

---

## 🟢 Nivå 0 — FAKTISKE PnL-KONTROLLERE (AUTORITET)

**Komponenter som direkte kan endre realized PnL**

### � RESTRICTED CONTROLLER MODE (ACTIVE)

**Status per Feb 10, 2026 22:45 UTC:**

```
██████╗ ███████╗ ██████╗
██╔══██╗██╔════╝██╔════╝
██████╔╝███████╗██║     
██╔══██╗╚════██║██║     
██████╔╝███████║╚██████╗
╚═════╝ ╚══════╝ ╚═════╝

BASELINE SAFETY CONTROLLER
EXIT-ONLY EMERGENCY MODE
```

**Current CONTROLLER:**
- **Baseline Safety Controller (BSC)** - CONTROLLER (RESTRICTED)
- Deployment: A2.2 Emergency Fallback (2026-02-10 22:04 UTC)
- Verification: `BSC_VERIFICATION_REPORT_FEB10_2026.md`
- Authority: Exit-only, fixed thresholds, 30-day sunset
- Status: ✅ TECHNICALLY VERIFIED | ⚠️ OPERATIONALLY BLOCKED (API access)
- Canonical: `BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md`

**DEMOTED (Former CONTROLLER):**
- **Harvest Proposal** - DEMOTED to OBSERVER (2026-02-10 18:54 UTC)
- Audit: `HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md`
- Reason: Execution pipeline catastrophically broken
- Evidence: Harvest Brain consumer dead 2+ days (161K+ event lag)

**Current Reality:**
- ✅ BSC service running (41min uptime, 0 crashes)
- ⚠️ BSC operationally blocked (Binance API error -2015)
- ✅ 95 poll cycles completed (FAIL OPEN verified)
- ✅ 97 audit events logged (quantum:stream:bsc.events)
- ❌ No automated entry control (BSC exit-only)
- ⚠️ **Open positions require API credentials fix for BSC protection**

**Tillatt aktivitet:** Exit-only (manual close OR BSC when API operational)

**Exit from Restricted Mode:** 
- PATH 1: Repair harvest_brain:execution → restore Harvest Proposal authority
- PATH 2: BSC reaches 30-day sunset → automatic demotion
- PATH 3: New CONTROLLER candidate provides 5 BEVISKRAV

**CRITICAL WARNING:** System operates with ZERO automated PnL control. All trading decisions require manual intervention.

---

## 🟡 Nivå 1 — GATEKEEPERS (INDIREKTE KONTROLL)

**Komponenter som kan hindre eller tillate PnL, men ikke bestemme utfallet**

### 🟡 Universe

**Type**: GATEKEEPER  
**Status**: AKTIV  
**Autoritet**: SYMBOL-ALLOWLIST  

**Bevis**:
- `quantum:cfg:universe:active` → 567 Binance perpetual symbols
- `quantum-universe-service.service` (PID 891810, running 13h)
- Intent Executor enforces allowlist via hard runtime check:
  ```python
  if symbol not in self.allowlist:
      return False  # symbol_not_in_allowlist:{symbol}
  ```
- Refresh rate: Every 60 seconds
- Metadata: `stale=0`, `count=567`, `asof_epoch=1770719138`

**Fallback Behavior** (fail-open):
```python
# If universe unavailable:
return ["BTCUSDT", "ETHUSDT", "TRXUSDT"]  # Hardcoded fallback
```

**Production Usage**:
- Universe provides: **567 symbols** (all Binance perpetuals)
- AI engine scope: **30 symbols** (QT_ALLOWED_SYMBOLS env var)
- Active trading: **BTC + ETH only** (current harvest activity)

**Counterfactual Test**:
- **Scenario**: AI engine generates intent for `ZZZUSDT` (not in universe)
- **Result**: Intent executor blocks with `symbol_not_in_allowlist:ZZZUSDT`
- **PnL Impact**: Order never reaches exchange → position never opened

**Hvis fjernet / krasjer**:  
➡️ 97% av universet forsvinner (567 → 3 symbols)  
➡️ PnL endrer seg indirekte (færre muligheter, ikke smartere beslutninger)

**Konklusjon**: ⚠️ **Har makt, men ingen intelligens** (pure allowlist filter)

---

## 🔵 Nivå 2 — SCORERS (INGEN FUNNET)

**Komponenter som påvirker beslutninger uten å ha veto eller direkte kontroll**

### ❌ Ingen komponenter i denne klassen i dag

**Forklaring**: Ingen ML/AI-komponenter har advisory role uten blocking authority. Alle scorers er enten:
- **Disabled** (RL Agent)
- **Blocked fra execution** (AI Ensemble)
- **Unread output** (AI Exit Evaluator)

---

## ⚪ Nivå 3 — OBSERVERS (TELEMETRI / ILLUSJON)

**Komponenter som ser viktige ut, men ikke påvirker penger**

### ⚪ AI Ensemble Predictor

**Type**: OBSERVER  
**Status**: KJØRER (men irrelevant)  
**Output**: `trade.intent` → Redis Stream  

**Hvorfor OBSERVER**:
- ML-modeller **feiler** (NHiTS/PatchTST return NoneType)
- Consensus threshold **ikke møtt** (1/4 models active)
- Fallback trigger: **RSI/MACD rules** (NOT ML)
- `MANUAL_LANE_OFF` blokkerer **all AI intent execution**
- Metrics: `executed_false=18658`, `executed_true=3948` (only harvest bypasses)

**Bevis fra runtime**:
```json
"model_breakdown": {
  "nhits": {"action": "HOLD", "model": "vunknown"},
  "patchtst": {"action": "HOLD", "model": "vunknown"},
  "fallback": {"action": "SELL", "triggered_by": "rsi_macd_rules"}
}
```

**Hvis fjernet**: ➡️ PnL = identisk (decisions come from fallback rules, not ML)

---

### ⚪ AI Exit Evaluator

**Type**: OBSERVER  
**Status**: KJØRER (produserer output)  
**Output**: `ai.exit.decision` → Redis Stream  

**Hvorfor OBSERVER**:
- Har produsert **39,000+ exit evaluations**
- **INGEN consumers** av `ai.exit.decision` stream
- Aldri lest av Harvest Proposal eller Intent Executor
- Governor consumer group: **lag=39446** (unread backlog)

**Bevis**:
```bash
redis-cli XINFO GROUPS quantum:stream:ai.exit.decision
# Result: governor group has 39446 pending messages (never processed)
```

**Hvis fjernet**: ➡️ PnL = identisk (output ikke konsumert)

---

### ⚪ CLM (SimpleCLM)

**Type**: OBSERVER  
**Status**: DATA STARVATION  
**Output**: Ingen (pure data collection)  

**Hvorfor OBSERVER**:
- Samler `trade.closed` events (currently broken: bytes decoding bug)
- **Har ingen beslutningsmetoder** (no predict(), no adjust_policy())
- Historisk expectancy: **-0.06% avg PnL** (negative)
- Ikke koblet til exits, sizing, eller leverage
- Data corruption: 25+ hours uten nye trades (decoding failure)

**Bevis**:
```python
# SimpleCLM has ZERO control methods:
def collect_feedback()  # ← Data only
def _compute_metrics()  # ← Analysis only
# NO: execute(), adjust(), predict() ❌
```

**Hvis fjernet**: ➡️ PnL = identisk (ingen kontrollmetoder)

---

### ⚪ RL Trainer

**Type**: OBSERVER  
**Status**: OFFLINE TRAINING ONLY  
**Output**: Policy checkpoints (.pt files)  

**Hvorfor OBSERVER**:
- Kjører offline batch training
- **Ingen runtime decision path**
- Output consumed by RL Agent (but RL Agent is DISABLED)
- Training data: Stale (last CLM update 25+ hours ago due to bug)

**Hvis fjernet**: ➡️ PnL = identisk (no runtime influence)

---

### ⚪ Harvest Proposal System

**Type**: OBSERVER (DEMOTED from CONTROLLER)  
**Status**: PRODUCING INTENTS (execution broken)  
**Output**: `harvest.intent` → Redis Stream  
**Demotion Date**: 2026-02-10 18:54 UTC  

**Hvorfor OBSERVER (runtime audit findings)**:
- ✅ Produserer `harvest.intent` events (3,054 total, continuous)
- ✅ Intent executor consumes stream (lag=0, 13 consumers)
- ❌ **Apply Layer publishes `executed=False`** (no actual execution)
- ❌ **Harvest Brain consumer DEAD** (157,777 event lag, stuck since Feb 8)
- ❌ **Execution service: 0 CLOSE orders in 24h**
- ❌ **System-wide: 0 Binance exit orders found**
- ❌ **Phantom closes** (trying to close BTCUSDT/ETHUSDT positions that don't exist)

**Execution Chain Reality**:
```
Harvest Proposal → harvest.intent ✅ (3,054 events)
   ↓
Intent Executor → harvest_executed=971 ⚠️ (FROZEN metrics)
   ↓
Apply Layer → "CLOSE allowed" but executed=False ❌
   ↓
Harvest Brain Consumer → LAG 157,777 ❌ (DEAD 2 days)
   ↓
Execution Service → NO LOGS ❌
   ↓
Binance Exchange → 0 ORDERS ❌
```

**Demotion Criteria Met**:
- **KRITERIUM A**: Execution path broken (decisions never reach exchange)
- **KRITERIUM B**: Ghost controller (system stable without Harvest execution)
- **KRITERIUM E**: Counterfactual collapse (no CLM data, phantom closes)

**Hvis fjernet**:  
➡️ PnL = identisk (already non-functional for 48+ hours)

**Re-escalation requirements**:
1. Repair Harvest Brain consumer (clear 157k lag)
2. Fix Apply Layer execution submission
3. Verify CLOSE orders reach Binance
4. Repair position tracking (fix phantom closes)
5. Wait 24-48h for stable operation
6. Submit new escalation audit starting from OBSERVER

**Concrete Evidence**: See `HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md`

---

### ⚪ RL Feedback

**Type**: OBSERVER  
**Status**: DATA COLLECTION  
**Output**: Feedback events → Redis  

**Hvorfor OBSERVER**:
- Samler state/action/reward for RL training
- Feeds RL Trainer (offline)
- **Ingen direkte kontroll over exits eller entries**

**Hvis fjernet**: ➡️ PnL = identisk (pure telemetry)

---

### ⚪ Governor

**Type**: OBSERVER  
**Status**: TELEMETRI ONLY  
**Output**: Timestamps → `quantum:governor:exec:{SYMBOL}` (lists)  

**Hvorfor OBSERVER**:
- Konsumerer `apply.plan` stream (53,972 events processed, lag=0)
- Skriver **KUN timestamps** til Redis lists (not decisions)
- Logs show "ALLOW plan {id}" but these are **audit logs**, not control signals
- **INGEN consumers** av governor output keys
- Feature flag: `quantum:governor:execution` = **nil** (disabled)

**Bevis**:
```bash
# Governor output:
redis-cli LRANGE quantum:governor:exec:BTCUSDT 0 2
# Result: ["1770718939.26", "1770718880.19", "1770718820.92"]  ← TIMESTAMPS ONLY

# No consumers:
grep -r "quantum:governor:exec" --exclude="governor/main.py"
# Result: ZERO matches in executor/apply_layer ❌
```

**Counterfactual**:
- Recent logs: All decisions = "ALLOW" with `qty=0.0000, notional=$0.00`
- Intent executor metrics show **ZERO references to governor decisions**
- Executor uses `p35_guard_blocked=1975`, `blocked_source=827` (NOT governor)

**Hvis fjernet**: ➡️ PnL = identisk (pure timestamp logger, no consumers)

---

## 🔴 Nivå 4 — DEAD / DISABLED

**Komponenter som kjører, men er funksjonelt døde**

### 🔴 RL Agent

**Type**: DEAD  
**Status**: SERVICE ACTIVE, FEATURE DISABLED  
**Output**: Ingen (blocked by flag)  

**Bevis**:
- `rl_influence_enabled = false` (hardcoded in AI engine)
- All intents show: `"rl_gate_pass": false, "rl_gate_reason": "rl_disabled"`
- Service process exists but produces zero decisions
- Even if enabled, output goes to AI Ensemble (blocked by MANUAL_LANE_OFF)

**Hvis aktivert**: ➡️ PnL still unchanged (output blocked by MANUAL_LANE_OFF)

---

## 🗺️ FULLSTENDIG PnL AUTHORITY TREE

```
REALIZED PnL
│
├── EXIT CONTROL (DIRECT) ✅
│   └── Harvest Proposal System (RULE-BASED)
│       ├── R-value triggers (R_net thresholds)
│       ├── Time limits (max_time_sec)
│       └── Emergency stop-loss
│
├── SYMBOL ACCESS (INDIRECT) 🟡
│   └── Universe (ALLOWLIST)
│       ├── 567 Binance perpetuals
│       └── Fail-open fallback: [BTC, ETH, TRX]
│
├── ENTRY / PREDICTION ❌
│   ├── AI Ensemble (blocked + fallback)
│   ├── RL Agent (disabled)
│   └── CLM (observer)
│
├── GOVERNANCE ❌
│   └── Governor (telemetry only)
│
└── LEARNING ❌
    ├── RL Trainer (offline)
    └── RL Feedback (data collection)
```

---

## 🧠 KJERNEERKJENNELSE (FASTLÅST)

**Systemet ditt er ikke AI-drevet.**  
**Det er regelbasert med AI-telemetri rundt seg.**

Dette er ikke en feil.  
Dette er en konservativ, fail-safe runtime-virkelighet.

### Authority Distribution:
- **Direct PnL Control**: 100% rule-based (Harvest Proposal)
- **Indirect PnL Control**: 100% static filter (Universe allowlist)
- **AI/ML Influence**: 0% (all observers or disabled)

### Architectural Reality:
```
DESIGN INTENT           RUNTIME REALITY
─────────────           ───────────────
AI Ensemble    →        Fallback RSI/MACD rules
RL Agent       →        Disabled (flag=false)
CLM Learning   →        Data starvation (bug)
AI Exit Brain  →        Unread output (39k backlog)
Governor       →        Timestamp logger (no consumers)
```

---

## 🧾 ENSETNINGS-SUMMARY (EXECUTIVE TRUTH)

> **"I produksjon styres penger kun av regelbaserte exits og symbol-allowlist.  
> Alle AI-komponenter er observatører uten beslutningsmyndighet."**

---

## 📊 VERIFICATION COMMANDS (REPEATABLE AUDIT)

For å verifisere denne mappen i fremtiden:

```bash
# 1. Verify Harvest Proposal authority:
journalctl -u quantum-intent-executor.service --since "1 hour ago" | grep "harvest_executed"

# 2. Verify Universe gatekeeper:
redis-cli HGETALL quantum:cfg:universe:meta
redis-cli GET quantum:cfg:universe:active | python3 -c "import sys,json; print(len(json.load(sys.stdin)['symbols']))"

# 3. Verify AI Ensemble blocked:
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | grep "rl_influence_enabled"

# 4. Verify AI Exit Evaluator unread:
redis-cli XINFO GROUPS quantum:stream:ai.exit.decision | grep "pending"

# 5. Verify CLM data starvation:
ls -lh /root/quantum_trader/microservices/clm/clm_trades.jsonl
stat -c %Y /root/quantum_trader/microservices/clm/clm_trades.jsonl

# 6. Verify Governor telemetry-only:
redis-cli KEYS "quantum:governor:exec:*"
grep -r "quantum:governor:exec" /root/quantum_trader --include="*.py" | grep -v governor/main.py

# 7. Verify RL Agent disabled:
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | grep "rl_gate_reason"
```

---

## 🔒 AUDIT PROVENANCE

**Audit 1 - CLM Eligibility** (Feb 8-9, 2026):
- Result: **REJECTED** (data integrity failure + negative expectancy + no control)
- Bugfix: Implemented bytes→str decoder (NOT YET DEPLOYED)
- Documentation: `AI_CLM_AUDIT_COMPLETE_FEB8_2026.md`

**Audit 2 - Prediction Agents** (Feb 9-10, 2026):
- Result: **ALL OBSERVERS OR DEAD** (zero PnL influence)
- Key Finding: MANUAL_LANE_OFF blocks all AI intents
- Key Finding: Ensemble models failed, fallback rules active
- Documentation: `AI_PREDICTION_AGENTS_AUDIT_COMPLETE_FEB9_2026.md`

**Audit 3 - Governance & Universe** (Feb 10, 2026):
- Governor: **OBSERVER** (timestamps only, no consumers)
- Universe: **GATEKEEPER** (allowlist enforcement with fail-open)
- Documentation: This document

---

## ✅ CANONICAL STATUS

This document represents the **authoritative, evidence-based truth** about PnL control authority in quantum_trader as of February 10, 2026.

**Approval Date**: February 10, 2026  
**Next Review**: On any deployment that changes:
- Intent execution flow
- AI model integration
- Harvest proposal logic
- Universe enforcement

**Signed**: Runtime Evidence (systemd logs, Redis state, execution traces)

---

## 🔗 RELATED DOCUMENTS

**Authority Governance Framework:**

1. **PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md**  
   How to gain authority (5 BEVISKRAV)

2. **PNL_AUTHORITY_DEMOTION_PROMPT_CANONICAL.md**  
   How to lose authority (5 demotion criteria)

3. **AUTHORITY_FREEZE_PROMPT_CANONICAL.md**  
   What's allowed/forbidden when 0 CONTROLLERS

4. **BASELINE_SAFETY_CONTROLLER_PROMPT_CANONICAL.md**  
   Emergency fallback controller specification (PATH 2)

5. **BASELINE_SAFETY_CONTROLLER_AUDIT_PROMPT_CANONICAL.md**  
   BSC comprehensive weekly audit (boundary enforcement)

6. **BSC_SCOPE_GUARD_DAILY_AUDIT.md**  
   BSC daily operational verification (scope guard)

7. **HARVEST_PROPOSAL_AUDIT_PROMPT_CANONICAL.md**  
   Reusable CONTROLLER audit procedure

**Current System State:**

8. **NO_CONTROLLER_MODE_DECLARATION_FEB10_2026.md**  
   Formal declaration of current state (3 re-entry paths)

9. **HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md**  
   Reference demotion audit (Feb 10, 2026)

---

**END OF CANONICAL MAP**
