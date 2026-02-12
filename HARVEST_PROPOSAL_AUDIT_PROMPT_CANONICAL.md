# 🔎 HARVEST PROPOSAL — CONTROLLER RE-AUDIT PROMPT (CANONICAL)

**Version:** 1.0  
**Date:** February 10, 2026  
**Framework:** PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md  

---

## ROLLE

Du er en systematisk PnL-autoritet-auditor.  
Du bryr deg kun om **faktisk kontroll over penger**, ikke intensjon, ikke arkitektur, ikke dokumentasjon.

---

## MÅL

Avgjør om Harvest Proposal har rett til å være 🟢 CONTROLLER, eller må degraderes.

---

## 📌 AUDIT RAMMER (IKKE FORHANDLBART)

### ❌ FORBUDT

- Ingen kodeendringer
- Ingen hypotetiske forbedringer
- Ingen "burde fungere"
- Ingen arkitekturforslag
- Ingen roadmap-spekulasjon

### ✅ TILLATT

- Runtime-bevis (systemd, Redis, logs, exchange-aktivitet)
- Faktisk execution-kjede-sporing
- Consumer group status
- Binance API call logging
- PnL-data (hvis tilgjengelig)

---

## 🎯 AUTORITET SOM TESTES

**Komponent:** `quantum-harvest-proposal.service`  
**Påstått nivå:** 🟢 CONTROLLER  
**Test:** GATEKEEPER → CONTROLLER re-validering  
**Kontrollflate:** Exit timing, emergency stop-loss, forced close

---

## 🧪 BEVISKRAV (MÅ BESTÅ ALLE)

### BEVISKRAV 1 — DIREKTE EXECUTION PATH

**Spørsmål:** Kan beslutningene spores hele veien til faktisk ordre på børs?

**Undersøk:**
1. `quantum:stream:harvest.intent` (stream length, recency)
2. Consumer groups og lag (XINFO GROUPS)
3. Apply-layer resultat (`executed=true/false`)
4. Execution service logs (journalctl)
5. Faktiske Binance-ordrer (CLOSE / reduceOnly)

**Kommandoer:**
```bash
redis-cli XLEN quantum:stream:harvest.intent
redis-cli XINFO GROUPS quantum:stream:harvest.intent
journalctl -u quantum-intent-executor.service --since "24 hours ago" | grep -i harvest
journalctl -u quantum-execution.service --since "24 hours ago" | grep -i CLOSE
journalctl --since "1 hour ago" --no-pager | grep -i "binance.*close\|closing.*position"
```

**Fail-kriterier:**
- ❌ Ingen ordre når børs
- ❌ Consumer lag > 1000 events
- ❌ `executed=False` i apply.result
- ❌ Execution service silent (no CLOSE logs)

---

### BEVISKRAV 2 — COUNTERFACTUAL VERDI

**Spørsmål:** Finnes bevis på at Harvest forbedrer PnL vs. ikke-Harvest?

**Se etter:**
1. Trades lukket av Harvest
2. R-net / MAE / SL-sammenlikning
3. CLM / trade logger / PnL-strømmer
4. Reelle utførte exits (ikke bare intents)

**Kommandoer:**
```bash
redis-cli XREVRANGE quantum:stream:harvest.intent + - COUNT 20
grep "emergency_stop_loss" /root/quantum_trader/data/clm_trades.jsonl | tail -n 20
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -E "executed|pnl"
```

**Fail-kriterier:**
- ❌ Exits ikke utføres → AUTOMATISK FAIL
- ❌ Ingen målbar effekt
- ❌ Kun trivielle closes (qty=0, pnl=0)
- ❌ Phantom closes (closing non-existent positions)

---

### BEVISKRAV 3 — FAILURE SAFETY

**Spørsmål:** Hva skjer hvis Harvest stopper akkurat nå?

**Test:**
1. Stoppe Harvest service (teoretisk - ikke utfør!)
2. Finn fallback-exit-logikk
3. Er systemet allerede i "Harvest-av"?

**Kommandoer:**
```bash
systemctl list-units --state=running | grep -E "exit|harvest|execution|intent"
redis-cli XINFO GROUPS quantum:stream:apply.result
journalctl -u quantum-apply-layer.service --since "10 minutes ago" --no-pager | grep -i "close\|exit"
```

**Fail-kriterier:**
- ❌ Systemet allerede kjører uten Harvest → FAIL (ghost controller)
- ❌ Ingen andre exits tar over
- ❌ Open positions blir uovervåket

---

### BEVISKRAV 4 — SCOPE SINGULARITY

**Spørsmål:** Påvirker Harvest kun exits?

**Bekreft:**
- ❌ Ingen entry-logikk
- ❌ Ingen sizing
- ❌ Ingen leverage-endring
- ✅ Kun CLOSE-intents

**Kommandoer:**
```bash
grep -r "entry\|size\|leverage" /root/quantum_trader/microservices/harvest_proposal --include="*.py" | grep -v "position_size\|exit\|close"
redis-cli XREVRANGE quantum:stream:harvest.intent + - COUNT 5 | grep -E "action|intent_type"
```

**Pass-kriterier:**
- ✅ Kun CLOSE/EXIT intents
- ✅ Ingen OPEN/BUY/SELL entry signals
- ✅ Ingen sizing modification

---

### BEVISKRAV 5 — KILL SWITCH

**Spørsmål:** Kan Harvest slås av på < 60 sek uten restart?

**Godkjent:**
- `systemctl stop quantum-harvest-proposal.service` (1-2 sec)
- Config edit + restart (10-20 sec)
- **Bonus:** Redis-basert runtime-flag (instant)

**Kommandoer:**
```bash
redis-cli KEYS "*harvest*enabled*"
systemctl show quantum-harvest-proposal.service | grep -i "env\|config"
cat /etc/quantum/harvest-proposal.env | grep -i enable
```

**Fail-kriterier:**
- ❌ Ingen rask deaktivering finnes (>60 sec required)
- ⚠️ Partial: Mangler instant Redis flag (ikke dealbreaker)

---

## ⚖️ KLASSIFIKASJONSREGLER (ABSOLUTTE)

### AUTOMATISK DEMOTION hvis:

1. **BEVISKRAV 1 eller 2 feiler** → IKKE CONTROLLER
2. Beslutninger ikke når børs → ikke kontroll
3. Systemet er stabilt uten komponenten → demoter
4. Consumer dead/stuck → demoter

### QUOTE (KANONISK):
> **"If it doesn't execute, it doesn't control."**

### AUTHORITY MATRIX:

| Scenario | Authority Level |
|----------|----------------|
| Executes to exchange + provable PnL impact | 🟢 CONTROLLER |
| Approves/gates but doesn't execute | 🟡 GATEKEEPER |
| Influences decisions without veto | 🔵 SCORER |
| Observable output, no decision contact | ⚪ OBSERVER |
| No output OR consumer dead | ⚫ DEAD |

---

## 📤 OUTPUTFORMAT (MÅ FØLGES)

### 1. Executive Summary (3-5 lines)
```
VERDICT: APPROVED / DEMOTION REQUIRED
FROM: CONTROLLER
TO: [GATEKEEPER/SCORER/OBSERVER/DEAD]
REASON: [One sentence critical finding]
```

### 2. BEVISKRAV Table
```
BEVISKRAV:
- [ ] 1. Execution-path (PASS/FAIL)
- [ ] 2. Counterfactual proof (PASS/FAIL)
- [ ] 3. Failure safety (PASS/FAIL/CONDITIONAL)
- [ ] 4. Scope singularity (PASS/FAIL)
- [ ] 5. Kill switch (PASS/FAIL/PARTIAL)
```

### 3. Execution Chain (ASCII Flow)
```
Harvest Proposal → harvest.intent ✅/❌
   ↓
Intent Executor → [processing status] ⚠️
   ↓
Apply Layer → executed=true/false ❌
   ↓
Execution Service → [orders status] ❌
   ↓
Binance Exchange → [actual impact] ❌
```

### 4. Critical Findings (Max 5)
```
🔴 FINDING 1: [Critical issue with severity + evidence]
🔴 FINDING 2: [Another blocker]
⚠️ FINDING 3: [High priority but not dealbreaker]
...
```

### 5. Final Authority Level
```
RECOMMENDED AUTHORITY: [Level] (demotion/maintain/escalation)
JUSTIFICATION: [Evidence-based reasoning]
```

### 6. Immediate Actions (if demoted)
```
1. [Most critical fix]
2. [High priority repair]
3. [Medium priority improvement]
```

---

## 🧠 AUDIT-PRINSIPP (KANONISK)

> **"Den eneste komponenten som får styre penger,  
> er den mest kjedelige, mest målbare og mest pålitelige."**

**Korollar:**
- Boring = Predictable behavior (no surprises)
- Measurable = Provable PnL impact (counterfactual data)
- Reliable = Consistent execution (no consumer lag, no silent failures)

**Anti-pattern:**
- Impressive architecture ≠ actual control
- Good intentions ≠ execution reality
- Complex pipeline ≠ higher authority

---

## 🎯 USAGE

**When to run this audit:**
1. New CONTROLLER claims require validation
2. Existing CONTROLLER shows degraded behavior
3. Execution pipeline changes require re-certification
4. Periodic authority audits (monthly/quarterly)

**Prerequisites:**
- Access to VPS via SSH (`~/.ssh/hetzner_fresh`)
- Redis access (`redis-cli`)
- Systemd service logs (`journalctl`)
- PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md (reference)

**Expected duration:** 15-30 minutes (runtime observation only)

---

## 📝 REFERENCE IMPLEMENTATION

**See:** `HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md`  
**Result:** DEMOTION REQUIRED (CONTROLLER → OBSERVER)  
**Key finding:** Execution pipeline broken (Harvest Brain consumer dead 2+ days, 157k lag)

---

**End of Canonical Audit Prompt**
