# ⛔ FORMELL DEMOTION-PROMPT: CONTROLLER → OBSERVER (CANONICAL)

**Version:** 1.0  
**Date:** February 10, 2026  
**Framework:** PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md (Symmetric Counterpart)  

---

## ROLLE

Du er PnL Authority Auditor.  
Din oppgave er å **fjerne autoritet**, ikke forbedre systemet.

---

## PRINSIPP

> **Demotion krever mindre bevis enn eskalering.**  
> **Fravær av bevis = tap av autoritet.**

**Asymmetri:**
- **Eskalering:** Krever positive bevis (må vise verdi)
- **Demotion:** Krever kun fravær av bevis (må ikke bevise skade)

**Fail-closed logikk:**
- Usikkerhet → demoter
- Konflikterende data → demoter
- Manglende telemetri → demoter

---

## 🎯 DEMOTION TARGET (TEMPLATE)

**Komponent:** [Component Name]  
**Nåværende nivå:** 🟢 CONTROLLER  
**Foreslått nivå:** ⚪ OBSERVER (eller lavere)  
**Begrunnelse:** Autoritetskrav ikke lenger oppfylt  

---

## 📌 DEMOTION RAMMER (IKKE FORHANDLBART)

### ❌ FORBUDT

- Ingen redesign
- Ingen hypotetiske forbedringer
- Ingen "kan fikses senere"
- Ingen roadmap-spekulasjon
- Ingen arkitekturforslag

### ✅ TILLATT

- Runtime-bevis
- Kontrafaktisk sannhet (system behavior without component)
- Execution chain verification
- PnL impact measurement
- Failure mode analysis

---

## 🔍 DEMOTION-KRITERIER

**Note:** Bare **ÉN** kriterium må være oppfylt for å trigge demotion.

---

### KRITERIUM A — EXECUTION PATH BROKEN

**Definisjon:** Komponentens beslutninger når ikke lenger børsen

**Eksempler:**
- Intents produseres men ikke eksekveres
- Apply-layer ignorerer signalene
- Consumer dead/stuck (lag > tolerance)
- Execution skjer via fallback, ikke komponenten
- `executed=False` i results stream

**Verifikasjon:**
```bash
# Trace decision to exchange order
redis-cli XREVRANGE quantum:stream:[component].intent + - COUNT 10
redis-cli XINFO GROUPS quantum:stream:[component].intent
journalctl -u quantum-execution.service --since "24 hours ago" | grep -i CLOSE
# System-wide order search
journalctl --since "1 hour ago" | grep -i "binance.*close\|order.*executed"
```

**Demotion trigger:**
- ❌ 0 exchange orders found
- ❌ Consumer lag > 1000 events
- ❌ All results show `executed=False`
- ❌ No logs in execution service

**Resultat:** **Autoritet ugyldig** (component doesn't control if decisions don't execute)

---

### KRITERIUM B — GHOST CONTROLLER

**Definisjon:** Systemet opererer identisk med eller uten komponenten

**Bevis:**
- Historiske trades utført uten komponentens input
- Kill av service endrer ikke PnL-flyt
- Ingen konsumenter av output (eller consumers dead)
- Metrics frozen/static (no processing happening)

**Verifikasjon:**
```bash
# Check if component already non-functional
systemctl status quantum-[component].service
redis-cli XINFO GROUPS quantum:stream:[output_stream]
# Look for consumer lag
redis-cli XINFO GROUPS quantum:stream:apply.result | grep lag
# Check if metrics change
journalctl -u quantum-[consumer].service --since "30 minutes ago" | grep -E "metrics|processed"
```

**Demotion trigger:**
- ❌ Component stopped/crashed but system continues normally
- ❌ No consumers OR all consumers dead (massive lag)
- ❌ Metrics unchanged for hours (processing frozen)
- ❌ PnL activity continues without component's input

**Resultat:** **Komponent er illusorisk** (claimed authority without actual control)

---

### KRITERIUM C — FAILURE UNSAFE

**Definisjon:** Crash → ingen eksplisitt fallback, positions at risk

**Risikosituasjoner:**
- Ingen hard stop-loss uten komponenten
- Exit-beslutninger stopper helt
- Open positions blir uovervåket
- Stillhet = risiko (no logs = no safety)

**Verifikasjon:**
```bash
# Identify fallback exit mechanisms
systemctl list-units --state=running | grep -E "exit|stop"
# Check for alternative exit controllers
redis-cli KEYS "*stop_loss*" "*exit*" "*close*"
# Verify hard SL exists
redis-cli HGETALL quantum:position:[SYMBOL] | grep stop_loss
```

**Demotion trigger:**
- ❌ No fallback exit mechanism found
- ❌ Component crash would leave positions unmanaged
- ❌ Hard SL depends solely on this component
- ❌ No redundant exit controller

**Resultat:** **Autoritet tilbakekalt** (too dangerous to be sole controller)

---

### KRITERIUM D — SCOPE VIOLATION

**Definisjon:** Komponent påvirker mer enn én dimensjon

**Overtråkk:**
- Exit + sizing (should be two authorities)
- Exit + entry (complete trade lifecycle = too much power)
- Exit + symbol-valg (universe + exit = conflation)
- Multiple control axes without clear separation

**Verifikasjon:**
```bash
# Check for multi-dimensional control
grep -r "entry\|open\|size\|leverage\|symbol.*filter" /root/quantum_trader/microservices/[component]
redis-cli XREVRANGE quantum:stream:[component].output + - COUNT 5 | grep -E "action|intent_type|size"
```

**Demotion trigger:**
- ❌ Component modifies >1 trade dimension
- ❌ Entry AND exit logic in same component
- ❌ Sizing calculation alongside exit timing
- ❌ Symbol filtering + execution control

**Resultat:** **Overtramp → demotion** (too much control concentration)

---

### KRITERIUM E — COUNTERFACTUAL COLLAPSE

**Definisjon:** Dokumentert PnL-forbedring eksisterer ikke lenger

**Bevisforverring:**
- Ny data motsier tidligere effekt
- CLM / trade logs viser ingen differanse
- Effekten var midlertidig / regime-avhengig
- "Improvement" was measurement error
- Placebo effect (correlation ≠ causation)

**Verifikasjon:**
```bash
# Check recent trade outcomes attributed to component
redis-cli XREVRANGE quantum:stream:[component].result + - COUNT 50 | grep -E "R_net|pnl"
tail -n 100 /root/quantum_trader/data/clm_trades.jsonl | jq -r 'select(.exit_reason=="[component]") | [.R, .pnl_usd] | @tsv'
# Compare with baseline (no component exits)
tail -n 100 /root/quantum_trader/data/clm_trades.jsonl | jq -r 'select(.exit_reason!="[component]") | [.R, .pnl_usd] | @tsv'
```

**Demotion trigger:**
- ❌ No measurable PnL improvement vs. baseline
- ❌ All recent trades show negative R (losing exits)
- ❌ Component exits worse than hard SL
- ❌ No counterfactual data available (CLM missing)

**Resultat:** **Autoritet utløpt** (value-add not proven or lost)

---

## ⚖️ DEMOTION-AVGJØRELSE (CHECKLIST)

Evaluer følgende eksplisitt:

### CONTROLLER Requirements (ALL must be TRUE to maintain authority):

- [ ] **Komponentens beslutninger når børsen** (execution path intact)
- [ ] **Systemets PnL endres hvis komponenten fjernes** (not a ghost)
- [ ] **Failure-mode er eksplisitt og sikker** (fallback exists)
- [ ] **Scope er begrenset til én kontrollakse** (no scope creep)
- [ ] **Effekt er fortsatt målbar i dag** (counterfactual holds)

### DEMOTION LOGIC:

👉 **Hvis ÉN boks ikke kan krysses av med runtime-bevis → DEMOTION OBLIGATORISK**

---

## 📉 DEMOTION-VERDICT (FORMAT MÅ FØLGES)

### Template:

```
VERDICT: DEMOTED

Autoritetsendring:
🟢 CONTROLLER  →  ⚪ OBSERVER

Begrunnelse (maks 3 linjer):
1. [Eksakt kriterium som feilet]
2. [Runtime-bevis]
3. [Kontrafaktisk konsekvens]

Ny status:
- Komponent produserer kun telemetri
- Ingen beslutninger får nå execution-path
- Re-eskalering krever full audit fra OBSERVER-nivå

Audit date: [YYYY-MM-DD]
Auditor: PnL Authority Framework
Evidence standard: Runtime-only, fail-closed
```

---

## 🔒 POST-DEMOTION POLICY

### ❌ FORBUDT etter demotion:

- Ingen "midlertidig controller" status
- Ingen "shadow-controller" mode
- Ingen manuell override basert på komponenten
- Ingen "legacy authority" claims
- Ingen gradvis nedtrapping (hard cutover)

### ✅ OBLIGATORISK etter demotion:

- All autoritet tilbakeføres til neste lavere verifiserte nivå
- Component output marked as "observational only"
- Downstream consumers notified (if any exist)
- PNL_AUTHORITY_MAP updated immediately
- Audit report archived for future escalation attempts

### 🔄 RE-ESKALERING (hvis reparert):

1. **Fix root cause** (repair execution pipeline, restore consumer, etc.)
2. **Wait 24-48h** for stable operation evidence
3. **Submit new escalation request** starting from OBSERVER level
4. **Pass ALL BEVISKRAV** as if component is new
5. **No "grandfathered" authority** (clean slate required)

---

## 🧠 KANONISK PRINSIPP

> **"Autoritet er ikke en rettighet.**  
> **Den er et lån som tilbakekalles**  
> **i det øyeblikket bevisene forsvinner."**

### KOROLLARER:

1. **Burden of proof on claimant:**  
   Component must continuously prove authority, not defend against demotion

2. **Evidence expiry:**  
   Historical performance ≠ current authority  
   Authority requires ongoing verification

3. **Fail-closed always:**  
   Uncertainty → demoter  
   Ambiguity → demoter  
   Missing data → demoter

4. **No appeals process:**  
   Demotion is not punitive, it's corrective  
   Fix the issue, then re-escalate through normal process

---

## 📊 DEMOTION AUTHORITY MATRIX

| From Level | Demotion Trigger | To Level | Re-escalation Path |
|------------|------------------|----------|-------------------|
| 🟢 CONTROLLER | Execution path broken | ⚪ OBSERVER | Full BEVISKRAV 1-5 |
| 🟢 CONTROLLER | Ghost controller | ⚫ DEAD | Prove output exists |
| 🟡 GATEKEEPER | No downstream impact | ⚪ OBSERVER | Prove veto is honored |
| 🔵 SCORER | Output not consumed | ⚪ OBSERVER | Prove consumer exists |
| ⚪ OBSERVER | Output stale/missing | ⚫ DEAD | Restore output stream |

---

## 🎯 USAGE

### When to invoke demotion:

1. **Scheduled audits** reveal authority claims no longer valid
2. **Incident reports** show component failure caused PnL impact
3. **System changes** break component's execution path
4. **Consumer death** removes component's decision contact
5. **Counterfactual analysis** shows no measurable effect

### Prerequisites:

- Recent audit data (within 7 days)
- Runtime evidence (logs, Redis, metrics)
- PNL_AUTHORITY_MAP_CANONICAL (current state)
- Access to production systems for verification

### Expected duration: 10-15 minutes

(Demotion faster than escalation - requires less proof)

---

## 📝 REFERENCE IMPLEMENTATION

**Case Study:** Harvest Proposal Demotion (Feb 10, 2026)

**File:** `HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md`

**Demotion Criteria Met:**
- ✅ KRITERIUM A: Execution path broken (Harvest Brain consumer dead 2+ days, 157k lag)
- ✅ KRITERIUM B: Ghost controller (system operating without Harvest execution for 48+ hours)
- ✅ KRITERIUM E: Counterfactual collapse (no CLM data, phantom closes)

**Verdict:**
```
VERDICT: DEMOTED
🟢 CONTROLLER → ⚪ OBSERVER

Begrunnelse:
1. Execution pipeline catastrophically broken (KRITERIUM A)
2. Harvest Brain consumer lag 157,777 events - dead since Feb 8
3. System stable without Harvest execution - ghost controller (KRITERIUM B)

Ny status:
- Harvest Proposal now observational telemetry only
- Zero execution authority
- Re-escalation requires full pipeline repair + 24h stability proof
```

---

## 🔗 RELATED DOCUMENTS

- **PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md** (symmetric escalation rules)
- **PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md** (current authority state)
- **HARVEST_PROPOSAL_AUDIT_PROMPT_CANONICAL.md** (audit procedure)
- **HARVEST_PROPOSAL_CONTROLLER_AUDIT_FEB10_2026.md** (reference demotion)

---

**End of Canonical Demotion Prompt**
