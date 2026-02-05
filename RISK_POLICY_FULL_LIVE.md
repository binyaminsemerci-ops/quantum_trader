# RISK POLICY — QUANTUM TRADER (FULL LIVE)

**Status:** BINDENDE  
**Scope:** Live trading med ekte kapital  
**Authority:** Systemisk (overordnet modell, RL og menneskelig operatør)  
**Effective date:** 2026-02-05

---

## 1. PURPOSE & AUTHORITY

Denne risk policyen definerer **hvem som har rett til kapital**, og under hvilke
betingelser Quantum Trader får lov til å handle i live marked.

> Kapital er et privilegium, ikke en rettighet.

Denne policyen:
- overstyrer RL-policyer
- overstyrer optimaliseringsmål
- kan ikke overstyres manuelt i runtime

---

## 2. SYSTEM IDENTITY (FULL LIVE)

Systemet opererer i **FULL LIVE MODE** når:
- ekte kapital er deployert
- handler eksponeres mot eksternt marked
- tap er irreversible

I denne modusen gjelder **strengeste risiko-regime**.

---

## 3. CAPITAL PHASES (KONTRAKT)

### Phase C — FULL LIVE

- Kapital: Ekte, begrenset
- Mål: Risikojustert avkastning
- Prioritet: Overlevelse > læring > avkastning

Overgang *inn* i Full Live krever eksplisitt GO-beslutning.  
Overgang *ut* skjer automatisk ved policy-brudd.

---

## 4. LAYER 0 — SYSTEMIC SAFETY (HARD INVARIANTS)

**Disse reglene er absolutte.**

### 4.1 Infrastruktur Kill-Switch
Trading **MÅ STOPPE UMIDDELBART** hvis én av følgende er sann:

- Redis utilgjengelig
- `quantum-rl-feedback-v2` inaktiv
- `quantum-rl-trainer` inaktiv
- Heartbeat ikke oppdatert innen terskel

➡️ Enforcement: runtime hard-stop (fail-closed)

---

### 4.2 Execution Integrity
- Ingen trades uten bekreftet system-helse
- Ingen delvis degraderte tilstander
- Ingen "best effort"-modus

➡️ Enforcement: blokkering før ordreutsendelse

---

## 5. LAYER 1 — CAPITAL PROTECTION

### 5.1 Max Leverage
- Hard cap på leverage (global)
- RL kan aldri overskride cap

### 5.2 Daily Loss Limit
- Absolutt tapsgrense per dag
- Brudd → trading stoppes ut dagen

### 5.3 Rolling Drawdown
- Maks tillatt drawdown over glidende vindu
- Brudd → automatisk pause + review

### 5.4 Loss Streak Breaker
- N sammenhengende tap → cooldown
- Cooldown er tidsbasert, ikke trade-basert

---

## 6. LAYER 2 — MARKET & REGIME GATING

Systemet får kun handle når markedet oppfyller minimumskrav.

### 6.1 Volatility Gate
- Minimum og maksimum volatilitet
- Ekstreme regimer = ingen nye trades

### 6.2 Liquidity Gate
- Kun symbols med bekreftet likviditet
- Illikvide perioder = flat eksponering

### 6.3 Symbol Whitelist
- Kun forhåndsgodkjente symboler
- Nye symboler krever eksplisitt godkjenning

---

## 7. LAYER 3 — MODEL AUTONOMY (BEGRENSET)

RL-modellen har kun autonomi innenfor:
- posisjonssizing (innen rammer)
- timing av entry/exit
- policy-oppdatering

RL har **ingen kontroll** over:
- maksimal eksponering
- systemtilstand
- kill-switcher
- kapitalfase

---

## 8. FAILURE HANDLING POLICY

### 8.1 Failure Classification

| Type | Respons |
|-----|--------|
| Infrastruktur | Hard stop |
| Læringsplan | Pause |
| Markedsregime | No new trades |
| Modellavvik | Rollback / freeze |

### 8.2 Guiding Principle

> Ved tvil i runtime, tas **ingen trade**.

---

## 9. HUMAN OVERRIDE

- Menneskelig override er tillatt **kun** for:
  - å stoppe systemet
  - å redusere eksponering
- Menneske kan **aldri** tvinge systemet til å trade

---

## 10. GO / NO-GO CONDITIONS (LIVE)

Systemet er **NO-GO for live kapital** hvis én av disse er sann:
- Kill-switch trigget siste 24t
- Drawdown-grense brutt
- Learning plane ustabil
- Uforklarlig modelladferd

GO-status må re-godkjennes eksplisitt.

---

## 11. FINAL AXIOM

> Et system som ikke kan stoppe seg selv,
> har ikke rett til å styre kapital.

---

**Denne policyen er aktiv når filen er til stede i produksjonsmiljøet.**  
Endringer krever ny eksplisitt godkjenning.
