# Emergency Exit System - Complete Documentation

> **Institusjonelt forsvarlig design for fail-closed sikkerhet**

**Opprettet:** 13. Februar 2026  
**Status:** Implementert, klar for deployment

---

## Oversikt

Systemet består av to komponenter som sammen gir fail-closed sikkerhet uten exchange stop-loss:

| Komponent | Formål |
|-----------|--------|
| **Emergency Exit Worker (EEW)** | Lukker ALLE posisjoner på `system.panic_close` |
| **Exit Brain Watchdog** | Overvåker Exit Brain, trigger panic_close ved feil |

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIKKERHETSKJEDE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exit Brain ──[heartbeat]──► Watchdog                          │
│                                  │                              │
│                          [feil detektert?]                      │
│                                  │                              │
│                                  ▼                              │
│  Risk Kernel ◄─────────► system.panic_close                    │
│                                  │                              │
│                                  ▼                              │
│                     Emergency Exit Worker                       │
│                          │                                      │
│                    [MARKET close all]                           │
│                          │                                      │
│                          ▼                                      │
│                    System HALTED                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Emergency Exit Worker (EEW)

### Mandat
- Eier KUN én handling: **MARKET close alle posisjoner**
- Ignorerer: signaler, AI, strategi, PnL
- Kan kun trigges, aldri diskutere

### Trigger: `system.panic_close`

**Autoriserte publishere:**
1. Risk Kernel (automatisk)
2. Exit Brain (kun ved fatal health-failure)
3. Ops / Manuell (sjeldent, logget)

### Eksekveringsflyt
```
1. Mottar system.panic_close event
2. Validerer trigger (source, timestamp)
3. Henter ALLE åpne posisjoner fra exchange
4. For hver posisjon:
   - MARKET order
   - reduceOnly=true
   - Fortsett umiddelbart (ingen venting)
5. Publiserer panic_close.completed
6. Setter system til HALTED
```

### Filer
```
services/emergency_exit_worker/
├── README.md                    # Oversikt
├── policy_ref.md                # Policy og autorisasjon
├── trigger_conditions.md        # Trigger-betingelser
├── execution_rules.md           # Eksekveringsregler
├── emergency_exit_worker.py     # Hovedimplementasjon
└── tests/
    ├── test_panic_close_all.md
    ├── test_partial_failure.md
    └── test_idempotency.md
```

---

## 2. Exit Brain Watchdog

### Mål
Oppdage Exit Brain-feil **raskere** enn markedet kan skade deg.

### Heartbeat-spesifikasjon

**Stream:** `quantum:stream:exit_brain.heartbeat`

**Frekvens:** 1-2 sekund

**Innhold:**
```json
{
  "timestamp": 1707840000.123,
  "status": "OK|DEGRADED",
  "active_positions_count": 5,
  "last_decision_ts": 1707839999.456,
  "loop_cycle_ms": 245,
  "pending_exits": 2
}
```

### Trigger-betingelser

| Betingelse | Terskel | Handling |
|------------|---------|----------|
| Heartbeat mangler | > 5 sek | panic_close |
| Status = DEGRADED | > 10 sek | panic_close |
| last_decision stagnerer | > 30 sek (med posisjoner) | panic_close |
| Posisjoner UBESKYTTET | > 3 sek (HB mangler + pos > 0) | panic_close |

### Viktig regel

> **Ingen grace-perioder i volatilitet.**
> 
> False positives er OK. False negatives er IKKE.

### Filer
```
services/exit_brain/
├── README.md                    # Eksisterende
├── heartbeat.md                 # Heartbeat-spesifikasjon
├── watchdog_rules.md            # Watchdog-regler
├── recovery_flow.md             # Recovery-prosedyre
├── exit_brain_watchdog.py       # Watchdog-implementasjon
└── heartbeat_mixin.py           # Mixin for Exit Brain
```

---

## 3. Recovery Flow

Etter panic_close:

```
1. Exit Brain STOPPET
2. Watchdog detekterer heartbeat mangler
3. panic_close TRIGGET
4. EEW lukker ALLE posisjoner
5. System er FLAT
6. Exit Brain RESTARTES
7. Exit Brain i SHADOW MODE
8. Manuell godkjenning før live
```

### Shadow Mode
- Monitorer (hvis nye posisjoner)
- Beregner beslutninger
- Logger beslutninger
- UTFØRER IKKE

---

## 4. Deployment

### Systemd Services
```
ops/systemd/quantum-emergency-exit-worker.service
ops/systemd/quantum-exit-brain-watchdog.service
```

### Deploy-kommando
```bash
sudo bash /home/qt/quantum_trader/ops/deploy_emergency_exit_system.sh
```

### Verifiser
```bash
# Sjekk services
systemctl status quantum-emergency-exit-worker
systemctl status quantum-exit-brain-watchdog

# Sjekk streams
redis-cli XINFO STREAM quantum:stream:system.panic_close
redis-cli XINFO STREAM quantum:stream:exit_brain.heartbeat
```

---

## 5. Sikkerhetseffekt

| Risiko | Før | Etter |
|--------|-----|-------|
| Exit Brain crash | 🔴 Fatal | 🟢 Kontrollert |
| API stall | 🔴 Fatal | 🟡 Begrenset |
| Black Swan | 🔴 Fatal | 🟡 Overlevbar |
| Stop-hunting | 🟢 Unngått | 🟢 Unngått |

---

## 6. GO / NO-GO Sjekkliste

### GO (micro-capital) hvis:

- [ ] Emergency Exit Worker deployert
- [ ] `system.panic_close` testet (testnet)
- [ ] Exit Brain heartbeat < 2s
- [ ] Watchdog trigger < 5s
- [ ] Capital limits konservative

### Test-prosedyre (TESTNET ONLY)
```bash
# 1. Opprett test-posisjon
# 2. Trigger panic_close
redis-cli XADD quantum:stream:system.panic_close '*' \
  source ops \
  reason 'DEPLOYMENT_TEST' \
  timestamp $(date +%s)

# 3. Verifiser alle posisjoner lukket innen 5 sekunder
# 4. Verifiser panic_close.completed publisert
# 5. Verifiser system i HALTED state
```

---

## 7. Viktige Regler

### ALDRI
- ❌ Legg til retry-logikk i EEW
- ❌ Legg til optimalisering
- ❌ Legg til betingelser
- ❌ Relakser watchdog-terskler under volatilitet

### ALLTID
- ✅ Log alle panic_close events
- ✅ Krever manuell inspeksjon etter EEW-feil
- ✅ Shadow mode før live etter recovery
- ✅ Test på testnet før produksjon

---

## 8. Kontakt ved Feil

Hvis EEW eller Watchdog feiler kritisk:

1. **STOPP all trading umiddelbart**
2. Sjekk VPS tilgang
3. Sjekk exchange-posisjoner manuelt
4. Lukk posisjoner manuelt hvis nødvendig
5. Undersøk årsak før restart

---

*Dokumentasjon generert: 13. Februar 2026*
