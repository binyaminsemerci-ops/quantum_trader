# PRODUCTION RUNBOOK - QSC / CANARY DEPLOY

**Formål:** Sikker utrulling av nye AI-modeller med automatisk kvalitetskontroll og rollback

**Varighet:** 6 timer overvåkning  
**Kritikalitet:** HIGH - Automatisk rollback ved brudd

---

## 🔒 ABSOLUTTE REGLER

```
✗ Ingen PASS = ingen deploy
✗ Ingen telemetry = ingen beslutning  
✗ Ingen manuell overstyring
```

---

## STEG 1: BEKREFT TELEMETRY

### Sjekk antall events

```bash
redis-cli XLEN quantum:stream:trade.intent
```

**Krav:** ≥ 200 events

**Hvis <200:**
- STOPP - Vent på mer data
- Ingen deploy tillatt

---

## STEG 2: KJØR QUALITY GATE

### Hent cutover timestamp

```bash
# Hent AI engine restart-tidspunkt
systemctl show quantum-ai_engine.service -p ActiveEnterTimestamp

# Konverter til ISO format (eksempel)
# Fri 2026-01-10 05:43:15 UTC → 2026-01-10T05:43:15Z
```

### Kjør quality gate (post-cutover analyse)

```bash
CUTOVER_TS="2026-01-10T05:43:15Z"  # Bytt med faktisk timestamp

python3 ops/model_safety/quality_gate.py --after $CUTOVER_TS
```

### Vurder resultat

**Exit code 0 = GRØNT (gå videre)**
```
✅ QUALITY GATE: PASS
   Safe to proceed with canary activation
```

**Exit code 2 = RØDT (STOPP)**
```
❌ QUALITY GATE: FAIL (BLOCKER)
   Violations detected = NO ACTIVATION
```

**Hvis FAIL:**
- STOPP deployment
- Les rapport: `reports/safety/quality_gate_*.txt`
- Feilsøk modell-issue
- Retry etter fix

---

## STEG 3: START CANARY (10% TRAFIKK)

### Dry run (anbefalt først)

```bash
python3 ops/model_safety/qsc_mode.py \
  --model patchtst \
  --cutover $CUTOVER_TS \
  --dry-run
```

**Forventet output:**
```
✅ QUALITY GATE PASSED
   342 post-cutover events analyzed
```

### Aktiver canary

```bash
# Velg modell: xgb, lgbm, nhits, eller patchtst
MODEL="patchtst"

python3 ops/model_safety/qsc_mode.py \
  --model $MODEL \
  --cutover $CUTOVER_TS
```

**Forventet output:**
```
[ACTIVATED] CANARY ACTIVATED
================================================================================
  Model:         patchtst
  Weight:        10.0%
  Start Time:    2026-01-10T12:00:00Z
  Monitor For:   6 hours

Weights:
  [*] patchtst    10.0%  (canary)
  [ ] xgb         28.1%
  [ ] lgbm        28.1%
  [ ] nhits       33.8%

Rollback Command:
  python3 ops/model_safety/qsc_rollback.sh
================================================================================
```

**Filer opprettet:**
- `data/baseline_model_weights.json` - Backup av originale vekter
- `data/qsc_canary_weights.json` - Nye canary-vekter
- `data/systemd_overrides/qsc_canary.conf` - Systemd override
- `logs/qsc_canary.jsonl` - Aktiveringslogg

### Restart AI engine

```bash
sudo systemctl restart quantum-ai_engine.service
```

### Verifiser canary er aktiv

```bash
# Sjekk at service startet OK
sudo systemctl status quantum-ai_engine.service

# Sjekk logs for canary-melding
sudo journalctl -u quantum-ai_engine.service -n 50 | grep -i canary
```

---

## STEG 4: OVERVÅK 6 TIMER

### Start monitoring daemon

```bash
# Foreground (se output direkte)
python3 ops/model_safety/qsc_monitor.py

# Eller background
nohup python3 ops/model_safety/qsc_monitor.py > logs/qsc_monitor.log 2>&1 &
echo $! > logs/qsc_monitor.pid
```

### Overvåkingsdetaljer

**Intervall:** 30 sekunder  
**Varighet:** 6 timer (720 checks)  
**Auto-rollback:** JA - ved ethvert brudd

### Følg med på status

```bash
# Watch canary log (real-time)
tail -f logs/qsc_canary.jsonl | jq .

# Sjekk scoreboard (oppdatert hver 30s)
watch -n 30 cat reports/safety/scoreboard_latest.md

# Monitor daemon output
tail -f logs/qsc_monitor.log
```

---

## STEG 5: VURDER RESULTAT

### Scenario A: INGEN BRUDD (6 timer fullført)

**Output:**
```
================================================================================
✅ MONITORING COMPLETED - NO VIOLATIONS
================================================================================
Canary Model:      patchtst
Duration:          6.0 hours
Total Checks:      720

Canary is safe to promote to full production.
```

**Aksjon:**
```bash
# Canary er sikker - promoter til høyere %
# Manuell prosess (ikke automatisk)

# Eksempel: Øk til 25%
# 1. Rediger data/qsc_canary_weights.json
# 2. Restart AI engine
# 3. Overvåk på nytt i 6 timer
# 4. Gjenta til 100%
```

### Scenario B: BRUDD OPPDAGET (auto-rollback utført)

**Output:**
```
================================================================================
🚨 VIOLATION DETECTED - EXECUTING ROLLBACK
================================================================================
Canary Model: patchtst

Violations:
  ❌ Action collapse: HOLD=78.3% (>70%)
  ❌ Flat predictions: conf_std=0.0412 (<0.05)

Rollback Script: ops/model_safety/qsc_rollback.sh

✅ Rollback completed successfully
```

**Aksjon:**
```bash
# Rollback allerede utført automatisk
# Verifiser at baseline er gjenopprettet:

cat data/baseline_model_weights.json

# Sjekk at AI engine kjører med baseline:
sudo systemctl status quantum-ai_engine.service
```

**Feilsøking:**
```bash
# Les full rollback-logg
cat logs/qsc_canary.jsonl | jq 'select(.action=="rollback_executed")'

# Kjør quality gate for diagnose
python3 ops/model_safety/quality_gate.py --after $CUTOVER_TS

# Sjekk scoreboard for detaljer
cat reports/safety/scoreboard_latest.md
```

---

## MANUELL ROLLBACK (VED BEHOV)

### Hvis du må stoppe canary manuelt

```bash
# Utfør rollback
bash ops/model_safety/qsc_rollback.sh

# Eller Python-versjon
python3 ops/model_safety/qsc_mode.py --rollback
```

**Hva skjer:**
1. Fjerner systemd override (`qsc_canary.conf`)
2. Gjenoppretter baseline vekter
3. Restarter AI engine
4. Logger rollback-event

### Verifiser rollback

```bash
# Sjekk at override er fjernet
ls /etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf
# Skal gi: No such file or directory

# Sjekk at baseline er aktiv
sudo journalctl -u quantum-ai_engine.service -n 50 | grep -i weight
```

---

## VIOLATION TRIGGERS (AUTO-ROLLBACK)

Monitor oppdager disse bruddene automatisk:

| Violation | Terskel | Beskrivelse |
|-----------|---------|-------------|
| **Action Collapse** | >70% | En action-klasse dominerer |
| **Dead Zone** | HOLD >85% | Modellen holder for mye |
| **Flat Predictions** | conf_std <0.05 | Ingen variasjon i confidence |
| **Narrow Range** | P10-P90 <0.12 | For lite spredning |
| **Quality Gate Fail** | Status=NO-GO | Feiler kvalitetssjekk |
| **Ensemble Dysfunction** | Agreement <55% eller >80% | Modellene er ikke enige |
| **Model Chaos** | Hard disagree >20% | For mye uenighet |
| **Constant Output** | std <0.01 | Modellen er "frossen" |

**Ethvert brudd → Øyeblikkelig rollback**

---

## TROUBLESHOOTING

### Problem: "Quality gate failed"

**Symptomer:** qsc_mode.py exit code 1

**Diagnose:**
```bash
# Kjør quality gate direkte
python3 ops/model_safety/quality_gate.py --after $CUTOVER_TS

# Les rapport
cat reports/safety/quality_gate_*.txt
```

**Løsninger:**
- <200 events → Vent på mer data
- Collapse oppdaget → Retrain modell
- Flat predictions → Sjekk confidence-kalibrering

### Problem: "Systemd override not installed"

**Symptomer:** Override finnes i `data/systemd_overrides/` men ikke i `/etc/systemd/`

**Løsning:**
```bash
# Installer manuelt (krever sudo)
sudo cp data/systemd_overrides/qsc_canary.conf \
  /etc/systemd/system/quantum-ai_engine.service.d/

sudo systemctl daemon-reload
sudo systemctl restart quantum-ai_engine.service
```

### Problem: "Monitoring check failed"

**Symptomer:** qsc_monitor.py rapporterer feil i checks

**Diagnose:**
```bash
# Test scoreboard manuelt
python3 ops/model_safety/scoreboard.py

# Sjekk Redis
redis-cli PING
redis-cli XLEN quantum:stream:trade.intent

# Les siste events
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 10
```

### Problem: "Redis connection refused"

**Symptomer:** `ConnectionError: Error 10061 connecting to localhost:6379`

**Løsning:**
```bash
# Start Redis
sudo systemctl start redis

# Verifiser
redis-cli PING
# Skal svare: PONG
```

---

## LOGGING & AUDIT

### Canary log (JSONL format)

```bash
# Full log
cat logs/qsc_canary.jsonl | jq .

# Kun aktiveringer
cat logs/qsc_canary.jsonl | jq 'select(.action=="canary_activated")'

# Kun rollbacks
cat logs/qsc_canary.jsonl | jq 'select(.action=="rollback_executed")'

# Kun vellykkede monitors
cat logs/qsc_canary.jsonl | jq 'select(.action=="monitoring_completed")'
```

### AI engine logs

```bash
# Følg live
sudo journalctl -u quantum-ai_engine.service -f

# Søk etter canary-meldinger
sudo journalctl -u quantum-ai_engine.service | grep -i canary

# Sjekk vekt-konfigurasjon
sudo journalctl -u quantum-ai_engine.service | grep -i weight
```

### Redis telemetry

```bash
# Antall events
redis-cli XLEN quantum:stream:trade.intent

# Siste 100 events
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100

# Events etter cutover
redis-cli XRANGE quantum:stream:trade.intent $CUTOVER_STREAM_ID + COUNT 500
```

---

## QUICK REFERENCE

### Kommandoer (copy-paste)

```bash
# 1. Sjekk telemetry
redis-cli XLEN quantum:stream:trade.intent

# 2. Quality gate
CUTOVER_TS="2026-01-10T05:43:15Z"  # BYTT DETTE
python3 ops/model_safety/quality_gate.py --after $CUTOVER_TS

# 3. Aktiver canary (hvis PASS)
python3 ops/model_safety/qsc_mode.py --model patchtst --cutover $CUTOVER_TS
sudo systemctl restart quantum-ai_engine.service

# 4. Start overvåkning
python3 ops/model_safety/qsc_monitor.py

# 5. Følg status
tail -f logs/qsc_canary.jsonl | jq .
```

### Filer å overvåke

```
logs/qsc_canary.jsonl              - Canary events
logs/qsc_monitor.log               - Monitor output
reports/safety/scoreboard_latest.md - Model status
data/baseline_model_weights.json   - Backup weights
data/qsc_canary_weights.json       - Active canary weights
```

### Emergency rollback

```bash
bash ops/model_safety/qsc_rollback.sh
```

---

## SIKKERHETSKONTROLLER

Før hver deploy, verifiser:

- [ ] Redis kjører: `redis-cli PING` → PONG
- [ ] ≥200 events: `redis-cli XLEN quantum:stream:trade.intent`
- [ ] Cutover timestamp er korrekt
- [ ] Quality gate exit 0
- [ ] Backup av baseline finnes
- [ ] Rollback-script er kjørbart: `bash -n ops/model_safety/qsc_rollback.sh`

Under deploy:

- [ ] Canary aktivert OK
- [ ] AI engine restartet
- [ ] Monitor kjører
- [ ] Scoreboard oppdateres (hver 30s)

Etter 6 timer:

- [ ] Monitor fullført ELLER rollback utført
- [ ] Logg er komplett i `logs/qsc_canary.jsonl`
- [ ] AI engine kjører stabilt

---

## ESKALERING

### Ved akutt problem

1. **Stopp canary øyeblikkelig:**
   ```bash
   bash ops/model_safety/qsc_rollback.sh
   ```

2. **Verifiser baseline gjenopprettet:**
   ```bash
   sudo systemctl status quantum-ai_engine.service
   ```

3. **Samle diagnostikk:**
   ```bash
   # Full canary log
   cat logs/qsc_canary.jsonl > /tmp/qsc_incident_$(date +%s).json
   
   # Scoreboard
   cp reports/safety/scoreboard_latest.md /tmp/scoreboard_incident_$(date +%s).md
   
   # AI engine logs (siste 1000 linjer)
   sudo journalctl -u quantum-ai_engine.service -n 1000 > /tmp/ai_engine_incident_$(date +%s).log
   ```

---

## CONTACT

**Dokumentasjon:**
- Runbook: `PRODUCTION_RUNBOOK_QSC_CANARY.md` (denne filen)
- Full guide: [QSC_MODE_DOCUMENTATION.md](QSC_MODE_DOCUMENTATION.md)
- Quick ref: [QSC_MODE_README.md](QSC_MODE_README.md)

**Scripts:**
- Aktivering: `ops/model_safety/qsc_mode.py`
- Overvåkning: `ops/model_safety/qsc_monitor.py`
- Rollback: `ops/model_safety/qsc_rollback.sh`
- Test: `ops/model_safety/qsc_test.py`

---

**Versjon:** 1.0  
**Dato:** 2026-01-10  
**Status:** PRODUCTION READY  
**Språk:** Norsk (Norwegian Bokmål)
