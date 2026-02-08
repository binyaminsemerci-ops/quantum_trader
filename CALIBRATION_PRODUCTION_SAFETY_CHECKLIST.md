# 🔒 PROD-SIKKERHETSSJEKK — CALIBRATION-ONLY

**Dette er bindende. Hvis én sjekk feiler → STOPP.**

---

## FASE 0 — FORUTSETNINGER (MÅ VÆRE SANT FØR NOE KAN KJØRES)

### 0.1 Systemhelse

**Alle disse må være grønne:**

```bash
systemctl status quantum-ai-engine       # ✅ ACTIVE
systemctl status quantum-execution       # ✅ ACTIVE
systemctl status quantum-learning-monitor # ✅ ACTIVE
systemctl status quantum-learning-api    # ✅ ACTIVE

# Ingen ERROR/TRACEBACK siste 30 min
journalctl -u quantum-ai-engine --since "30 minutes ago" | grep -i error

# Redis latency < 50ms
redis-cli --latency-history
```

- [ ] quantum-ai-engine = ACTIVE
- [ ] quantum-execution = ACTIVE
- [ ] quantum-learning-monitor = ACTIVE
- [ ] quantum-learning-api = ACTIVE
- [ ] Ingen ERROR/TRACEBACK siste 30 min i AI Engine
- [ ] Redis latency < 50ms (ingen backlog)

**❌ Hvis én feiler → IKKE START**

---

### 0.2 Data-integritet (CLM)

```bash
# Sjekk CLM-data
wc -l /home/qt/quantum_trader/data/clm_trades.jsonl
python -c "
import json
from pathlib import Path

file = Path('/home/qt/quantum_trader/data/clm_trades.jsonl')
trades = [json.loads(line) for line in file.read_text().splitlines()]

print(f'Trades: {len(trades)}')
print(f'Symbols: {len(set(t[\"symbol\"] for t in trades))}')
wins = sum(1 for t in trades if t['actual_pnl_pct'] > 0)
losses = sum(1 for t in trades if t['actual_pnl_pct'] < 0)
print(f'Wins: {wins}, Losses: {losses}')

# Sjekk for bad data
for i, t in enumerate(trades, 1):
    if not t.get('entry_price') or not t.get('exit_price'):
        print(f'❌ Trade {i}: Missing price')
    if t.get('actual_pnl_pct') == 0.0:
        print(f'⚠️ Trade {i}: Zero PnL')
"
```

- [ ] `/data/clm_trades.jsonl` eksisterer
- [ ] `line_count >= cadence.min_trades`
- [ ] Minst 2 symbols
- [ ] Minst 1 WIN og 1 LOSS
- [ ] Ingen NaN, null, 0.0 i entry/exit/pnl

**❌ Hvis data må "renses manuelt" → STOPP**

---

### 0.3 Learning Cadence Gate (HARD)

```bash
# Fra API
curl -s http://localhost:8003/api/learning/readiness/simple | jq '.'
```

Forventet output:
```json
{
  "ready": true,
  "actions": ["calibration"]
}
```

- [ ] `ready == true`
- [ ] `"calibration"` i `actions`
- [ ] IKKE `"retrain"`, `"weights_update"` eller lignende

**❌ Hvis cadence ikke er READY → STOPP**

---

## FASE 1 — PRE-FLIGHT LOCKDOWN (FØR CALIBRATION START)

### 1.1 Skrivebeskyttelse (KONSEPTUELL)

**Bekreft at følgende IKKE skjer automatisk:**

- [ ] Ingen modell weights skrives
- [ ] Ingen ensemble weights endres
- [ ] Ingen config files oppdateres
- [ ] Ingen Meta-Agent parametere endres
- [ ] Ingen restart trigges

**Calibration = analyse + forslag, ikke handling.**

---

### 1.2 Isolasjon

- [ ] Calibration leser kun CLM-data
- [ ] Ingen live market data brukes
- [ ] Ingen redis streams konsumeres (read-only file)

---

## FASE 2 — CALIBRATION-KJØRING (INNENFOR RAMMER)

### 2.1 Tillatte operasjoner

**Calibration har kun lov til å:**

- [ ] Beregne confidence calibration (ex. isotonic)
- [ ] Sammenligne pre/post error (MSE/Brier)
- [ ] Generere rapport (markdown/json)
- [ ] Foreslå parametre (ikke anvende)

**Alt annet = brudd**

---

### 2.2 Hard Safety Bounds (MÅ OVERHOLDES)

| Parameter | Maks tillatt |
|-----------|-------------|
| Confidence endring | ±15% absolut |
| MSE-forbedring krav | ≥ 5% |
| Degradering tillatt | 0% |
| Antall modeller påvirket | ≤ 100% (ingen selektiv skjult endring) |

```bash
# Start calibration
python microservices/learning/calibration_cli.py run

# Les rapport
cat /tmp/calibration_*.md
```

**Sjekk i rapporten:**

- [ ] MSE improvement ≥ 5%
- [ ] Ingen modell blir dårligere
- [ ] Confidence endring innenfor ±15%

**❌ Hvis MSE forbedres < 5% → IKKE APPLY**  
**❌ Hvis én modell blir dårligere → STOPP**

---

## FASE 3 — RESULTATVALG (MENNESKE I LØKKA)

### 3.1 Rapportkrav (MÅ FINNES)

- [ ] Sammendrag (1 side)
- [ ] Før/etter metrics
- [ ] Confidence reliability plot (eller tabell)
- [ ] Risiko-seksjon
- [ ] "DO NOT APPLY IF …" seksjon

**Manglende rapport = ingen beslutning**

---

### 3.2 Beslutningsknapp (MANUELL)

**Kun én av disse er lov:**

```bash
# ✅ APPLY
python microservices/learning/calibration_cli.py approve <job_id>

# ❌ ABORT
# (gjør ingenting, job blir arkivert)
```

- [ ] Rapport gjennomgått
- [ ] Alle safety bounds OK
- [ ] Beslutningstaker tilgjengelig for 24t monitoring

**Ingen "midlertidig", ingen "delvis".**

---

## FASE 4 — APPLY-SIKKERHET (KUN HVIS DU VELGER APPLY)

### 4.1 Atomisk oppdatering

```bash
# Når approve kjøres:
# 1. Deployment skaper ny /config/calibration.json
# 2. Gammel config arkiveres til /config/calibration_archive/calibration_<version>.json
# 3. AI Engine hot-reloader config (ingen restart nødvendig)
```

- [ ] Endring lagres i ny versjon
- [ ] Gammel versjon intakt i archive
- [ ] Én restart maks (AI Engine) - kun hvis hot-reload feiler
- [ ] Tydelig versjons-ID i logs

**Verifiser deployment:**

```bash
# Sjekk at config er lastet
journalctl -u quantum-ai-engine -n 50 | grep -i calibration

# Forventet:
# [CalibrationLoader] ✅ Loaded config version: cal_20260220_143022
# [CalibrationLoader] Confidence calibration: enabled
# [CalibrationLoader] Ensemble weights: 4 models
```

---

### 4.2 Post-Apply Watch (24t)

```bash
# Monitor metrics:
# 1. PnL drift
tail -f /home/qt/quantum_trader/data/clm_trades.jsonl | jq '.actual_pnl_pct'

# 2. Win-rate delta
python -c "
import json
from pathlib import Path
trades = [json.loads(line) for line in Path('/home/qt/quantum_trader/data/clm_trades.jsonl').read_text().splitlines()]
recent_20 = trades[-20:]
win_rate = sum(1 for t in recent_20 if t['actual_pnl_pct'] > 0) / len(recent_20)
print(f'Win rate (last 20): {win_rate:.1%}')
"

# 3. HOLD-rate
journalctl -u quantum-ai-engine --since "1 hour ago" | grep "HOLD" | wc -l

# 4. Confidence distribution
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -oP "confidence=\K[0-9.]+" | python -c "
import sys
confs = [float(line.strip()) for line in sys.stdin]
if confs:
    print(f'Avg: {sum(confs)/len(confs):.3f}')
    print(f'Min: {min(confs):.3f}')
    print(f'Max: {max(confs):.3f}')
"
```

**Monitor i 24 timer:**

- [ ] PnL drift innenfor normalt (<10% standardavvik)
- [ ] Win-rate delta <5% endring
- [ ] HOLD-rate ikke dramatisk økt (>50% mer enn baseline)
- [ ] Confidence distribution rimelig (0.50-0.95 range)

**Hvis avvik > terskel → ROLLBACK**

```bash
# Rollback
python microservices/learning/calibration_cli.py rollback
# eller
python microservices/learning/calibration_cli.py rollback <specific_version>
```

---

## 🚨 ABSOLUTTE STOPP-SIGNALER

**Calibration MÅ IKKE kjøres hvis:**

- [ ] Data < cadence minimum
- [ ] Marked ekstremt (flash crash, halt, voldum >300% normalt)
- [ ] Execution har errors (siste 1t)
- [ ] Du ikke er tilgjengelig for review og 24t monitoring
- [ ] Risk-Safety Service rapporterer HALT
- [ ] SimpleCLM viser STARVATION (>2t uten trades)

---

## 🧠 FILOSOFI (LÅST)

> **Calibration er korreksjon av ærlighet,  
> ikke optimalisering av profitt.**

> **Hvis vi er i tvil → vi lærer senere, ikke nå.**

---

## 📋 QUICK CHECKLIST

**Pre-flight (5 min):**
```
[ ] Fase 0.1: Alle services ACTIVE
[ ] Fase 0.2: CLM data valid (≥50 trades, 2+ symbols, wins+losses)
[ ] Fase 0.3: Learning Cadence READY
[ ] Fase 1: Skrivebeskyttelse konseptuelt forstått
```

**Run (10 min):**
```
[ ] calibration_cli.py run
[ ] Fase 2.2: MSE improvement ≥5%, ingen degradering
[ ] Fase 3.1: Rapport komplett og gjennomgått
```

**Deploy (5 min):**
```
[ ] calibration_cli.py approve <job_id>
[ ] Fase 4.1: Verifiser CalibrationLoader lastet config
```

**Monitor (24t):**
```
[ ] Fase 4.2: PnL, win-rate, HOLD-rate, confidence OK
[ ] Rollback hvis nødvendig
```

---

**Total tid:** 20 min aktiv + 24t passiv monitoring  
**Blocker:** Learning Cadence (50+ trades)  
**Ansvarlig:** Menneske (ikke automatisert)
