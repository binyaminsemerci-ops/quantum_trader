# Quantum Trader – Systemarkitektur (Autonom Versjon)

Dette dokumentet beskriver hvordan plattformen er strukturert for å støtte en autonom, kontinuerlig lærende AI-handelsprosess. Fokus: sporbarhet, sikker aktivering av real trading, og modulær evolusjon (strategi → evaluering → promotering → utførelse → overvåking).

---

## 1. Domener & Lag

| Lag | Ansvar | Artefakter / Katalog |
|-----|--------|----------------------|
| Presentasjon | Dashboard, monitorering, modellstatus, signaler | `frontend/` |
| API / Orkestrering | REST + WS, oppgaver, status, risk gating | `backend/` (FastAPI) |
| Feature / Data | Henter & beriker pris-/sentimentdata, genererer features | `ai_engine/feature_engineer.py` |
| Modellering | Trening, evaluering, backtest, versjonering | `main_train_and_backtest.py`, `ai_engine/models/` |
| Modellregister | Metadata, parametre, metrics, aktiv modell | (Tabell: `model_registry`) |
| Strategi Evolusjon (future) | Genetisk / RL / ensemble pipelines | Planlagt (fase 2) |
| Utførelse | Simulert / testnet / live ordre, posisjonslogikk | (Backend loops / kommende modul) |
| Observability | Logging, heartbeats, metrics, drift alerts | `backend/`, (Prometheus plan) |
| Persistens | SQLite (default) / Postgres (valg), artifacts disk | `backend/database.py`, `ai_engine/models/` |

---

## 2. Autonom livssyklus (nå → mål)

```text
┌──────────────┐   train/backtest   ┌──────────────┐   promote   ┌──────────────┐
│ Data + Feats │ ─────────────────> │  Model Build  │ ──────────> │ ModelRegistry │
└──────┬───────┘                    └──────┬───────┘             └──────┬───────┘
       │  live prices / sentiment         │ metrics JSON                 │ active model path
       ▼                                  ▼                             ▼
   Signal Engine  <──── evaluate loop ─── Backtest / Eval ─── risk gates ───▶ Execution (sim/testnet/live)
       │                                                                   │
       └───────> WebSocket / REST ───▶ Frontend Dashboard  ◀───────────────┘
```

Promoteringskriterier (grunnlag – kan utvides):
1. Minimum Sharpe / Sortino.
2. Ingen kritisk drift (feature distribution) siste N sykluser.
3. Risiko-regler (maks drawdown, konsentrasjon) passerte i simulering.

---

## 3. Modellregister (ny tabell)
Kolonner (første versjon):
| Felt | Type | Beskrivelse |
|------|------|------------|
| id | INT PK | Sekvens |
| version | TEXT | Semver / timestamp-baserte versjoner |
| tag | TEXT | Menneskelesbar etikett (f.eks. `exp-vol-adj`) |
| path | TEXT | Relativ sti til artefakt (modellfil / katalog) |
| params_json | TEXT | JSON serialiserte hyperparametre |
| metrics_json | TEXT | JSON (Sharpe, drawdown, accuracy etc.) |
| trained_at | DATETIME | UTC sluttid for trening |
| is_active | INT (0/1) | Flag for «promotert» modell |

Aktivering skjer ved å sette `is_active=1` på én rad og `0` på andre.

---

## 4. Signal & Utførelsesflyt (nåværende status)
1. Live priser (Binance public) hentes og caches.
2. Feature pipeline beregner tekniske indikatorer (MA, RSI – utvides med volatilitet & sentiment).
3. Modell (aktiv) genererer signal (retur / score / retning).
4. Risiko-baseline filtrerer (f.eks. maks samtidige posisjoner – planlagt).
5. I demo-modus: pseudo-handler logges; i fremtid: testnet ordre.
6. Eventer publiseres til frontend (WS) + logges strukturert.

---

## 5. Planlagte utvidelser (faseinndeling)
| Fase | Fokus | Innhold |
|------|-------|---------|
| 1 | Robust grunnmur | Modellregister, ekstra metrics, baseline risk, CLI (`qtctl`) |
| 2 | Strategi Evolusjon | Genetisk pipeline + ensemble-blending |
| 3 | Testnet Utførelse | Ordre-routing, fill tracking, latency måling |
| 4 | Drift & Driftvern | Drift-deteksjon, regimeklassifisering, auto fallback |
| 5 | RL / Online | Replay buffer, off-policy evaluering, sikker rollout |

---

## 6. Kodeankre (oppdatert)
| Fil / Mappe | Rolle |
|-------------|-------|
| `backend/main.py` | FastAPI app + bakgrunnsjobber |
| `backend/database.py` | ORM-tabeller (inkl. `ModelRegistry`) |
| `ai_engine/feature_engineer.py` | Feature-generering |
| `main_train_and_backtest.py` | Trening + backtest pipeline |
| `ai_engine/models/` | Lagrede modellartefakter |
| `scripts/qtctl.py` | CLI for retrain, liste, promotere modeller |

---

## 7. Observability (målbilde)
- Heartbeats (implementert) → status-endepunkt.
- Signal staleness métric.
- Treningslatens & køtid.
- Drift-métrics (PSI / KL) – plan.
- Prometheus + Grafana dashboard – plan.

---

## 8. Risiko / Sikkerhet (inkrementell innføring)
- Kill-switch (manuell + automatisk ved overskredet drawdown).
- Audit-logg med hash-kjede (trade & beslutninger) – planlagt.
- Policy: Ingen real trading uten verifiserte quality gates (se `TODO.md`).

---

## 9. Sekvens (eksempel) – Trening → Promotering (mål)
```text
User/cron -> qtctl retrain -> pipeline (feature build, train, backtest) ->
metrics JSON -> registry insert (is_active=0) -> promoter sjekker gates ->
qtctl promote <id> -> set is_active=1 -> backend hot-reloads active model reference.
```

---

## 10. Videre arbeid
Se `TODO.md` for prioriterte neste steg. Dokumentet oppdateres når nye subsystemer materialiseres.

---

_Arkitekturfilen er kuratert etter autonom visjon – tidligere demo-fokuserte beskrivelser er arkivert i git historikk._
