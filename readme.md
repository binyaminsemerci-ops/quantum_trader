# Quantum Trader 🚀 – Fully Autonomous AI Crypto Trading Platform

> Helhetlig system: AI-drevet, kontinuerlig lærende, selv-optimaliserende trading plattform som analyserer markedet, utvikler strategier, utfører handler og evaluerer resultat – helt autonomt.

---

## 🧠 Kjerneidé

Quantum Trader er ikke bare en bot – det er et adaptivt AI-trading økosystem. Systemet:

| Kapabilitet | Status | Kort forklaring |
|-------------|--------|-----------------|
| Live markedsdata (Binance public) | ✅ | Henter priser, candlesticks, volum |
| AI signal-generering (demo + XGBoost kjerne) | ✅ | Genererer strukturert signals feed med caching |
| Portefølje P&L live | ✅ | Dynamisk P&L basert på posisjoner og live priser |
| Auto-trading service (rammeverk) | ✅ | Tjeneste/loop for AI styrt handel (demo-modus) |
| Kontinuerlig auto-trening | ✅ | API for planlagt retrening (intervaller) |
| Risiko-motor (baseline) | 🟡 | Posisjonsbegrensning + konfigurerbar kapitalbruk |
| Strategi-evolusjon (fase 2) | 🔄 | Planlagt genetisk / RL adaptiv modul |
| Multi-kilde feature pipeline | 🔄 | Tekniske + sentiment + struktur (delvis) |
| Full real trading (binance orders) | 🔜 | Klargjort i arkitektur (testnet først) |
| Observability (logging, metrics, heartbeats) | ✅ | Structured logs + periodic heartbeat |
| Frontend crash-safety | ✅ | Null-sikring + safeToFixed overalt |
| WebSockets (dash / watchlist / alerts) | ✅ | Live strømmer med fallback |

---

## 🗺️ Arkitektur Oversikt

```text
┌───────────────────────────┐
│        Frontend (React)   │
│  - FullSizeDashboard      │
│  - AI Trading Monitor     │
│  - Watchlist + Chart      │
└──────────────┬────────────┘
       │ REST / WS
┌──────────────▼────────────┐
│        FastAPI Backend    │
│  Routes: prices, signals, │
│  watchlist, portfolio, AI │
│  trading, training, ws    │
├──────────────┬────────────┤
│ Background   │ Task loops │
│ - Heartbeat  │ - Auto ML  │
│ - Evaluator  │ - AI Trade │
└──────┬───────┴───────┬────┘
   │ Feature + ML  │
┌──────▼────────────────▼───┐
│        AI Engine           │
│  Feature Engineering       │
│  XGBoost / Model Store     │
│  Strategy Framework        │
└──────────┬────────────────┘
       │ Data / State
┌──────────▼────────────────┐
│   Storage Layer (SQLite)  │
│  trades / signals / logs  │
└───────────────────────────┘
```

Se også `ARCHITECTURE.md` og `AI_TRADING_README.md` for dypdykk.

---

## 📂 Katalogstruktur (Hoved)

```text
backend/               FastAPI app, AI trading & API routes
frontend/              React/Vite dashboard (TypeScript + Tailwind)
ai_engine/             Feature eng., modeller, agenter
scripts/               Stress tester, system scripts, orchestration
config/                Konfigurasjon og miljøstyring
tests/                 Integrasjon, system- og AI-tester
artifacts/             Modell- og evalueringsartefakter
docs/                  Dokumentasjonsekstra (hvis utfylt)
```

---

## 🚀 Hurtigstart

### 1. Orkestrert oppstart (anbefalt)

PowerShell:

```powershell
./start_orchestrator.ps1
```

Velg:

```text
A = Clean   · B = Bootstrap  · C = Start  · D = Test  · F = Train AI
```

### 2. Manuell Backend

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r backend/requirements-dev.txt
uvicorn backend.main:app --reload --port 8000

```

### 3. Manuell Frontend

```powershell
cd frontend
npm install
npm run dev

```
Dashboard: <http://127.0.0.1:5173>

### 4. Docker Compose (full stack)


```powershell
docker compose up --build

```

---

## 🔐 Miljø / Konfigurasjon

Lag `.env` (se `.env.example` hvis tilgjengelig):

```env
ENABLE_LIVE_MARKET_DATA=1
BINANCE_API_KEY=...           # (valgfritt for private/trade ops)
BINANCE_API_SECRET=...
REAL_TRADING=0                # Sett til 1 for ekte (testnet først)
MODEL_REFRESH_INTERVAL=600
```

Konfig konsolideres gradvis i `backend/config.py`.

---

## 🤖 AI & Auto-Trading

Se `AI_TRADING_README.md` for komplett detalj, men nøkkelkommandoer:

Start dedikert AI trading backend:

```powershell
python start_ai_trading_backend.py
```

API (eksempler):

```http
GET  /api/v1/ai-trading/status
POST /api/v1/ai-trading/start
POST /api/v1/ai-trading/stop
POST /api/v1/ai-trading/config
GET  /api/v1/ai-trading/signals
```

Auto-trening:

```http
POST /api/v1/training/auto/enable?interval_minutes=10
POST /api/v1/training/run-once?limit=1200
```

Tren & backtest lokalt:

```powershell
python main_train_and_backtest.py train
python main_train_and_backtest.py backtest --symbols BTCUSDC ETHUSDC
```

---

## 📡 API Høydepunkter

| Endpoint | Beskrivelse |
|----------|-------------|
| GET /watchlist/full | Utvidet watchlist m/ live priser + sparkline |
| GET /portfolio/ | P&L, posisjoner, total verdi |
| GET /signals/recent | AI/demosignaler (cache 30s) |
| GET /api/v1/system/status | Uptime, CPU, memory, eval status |
| WS /ws/watchlist | Live prisoppdateringer |
| WS /ws/alerts | Signal / alert push |
| GET /api/v1/training/status | ML retrening status |
| POST /api/v1/trading/auto/enable | Starter demo trading loop |
| GET /api/v1/model/active | Aktiv modell metadata (version, tag, id) |

Swagger/OpenAPI: `http://localhost:8000/docs`

---

## 🧪 Testing

Backend:

```powershell
pytest -q
```
Frontend:

```powershell
cd frontend
npm run test
```
Stress / e2e:

```powershell
python scripts/stress/harness.py --count 1
```

---

## 🛠️ Utvikling & Kvalitet

- Lint: `ruff`, `mypy`
- Typecheck: `npm run typecheck`
- Pre-commit: se `.pre-commit-config.yaml`
- Logging: strukturert + heartbeat hver 15s
- Metrics: basis (kan utvides til Prometheus)
       - Modell: Sharpe, Sortino & max_drawdown gauges (oppdateres ved ny trening)
       - Prometheus: /metrics (inkluderer quantum_model info)

---

## 📈 Observability

- Heartbeat logger i `backend/main.py`
- System status endpoint (`/api/v1/system/status`)
- Plan: Prometheus + grafana dashboard (TODO)

---

## 🛡️ Risiko & Sikkerhet

- Ingen ekte ordre sendes med mindre `REAL_TRADING=1` + gyldige keys
- Fallback til demo-data hvis eksterne kilder feiler
- Planlagt: rate limiting, signert audit logg, secrets vault

---

## 🔭 Roadmap (Kort)

| Tema | Neste Milepæl |
|------|---------------|
| AI Strategy Evolution | Genetisk pipeline + RL scoring |
| Full Trading | Binance testnet ordre + fills tracking |
| Risk Engine | Volatilitet-adaptive posisjoner |
| Sentiment | Integrere CryptoPanic + klassifisering |
| Charting | Candlestick + indikator overlays |
| Monitoring | Prometheus + alert regler |
| Portfolio | Multi-asset capital allocation modeller |

Detaljert liste i `TODO.md`.

---

## 🤝 Bidra

1. Fork & branch: `feature/<navn>`
2. Kjør tester før PR
3. Oppdater `TODO.md` ved funksjonell scope-endring
4. Legg til korte notater i `CHANGELOG.md` hvis relevant

---


## ⚠️ Disclaimer

Dette prosjektet er for forskning og eksperimentering. **Ingen garanti** for fortjeneste. All bruk skjer på eget ansvar. Test alltid i *demo / testnet* før ekte kapital eksponeres.

---


## 📄 Lisens

(Legg inn lisensfil hvis ønskelig – MIT / Apache 2.0 anbefales.)

---

## 🧾 Hurtig Kommandoreferanse

```powershell
# Start backend
uvicorn backend.main:app --reload

# Start AI trading backend
python start_ai_trading_backend.py

# Start frontend
cd frontend; npm run dev

# Trening + backtest
python main_train_and_backtest.py train

# Auto-trening hver 10 min
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/training/auto/enable?interval_minutes=10

# Hente watchlist
curl http://localhost:8000/watchlist/full

# Modellregister (CLI)
python scripts/qtctl.py list
python scripts/qtctl.py active
python scripts/qtctl.py register --version v2025.10.03.1 --path ai_engine/models/model-x.bin --activate
python scripts/qtctl.py promote --id 5
```

---
*Denne README er generert som helhetlig status + visjon. Synkroniser alltid med `TODO.md` for oppgaver.*
