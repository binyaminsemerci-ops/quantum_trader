# Quantum Trader — Systemarkitektur

Dette dokumentet gir en kort oppsummering av hvordan Quantum Trader er bygd, hvilke komponenter som finnes og hvor i repoet de befinner seg.

## Hovedkomponenter

- Backend (FastAPI + Python)
  - Katalog: `backend/` (filer: `main.py`, `routes/`, `database.py`, `train_and_save.py`)
  - Ansvar: Ingest, API for predict/scan/train, bakgrunnsjobber, model serving

- Database (Postgres)
  - Konfigurasjon: `docker-compose.yml` og `backend/database.py`
  - Tabeller: prices, features, signals, trades, trade_logs, settings

- AI Engine
  - Katalog: `ai_engine/`
  - Ansvar: Feature engineering (`feature_engineer.py`), treningsskall (`train_and_save.py`), agenter (`agents/`)

- Frontend (React + TypeScript + Vite + Tailwind)
  - Katalog: `frontend/` (src/ med `components/`, `api/`, `pages/`)
  - Ansvar: Dashboard, grafer, trade-logg, realtime visning

## Dataflyt (kort)
1. Ingest jobber henter data fra Binance / CryptoPanic / Twitter.
2. Data blir lagret i Postgres.
3. Feature engineering bruker data fra Postgres til å lage features.
4. AI-modellen (XGBoost) trenes og lagres (pickle eller JSON fallback).
5. Backend eksponerer predict/scan/train API-er som frontend bruker.
6. Handelsmotor oversetter signaler til ordre mot Binance.

## Lokalt kjøreeksempel (Docker Compose)

1. Bygg og start alle tjenester:

```powershell
docker-compose up --build
```

2. Backend: http://localhost:8000
3. Frontend: http://localhost:3000
4. Helse-endepunkt: http://localhost:8000/api/health

## Frontend utvikling (lokalt)

Gå til `frontend/` og kjør:

```bash
npm install
npm run dev
```

## Fil-tilknytning (hurtigreferanse)
- `ai_engine/feature_engineer.py` → Feature engineering pipeline
- `ai_engine/train_and_save.py` → Training harness + artifact saving
- `backend/routes/ai.py` → API-endepunkter for predict/scan/train
- `backend/database.py` → SQLAlchemy, session factory, get_db
- `frontend/src/components/CandlesChart.tsx` → Candles UI (henter fra /api/candles)

## Diagram
- `frontend/src/assets/system-architecture.svg` inneholder en visuell oversikt (SVG).

---

Vil du at jeg skal:
- Generere en mer detaljert sekvensdiagram for én trade?
- Legge til en CI/CD pipeline-fil i `frontend/` som bygger og deployer til GitHub Pages / Vercel?
- Lage en fullstendig `frontend/` scaffold (komponenter og TS-typer) som kan bygges direkte?

Gi beskjed hva jeg skal lage videre — jeg kan begynne å generere frontend-komponenter, Recharts-candlestick, eller en detaljert roadmap med tickets og prioriteringer.
