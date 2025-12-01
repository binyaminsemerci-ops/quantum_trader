# Quantum Trader – Oppsummering

Denne filen er en samlet oppsummering av Quantum Trader-prosjektet og en prioritert "to do"-liste for å ta prosjektet fra nåværende til et profesjonelt, produksjonsklart system.

## 1. Hva er Quantum Trader?

Quantum Trader er en AI-drevet kryptohandelsplattform bestående av tre hoveddeler:

- Backend (FastAPI + Python) → styrer datainnsamling, AI-modell og ordreutførelse.
- Database (PostgreSQL + Alembic) → lagrer trades, signaler, nyheter, tweets og historiske data.
- Frontend (React + Vite + TypeScript + Tailwind) → dashboard med grafer, signaler og oversikt.

Kort sagt: en tradingbot + analyseterminal i én pakke.

## 2. Hva skal den brukes til?

Quantum Trader har to bruksområder:

1) Analyseplattform
- Henter data fra børser (e.g. Binance).
- Henter nyheter (CryptoPanic) og sentiment fra Twitter/X.
- Kjører AI-modeller (f.eks. XGBoost) for BUY/SELL/HOLD-prediksjoner.
- Viser resultater i frontend for beslutningsstøtte uten å koble til konto.

2) Fullverdig Tradingbot
- Kobles til børser via API-nøkler.
- Når modellen sier BUY/SELL, kan botten utføre faktiske ordre.
- Logger alt i DB (ordre, tidspunkt, pris, mengde) og viser handelshistorikk i frontend.

## 3. Hvordan fungerer den teknisk? (Flyt steg for steg)

1. Datainnsamling
   - Binance API → priser og OHLCV.
   - CryptoPanic → nyheter.
   - Twitter/X → tweets for sentiment.
   - Alt lagres i PostgreSQL-tabeller (prices, news, tweets).

2. Feature engineering
   - Beregn tekniske indikatorer (MA, EMA, RSI, MACD, Bollinger Bands).
   - Beregn sentimentscore og aggreger over tid.
   - Lag en features-tabell som brukes av AI-modellen.

3. AI-modell
   - XGBoost (eller lignende) trenes på historiske features.
   - Output: BUY / SELL / HOLD.
   - Signalene lagres i en signals-tabell.

4. Handelsmotor
   - Les siste modell-signal.
   - Beslutningslogikk (risikoreglene, posisjon sizing).
   - Utfør ordre via exchange API (først testnet, så live hvis ønsket).
   - Logg resultat i trades-tabellen.

5. Frontend dashboard
   - Sanntid visning av prisgraf (candlesticks), AI-signaler, sentiment og trade-logg.

6. Kontinuerlig loop
   - Systemet kjører periodisk (minutt / time) og oppdaterer data, features, modeller og utfører trading ved behov.

## 4. Hvorfor bygge Quantum Trader?

- Automatisering – AI kan handle 24/7.
- Datadrevet beslutning – kombinerer markedsdata + sentimentanalyse.
- Testbarhet – backtesting og testnet først.
- Utvidbarhet – flere børser, strategier og AI-modeller.

## 5. Målbilde

Quantum Trader skal kunne fungere både som:
- Trading decision support terminal for manuelt bruk.
- Autonom tradingbot koblet mot børs(er) som kan handle på vegne av brukeren.

---

## Systemarkitektur (kort)

- Backend: FastAPI + SQLAlchemy + Alembic, scripts for datainnsamling og modelltrening.
- Database: PostgreSQL, migrasjoner med Alembic.
- Exchanges: Adapter-mønster (støtte for Binance, Coinbase, KuCoin) — ccxt brukes i adaptere, men kan holdes som en valgfri avhengighet.
- Frontend: React + Vite + TypeScript + Tailwind. Komponenter i `.tsx`/`.ts`.
- CI: GitHub Actions, med egen integrasjonsjobb for tunge avhengigheter (f.eks. ccxt).
- Containerisering: Docker + docker-compose for lokal utvikling.

---

## Forslag til fil- og frontend-struktur (TS/TSX)

frontend/
 ├── src/
 │   ├── components/
 │   │   ├── Header.tsx
 │   │   ├── Sidebar.tsx
 │   │   ├── Dashboard.tsx
 │   │   ├── PriceChart.tsx  (Recharts / TradingView)
 │   │   ├── TradeLog.tsx
 │   │   ├── SignalFeed.tsx
 │   │   └── SentimentFeed.tsx
 │   ├── pages/
 │   │   ├── Home.tsx
 │   │   ├── Trades.tsx
 │   │   └── Signals.tsx
 │   ├── api/
 │   │   └── client.ts
 │   ├── types/
 │   └── App.tsx

---

## Docker-compose (lokal utvikling)

Se `docker-compose.yml` i repo for et eksempel-oppsett som starter backend, frontend og en PostgreSQL database.

---

## Flytscenario for én trade (step-by-step)

1. Hent sanntids OHLCV fra Binance.
2. Feature engineering (indikatorer + sentiment).
3. Modell predikerer BUY/SELL/HOLD.
4. Handelsmotor vurderer risiko og posisjon-sizing.
5. Hvis BUY/SELL → send ordre til exchange via API.
6. Logg resultat i DB og oppdater frontend i real-time.
7. Repeter kontinuerlig.

---

## TODO-liste (prioritert)

Denne TODO-listen er i prioritert rekkefølge. Hver oppgave har et forslag til filer/mapper å opprette eller oppdatere.

1. Kritisk: Sikkerhet og hemmelighetshåndtering
   - [ ] Centraliser secrets (bruk env + .env, ikke sjekk inn nøkler).
     - Filer: `config/config.py`, `backend/.env.example`, `frontend/.env.example`
   - [ ] Implementer masking av nøkler i logger og admin-UI.
     - Filer: `backend/utils/startup.py`, `frontend/src/pages/Settings.tsx`

2. Kritisk: CI & Integrasjonspolicy
   - [ ] Hold tunge avhengigheter valgfrie (flytt ccxt til `backend/requirements-ccxt.txt`).
   - [ ] Lag en separat GitHub Actions job for integrasjonstester (kun triggered/dispatch eller for maintainers).
     - Fil: `.github/workflows/ci.yml`

3. Høy: Backend testdekning og adapter-tester
   - [ ] Unit-tester for exchange-adapter factory (mock ccxt).
     - Filer: `backend/tests/test_exchanges.py`.
   - [ ] Integrasjonstest som kjører `backend/scripts/adapter_smoke.py` mot testnet (kjøres i integrasjonsjobben).

4. Høy: AI, feature engineering og treningspipeline
   - [ ] Implementer feature-engineering scripts (lag `ai_engine/feature_engineer.py`).
   - [ ] Treningspipeline: `ai_engine/train.py` (lagre modeller til `artifacts/models/`).
   - [ ] Legg til automatiske backtesting-scripts.

5. Høy: Frontend funksjonalitet (TypeScript / TSX)
   - [ ] Implementer `PriceChart.tsx` med Recharts/TradingView candlesticks.
     - Filer: `frontend/src/components/PriceChart.tsx`, `frontend/src/api/prices.ts`.
   - [ ] Implementer `SignalFeed.tsx` (realtidssignal feed via websockets eller polling).
     - Filer: `frontend/src/components/SignalFeed.tsx`, `frontend/src/api/signals.ts`.
   - [ ] Implementer `SentimentFeed.tsx`.
   - [ ] Settings-side for exchange-API nøkler og DEFAULT_EXCHANGE (`frontend/src/pages/Settings.tsx`).

6. Medium: Database & migrasjoner
   - [ ] Sjekk Alembic-migrasjoner og legg til migrations for alle tabeller (`migrations/`).
   - [ ] Legg til DB-seed for demo data (`backend/seed_trades.py`).

7. Medium: Observability & produksjonsklarhet
   - [ ] Legg til logging/metrics (prometheus client, structured logging).
     - Filer: `backend/utils/logging.py`, `backend/metrics.py`.
   - [ ] Health checks og readiness probes (`/api/health`).

8. Medium: Deployment
   - [ ] Dockerfile for backend og frontend (prod builds).
   - [ ] Kubernetes manifests / Helm chart (valgfritt for produksjon).
   - [ ] CI/CD: automatisk build & push av docker images til registry.

9. Low: UX & ekstra polering
   - [ ] Toast/notification system i frontend for "saved keys", "order executed" osv.
   - [ ] Kopier-til-clipboard og masked key preview (frontend Settings).

10. Low: Dokumentasjon og onboarding
    - [ ] Lag `README.md` (dette dokumentet).
    - [ ] Lag `CONTRIBUTING.md`, `DEVELOPMENT.md` med oppsett for lokalt dev (docker-compose, env, init DB).
    - [ ] Legg til eksempelfiler `.env.example` for backend/frontend.

---

## Forslag til første konkrete leveranser (sprint 1)

- Sprintmål (2 uker):
  1. Lage `PriceChart.tsx` og koble denne mot backend-priser (mock først).
  2. Lage `SignalFeed.tsx` og vise eksisterende signals fra API.
  3. Sette opp CI-jobb som kjører frontend typecheck + vitest, og backend pytest (uten ccxt).
  4. Sentralisere secrets med `config/config.py` og `.env.example`.

## Hvordan jeg kan hjelpe videre

- Jeg kan lage konkrete filer for sprint-1 (komponenter, API-klienter, tests) i TS/TSX.
- Jeg kan legge til backend tests for adapterene og lage fixtures for mocked ccxt.
- Jeg kan tegne et arkitekturdiagram (SVG eller PlantUML) som viser hele flyten fra datakilder til ordreutførelse.

---

Hvis du vil at jeg skal generere Sprint-1-filene (f.eks. `PriceChart.tsx`, `SignalFeed.tsx`, `frontend/src/api/prices.ts`), si fra hvilken del jeg skal starte med — jeg kan begynne med frontend PriceChart-komponenten i TSX med Recharts, eller jeg kan starte med backend adapter-tester. 🚀

---

## Full Reset / Rebuild (Windows)

For å starte helt på nytt og sikre at styling (Tailwind/dark mode) faktisk bygger korrekt:

```powershell
./scripts/full-reset.ps1
```

Flagg:
```powershell
./scripts/full-reset.ps1 -PreserveVenv -PreserveDB   # Behold venv og eksisterende databaser
./scripts/full-reset.ps1 -Fast                      # Hopper over reinstall av deps
```

Verifisering i nettleser-konsoll:
```js
document.documentElement.classList.add('dark')
document.documentElement.classList.add('compact-mode')
```

Sjekk at `frontend/tailwind-debug.css` inneholder `.dark:` og `.compact-mode` (grid-gap justeringer).

Hvis ikke: dobbeltsjekk at du kjører i `c:\quantum_trader\frontend` og at `tailwind.config.ts` finnes, ikke den gamle `.tsx`.

