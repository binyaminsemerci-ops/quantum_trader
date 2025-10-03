# 🚀 Quantum Trader Auto Trading & Auto Training

Dette dokumentet forklarer hvordan du aktiverer:
1. Kontinuerlig modell-trening hvert 10. minutt (innebygd i backend)
2. (Demo) Auto-trading loop som kjører hvert 5. minutt
3. Hvordan vi senere kan koble på ekte Binance engine (`backend/services/binance_trading.py`)

## 🔁 10-minutters Auto-Training (innebygd)

Backend fikk nye API endpoints i `simple_main.py`:

| Endpoint | Metode | Beskrivelse |
|----------|--------|-------------|
| `/api/v1/training/auto/enable?interval_minutes=10` | POST | Aktiver kontinuerlig trening (default 10 min) |
| `/api/v1/training/auto/disable` | POST | Slå av |
| `/api/v1/training/status` | GET | Sjekk status & siste metrics |
| `/api/v1/training/run-once?limit=1200` | POST | Kjør én treningsjobb i bakgrunnen |

### Eksempel (PowerShell)
```powershell
# Aktiver auto-trening hvert 10. minutt
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/training/auto/enable?interval_minutes=10

# Sjekk status
Invoke-RestMethod http://localhost:8000/api/v1/training/status

# Kjøre en engangs-trening
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/training/run-once?limit=1200
```

## 🤖 Auto-Trading (demo mode foreløpig)
Disse endpoints styrer en enkel demo-loop (ingen ekte Binance-handler ennå):

| Endpoint | Metode | Beskrivelse |
|----------|--------|-------------|
| `/api/v1/trading/auto/enable?interval_minutes=5` | POST | Start trading loop (demo) |
| `/api/v1/trading/auto/disable` | POST | Stopp trading loop |
| `/api/v1/trading/auto/status` | GET | Se status |
| `/api/v1/trading/start?interval_minutes=5` | POST | (Alias) Start loop |
| `/api/v1/trading/stop` | POST | (Alias) Stopp loop |
| `/api/v1/trading/run-cycle` | POST | Manuell enkel syklus (demo) |

### Eksempel
```powershell
# Start auto trading (demo)
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/trading/auto/enable?interval_minutes=5

# Sjekk status
Invoke-RestMethod http://localhost:8000/api/v1/trading/auto/status

# Stopp
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/trading/auto/disable
```

## 🏦 Når kan vi aktivere ekte XGBoost + Binance auto-handel?
Ekte handel via `BinanceTradeEngine` krever:
1. Binance API keys i config (sjekk `config/config.py` eller `.env`):
   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
2. Installert `python-binance` i miljøet (`pip install python-binance`)
3. Lasting av ekte modell fra `ai_engine/models/` (genereres av auto-training)
4. Bytte ut demo-trading loop med ekte motor:

### Plan for aktivering (fase 2)
1. Lage endpoint: `/api/v1/trading/real/enable` som:
   - Importerer `from backend.services.binance_trading import get_trading_engine`
   - Starter `engine.start_trading(cycle_interval_minutes=5)` i en asyncio-task
2. Lage endpoint: `/api/v1/trading/real/disable` som kaller `engine.stop_trading()`
3. Oppdatere `/api/v1/trading/status` til å spørre motoren når aktiv
4. Egen feilhåndtering + rate limit + logging

## 🔐 Sikkerhet & Forsiktighet
- Ikke aktiver ekte trading uten testnet først
- Legg inn env flagg: `REAL_TRADING=0/1`
- Logg alle ordre til DB før sending for revisjon

## 🧪 Rask verifisering
```powershell
# Start backend (hvis ikke kjører)
uvicorn backend.simple_main:app --reload

# Aktiver auto-trening
Invoke-RestMethod -Method Post http://localhost:8000/api/v1/training/auto/enable?interval_minutes=10

# Vent ~10 min og sjekk status
Invoke-RestMethod http://localhost:8000/api/v1/training/status
```

## ✅ Neste steg forslag
- [ ] Implementer ekte trading endpoints
- [ ] Legg til persistente logs for hver trening (`logs/training/`)
- [ ] Eksponér siste metrics i WebSocket push
- [ ] Frontend toggle for auto-training & trading
- [ ] Varsler ved feil (Slack / Telegram)

---
Spør hvis du vil at jeg skal aktivere fase 2 (ekte handel). 🚀
