# AI Auto Trading System

Dette dokumentet beskriver hvordan du bruker det nye AI Auto Trading systemet i Quantum Trader.

## 🚀 Hva er AI Auto Trading?

AI Auto Trading systemet bruker maskinlæring (XGBAgent) til å automatisk:

- Analysere markedsdata og generere handelssignaler
- Utføre kjøp og salg av kryptovaluta basert på AI-prediksjoner
- Administrere risiko med stop-loss og take-profit ordrer
- Overvåke og rapportere handelsresultater i sanntid

## 🏗️ Systemarkitektur

```text
Frontend (React/TypeScript)
├── AITradingControls.tsx     # Kontroller for å starte/stoppe AI trading
├── AISignals.tsx             # Viser AI signaler og utførelser i sanntid
└── SimpleDashboard.tsx       # Hovedgrensesnittet

Backend (Python/FastAPI)
├── ai_auto_trading_service.py # Hoved AI trading tjeneste
├── simple_main.py            # Backend med AI trading API endpoints
└── ai_engine/                # Eksisterende XGBAgent og treningsinfrastruktur

AI Engine
├── XGBAgent                  # Maskinlæringsmodell for handelsprediksjon
├── Feature Engineering       # Databehandling og funksjonsutvinning
└── Training Pipeline         # Automatisk retrening av modeller
```

## 🛠️ Installasjon og Oppsett

### 1. Forutsetninger

```bash
# Python miljø med alle avhengigheter installert
cd backend
pip install -r requirements.txt

# Node.js miljø for frontend
cd frontend
npm install
```

### 2. Start Backend med AI Trading

```bash
# Fra quantum_trader rot katalog
python start_ai_trading_backend.py
```

### 3. Start Frontend

```bash
cd frontend
npm run dev
```

## 📡 API Endpoints

### AI Trading Kontroll

```text
GET    /api/v1/ai-trading/status          # Få AI trading status
POST   /api/v1/ai-trading/start           # Start AI trading
POST   /api/v1/ai-trading/stop            # Stopp AI trading
POST   /api/v1/ai-trading/config          # Oppdater AI konfigurasjon
```

### AI Data og Signaler

```text
GET    /api/v1/ai-trading/signals         # Få siste AI signaler
GET    /api/v1/ai-trading/executions      # Få siste handelsutførelser
```

### WebSocket for Sanntidsdata

```text
ws://127.0.0.1:8001/ws/ai-trading        # Sanntids AI signaler og utførelser
```

## 🎮 Slik Bruker Du AI Trading

### 1. Via Frontend (Anbefalt)

1. Åpne <http://localhost:3000> i nettleseren
2. Finn "AI Auto Trading" seksjonen på dashboard
3. Konfigurer innstillinger (valgfritt):
   - Trading symbols (standard: BTCUSDC, ETHUSDC)
   - Posisjonsstørrelse ($)
   - Stop loss (%)
   - Take profit (%)
   - Minimum tillitsnivå
   - Maksimale posisjoner
   - Risikogrense ($)
4. Klikk "Start AI Trading"
5. Overvåk AI aktivitet i sanntid

### 2. Via API (For utviklere)

```bash
# Start AI trading med standard symboler
curl -X POST http://127.0.0.1:8001/api/v1/ai-trading/start \
  -H "Content-Type: application/json" \
  -d '["BTCUSDC", "ETHUSDC"]'

# Få status
curl http://127.0.0.1:8001/api/v1/ai-trading/status

# Stopp AI trading
curl -X POST http://127.0.0.1:8001/api/v1/ai-trading/stop
```

## ⚙️ Konfigurasjon

### Standard AI Trading Innstillinger

```json
{
  "position_size": 1000.0,        // Posisjonsstørrelse i USD
  "stop_loss_pct": 2.0,           // Stop loss i prosent
  "take_profit_pct": 4.0,         // Take profit i prosent
  "min_confidence": 0.7,          // Minimum AI tillitsnivå (0.0-1.0)
  "max_positions": 5,             // Maksimale samtidige posisjoner
  "risk_limit": 10000.0,          // Total risikogrense i USD
  "signal_interval": 30           // Signalgenerering intervall (sekunder)
}
```

### Tilpasse Konfigurasjon

Du kan oppdatere konfigurasjonen via:

- Frontend: "AI Configuration" panel
- API: POST til `/api/v1/ai-trading/config`

## 📊 Overvåking og Rapportering

### Sanntids Metrikker

- **Total Signaler**: Antall AI signaler generert
- **Vellykkede Handler**: Antall utførte handler
- **P&L**: Total profit og tap
- **Vinnrate**: Prosent vellykkede handler
- **Aktive Posisjoner**: Nåværende åpne posisjoner

### Loggfiler

- AI trading aktivitet logges til backend/logs/
- Handelsutførelser lagres i databasen
- Signaler og prediksjoner arkiveres for analyse

## 🛡️ Risikohåndtering

### Innebygd Risikokontroll

- **Position Sizing**: Automatisk posisjonsstørrelse basert på kontosaldo
- **Stop Loss**: Automatisk stans av tapende posisjoner
- **Take Profit**: Automatisk realisering av gevinster
- **Maksimale Posisjoner**: Begrenser antall samtidige handler
- **Risikogrenser**: Stopper handel ved overskriding av grenser

### Overvåking

- Kontinuerlig overvåking av AI ytelse
- Automatisk stopp ved dårlig ytelse
- Sanntids risikometriske analyser

## 🧪 Testing

### Integrasjonstest

```bash
# Test hele AI trading systemet
python test_ai_trading_integration.py
```

### Manuell Testing

1. Start backend: `python start_ai_trading_backend.py`
2. Test API endpoints manuelt
3. Verifiser frontend integrasjon
4. Sjekk WebSocket tilkoblinger

## 🔧 Feilsøking

### Vanlige Problemer

**Backend starter ikke**

- Sjekk at alle avhengigheter er installert
- Verifiser at port 8001 er ledig
- Sjekk backend/logs/ for feilmeldinger

**AI Service ikke tilgjengelig**

- Sjekk at `ai_auto_trading_service.py` finnes i rot katalogen
- Verifiser at XGBAgent kan lastes
- Sjekk at treningsdata finnes

**Ingen AI signaler genereres**

- Verifiser markedsdata tilkobling
- Sjekk AI modell konfigurasjon
- Kontroller minimum tillitsnivå innstillinger

**Frontend viser ikke AI data**

- Sjekk WebSocket tilkobling i browser DevTools
- Verifiser backend API endepunkt tilgjengelighet
- Kontroller CORS innstillinger

### Debug Modus

Sett `DEBUG=true` miljøvariabel for detaljert logging:

```bash
DEBUG=true python start_ai_trading_backend.py
```

## 📚 Ytterligere Ressurser

- **AI Engine Dokumentasjon**: `ai_engine/README.md`
- **Backend API Docs**: <http://127.0.0.1:8001/docs> (når server kjører)
- **Frontend Komponenter**: `frontend/src/components/`
- **System Tests**: `tests/` katalogen

## 🤝 Bidrag

For å bidra til AI trading systemet:

1. Opprett en feature branch
2. Implementer endringer med tester
3. Kjør integrasjonstester
4. Send pull request

---

**⚠️ Viktig Disclaimer**: AI trading involverer finansiell risiko.
Test grundig i et sikkert miljø før bruk med ekte midler.
