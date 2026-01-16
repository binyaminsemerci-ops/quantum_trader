# 🚀 TESTNET QUICKSTART GUIDE

## ✅ NÅVÆRENDE STATUS

**Backend**: RUNNING ✅  
**Exchange**: Binance Testnet (Demo penger)  
**Balance**: $15,000 USDT (testnet demo)  
**Max Positions**: 20 samtidig  
**Leverage**: 30x maks  

---

## 🔍 VERIFISERE AT DU ER PÅ TESTNET

### 1. Sjekk Environment Variabler
```powershell
docker exec quantum_backend printenv | Select-String "TESTNET"
```

**Forventet output**:
```
BINANCE_TESTNET=true
USE_BINANCE_TESTNET=true
```

### 2. Sjekk Backend Health
```powershell
curl http://localhost:8000/health | ConvertFrom-Json
```

### 3. Sjekk Container Logs
```powershell
journalctl -u quantum_backend.service -f
```

**Se etter**:
- `"exchange": "binance-testnet"`
- `"testnet": true`
- Ingen `mainnet` eller `production` meldinger

---

## 📊 MONITORING KOMMANDOER

### Backend Status
```powershell
# Se om containeren kjører
systemctl list-units --filter "name=quantum_backend"

# Se siste logs
journalctl -u quantum_backend.service --tail 100

# Følg logs live
journalctl -u quantum_backend.service -f

# Restart backend
systemctl restart backend

# Stopp backend
systemctl down backend
```

### Metrics Endpoint
```powershell
# Hent trading metrics
curl http://localhost:8000/metrics
```

---

## 🎯 LEGGE TIL BYBIT TESTNET

### Steg 1: Få Bybit Testnet API Keys
1. Gå til: https://testnet.bybit.com/
2. Opprett konto (gratis)
3. Gå til **API Management**
4. Opprett API key med **Futures Trading** permissions
5. Kopier **API Key** og **Secret Key**

### Steg 2: Legg til Keys i .env
Åpne `c:\quantum_trader\.env` og finn:
```bash
# BYBIT TESTNET (Secondary Exchange)
BYBIT_TESTNET=true
BYBIT_API_KEY=YOUR_BYBIT_TESTNET_KEY_HERE      # ← Lim inn din key her
BYBIT_API_SECRET=YOUR_BYBIT_TESTNET_SECRET_HERE # ← Lim inn din secret her

# MULTI-ACCOUNT TRADING
MULTI_EXCHANGE_ENABLED=false  # ← Sett til true
```

### Steg 3: Restart Backend
```powershell
systemctl restart backend
```

### Steg 4: Verifiser Bybit Connection
```powershell
curl http://localhost:8000/health | ConvertFrom-Json
```

**Se etter**:
```json
{
  "capabilities": {
    "exchanges": {
      "binance": true,
      "bybit": true    // ← Skal være true
    }
  }
}
```

---

## 🛡️ SIKKERHET - TESTNET vs MAINNET

### ✅ TESTNET (Demo Penger)
- `BINANCE_TESTNET=true` - Du er trygg!
- Bruker demo keys fra testnet.binancefuture.com
- Ingen ekte penger involvert
- Kan teste alle strategier uten risiko

### ⚠️ MAINNET (Ekte Penger)
**IKKE BRUK ENDA!** For å bruke ekte penger må du:
1. Sette `BINANCE_TESTNET=false`
2. Bruke production API keys fra binance.com
3. Ha gjennomført minst 2 uker testnet trading
4. Ha dokumentert profitt på testnet
5. Starte med MICRO profile (-0.5% daily limit)

---

## 📈 TESTNET TRADING METRICS

### Daglig Monitoring
```powershell
# Position count
curl http://localhost:8000/positions | ConvertFrom-Json

# Daily PnL
curl http://localhost:8000/metrics | Select-String "pnl"

# Win rate
curl http://localhost:8000/metrics | Select-String "win_rate"
```

### Forventet Performance (Testnet)
- **Win Rate**: 55-70% (AI-drevet)
- **Daily PnL**: +0.5% til +2.0% (varierer)
- **Max Drawdown**: -5% (før ESS trigger)
- **Avg Hold Time**: 30 min - 4 timer

---

## 🔧 TROUBLESHOOTING

### Backend starter ikke
```powershell
# Sjekk Docker logs
journalctl -u quantum_backend.service --tail 100

# Restart
systemctl restart backend

# Full rebuild
systemctl down
systemctl up backend -d
```

### Ingen trades plasseres
**Mulige årsaker**:
1. Confidence threshold for høy (sjekk `QT_CONFIDENCE_THRESHOLD`)
2. RiskGate blokkerer (for mange posisjoner)
3. Ingen gode signal (AI venter på muligheter)
4. Exchange connectivity issues

**Sjekk logs**:
```powershell
journalctl -u quantum_backend.service -f | Select-String "signal|trade|position"
```

### Testnet keys virker ikke
1. Verifiser at keys er fra **testnet.binancefuture.com** (ikke binance.com)
2. Sjekk at API permissions inkluderer **Futures Trading**
3. Sjekk at `BINANCE_TESTNET=true` i `.env`
4. Restart backend: `systemctl restart backend`

---

## 🎓 NESTE STEG

### Kortsiktig (1-2 uker)
1. ✅ Kjør på Binance testnet (FERDIG)
2. 🔄 Legg til Bybit testnet (KLAR TIL Å KONFIGURERE)
3. 📊 Samle performance data (min 50 trades)
4. 🤖 La AI lære fra testnet trading

### Mellomlang (2-4 uker)
1. Oppnå konsistent profitabilitet på testnet
2. Test multi-exchange trading (Binance + Bybit samtidig)
3. Optimiser AI confidence thresholds
4. Dokumenter win rate, sharpe ratio, drawdown

### Langsiktig (1-2 måneder)
1. Flytt til mainnet med MICRO profile (-0.5% daily limit)
2. Start med $500-1000 ekte penger
3. Gradvis øk til SMALL profile (-1.0% daily limit)
4. Scale opp etter dokumentert suksess

---

## 📞 QUICK COMMANDS

```powershell
# Start trading
systemctl up backend -d

# Stop trading
systemctl down backend

# Watch live
journalctl -u quantum_backend.service -f

# Check health
curl http://localhost:8000/health

# Restart
systemctl restart backend
```

---

## ✅ TESTNET VERIFICATION CHECKLIST

- [x] Backend kjører (port 8000)
- [x] `BINANCE_TESTNET=true` verified
- [x] Health endpoint responds
- [x] Binance testnet keys configured
- [ ] Bybit testnet keys configured (OPTIONAL)
- [ ] Multi-exchange enabled (OPTIONAL)
- [ ] First trades executed successfully
- [ ] Metrics tracking working

---

**🎉 DU ER NÅ PÅ TESTNET MED DEMO PENGER!**

Systemet kan nå trade autonomt med AI-drevne beslutninger. Alle trades bruker demo USDT ($15,000 balance). Ingen ekte penger er i risiko.

*Opprettet: 2025-12-04*  
*Backend Status: RUNNING*  
*Exchange: Binance Testnet*

