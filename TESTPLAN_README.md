# AI Hedge Fund OS - Helhetlig Testplan

Omfattende systemvalidering for AI Hedge Fund OS med automatiserte tester for alle kritiske komponenter.

## 📋 Oversikt

Denne testplanen validerer:
- ✅ Container helse og tilgjengelighet
- ✅ API endpoints (backend og AI engine)
- ✅ Redis dataintegritet
- ✅ AI-modeller (xgb, lgbm, nhits, patchtst)
- ✅ Regime forecasting
- ✅ Governance og System Stress Index (SSI)
- ✅ End-to-end signal generering
- ✅ Logganalyse

## 🚀 Kjøring

### Linux/WSL
```bash
chmod +x comprehensive_system_test.sh
./comprehensive_system_test.sh
```

### Windows PowerShell
```powershell
.\comprehensive_system_test.ps1
```

## 📊 Teststruktur

### Trinn 1: Container Helse
Verifiserer at alle nødvendige Docker containers kjører:
- backend
- ai_engine
- redis
- rl_optimizer
- strategy_evaluator
- strategy_evolution
- quantum_policy_memory
- global_policy_orchestrator
- federation_stub

### Trinn 2: Interne API-er
Tester health endpoints:
- `http://localhost:8000/health` (Backend)
- `http://localhost:8001/health` (AI Engine)

Validerer at begge returnerer `status: "ok"` og viser lastede modeller.

### Trinn 3: Redis Dataintegritet
Sjekker Redis for kritiske nøkler:
- `governance_weights`
- `current_policy`
- `meta_best_strategy`
- `quantum_regime_forecast`
- `system_ssi`

Verifiserer også minnebruk og generell Redis helse.

### Trinn 4: AI-Modell Sanity-Check
Tester at alle fire AI-modeller er lastet:
- XGBoost (xgb)
- LightGBM (lgbm)
- NHiTS (nhits)
- PatchTST (patchtst)

### Trinn 5: Regime-Forecast Validering
Validerer `quantum_regime_forecast` i Redis:
- Sjekker tidsstempel (skal være < 6 timer gammelt)
- Validerer regime-sannsynligheter (bull, bear, neutral, volatile)

### Trinn 6: Governance og SSI
Verifiserer:
- System Stress Index (SSI) ligger mellom -2 og 2
- Governance weights eksisterer og summerer til ~1.0

### Trinn 7: End-to-End Simulering
Sender syntetiske trading signals for:
- BTCUSDT
- ETHUSDT
- SOLUSDT

Validerer:
- Action er BUY, SELL eller HOLD
- Confidence > 0.4

### Trinn 8: Logg-Evaluering
Søker etter kritiske feil i de siste 1000 logg-linjene:
- ERROR
- CRITICAL
- Exception

## 📈 Output

Skriptet gir fargekodet output:
- 🔵 **[INFO]** - Informasjon
- ✅ **[✓]** - Test bestått
- ❌ **[✗]** - Test feilet
- ⚠️ **[!]** - Advarsel

### Eksempel på vellykket kjøring:
```
═══════════════════════════════════════════════════════════════
  TEST SAMMENDRAG
═══════════════════════════════════════════════════════════════
Total tester: 45
Bestått: 45
Feilet: 0

✅ ALLE TESTER BESTÅTT!
Systemet er verifisert funksjonelt og stabilt i dry-run mode.
```

## 🏁 Etter Vellykket Testing

Når alle tester passerer, kan du manuelt koble til Binance Testnet:

### 1. Opprett Testnet API-nøkler
Gå til https://testnet.binance.vision og opprett API-nøkler

### 2. Oppdater .env-filen
```env
BINANCE_API_KEY=din_testnet_key
BINANCE_API_SECRET=din_testnet_secret
BINANCE_BASE_URL=https://testnet.binance.vision/api
MODE=testnet
```

### 3. Start systemet på nytt
```bash
docker compose down && docker compose up -d
```

## ⚠️ Viktige Sikkerhetsmerknader

1. **Bruk kun små posisjoner på testnet** - Dette er for testing!
2. **Aldri bruk live API-nøkler** før full bekreftet oppførsel
3. **Overvåk systemet nøye** de første timene på testnet
4. **Sjekk logs regelmessig** for uventede feil

## 🔧 Feilsøking

### Docker containers kjører ikke
```bash
docker compose up -d
docker compose ps
```

### API ikke tilgjengelig
```bash
docker compose logs backend
docker compose logs ai_engine
```

### Redis tom for data
```bash
# Normal ved første oppstart
# Vent 5-10 minutter for initialisering
docker compose logs redis
```

### AI-modeller ikke lastet
```bash
docker compose logs ai_engine | grep -i "model"
# Sjekk at modellene er lastet under oppstart
```

## 📝 Loggfiler

Testresultater kan også lagres til fil:

### Linux/WSL
```bash
./comprehensive_system_test.sh | tee test_results_$(date +%Y%m%d_%H%M%S).log
```

### PowerShell
```powershell
.\comprehensive_system_test.ps1 | Tee-Object -FilePath "test_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
```

## 🔄 Automatisering

For kontinuerlig overvåking, sett opp en cron job (Linux) eller Scheduled Task (Windows):

### Linux Cron
```bash
# Kjør hver time
0 * * * * /path/to/comprehensive_system_test.sh >> /var/log/quantum_trader_tests.log 2>&1
```

### Windows Task Scheduler
```powershell
$Action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\quantum_trader\comprehensive_system_test.ps1"
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
Register-ScheduledTask -Action $Action -Trigger $Trigger -TaskName "QuantumTraderHealthCheck"
```

## 📞 Support

Ved problemer:
1. Sjekk loggene først: `docker compose logs`
2. Verifiser at alle containers kjører: `docker compose ps`
3. Test individuell komponent med curl/Invoke-RestMethod
4. Review denne README for feilsøkingstips

## 📚 Relatert Dokumentasjon

- [AI_HEDGEFUND_OS_GUIDE.md](./AI_HEDGEFUND_OS_GUIDE.md)
- [AI_FULL_SYSTEM_OVERVIEW_DEC13.md](./AI_FULL_SYSTEM_OVERVIEW_DEC13.md)
- [AI_DEPLOYMENT_CHECKLIST.md](./AI_DEPLOYMENT_CHECKLIST.md)

---

**Versjon:** 1.0  
**Sist oppdatert:** 2025-12-20  
**Maintainer:** Quantum Trader DevOps Team

