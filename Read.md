# Quantum Trader 🚀

![CI](https://github.com/<Gemgeminay>/quantum_trader/actions/workflows/ci.yml/badge.svg)

Et fullstack trading-system med **FastAPI (backend)**, **React + Vite (frontend)** og **Docker**.
Prosjektet inkluderer plan for Binance-integrasjon, AI-modell, trading-motor og et dashbord for live trading.

---

## 📌 Status
- ✅ Backend kjører i Docker (`17/17 tester passerer`).
- ✅ Frontend bygger med Vite + React.
- ✅ API-endepunkter for stats, trades, settings osv. er på plass.
- ✅ **XGBoost ML-integrasjon komplett** - Agent genererer live handelssignaler med metadata
- ✅ **Signal-prioritering** - ML-prediksjoner prioriteres over tekniske indikatorer
- 🔄 Neste steg: Binance API-wrapper optimalisering + real-time streaming

---

## 🤖 XGBoost ML Integration

Quantum Trader bruker nå machine learning aktivt for å generere handelssignaler:

- **XGBoost Agent** - 80.5% accuracy på 921 samples
- **Ensemble Support** - 5 modeller (XGBoost, LightGBM, RandomForest, GradientBoost, MLP)
- **Metadata Tracking** - Alle signaler merket med kilde (`XGBAgent` vs `LiveAIHeuristic`)
- **Graceful Fallback** - Heuristikk brukes når agent ikke genererer signaler

### Quick Test

```powershell
# Test signal-generering
python demo_integration.py

# Kjør integrasjonstester
pytest backend/tests/test_xgb_integration_demo.py -v
```

📚 **Full dokumentasjon:** Se [XGBOOST_INTEGRATION.md](XGBOOST_INTEGRATION.md)

---

## 🚀 Kom i gang

### 1. Klon repoet
```bash
git clone https://github.com/<din-bruker>/quantum_trader.git
cd quantum_trader
```

<!-- CI trigger: noop edit to retrigger workflows on 2025-09-23 -->
<!-- CI trigger: second noop edit to retrigger workflows on 2025-09-23 -->
