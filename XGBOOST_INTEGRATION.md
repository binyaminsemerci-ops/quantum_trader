# XGBoost ML Integration - Komplett Dokumentasjon

## 📊 Oversikt

Quantum Trader bruker nå XGBoost machine learning-modeller aktivt for å generere handelssignaler. Integrasjonen prioriterer ML-prediksjoner over tekniske indikatorer og propagerer metadata om signal-kilder gjennom hele systemet.

## ✅ Hva er implementert

### 1. XGBAgent-integrasjon i Live Signaler

**Fil:** `backend/routes/live_ai_signals.py`

- **Lazy loading:** Agent lastes kun ved første bruk (reduserer oppstartstid)
- **Timeout-håndtering:** 70s timeout for agent-prediksjoner
- **Backoff-logikk:** 90s backoff hvis agent feiler (unngår cascade failures)
- **Graceful fallback:** Heuristikk-signaler brukes hvis agent feiler

```python
async def _get_agent() -> Optional[Any]:
    """Return a cached XGBAgent instance, creating it lazily when available."""
    global _AGENT
    if _AGENT_DISABLED_UNTIL and _loop_time() < _AGENT_DISABLED_UNTIL:
        return None
    # ... lazy load with backoff
```

### 2. Signal-prioritering i Trading Bot

**Fil:** `backend/trading_bot/autonomous_trader.py`

Trading bot prioriterer ML-signaler foran heuristikk:

```python
def _prioritize_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize agent signals over heuristic signals"""
    agent_signals = [s for s in signals if s.get("source") == "XGBAgent"]
    heuristic_signals = [s for s in signals if s.get("source") != "XGBAgent"]
    return agent_signals + heuristic_signals
```

### 3. Metadata-sporing

Alle signaler har nå:

- **`source`**: Signal-kilde (`XGBAgent` eller `LiveAIHeuristic`)
- **`model`**: Modell-type (`ensemble`, `xgboost`, eller `technical`)
- **`confidence`**: Prediksjons-sikkerhet (0.0 - 1.0)

**Eksempel signal:**
```json
{
  "id": "xgb_BTCUSDT_1731552000",
  "symbol": "BTCUSDT",
  "type": "BUY",
  "confidence": 0.75,
  "price": 42000.0,
  "source": "XGBAgent",
  "model": "ensemble",
  "reason": "ML prediction",
  "timestamp": "2025-11-14T03:30:00Z"
}
```

### 4. Signal-generering Flow

```
┌─────────────────────────────────────────────────────────────┐
│ get_live_ai_signals(limit=10, profile="mixed")              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │ _agent_signals()        │ ← Prioritet #1: ML Agent
        │ - XGBAgent.scan()       │
        │ - Timeout: 70s          │
        │ - Returns: BUY/SELL     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────┐
        │ If < limit signals:         │
        │   SimpleAITrader.generate() │ ← Fallback: Heuristikk
        │   - RSI, SMA, momentum      │
        │   - Technical indicators    │
        └────────────┬────────────────┘
                     │
        ┌────────────▼────────────┐
        │ _merge_signals()        │ ← Kombiner og dedupliser
        │ - Agent først           │
        │ - Heuristikk hvis nødv. │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ Return med metadata     │
        │ - source: XGBAgent/Heur │
        │ - model: ensemble/tech  │
        └─────────────────────────┘
```

## 🧪 Testing

### Kjør alle tester:

```powershell
# Alle signal-relaterte tester
pytest backend/tests/ -k signal -v

# XGBoost-integrasjonstester
pytest backend/tests/test_xgb_integration_demo.py -v

# Spesifikke tester
pytest backend/tests/test_signals_api.py::test_latest_ai_signals_include_source -v
```

### Test-dekning:

| Test | Beskrivelse | Status |
|------|-------------|--------|
| `test_xgb_agent_generates_signals_with_metadata` | Agent genererer signaler med metadata | ✅ PASS |
| `test_heuristic_signals_have_metadata` | Heuristikk har korrekt metadata | ✅ PASS |
| `test_get_live_ai_signals_prioritizes_agent` | Agent prioriteres over heuristikk | ✅ PASS |
| `test_metadata_propagation_through_api` | Metadata flyter til API | ✅ PASS |
| `test_trading_bot_prioritizes_agent_signals` | Bot prioriterer agent-signaler | ✅ PASS |
| `test_xgb_model_loads_successfully` | Modell laster uten feil | ✅ PASS |

### Test-resultater:

```
✅ 6/6 XGBoost-integrasjonstester passerer
✅ 8/8 Signal API-tester passerer
✅ XGBoost-modell laster med 80.5% accuracy
✅ Ensemble med 5 modeller (XGBoost, LightGBM, RandomForest, GradientBoost, MLP)
```

## 🔧 Konfigurasjon

### Signal-terskler

**Fil:** `backend/routes/live_ai_signals.py`

```python
# XGBAgent signal threshold
if score <= 0.0001:  # Svært lav terskel for å fange svake signaler
    continue

# Heuristikk BUY/SELL threshold
if score > 0.15:  # BUY
    action = "BUY"
elif score < -0.15:  # SELL
    action = "SELL"

# Score filter
if analysis["action"] != "HOLD" and analysis["score"] > 0.05:
    # Inkluder signal
```

### Trading Bot konfigurasjon

**Fil:** `backend/trading_bot/autonomous_trader.py`

```python
def __init__(
    self,
    balance: float = 10000.0,
    risk_per_trade: float = 0.01,  # 1% risk per trade
    min_confidence: float = 0.4,   # Minimum confidence for execution
    dry_run: bool = True,
    enabled_markets: Optional[List[str]] = None,
):
```

## 📈 Modell-informasjon

### Trenede modeller

**Plassering:** `ai_engine/models/`

- `xgb_model.pkl` - XGBoost classifier (80.5% accuracy, 921 samples)
- `scaler.pkl` - StandardScaler for feature normalization
- `model_metadata.json` - Treningsdetaljer og metrikker

### Feature Engineering

**Fil:** `ai_engine/feature_engineer.py`

Modellen bruker følgende features:
- Tekniske indikatorer (RSI, MACD, Bollinger Bands)
- Volum-analyse
- Prismomentum
- Trendstyrke

### Ensemble-støtte

Systemet støtter 6 modeller i ensemble:
1. **XGBoost** (primær)
2. **LightGBM**
3. **CatBoost** (optional)
4. **RandomForest**
5. **GradientBoost**
6. **MLP** (Neural Network)

Faller tilbake til enkelt XGBoost-modell hvis ensemble-modeller mangler.

## 🚀 Bruk

### API Endpoints

#### Get Live Signals
```http
GET /api/ai/signals/latest?limit=10&profile=mixed
```

**Response:**
```json
[
  {
    "id": "xgb_BTCUSDT_1731552000",
    "symbol": "BTCUSDT",
    "type": "BUY",
    "confidence": 0.75,
    "price": 42000.0,
    "timestamp": "2025-11-14T03:30:00Z",
    "reason": "ML prediction",
    "source": "XGBAgent",
    "model": "ensemble"
  }
]
```

**Profiles:**
- `left` - Konservativ (BTC, ETH, ADA, DOT, LTC)
- `right` - Aggressiv (SOL, AVAX, MATIC, LINK, UNI, AAVE, SUSHI)
- `mixed` - Balansert (BTC, ETH, BNB, SOL, ADA, DOT, AVAX, MATIC)

### Programmatisk bruk

```python
from backend.routes.live_ai_signals import get_live_ai_signals

# Get signals
signals = await get_live_ai_signals(limit=10, profile="mixed")

# Filter by source
agent_signals = [s for s in signals if s.get("source") == "XGBAgent"]
heuristic_signals = [s for s in signals if s.get("source") == "LiveAIHeuristic"]

# Filter by confidence
high_conf = [s for s in signals if s.get("confidence", 0) > 0.6]
```

## 🐛 Feilsøking

### Agent genererer ingen signaler

**Problem:** XGBAgent returnerer 0 signaler

**Årsaker:**
1. Markedet er for nøytralt (alle prediksjoner er HOLD)
2. Confidence-terskler er for høye
3. Agent er i backoff-modus (etter feil)

**Løsninger:**
```python
# 1. Senk confidence threshold
if score <= 0.0001:  # Fra 0.001 til 0.0001

# 2. Sjekk agent status
agent = await _get_agent()
if agent is None:
    print("Agent disabled eller i backoff")

# 3. Sjekk logs
logger.info("XGBAgent scan failed: %s", exc)
```

### Signaler mangler metadata

**Problem:** `source` og `model` er tomme strenger

**Årsak:** Mock-signaler brukes som fallback

**Løsning:**
```python
# Sjekk om live signals genereres
raw_signals = await get_live_ai_signals(limit=10)
if len(raw_signals) == 0:
    # Ingen live signals -> mock data brukes
    # Senk terskler eller vent på bedre markedsforhold
```

### Import-feil ved backend-start

**Problem:** `ModuleNotFoundError: No module named 'config.config'`

**Løsning:**
```powershell
# Kjør fra project root
cd C:\quantum_trader
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# IKKE fra backend-mappen:
cd backend
uvicorn main:app  # ← Dette feiler med import errors
```

## 📊 Monitoring

### Backend logs

```powershell
# Start med logging
python -m uvicorn backend.main:app --log-level info

# Nøkkel log-meldinger:
# - "XGBAgent loaded for live signal generation"
# - "AI signals generated for X symbols: BUY=Y SELL=Z HOLD=W"
# - "Agent signals: X, Heuristic signals: Y"
```

### Signal-statistikk

```python
from backend.main import _read_cached_signals

# Les cached signals
cached = _read_cached_signals("mixed")

# Count by source
sources = {}
for sig in cached:
    src = sig.get("source", "unknown")
    sources[src] = sources.get(src, 0) + 1

print(f"XGBAgent: {sources.get('XGBAgent', 0)}")
print(f"Heuristic: {sources.get('LiveAIHeuristic', 0)}")
```

## 🔄 Workflow

### Trading Cycle

```
1. Scheduler starter _run_execution_cycle (hver 5. minutt)
2. get_live_ai_signals() kalles
3. XGBAgent scanner top volum-symboler
4. Heuristikk genererer backup-signaler hvis nødvendig
5. Signaler merges og dedupliseres
6. _prioritize_signals() sorterer agent-signaler først
7. Trading bot utfører high-confidence signaler
8. Resultater logges og caches
```

### Retraining

```powershell
# Manuell retrening
python train_ai.py

# Automatisk (daglig kl. 03:00 UTC)
# Konfigurert i backend/utils/scheduler.py
```

## 📝 Endringsoversikt

### Filer modifisert:

1. **`backend/routes/live_ai_signals.py`**
   - Lagt til `_get_agent()`, `_agent_signals()`, `_merge_signals()`
   - XGBAgent lazy loading med backoff
   - Metadata-propagering (`source`, `model`)
   - Senket terskler (0.0001, 0.15, 0.05)

2. **`backend/trading_bot/autonomous_trader.py`**
   - Lagt til `_prioritize_signals()` metode
   - Sorterer XGBAgent-signaler først
   - Logger signal-kilder

3. **`backend/main.py`**
   - Forbedret `_normalise_signals()` for metadata
   - Ekstraher `source` og `model` fra raw signals
   - Debug-logging for signal flow

4. **`backend/tests/test_signals_api.py`**
   - Lagt til `test_latest_ai_signals_include_source()`
   - Regression test for metadata

5. **`backend/tests/test_xgb_integration_demo.py`** (NY)
   - 6 integrasjonstester
   - Demonstrerer agent-funksjonalitet
   - Verifiserer metadata-flow

6. **`ai_engine/agents/xgb_agent.py`**
   - Fikset import fallback for `external_data`

## 🎯 Neste steg

### Forbedringsmuligheter:

1. **Multi-model ensemble voting**
   - Kombiner prediksjoner fra alle modeller
   - Vektet stemming basert på modell-accuracy

2. **Real-time feature updates**
   - Strømme markedsdata kontinuerlig
   - Reduser latency i predictions

3. **A/B testing framework**
   - Sammenlign agent vs heuristikk ytelse
   - Automatisk velg beste strategi

4. **Adaptive thresholds**
   - Juster terskler basert på markedsvolatilitet
   - Dynamisk risk management

5. **Model versioning**
   - Spor modell-versjoner i produksjon
   - Rollback ved dårlig ytelse

---

**Versjon:** 1.0  
**Dato:** 14. November 2025  
**Status:** ✅ Produksjonsklar
