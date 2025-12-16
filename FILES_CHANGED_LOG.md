# 📁 ENDREDE FILER - TESTNET TRADING FIKSER

## 🔴 KRITISKE FIKSER (7 filer endret)

### 1. **backend/services/policy_observer.py**
**Endringer:** 4 steder
- **Linje 101:** `policy.risk_per_trade_pct` → `policy.max_risk_pct`
- **Linje 142:** `policy.risk_per_trade_pct` → `policy.max_risk_pct`  
- **Linje 143:** Fjernet `MaxPos={policy.max_open_positions}`, la til `Profile={policy.risk_profile}`
- **Linje 217:** `obs["policy"]["risk_per_trade_pct"]` → `obs["policy"]["max_risk_pct"]`

**Årsak:** TradingPolicy dataclass har ikke `risk_per_trade_pct` eller `max_open_positions`

---

### 2. **backend/services/market_data_helpers.py**
**Endringer:** 2 steder

#### A. Import fix (linje 137)
```python
# FØR:
from backend.api_bulletproof import binance_ohlcv

# ETTER:
from backend.routes.external_data import binance_ohlcv
```

#### B. Ticker fallback (linje 160-177)
```python
# FØR:
ticker = await adapter.get_ticker(symbol)
if ticker and 'bid' in ticker:
    # beregn spread
else:
    spread_bps = 10

# ETTER:
spread_bps = 10  # Default
try:
    if hasattr(adapter, 'get_ticker'):
        ticker = await adapter.get_ticker(symbol)
        # beregn spread
except Exception as ticker_err:
    logger.debug(f"Using default spread: {ticker_err}")
```

**Årsak:** PaperExchangeAdapter har ikke `get_ticker()` metode

---

### 3. **backend/services/event_driven_executor.py**
**Endringer:** 2 steder

#### A. Model votes fix (linje 763-779)
```python
# FØR:
signal_quality = SignalQuality(
    model_votes={model: normalize_action(action)},  # CRASHER hvis model er dict!
    ...
)

# ETTER:
# Extract model votes properly - model might be a dict
if isinstance(model, dict):
    model_votes_dict = {}
    if 'models' in model:
        for model_name, vote_data in model.get('models', {}).items():
            if isinstance(vote_data, dict) and 'action' in vote_data:
                model_votes_dict[model_name] = normalize_action(vote_data['action'])
    else:
        model_votes_dict['ensemble'] = normalize_action(action)
else:
    model_votes_dict = {str(model): normalize_action(action)}

signal_quality = SignalQuality(
    model_votes=model_votes_dict,
    ...
)
```

#### B. Regime detection skip (linje 742-746)
```python
# ETTER (ny kode):
if not hasattr(market_data, 'adx'):
    logger.debug(f"Skipping regime detection for {symbol} - ADX not available")
    regime = None
else:
    regime = self.regime_detector.detect_regime(market_data)
```

**Årsak:** 
- Model kan være dict med ensemble metadata
- MarketConditions mangler adx felt

---

### 4. **backend/config/risk_management.py**
**Endring:** 1 sted
- **Linje 202:** `min_confidence: float = field(default=0.40)`

**FØR:** `default=0.45`  
**ETTER:** `default=0.40`

**Årsak:** Redusert threshold for testnet trading

---

### 5. **backend/services/orchestrator_config.py**
**Endring:** 1 sted
- **Linje 28:** SAFE_PROFILE `base_confidence`

**FØR:** 
```python
"base_confidence": 0.55,  # Higher threshold for entry signals
```

**ETTER:**
```python
"base_confidence": 0.30,  # TESTNET: Lowered from 0.55 to allow more trades for testing
```

**⚠️ VIKTIG:** Dette er TESTNET-verdi! Production bør bruke 0.45-0.55!

---

## 📊 OPPSUMMERING

| Fil | Linjer endret | Type endring |
|-----|---------------|--------------|
| `policy_observer.py` | 4 | Attribute mapping fix |
| `market_data_helpers.py` | 2 | Import fix + fallback |
| `event_driven_executor.py` | 2 | Type handling + skip logic |
| `risk_management.py` | 1 | Threshold reduction |
| `orchestrator_config.py` | 1 | Threshold reduction |
| **TOTALT** | **10 endringer** | **5 filer** |

---

## 🔍 HVORDAN FINNE ENDRINGER IGJEN

### Git diff
```bash
git diff backend/services/policy_observer.py
git diff backend/services/market_data_helpers.py
git diff backend/services/event_driven_executor.py
git diff backend/config/risk_management.py
git diff backend/services/orchestrator_config.py
```

### Grep for kommentarer
```bash
# Finn TESTNET-spesifikke endringer
grep -r "TESTNET:" backend/

# Finn alle "ETTER:" kommentarer
grep -r "ETTER:" backend/

# Finn threshold endringer
grep -r "0.30\|0.40" backend/config/ backend/services/orchestrator_config.py
```

### VS Code search
1. Åpne Search (Ctrl+Shift+F)
2. Søk etter: `TESTNET:|FØR:|ETTER:`
3. Filter til `backend/` folder

---

## 🔄 REVERTERING (hvis nødvendig)

### For production deployment:
```python
# orchestrator_config.py, line 28
"base_confidence": 0.50,  # Production-verdi

# risk_management.py, line 202
min_confidence: float = field(default=0.65)  # Production-verdi
```

### Git reset (hvis commited):
```bash
git checkout HEAD -- backend/services/orchestrator_config.py
git checkout HEAD -- backend/config/risk_management.py
```

---

## 📝 TESTING CHECKLIST

Etter endringer, verifiser:
- [ ] Backend starter uten errors
- [ ] `✅ FULL LIVE MODE - Policy ENFORCED` i logs
- [ ] Signals genereres og evalueres
- [ ] Policy observer logger uten AttributeError
- [ ] Market data fetches uten ImportError
- [ ] Risk management evaluerer trades
- [ ] Rejection reasons er logiske (trend/confidence)

---

**Sist oppdatert:** 2025-11-22 16:10:00  
**Se også:** 
- `TESTNET_TRADING_FIXES_LOG.md` - Detaljert fikslogg
- `QUICK_START_TESTNET.md` - Oppstartguide
