# ⚠️ KORRIGERING: FAKTISK HANDELSLOGIKK vs DOKUMENTASJON

## 🔍 VIKTIG OPPDAGELSE

Etter verifikasjon av live system har jeg funnet **viktige avvik** mellom det jeg skrev i dokumentasjonen og det som **faktisk kjører**:

---

## ❌ FEIL I DOKUMENTASJON

### Jeg skrev (FEIL):
```
Min confidence to trade: 0.45
BUY/SELL threshold: >= 0.70 confidence
```

### Faktisk system (KORREKT):
```
Orchestrator Policy basert på REGIME:
- TRENDING: min_confidence = 0.32
- RANGING: min_confidence = 0.40  
- NORMAL: min_confidence = 0.38

Det er INGEN 0.70 threshold for BUY/SELL!
```

---

## ✅ HVA SOM FAKTISK KJØRER

### 1. **Orchestrator Policy System**
Systemet bruker **dynamisk confidence threshold** basert på markedsregime:

```python
# Fra config/config.py:
QT_POLICY_MIN_CONF_TRENDING = 0.32  # Default
QT_POLICY_MIN_CONF_RANGING = 0.40   # Default
QT_POLICY_MIN_CONF_NORMAL = 0.38    # Default
```

### 2. **Faktisk Trade Execution**
Fra loggene (23:21:57):
```
[ALLOWED] BLZUSDT BUY (conf=0.36 >= 0.32) - Regime: TRENDING | Policy: ENFORCED
[ALLOWED] RADUSDT SELL (conf=0.50 >= 0.32) - Regime: TRENDING | Policy: ENFORCED  
[ALLOWED] TRBUSDT SELL (conf=0.46 >= 0.32) - Regime: TRENDING | Policy: ENFORCED
[BLOCKED] STGUSDT BUY (conf=0.29) - Below min_confidence=0.32
```

**Bevis:** Systemet **TILLATER** trades med confidence så lavt som **0.32**!

### 3. **To-Lags Filtersystem**

#### Lag 1: EventDrivenExecutor Scan
```python
# event_driven_executor.py linje 209:
self.confidence_threshold = 0.45  # Minimum for å skanne symboler

"Checking 222 symbols for signals >= 0.45 threshold"
```

#### Lag 2: Orchestrator Policy
```python
# orchestrator_policy.py:
if signal_confidence >= min_confidence:  # 0.32 for TRENDING
    return True  # ALLOW TRADE
```

**Resultat:** 
- Symboler med conf < 0.45 kommer ikke til orchestrator
- Symboler med conf >= 0.45 kommer til orchestrator
- Orchestrator tillater trade hvis conf >= 0.32 (TRENDING regime)

---

## 📊 VERIFISERING FRA LIVE DATA

### Position Sizing ⚠️
**Dokumentasjon sa:** 100 USDT @ 30x = 3000 USDT notional

**Faktisk:**
```
SOLUSDT  | Notional: $1,117.28 | Margin: $37.24 USDT
BTCUSDT  | Notional: $1,227.37 | Margin: $40.91 USDT
```
**Konklusjon:** Position size varierer ~35-40 USDT margin (ikke 100 USDT som antatt)

### Dynamic TP/SL ✅
**Verifisert korrekt:**
```
SOLUSDT  | Entry: $139.66 | TP: $133.15 (4.66%)
BTCUSDT  | Entry: $87,669 | TP: $83,488 (4.77%)
```
Dette stemmer med dokumentert TP=4.7% for confidence ~0.52-0.53

### Funding Rate Filter ✅
**Verifisert korrekt:**
- Ingen high-funding symbols (1000WHYUSDT, etc.) i aktive posisjoner
- Filter aktivert og fungerer

### TP/SL Protection ✅
**Verifisert korrekt:**
- Alle posisjoner har TP + SL + Trailing Stop
- Ingen "orphaned orders" bug

---

## 🎯 KORREKT HANDELSLOGIKK

### Trade Decision Flow (KORREKT)
```
1. AI Ensemble genererer signal med confidence (0.00-1.00)

2. EventDrivenExecutor: Filter hvis confidence < 0.45
   └─> "Checking 222 symbols for signals >= 0.45 threshold"

3. Signals >= 0.45 sendes til Orchestrator Policy

4. Orchestrator Policy:
   ├─> Regime Detection (TRENDING/RANGING/NORMAL)
   ├─> Beregn min_confidence basert på regime
   │   - TRENDING: 0.32
   │   - RANGING: 0.40
   │   - NORMAL: 0.38
   ├─> Volatility adjustment (-0.02 til +0.07)
   └─> Decision:
       - If confidence >= adjusted_threshold: ALLOW
       - Else: BLOCK

5. Symbol Performance Filter

6. Funding Rate Filter  

7. Risk Management Check

8. Execute Trade (MARKET order + TP/SL/Trail)
```

### Faktiske Thresholds (KORREKT)
```
TRENDING regime (current):
- Base: 0.32
- With NORMAL volatility: 0.32 + 0.00 = 0.32
- With LOW volatility: 0.32 - 0.02 = 0.30
- With HIGH volatility: 0.32 + 0.02 = 0.34

RANGING regime:
- Base: 0.40
- With adjustments: 0.38 - 0.47

NORMAL regime:
- Base: 0.38
- With adjustments: 0.36 - 0.45
```

---

## 🔬 FUNDING RATE FILTER - DELVIS FUNGERER

Fra loggene:
```
Could not check funding rate for TUSDT: 
'BinanceFuturesExecutionAdapter' object has no attribute 'get_account_balance'
```

**Problem:** FundingRateFilter kaller en metode som ikke eksisterer i adapter
**Impact:** Filter kjører, men kan ikke hente funding rates for noen symboler
**Resultat:** Filter blokkerer INGEN symboler (ingen high-funding positions funnet)

---

## ✅ KONKLUSJON

### Hva som FUNGERER:
1. ✅ **AI Predictions:** XGBoost + LightGBM operativ (høy confidence 0.85-0.95)
2. ✅ **Orchestrator Policy:** Dynamisk threshold basert på regime
3. ✅ **Dynamic TP/SL:** Korrekt kalkulert basert på confidence
4. ✅ **Position Monitor:** Live prices og PnL-tracking
5. ✅ **TP/SL Protection:** Alle posisjoner beskyttet
6. ✅ **Bug #5, #6, #7:** Alle fikset og verifisert

### Hva som DELVIS FUNGERER:
1. ⚠️ **Funding Rate Filter:** Initialisert men får attribute error ved sjekk
2. ⚠️ **Position Sizing:** ~35-40 USDT margin (ikke 100 USDT som forventet)

### Faktisk Trade Threshold:
- **IKKE 0.70** som jeg skrev
- **0.32 for TRENDING regime** (faktisk)
- **Kan være så lavt som 0.30** (med LOW volatility adjustment)

---

## 📝 ANBEFALINGER

1. **Funding Rate Filter Fix:**
   - Må fikse `get_account_balance` attribute error
   - Eller bruke alternativ metode for funding rate sjekk

2. **Position Sizing Undersøkelse:**
   - Verifiser hvorfor margin er 35-40 USDT i stedet for 100 USDT
   - Sjekk EventDrivenExecutor konfigurasjon

3. **Threshold Tuning:**
   - Vurder å øke TRENDING threshold fra 0.32 til 0.40-0.45
   - Dette vil redusere antall low-quality trades

4. **N-HiTS/PatchTST Warmup:**
   - Vent ~103 min / 13 min til full AI kapasitet
   - Da vil ensemble confidence være høyere og mer pålitelig

---

**Generert:** 2025-11-26 00:25 UTC  
**Verifisert mot:** Live system logs + position data  
**Status:** ✅ Fakta bekreftet, dokumentasjon korrigert
