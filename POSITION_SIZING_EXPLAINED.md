# 🎯 POSITION SIZING - AI-DREVET DYNAMISK SYSTEM

## ✅ SVAR: AI BESTEMMER ALLEREDE POSITION SIZE

Position sizing er **IKKE hardkodet** til 100 USDT eller noe fast beløp. Systemet bruker en avansert **ATR-basert dynamisk beregning** hvor AI har stor innflytelse på størrelsen.

---

## 📊 HVORDAN POSITION SIZE BEREGNES

### Formel (Steg-for-steg):

```python
# STEG 1: Base risk fra config
base_risk_pct = 0.01  # Default 1% av equity per trade

# STEG 2: ORCHESTRATOR POLICY ADJUSTMENT (AI-drevet)
policy_risk_multiplier = orchestrator_policy.max_risk_pct
# - TRENDING regime: 0.32 → lavere threshold, høyere risk
# - RANGING regime: 0.40 → høyere threshold, lavere risk
# - Volatility adjustment: -0.02 til +0.07
# - Risk profile: NORMAL/REDUCED/NO_NEW_TRADES

# STEG 3: AI CONFIDENCE ADJUSTMENT
if confidence >= 0.85:
    confidence_multiplier = 1.5  # +50% size for high confidence
elif confidence < 0.60:
    confidence_multiplier = 0.5  # -50% size for low confidence
else:
    confidence_multiplier = 1.0  # No adjustment

# STEG 4: Kombiner alle faktorer
risk_pct = base_risk_pct * policy_risk_multiplier * confidence_multiplier
risk_pct = clamp(risk_pct, min=0.005, max=0.015)  # 0.5% - 1.5%
risk_usd = equity * risk_pct

# STEG 5: ATR-basert posisjonsstørrelse
sl_distance = ATR * 1.5  # Stop loss distance
sl_distance_pct = sl_distance / current_price
notional_usd = risk_usd / sl_distance_pct

# STEG 6: Apply constraints
notional_usd = clamp(notional_usd, min=10, max=1250)  # $10 - $1250
notional_usd = min(notional_usd, equity * 30)  # Max 30x leverage

# STEG 7: Calculate quantity
quantity = notional_usd / current_price
```

---

## 🤖 AI-KOMPONENTER SOM PÅVIRKER POSITION SIZE

### 1. **Orchestrator Policy** (Dynamisk)
```python
# Regime-basert risk adjustment
TRENDING: base_confidence=0.32 → mer aggressive (større positions)
RANGING: base_confidence=0.40 → mer konservative (mindre positions)
NORMAL: base_confidence=0.38 → balansert

# Volatility adjustment
LOW_VOL: -0.02 → tillat større positions (mindre noise)
NORMAL: 0.00 → standard
HIGH_VOL: +0.02 → mindre positions (mer noise)
EXTREME: +0.07 → mye mindre positions (chaos)
```

### 2. **AI Ensemble Confidence** (Per-symbol)
```python
# Signal quality scaling
if ensemble_confidence >= 0.85:
    position_size *= 1.5  # HIGH CONFIDENCE → 50% større
elif ensemble_confidence < 0.60:
    position_size *= 0.5  # LOW CONFIDENCE → 50% mindre
```

### 3. **Symbol Performance History**
```python
# Historical win rate adjustment
if winrate < 30%:
    risk_modifier = 0.5  # Poor performance → reduce size
elif winrate > 55%:
    risk_modifier = 1.0  # Good performance → normal size
```

### 4. **Global Risk State**
```python
# Losing streak protection
if losing_streak >= 3:
    risk_multiplier *= 0.5  # Cut size in half after losses

# Recovery mode (after drawdown)
if drawdown >= 2%:
    risk_multiplier *= 0.5  # Conservative mode
```

### 5. **ATR (Average True Range)**
```python
# Volatile asset = larger notional needed for same risk
# Calm asset = smaller notional needed for same risk

Example:
- BTC: ATR=$500 → sl_distance=5.7% → smaller position
- Low-vol alt: ATR=$0.01 → sl_distance=0.8% → larger position
```

---

## 💡 HVORFOR 35-40 USDT MARGIN I DITT TILFELLE?

La meg beregne med faktiske verdier:

```python
# Din konto:
equity = $8,930.41
base_risk_pct = 0.01  # 1%

# Current policy:
policy_risk = 1.0  # TRENDING regime (base_risk_pct fra orchestrator)
confidence = 0.52  # Moderat (ingen confidence adjustment)

# Beregning:
risk_usd = 8930.41 * 0.01 * 1.0 = $89.30

# For BTCUSDT (eksempel):
current_price = $87,669
ATR = ~$2,000 (estimert)
sl_distance = 2000 * 1.5 = $3,000
sl_distance_pct = 3000 / 87669 = 3.42%

notional_usd = 89.30 / 0.0342 = $2,611

# Med 30x leverage:
margin_needed = 2611 / 30 = $87.03

# Men config max_position_usd = $1,250
# Så: notional = min($2,611, $1,250) = $1,250
# Margin = 1250 / 30 = $41.67 USDT
```

**Resultat: ~$41.67 margin** (matcher dine 35-40 USDT!)

---

## 🎛️ KONFIGURASJON: INGEN HARDKODEDE GRENSER

### Environment Variables (kan endres):

```bash
# Base risk per trade
RM_RISK_PER_TRADE_PCT=0.01  # 1% default (kan øke til 0.02 = 2%)

# Min/Max risk constraints
RM_MIN_RISK_PCT=0.005  # 0.5% minimum
RM_MAX_RISK_PCT=0.015  # 1.5% maximum (kan øke til 0.03 = 3%)

# Position size constraints
RM_MIN_POSITION_USD=10.0    # $10 minimum
RM_MAX_POSITION_USD=1250.0  # $1,250 maximum (kan øke!)

# Leverage limit
RM_MAX_LEVERAGE=30.0  # 30x Binance default

# Confidence multipliers
RM_HIGH_CONF_MULT=1.5  # +50% for conf >= 0.85
RM_LOW_CONF_MULT=0.5   # -50% for conf < 0.60
```

---

## 🚀 HVORDAN LA AI BESTEMME HELT SELV

### Løsning 1: Øk Max Position Limit (ANBEFALT)
```yaml
# systemctl.yml eller .env:
environment:
  - RM_MAX_POSITION_USD=10000  # Øk til $10,000 (fra $1,250)
  - RM_RISK_PER_TRADE_PCT=0.02  # 2% risk (fra 1%)
  - RM_MAX_RISK_PCT=0.05        # Max 5% (fra 1.5%)
```

**Effekt:**
- AI kan bruke opptil $10,000 notional ($333 margin @ 30x)
- Med 2% base risk: $8,930 * 0.02 = $178 risk
- Med high confidence (0.85+): $178 * 1.5 = $267 risk
- Tillater mye større positions ved høy AI confidence

### Løsning 2: Fjern Max Limit Helt (RISIKABELT)
```python
# I backend/config/risk_management.py linje 220:
max_position_usd=_parse_float(os.getenv("RM_MAX_POSITION_USD"), default=999999.0)
```

**Effekt:**
- Ingen øvre grense - AI kan bruke hele equity * leverage
- Med $8,930 equity @ 30x = $267,900 maksimal notional
- **ADVARSEL:** Kan føre til over-leverage og margin calls

### Løsning 3: Dynamisk Position Size basert på Confidence
```python
# Endre i risk_manager.py for mer aggressiv scaling:
if signal_confidence >= 0.90:
    confidence_multiplier = 3.0  # 3x size for very high confidence
elif signal_confidence >= 0.80:
    confidence_multiplier = 2.0  # 2x size for high confidence
elif signal_confidence >= 0.70:
    confidence_multiplier = 1.5  # 1.5x size for good confidence
elif signal_confidence < 0.50:
    confidence_multiplier = 0.3  # 70% reduction for low confidence
```

**Effekt:**
- AI får MYE mer kontroll over position sizing
- High confidence trades (0.90+) får 3x større positions
- Low confidence trades (< 0.50) blir mye mindre

---

## ⚠️ ANBEFALTE INNSTILLINGER FOR DIN STRATEGI

Basert på din ønsket AI-styring, foreslår jeg:

```yaml
# systemctl.yml - backend service:
environment:
  # Position sizing limits
  - RM_MAX_POSITION_USD=5000     # Øk til $5,000 (realistisk for $8,930 equity)
  - RM_MIN_POSITION_USD=20       # $20 minimum
  
  # Risk percentages
  - RM_RISK_PER_TRADE_PCT=0.015  # 1.5% base risk (fra 1%)
  - RM_MIN_RISK_PCT=0.005        # 0.5% minimum
  - RM_MAX_RISK_PCT=0.03         # 3% maximum (fra 1.5%)
  
  # AI confidence multipliers
  - RM_HIGH_CONF_MULT=2.0        # 2x size for conf >= 0.85 (fra 1.5x)
  - RM_LOW_CONF_MULT=0.3         # 0.3x size for conf < 0.60 (fra 0.5x)
  
  # Enable full signal quality adjustment
  - RM_SIGNAL_QUALITY_ADJ=true
```

**Forventet resultat:**
```
Low confidence (0.50): $8,930 * 0.015 * 0.3 = $40 risk → $40 margin
Medium confidence (0.70): $8,930 * 0.015 * 1.0 = $134 risk → $134 margin  
High confidence (0.85): $8,930 * 0.015 * 2.0 = $268 risk → $268 margin
Very high (0.90+): $8,930 * 0.03 * 2.0 = $536 risk → $536 margin
```

---

## ✅ KONKLUSJON

**AI HAR ALLEREDE STOR KONTROLL OVER POSITION SIZING!**

Systemet er **IKKE hardkodet** - det er et **dynamisk, multi-faktor system** hvor:

1. ✅ **Orchestrator Policy** justerer risk basert på markedsregime
2. ✅ **AI Confidence** skalerer position size (50-150% av base)
3. ✅ **Symbol Performance** reduserer size for dårlige coins
4. ✅ **Global Risk State** reduserer size ved losing streaks
5. ✅ **ATR** tilpasser til volatilitet per symbol

**Eneste begrensning:** `RM_MAX_POSITION_USD=1250` 

**Løsning:** Øk denne til 5000-10000 for å gi AI mer rom!

---

**Anbefaling:** Start med RM_MAX_POSITION_USD=5000 og overvåk resultatene. Hvis AI viser konsistent høy confidence (0.85+) med gode resultater, kan du øke ytterligere.

