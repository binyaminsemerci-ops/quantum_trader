# 🚀 Trading Optimization Plan

## Nåværende problemer

### 1. For små posisjonsstørrelser
- **Problem**: Base notional = $250 per trade
- **Effekt**: Profitt blir for liten selv med gode trades
- **Løsning**: Øk til $500-$800 per trade

### 2. Dynamisk TP/SL ikke aggressiv nok
- **Problem**: Partial TP nivåer er for konservative
- **Effekt**: Profitt blir ikke realisert optimalt
- **Løsning**: Juster TP nivåer og aktiver trailing stop-loss

### 3. AI velger dårlige coins (tregg)
- **Problem**: Confidence threshold for lav, tar dårlige trades
- **Effekt**: Går inn i coins som ikke beveger seg
- **Løsning**: Øk confidence threshold og forbedre volume filtering

## Anbefalte endringer

### A. Øk posisjonsstørrelse (execution.py)
```python
# Fra:
base_notional = 250.0  # $250 per trade

# Til:
base_notional = 600.0  # $600 per trade
# Med 10x leverage = $6000 position size
# Max 6 posisjoner = $3600 margin = $36,000 exposure
```

### B. Juster confidence thresholds (ai_trading_engine.py)
```python
# Fra:
if confidence < 0.60:
    multiplier = 0.5

# Til:
if confidence < 0.70:
    multiplier = 0.3  # Mye mindre size for lav confidence
elif confidence >= 0.85:
    multiplier = 1.5  # Større size for høy confidence
```

### C. Forbedre coin selection (universe.py)
```python
# Legg til volatility filter
# Kun coins med minimum 3% daglig volatility
# Ekskluder stablecoins og low-movers
```

### D. Optimaliser TP/SL nivåer (position_monitor.py)
```python
# For HIGH leverage (10x+):
# TP1: 6% (partial 30%)
# TP2: 10% (partial 30%)
# TP3: 15% (partial 40%)
# Trailing SL: Aktiver etter TP1
```

### E. Øk .env konfigurasjoner
```bash
# Fra:
QT_MAX_NOTIONAL_PER_TRADE=2000
QT_MAX_POSITION_PER_SYMBOL=800
QT_MAX_GROSS_EXPOSURE=10000

# Til:
QT_MAX_NOTIONAL_PER_TRADE=5000
QT_MAX_POSITION_PER_SYMBOL=2000
QT_MAX_GROSS_EXPOSURE=25000
QT_EXECUTION_MIN_NOTIONAL=100.0  # Minimum $100 per order

# AI Thresholds:
QT_AI_MIN_CONFIDENCE=0.70  # Minimum 70% confidence
QT_AI_MIN_VOLATILITY=0.03  # Minimum 3% daily volatility
QT_PARTIAL_TP=0.5  # Aktivert partial TP
```

## Implementeringssteg

1. **Backup nåværende konfigurasjon**
2. **Oppdater .env.live med nye verdier**
3. **Endre base_notional i execution.py**
4. **Juster confidence thresholds i ai_trading_engine.py**
5. **Oppdater TP/SL nivåer i position_monitor.py**
6. **Test i paper trading først**
7. **Deploy til live trading**

## Forventet resultat

- **3x større posisjonsstørrelser**: $250 → $600 per trade
- **Bedre coin valg**: Kun high-confidence (>70%) trades
- **Mer aggressiv TP**: Raskere profit realisering
- **Trailing SL**: Beskytter profitt bedre

## Risikostyring

Med $10,000 balanse:
- Max 6 posisjoner × $600 = $3,600 margin
- Med 10x leverage = $36,000 exposure
- Stop-loss på 3% = max $1,080 risiko per dag
- Dette er innenfor QT_MAX_DAILY_LOSS=300 × 6 = $1,800

## Neste steg

Vil du at jeg skal implementere disse endringene nå?
