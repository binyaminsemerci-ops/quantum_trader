# BEREGNINGSFORMLER RAPPORT - Posisjonsstørrelse, Leverage, TP/SL
**Dato:** 2026-02-03  
**Formål:** Kartlegge alle beregningsformler for trading-parametere

---

## OVERSIKT

Systemet har **3 hoved-beregningsmoduler** for trading-parametere:

1. **Trading Mathematician** - AI-drevet Kelly Criterion + ATR-basert TP/SL
2. **RL Position Sizing Agent** - Reinforcement Learning for adaptiv sizing
3. **Adaptive Leverage Engine** - Leverage-sensitiv TP/SL (ikke aktivt)

---

## 1. TRADING MATHEMATICIAN (PRIMÆR MOTOR)

### 1A. Fil & Plassering
**Fil:** `backend/services/ai/trading_mathematician.py` (569 linjer)

**Brukes av:**
- `backend/services/ai/rl_position_sizing_agent.py` (linje 37, 190, 675)
- `backend/services/ai/math_ai_integration.py` (linje 9, 43, 80)
- `microservices/trading_bot/simple_bot.py` (indirekte via RL Agent)

### 1B. Trigger-flyt

```
trading_bot → RL Agent → Trading Mathematician → Resultat
```

**Konkret sekvens:**
1. `microservices/trading_bot/simple_bot.py:487` kaller `rl_sizing_agent.decide_sizing()`
2. `rl_position_sizing_agent.py:675` kaller `math_ai.calculate_optimal_parameters()`
3. Trading Mathematician beregner alle parametere
4. Resultatet sendes tilbake til trading_bot

### 1C. Beregningsformler

#### **FORMEL 1: Posisjonsstørrelse (Margin)**
```python
margin = balance * risk_per_trade_pct
```
**Standard:** `risk_per_trade_pct = 0.80` (80% av balanse)  
**Eksempel:** Balance $10,000 → Margin $8,000

**Kode (linje 180-194):**
```python
def _calculate_risk_amount(self, account: AccountState, available: float) -> float:
    # Base margin allocation - use configured risk percentage
    base_margin = account.balance * self.risk_per_trade_pct
    
    logger.info(f"💰 Using {self.risk_per_trade_pct*100:.1f}% of balance AS MARGIN: ${base_margin:.2f}")
    
    # Check if we have enough available
    if available < base_margin:
        logger.warning(f"⚠️  Available ${available:.2f} < target ${base_margin:.2f}, using what's available")
        return available * 0.95  # Use 95% of available
    
    return base_margin
```

---

#### **FORMEL 2: Stop Loss (SL)**
```python
base_sl = ATR_pct * 2.0  # Base: 2x ATR for mer rom

# Adjust for trend strength
if trend_strength > 0.7:
    base_sl *= 0.9  # Strong trend → tighter SL (10% reduksjon)
elif trend_strength < 0.3:
    base_sl *= 1.15  # Choppy market → wider SL (15% økning)

# Adjust based on historical losses
if avg_loss_pct > base_sl * 1.3:
    base_sl = avg_loss_pct * 0.95  # 95% of historical avg loss

# Conservative mode adjustment
if conservative_mode:
    base_sl *= 1.2  # +20% wider

# CLAMPS
min_sl = 0.025  # 2.5% minimum
max_sl = 0.030  # 3.0% maximum
final_sl = max(min_sl, min(max_sl, base_sl))
```

**Standard bounds:** 2.5% - 3.0%  
**Eksempel:** ATR=5% → base_sl=10% → clamped til 3.0%

**Kode (linje 196-244):**
```python
def _calculate_optimal_sl(
    self, market: MarketConditions, performance: PerformanceMetrics
) -> float:
    """
    Calculate optimal stop loss based on ATR and market conditions.
    
    NEW STRATEGY: Roomy initial SL (2.5-3%) to avoid whipsaw.
    """
    # Base SL on ATR (2.0x ATR for more room)
    base_sl = market.atr_pct * 2.0
    
    # Adjust for trend strength (less aggressive adjustment)
    if market.trend_strength > 0.7:
        base_sl *= 0.9
    elif market.trend_strength < 0.3:
        base_sl *= 1.15
    
    # Conservative mode: even wider SL
    if self.conservative_mode:
        base_sl *= 1.2
    
    # NEW BOUNDS: 2.5-3% range (roomy initial SL)
    min_sl = 0.025  # 2.5% minimum
    max_sl = 0.030  # 3.0% maximum
    
    return max(min_sl, min(max_sl, base_sl))
```

---

#### **FORMEL 3: Take Profit (TP)**
```python
base_tp = 0.035  # 3.5% base target (TP2)

# Adjust for volatility
if daily_volatility > 0.05:  # >5% daily volatility
    base_tp = 0.040  # 4.0% target
elif daily_volatility < 0.02:  # <2% daily volatility
    base_tp = 0.030  # 3.0% target

# Adjust for trend
if trend_strength > 0.7:
    base_tp *= 1.1  # +10% for strong trend

# Adjust based on historical wins
if avg_win_pct < base_tp:
    base_tp = avg_win_pct * 0.9  # 90% of historical avg

# CLAMPS
min_tp = 0.030  # 3.0% minimum
max_tp = 0.045  # 4.5% maximum
final_tp = max(min_tp, min(max_tp, base_tp))
```

**Standard bounds:** 3.0% - 4.5%  
**Partial TP:** `partial_tp = tp * 0.5` (50% av hovedmål)

**Kode (linje 246-299):**
```python
def _calculate_optimal_tp(
    self, sl_pct: float, performance: PerformanceMetrics, market: MarketConditions
) -> float:
    """
    Calculate optimal take profit for FIRST partial TP (TP2 target).
    
    NEW STRATEGY: Tighter TP targets for frequent profit taking.
    - TP1 (50%): 1.5-2.0% (handled by partial system)
    - TP2 (30%): 3.0-4.0% (this calculation)
    - TP3 (20%): Trailing from +5% (handled by trailing system)
    """
    # NEW: Target 3-4% for TP2 (main target)
    base_tp = 0.035  # 3.5% base target
    
    # Adjust based on volatility
    if market.daily_volatility > 0.05:
        base_tp = 0.040  # 4.0% target
    elif market.daily_volatility < 0.02:
        base_tp = 0.030  # 3.0% target
    
    # NEW BOUNDS: 3.0-4.5% range for TP2
    min_tp = 0.030
    max_tp = 0.045
    
    return max(min_tp, min(max_tp, base_tp))
```

---

#### **FORMEL 4: Leverage (Kelly Criterion)**
```python
# Kelly Criterion:
win_rate = historical_win_rate
avg_win = historical_avg_win_pct
avg_loss = historical_avg_loss_pct
loss_rate = 1 - win_rate

edge = (win_rate * avg_win) - (loss_rate * avg_loss)
variance = (win_rate * avg_win^2) + (loss_rate * avg_loss^2)
kelly_fraction = edge / variance

# Adjust for signal confidence
kelly_with_confidence = kelly_fraction * signal_confidence

# Apply safety cap
leverage = min(kelly_with_confidence, safety_cap)

# Fallback for limited history
if total_trades < 10:
    leverage = 10.0  # Conservative 10x
```

**Standard:** 10x (limited history)  
**Max:** 25x (safety_cap fra ENV/PolicyStore)  
**Kelly adjustment:** Multiplisert med signal confidence (0-1)

**Kode (linje 301-400+):**
```python
def _calculate_optimal_leverage(
    self,
    market: MarketConditions,
    performance: PerformanceMetrics,
    account: AccountState,
    signal_confidence: float = 0.70,
) -> float:
    """
    Calculate optimal leverage using Kelly Criterion.
    
    Formula: Optimal Leverage = Edge / Variance
    - Edge = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss)
    - Variance = (Win_Rate × Avg_Win²) + (Loss_Rate × Avg_Loss²)
    
    Adjusted by:
    - Signal confidence (AI ensemble agreement)
    - Position count (less aggressive if many open positions)
    - Safety cap (max leverage limit)
    """
    
    # If limited history, use conservative default
    if performance.total_trades < 10:
        default_lev = 10.0  # Conservative for new symbols
        logger.info(f"   📊 Limited history ({performance.total_trades} trades), using conservative {default_lev}x")
        return default_lev
    
    # Calculate Kelly Criterion
    win_rate = max(0.01, min(0.99, performance.win_rate))
    avg_win = max(0.001, performance.avg_win_pct)
    avg_loss = max(0.001, performance.avg_loss_pct)
    loss_rate = 1.0 - win_rate
    
    edge = (win_rate * avg_win) - (loss_rate * avg_loss)
    variance = (win_rate * (avg_win ** 2)) + (loss_rate * (avg_loss ** 2))
    
    if variance <= 0 or edge <= 0:
        return 1.0  # No edge, use minimum leverage
    
    kelly_fraction = edge / variance
    
    # Adjust Kelly by signal confidence
    kelly_with_confidence = kelly_fraction * signal_confidence
    
    # Apply safety cap
    leverage = min(kelly_with_confidence, self.safety_cap)
    leverage = max(1.0, leverage)  # Minimum 1x
    
    return leverage
```

---

### 1D. Input Parametere

**AccountState:**
```python
@dataclass
class AccountState:
    balance: float          # Total balance ($)
    equity: float           # Current equity ($)
    margin_used: float      # Used margin ($)
    open_positions: int     # Number of open positions
    max_positions: int      # Max allowed positions
```

**MarketConditions:**
```python
@dataclass
class MarketConditions:
    symbol: str                  # Trading symbol
    atr_pct: float              # ATR as percentage (0.05 = 5%)
    daily_volatility: float     # Daily volatility (0.03 = 3%)
    trend_strength: float       # 0-1 (0=choppy, 1=strong trend)
    liquidity_score: float      # 0-1 (0=illiquid, 1=liquid)
```

**PerformanceMetrics:**
```python
@dataclass
class PerformanceMetrics:
    total_trades: int           # Total completed trades
    win_rate: float            # Win rate 0-1 (0.65 = 65%)
    avg_win_pct: float         # Average win % (0.05 = 5%)
    avg_loss_pct: float        # Average loss % (0.03 = 3%)
    profit_factor: float       # Gross profit / Gross loss
    sharpe_ratio: float        # Risk-adjusted return
```

### 1E. Output

**OptimalParameters:**
```python
@dataclass
class OptimalParameters:
    margin_usd: float              # Margin to allocate ($200)
    leverage: float                # Optimal leverage (10x)
    notional_usd: float            # Notional position size ($2000)
    tp_pct: float                  # Take profit % (0.04 = 4%)
    sl_pct: float                  # Stop loss % (0.03 = 3%)
    partial_tp_pct: float          # Partial TP % (0.02 = 2%)
    expected_profit_usd: float     # Expected profit ($80)
    max_loss_usd: float            # Max loss ($60)
    risk_reward_ratio: float       # R:R ratio (1.33:1)
    confidence_score: float        # Confidence 0-1 (0.48)
```

---

## 2. RL POSITION SIZING AGENT (MELLOMMANN)

### 2A. Fil & Plassering
**Fil:** `backend/services/ai/rl_position_sizing_agent.py` (912 linjer)

**Rolle:** Wrapper rundt Trading Mathematician + Q-learning state management

### 2B. Trigger-flyt

```python
# Fra simple_bot.py linje 487-497:
sizing_decision = await asyncio.to_thread(
    self.rl_sizing_agent.decide_sizing,
    symbol=symbol,
    confidence=confidence,
    atr_pct=atr_value,
    current_exposure_pct=0.0,
    equity_usd=10000.0,
    adx=None,
    trend_strength=None
)
```

**Sekvens:**
1. Trading_bot samler market data (ATR, volatility, confidence)
2. Kaller `rl_sizing_agent.decide_sizing()` med market context
3. RL Agent bygger state (regime, confidence bucket, exposure)
4. Kaller `math_ai.calculate_optimal_parameters()` (linje 675)
5. Pakker resultat i `SizingDecision` objekt
6. Returnerer til trading_bot

### 2C. State Space (for Q-learning)

**5 dimensjoner:**
```python
class MarketRegime(Enum):
    HIGH_VOL_TRENDING    # High vol + strong trend
    LOW_VOL_TRENDING     # Low vol + strong trend
    HIGH_VOL_RANGING     # High vol + ranging
    LOW_VOL_RANGING      # Low vol + ranging
    CHOPPY               # Unstable/whipsaw

class ConfidenceBucket(Enum):
    VERY_HIGH  # >= 85%
    HIGH       # 70-85%
    MEDIUM     # 55-70%
    LOW        # 45-55%
    VERY_LOW   # < 45%

class PortfolioState(Enum):
    LIGHT      # < 30% exposure
    MODERATE   # 30-60% exposure
    HEAVY      # 60-80% exposure
    MAX        # >= 80% exposure
```

**Total state space:** 5 × 5 × 4 × 3 = 300 states

### 2D. Action Space

**Position Size:** [min, 10%, 25%, 50%, 100%] av max  
**Leverage:** [1x, 2x, 3x, 4x, 5x] (IKKE BRUKT - Math AI bestemmer)  
**TP/SL Strategy:** [Conservative, Balanced, Aggressive] (IKKE BRUKT - Math AI bestemmer)

### 2E. Exploration vs Exploitation

**Exploration rate:** 0.10 (10% explore, 90% exploit)  
**Kode (linje 134-139):**
```python
self.exploration_rate = 0.10  # 🎯 REDUCED from 0.50 to 0.10
# - 90% exploit (full size)
# - 10% explore
```

### 2F. Output

**SizingDecision:**
```python
@dataclass
class SizingDecision:
    position_size_usd: float      # Fra Math AI ($200)
    leverage: float               # Fra Math AI (10x)
    risk_pct: float              # Calculated risk %
    confidence: float            # Signal confidence
    reasoning: str               # Explanation
    state_key: str               # Q-learning state
    action_key: str              # Q-learning action
    q_value: float               # Q-table value
    # TP/SL parameters (fra Math AI):
    tp_percent: float            # 0.04 (4%)
    sl_percent: float            # 0.03 (3%)
    partial_tp_enabled: bool     # True
    partial_tp_percent: float    # 0.02 (2%)
    partial_tp_size: float       # 0.5 (50%)
```

---

## 3. ADAPTIVE LEVERAGE ENGINE (IKKE AKTIVT)

### 3A. Fil & Plassering
**Fil:** `microservices/exitbrain_v3_5/adaptive_leverage_engine.py` (250 linjer)

**Status:** ❌ Kode finnes, men IKKE integrert i live flow  
**Bevis:** Loggene viser statisk 3%/4% TP/SL, ikke adaptiv scaling

### 3B. Formler (teoretiske)

#### **Leverage Sensitivity Factor (LSF)**
```python
LSF = 1 / (1 + ln(leverage + 1))
```
**Konsept:** Higher leverage → Lower LSF → Tighter TP/SL

#### **TP Calculation (multi-stage)**
```python
tp1 = base_tp * (0.6 + LSF)
tp2 = base_tp * (1.2 + LSF / 2.0)
tp3 = base_tp * (1.8 + LSF / 4.0)
```

#### **SL Calculation**
```python
sl = base_sl * (1.0 + (1.0 - LSF) * 0.8)
```

#### **Volatility/Funding Adjustments**
```python
tp_scale = 1.0 + (funding_delta * 0.8)
sl_scale = 1.0 + (exchange_divergence * 0.4) + (volatility_factor * 0.2)
```

#### **Fail-safe Clamps**
```python
SL_CLAMP_MIN = 0.001   # 0.1%
SL_CLAMP_MAX = 0.02    # 2.0%
TP_MIN = 0.003         # 0.3%
SL_MIN = 0.0015        # 0.15%
```

#### **Harvest Schemes**
```python
# Low leverage (≤10x): Conservative gradual
harvest = [0.3, 0.3, 0.4]  # 30% @ TP1, 30% @ TP2, 40% @ TP3

# Medium leverage (11-30x): Aggressive front-load
harvest = [0.4, 0.4, 0.2]

# High leverage (>30x): Ultra-aggressive
harvest = [0.5, 0.3, 0.2]
```

### 3C. Hvorfor ikke aktivt?

**Bevis fra logs:**
```
💰 Margin target: $200.00 (2.0% of $10000.00)
🛡️  Optimal SL: 3.00%  (based on ATR=10.00%)
🎯 Optimal TP: 4.00% (R:R=1.33:1)
⚡ Optimal Leverage: 10.0x
```

→ **Statiske verdier** (3% SL, 4% TP), ikke adaptiv LSF-basert beregning

**Konklusjon:** Adaptive Leverage Engine eksisterer men er IKKE koblet til live flow.

---

## 4. FAKTISK FLYT (I PRODUKSJON)

### 4A. Komplett Sekvens

```
1. trading_bot/simple_bot.py starter trading-syklus (hver 60s)
   └─ Henter market data (price, volume, 24h change)
   └─ Beregner ATR: abs(price_change_24h) / 100
   └─ Beregner volatility: (high_24h - low_24h) / low * 10

2. Kaller AI Engine for signal (confidence, action, reason)
   └─ Fallback: 24h % change → BUY/SELL/HOLD

3. Kaller RL Agent: rl_sizing_agent.decide_sizing()
   └─ Byggger state: (regime, confidence_bucket, exposure, perf)
   └─ 90% exploit: Bruker beste Q-value action
   └─ 10% explore: Random action

4. RL Agent kaller Math AI: math_ai.calculate_optimal_parameters()
   └─ Input: AccountState, MarketConditions, PerformanceMetrics, confidence
   └─ Beregner:
       • Margin: balance * 0.80 = $8000 (80% av $10K)
       • SL: ATR * 2.0, clamped [2.5%, 3.0%] = 3%
       • TP: Base 3.5%, adjusted, clamped [3.0%, 4.5%] = 4%
       • Leverage: Kelly Criterion → 10x (limited history)
       • Notional: $8000 * 10x = $80,000
       • Expected profit: $80,000 * 4% = $3,200
       • Max loss: $80,000 * 3% = $2,400
   └─ Output: OptimalParameters

5. RL Agent pakker resultat i SizingDecision
   └─ position_size_usd: $8000
   └─ leverage: 10x
   └─ tp_percent: 0.04 (4%)
   └─ sl_percent: 0.03 (3%)
   └─ partial_tp_percent: 0.02 (2%)

6. trading_bot publiserer til quantum:stream:trade.intent
   └─ Inkluderer: symbol, side, confidence, entry_price, stop_loss, 
                  take_profit, position_size_usd, leverage, ATR, 
                  volatility, regime

7. intent-bridge konsumerer trade.intent
   └─ Policy filter: Kun tillatte symboler (9 stk)
   └─ Publiserer til quantum:stream:apply.plan

8. apply-layer konsumerer apply.plan
   └─ Anti-duplicate gate (10 min TTL)
   └─ Cooldown gate (15 min per symbol)
   └─ Eksekuterer på Binance testnet
   └─ Oppretter quantum:position:{symbol}
```

### 4B. Eksempel Output (fra logs)

```
💰 Using 80.0% of balance AS MARGIN: $8000.00 (of $10000.00)
💰 Margin target: $8000.00 (80.0% of $10000.00)
🛡️  Optimal SL: 3.00% (based on ATR=10.00%)
🎯 Optimal TP: 4.00% (R:R=1.33:1)
📊 Limited history (0 trades), using conservative 10.0x
⚡ Optimal Leverage: 10.0x
📊 Position Size: $8000.00 margin
💵 Notional: $80000.00
✅ Expected Profit: $3200.00
❌ Max Loss: $2400.00
🎲 Confidence: 48.0%

🔍 [DEBUG] Received from RL Agent: leverage=10.0, position_size=8000.0
[TRADING-BOT] [RL-SIZING] RIVERUSDT: $8000 @ 10.0x (ATR=10.00%, volatility=4.92)
```

---

## 5. KONFIGURASJON & TUNING

### 5A. Environment Variables

**PolicyStore (prioritert) eller ENV fallback:**

```bash
# Fra PolicyStore eller ENV:
RM_MAX_LEVERAGE=25.0           # Max Kelly safety cap (25x)
RM_RISK_PER_TRADE_PCT=0.80     # Risk per trade (80% av balance)
MIN_CONFIDENCE_THRESHOLD=0.45  # Min AI confidence (45%)
TRADING_SYMBOLS=ANKRUSDT,ARCUSDT,FHEUSDT,HYPEUSDT,RIVERUSDT,...
```

### 5B. Hardcoded Defaults (i TradingMathematician __init__)

```python
risk_per_trade_pct: float = 0.80    # 80% of balance per trade
target_profit_pct: float = 0.20     # 20% daily profit target
min_risk_reward: float = 2.0        # Minimum 2:1 R:R
safety_cap: float = 75.0            # Safety cap (overrides fra ENV)
conservative_mode: bool = False     # Aggressive mode
```

### 5C. Bounds & Clamps

**Stop Loss:**
- Minimum: 2.5% (`min_sl = 0.025`)
- Maximum: 3.0% (`max_sl = 0.030`)

**Take Profit:**
- Minimum: 3.0% (`min_tp = 0.030`)
- Maximum: 4.5% (`max_tp = 0.045`)

**Leverage:**
- Minimum: 1.0x
- Maximum: 25.0x (fra ENV/PolicyStore)
- Conservative fallback: 10.0x (hvis < 10 trades)

**Position Size:**
- Minimum: $10 (`min_position_usd = 10.0`)
- Maximum: $8000 (`max_position_usd = 8000.0`) - 80% av $10K

---

## 6. ADAPTIVE FEATURES (FREMTIDIG)

### 6A. Ikke Implementert (men kode finnes)

1. **Adaptive Leverage Engine** - LSF-basert TP/SL scaling
2. **Dynamic TP/SL Adjustment** - Juster TP/SL basert på P&L
3. **Trailing Stops** - Automatisk trail fra +5%
4. **Multi-stage Harvest** - TP1 (50%), TP2 (30%), TP3 (20%)

### 6B. Hvordan Aktivere

**For å aktivere Adaptive Leverage Engine:**

1. Endre `exitbrain_v3_5/exit_brain.py` til å bruke `adaptive_leverage_engine.py`
2. Kall `engine.compute_levels()` for hver posisjon
3. Erstatt statiske TP/SL med LSF-baserte verdier

**Eksempel (teoretisk):**
```python
from exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine

engine = AdaptiveLeverageEngine(base_tp=0.01, base_sl=0.005)
levels = engine.compute_levels(
    base_tp_pct=0.01,
    base_sl_pct=0.005,
    leverage=position_leverage,
    volatility_factor=atr_pct,
    funding_delta=funding_rate,
    exchange_divergence=0.0
)

# Use levels.tp1_pct, levels.sl_pct instead of static 4%, 3%
```

---

## 7. TESTING & VALIDERING

### 7A. Unit Tests

**Adaptive Leverage Engine har tests:**
```python
# microservices/exitbrain_v3_5/adaptive_leverage_engine.py
def test_adaptive_engine():
    # Test 1: Low leverage → higher TP
    # Test 2: High leverage → wider SL
    # Test 3: Clamps work for extreme leverage
    # Test 4: Harvest schemes correct
    # Test 5: TP minimums enforced
```

**Kjør:**
```bash
python -m microservices.exitbrain_v3_5.adaptive_leverage_engine --test
```

### 7B. Produksjon Validering

**Fra logs (Feb 3, 22:47 UTC):**
```
✅ ANKRUSDT: $200 @ 10.0x | TP=4.00% (partial@2.00%), SL=3.00%
✅ RIVERUSDT: $200 @ 10.0x | TP=4.00% (partial@2.00%), SL=3.00%
✅ FHEUSDT: $200 @ 10.0x | TP=4.00% (partial@2.00%), SL=3.00%
```

**Observasjoner:**
- ✅ Alle posisjoner bruker 10x leverage (limited history fallback)
- ✅ TP/SL er konsistent (3%/4%)
- ✅ Partial TP aktivert (2% = 50% av 4% mål)
- ❌ Ikke adaptiv (samme verdier for alle symboler)

---

## 8. KONKLUSJON

### 8A. Hva Som Fungerer (AKTIVT)

✅ **Trading Mathematician** - Fullstendig integrert og aktiv  
✅ **RL Position Sizing Agent** - Wrapper + Q-learning state  
✅ **Kelly Criterion Leverage** - Matematisk optimalisering  
✅ **ATR-basert SL** - Dynamisk basert på volatilitet  
✅ **Volatility-basert TP** - Justeres etter market conditions  
✅ **Fail-safe Clamps** - 2.5-3% SL, 3-4.5% TP bounds  
✅ **Conservative Fallback** - 10x leverage for nye symboler  

### 8B. Hva Som IKKE Fungerer (INAKTIVT)

❌ **Adaptive Leverage Engine** - Kode finnes, ikke integrert  
❌ **Dynamic TP/SL Adjustment** - Statiske verdier fra entry  
❌ **Multi-stage Harvest** - Kun én partial TP (50%)  
❌ **Trailing Stops** - Ikke aktivert  
❌ **LSF-basert Scaling** - Ikke i bruk  

### 8C. Formel-sammendrag

| Parameter | Formel | Standard | Bounds |
|-----------|--------|----------|---------|
| **Margin** | `balance * 0.80` | $8000 | $10 - $8000 |
| **SL** | `ATR * 2.0` → clamp | 3.0% | 2.5% - 3.0% |
| **TP** | `3.5%` → adjust → clamp | 4.0% | 3.0% - 4.5% |
| **Leverage** | Kelly / Conservative | 10x | 1x - 25x |
| **Partial TP** | `TP * 0.5` | 2.0% | - |
| **Notional** | `Margin * Leverage` | $80K | - |

### 8D. Trigger Chain (komplett)

```
trading_bot.py:487
  ↓
rl_sizing_agent.decide_sizing()
  ↓
rl_position_sizing_agent.py:675
  ↓
math_ai.calculate_optimal_parameters()
  ↓
trading_mathematician.py:120-400
  ↓
  • _calculate_risk_amount() → Margin
  • _calculate_optimal_sl() → SL
  • _calculate_optimal_tp() → TP
  • _calculate_optimal_leverage() → Leverage
  ↓
OptimalParameters object
  ↓
SizingDecision object
  ↓
trading_bot publishes to quantum:stream:trade.intent
```

---

**RAPPORT SLUTT**
