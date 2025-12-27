# 🚀 QUANTUM TRADER - KOMPLETT SYSTEMDOKUMENTASJON

> **AI-Drevet Kryptovaluta Trading Platform**  
> Autonom handel med 4-modell ensemble, 30x leverage, og intelligent risikostyring

**Sist oppdatert:** 26. november 2025  
**Versjon:** 3.0 (AI Hedge Fund OS)  
**Status:** 🟢 LIVE på Binance Futures Testnet

---

## 📋 INNHOLDSFORTEGNELSE

1. [Systemoversikt](#systemoversikt)
2. [Arkitektur](#arkitektur)
3. [AI Engine](#ai-engine)
4. [Trading System](#trading-system)
5. [Risikostyring](#risikostyring)
6. [AI Hedge Fund OS](#ai-hedge-fund-os)
7. [Dataflyt](#dataflyt)
8. [Konfigurasjoner](#konfigurasjoner)
9. [Deployment](#deployment)
10. [Feilsøking](#feilsøking)

---

## 🎯 SYSTEMOVERSIKT

### Hva er Quantum Trader?

Quantum Trader er en avansert, AI-styrt trading-platform som automatisk handler kryptovaluta futures på Binance. Systemet bruker fire separate AI-modeller som stemmer sammen for å ta handelsbeslutninger med høy presisjon.

### Nøkkelfunksjoner

- ✅ **4-Modell AI Ensemble** (XGBoost, LightGBM, N-HiTS, PatchTST)
- ✅ **Autonom Event-Driven Trading** (10s intervaller)
- ✅ **Dynamisk Position Sizing** (ATR-basert med AI confidence multipliers)
- ✅ **Intelligent TP/SL Management** (Confidence-basert, dynamisk)
- ✅ **Multi-Layer Risk Management** (6 lag sikkerhet)
- ✅ **AI Hedge Fund Operating System** (Meta-intelligence koordinering)
- ✅ **Continuous Learning** (Auto-retraining basert på resultater)
- ✅ **Self-Healing System** (Automatisk feildeteksjon og reparasjon)

### Nåværende Status (26. nov 2025)

**Live Trading på Binance Futures Testnet:**
- 💰 Balance: $8,930.41 USDT
- 📊 Leverage: 30x
- 🎯 Max posisjoner: 10 (5 aktive)
- 🤖 AI Status: 3/4 modeller aktive (N-HiTS warmup: ~15 min)
- ⏱️ Check interval: 10 sekunder
- 🔄 Cooldown: 120 sekunder mellom trades

**Aktive Posisjoner:**
1. PUNDIXUSDT (-2.42% PnL)
2. ZECUSDT (-10.37% PnL)
3. TAOUSDT (-23.00% PnL)
4. AAVEUSDT (-18.56% PnL)
5. BTCUSDT (-3.14% PnL, AI endret til SELL 83%)

---

## 🏗️ ARKITEKTUR

### Hovedkomponenter

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI HEDGE FUND OS (Øverste lag)              │
│  Koordinerer alle subsystemer, meta-intelligence                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR POLICY ENGINE                    │
│  Central "Conductor" - Unified policy for alle subsystemer      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
┌───────────────────┐                    ┌─────────────────────┐
│   AI ENGINE       │                    │   TRADING SYSTEM    │
│  (4 Modeller)     │◄──────────────────►│  (Execution Layer)  │
└───────────────────┘                    └─────────────────────┘
        ↓                                           ↓
┌───────────────────┐                    ┌─────────────────────┐
│  ENSEMBLE MANAGER │                    │  POSITION MONITOR   │
│  (Voting System)  │                    │  (TP/SL Management) │
└───────────────────┘                    └─────────────────────┘
        ↓                                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT LAYER                         │
│  6 Lag: Governor, Policy, Lifecycle, Performance, Cost, Funding │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      BINANCE FUTURES API                         │
│  REST + WebSocket, Market Data, Order Execution                 │
└─────────────────────────────────────────────────────────────────┘
```

### Katalogstruktur

```
quantum_trader/
├── backend/                  # FastAPI backend (Python)
│   ├── main.py              # Hovedapplikasjon, lifespan manager
│   ├── routes/              # API endpoints
│   │   ├── ai.py           # AI predictions, training
│   │   ├── trades.py       # Trade history, execution
│   │   ├── signals.py      # Live AI signals
│   │   └── risk.py         # Risk management API
│   ├── services/            # Forretningslogikk (50+ moduler)
│   │   ├── event_driven_executor.py      # ⭐ Hovedmotor
│   │   ├── ai_trading_engine.py          # AI signal generator
│   │   ├── orchestrator_policy.py        # Policy engine
│   │   ├── ai_hedgefund_os.py           # Meta-intelligence
│   │   ├── position_monitor.py           # TP/SL protection
│   │   ├── regime_detector.py            # Market regime detection
│   │   ├── risk_guard.py                 # Risk limits
│   │   ├── self_healing.py               # Auto-recovery
│   │   └── ...                           # 40+ andre moduler
│   └── config/              # Konfigurasjonsfiler
│       ├── config.py        # Environment config loader
│       └── risk.py          # Risk parameters
│
├── ai_engine/               # AI/ML modeller
│   ├── agents/              # Individuelle modell-agenter
│   │   ├── xgb_agent.py    # XGBoost (25% weight)
│   │   ├── lgbm_agent.py   # LightGBM (25% weight)
│   │   ├── nhits_agent.py  # N-HiTS (30% weight)
│   │   └── patchtst_agent.py # PatchTST (20% weight)
│   ├── ensemble_manager.py  # 4-modell voting system
│   ├── feature_engineer.py  # Technical indicators
│   └── train_and_save.py    # Model training pipeline
│
├── models/                  # Trente modeller (.pkl, .pth)
│   ├── xgb_model.pkl
│   ├── lightgbm_model.pkl
│   ├── nhits_model.pth
│   └── patchtst_model.pth
│
├── frontend/                # React dashboard (ikke i bruk)
├── database/                # SQLite schema
├── docker-compose.yml       # Container orchestration
├── .env                     # Environment variabler
└── README.md               # Prosjektdokumentasjon
```

---

## 🧠 AI ENGINE

### 4-Modell Ensemble System

Quantum Trader bruker fire komplementære AI-modeller som stemmer sammen:

#### 1️⃣ **XGBoost Agent** (25% vekt)
- **Type:** Gradient Boosted Trees
- **Styrke:** Feature interactions, non-linear patterns
- **Input:** 50+ tekniske indikatorer
- **Output:** BUY/SELL/HOLD + confidence (0-1)
- **Fil:** `ai_engine/agents/xgb_agent.py`

```python
class XGBAgent:
    def predict(self, symbol: str, features: Dict[str, float]) -> tuple[str, float, str]:
        # Preprocessor feature engineering
        # XGBoost model inference
        # Returns: (action, confidence, reason)
```

#### 2️⃣ **LightGBM Agent** (25% vekt)
- **Type:** Light Gradient Boosted Trees
- **Styrke:** Fast inference, sparse features, high efficiency
- **Spesialitet:** Volatility patterns, mean reversion
- **Fil:** `ai_engine/agents/lgbm_agent.py`

#### 3️⃣ **N-HiTS Agent** (30% vekt - høyest!)
- **Type:** Neural Hierarchical Interpolation for Time Series
- **Styrke:** Multi-rate temporal patterns, volatility forecasting
- **Arkitektur:** Multi-rate sampling blocks
- **Warmup:** 120 ticks (~20 min @ 10s intervals)
- **Fil:** `ai_engine/agents/nhits_agent.py`

#### 4️⃣ **PatchTST Agent** (20% vekt)
- **Type:** Patch Time Series Transformer
- **Styrke:** Long-range dependencies, trend detection
- **Arkitektur:** Transformer med patch-based encoding
- **Warmup:** 30 ticks (~5 min @ 10s intervals)
- **Fil:** `ai_engine/agents/patchtst_agent.py`

### Ensemble Voting System

**Fil:** `ai_engine/ensemble_manager.py`

```python
class EnsembleManager:
    def predict(self, symbol: str, features: Dict) -> tuple[str, float, Dict]:
        # 1. Samle predictions fra alle 4 modeller
        xgb_action, xgb_conf = xgb_agent.predict(symbol, features)
        lgbm_action, lgbm_conf = lgbm_agent.predict(symbol, features)
        nhits_action, nhits_conf = nhits_agent.predict(symbol, features)
        patchtst_action, patchtst_conf = patchtst_agent.predict(symbol, features)
        
        # 2. Skip modeller i warmup
        # 3. Weighted voting: 30% N-HiTS, 25% XGB, 25% LGBM, 20% PatchTST
        # 4. Consensus check: Minimum 3/4 modeller må være enige
        # 5. Aggregate confidence: Weighted average
        
        return (ensemble_action, ensemble_confidence, metadata)
```

**Eksempel output:**
```
[ENSEMBLE] BTCUSDT: SELL 83.18%
  XGB:    SELL/0.67
  LGBM:   BUY/0.89
  N-HiTS: HOLD/0.50 (warmup)
  PatchTST: SELL/0.99
→ Final: SELL (3/4 agree), Confidence: 0.83
```

### Feature Engineering

**Fil:** `ai_engine/feature_engineer.py`

Generer 50+ tekniske indikatorer fra OHLCV data:

**Momentum Indicators:**
- RSI (14, 28 periods)
- MACD (12, 26, 9)
- Stochastic RSI
- Williams %R
- CCI (Commodity Channel Index)

**Trend Indicators:**
- SMA (20, 50, 200)
- EMA (12, 26)
- ADX (Average Directional Index)
- Parabolic SAR

**Volatility Indicators:**
- Bollinger Bands (20, 2σ)
- ATR (Average True Range 14)
- Keltner Channels
- Standard Deviation

**Volume Indicators:**
- OBV (On-Balance Volume)
- Volume Rate of Change
- VWAP (Volume Weighted Average Price)

**Pattern Recognition:**
- Candlestick patterns (doji, engulfing, hammer)
- Support/Resistance levels
- Trend channels

---

## 💹 TRADING SYSTEM

### Event-Driven Executor

**Hjerte av systemet:** `backend/services/event_driven_executor.py`

Dette er hovedmotoren som kjører autonom trading:

```python
class EventDrivenExecutor:
    """
    Continuously monitors market and executes trades when AI detects
    high-confidence opportunities (10s check interval, 120s cooldown)
    """
    
    async def run_forever(self):
        while True:
            # 1. CHECK: Er vi i cooldown? (120s etter siste trade)
            if self._in_cooldown():
                await asyncio.sleep(10)
                continue
            
            # 2. AI SIGNALS: Generer predictions for alle 222 symbols
            ai_signals = await self._generate_ai_signals()
            
            # 3. ORCHESTRATOR FILTER: Policy-based filtering
            approved_signals = self.orchestrator.filter_signals(ai_signals)
            
            # 4. RISK CHECK: Position limits, exposure, daily loss
            safe_signals = self._apply_risk_filters(approved_signals)
            
            # 5. EXECUTION: Place orders for approved signals
            for signal in safe_signals:
                await self._execute_trade(signal)
            
            await asyncio.sleep(10)  # Check every 10 seconds
```

**Flow:**
```
Every 10 seconds:
  ↓
Check cooldown (120s)
  ↓ (if expired)
Generate AI signals (222 symbols)
  ↓
Orchestrator Policy filter
  ↓
Risk Management checks
  ↓
Execute approved trades
  ↓
Set cooldown timer
```

### AI Trading Engine

**Fil:** `backend/services/ai_trading_engine.py`

Genererer AI-signaler og dynamiske TP/SL targets:

```python
class AITradingEngine:
    def generate_signals(self) -> List[Signal]:
        """Generate AI signals for all symbols"""
        signals = []
        
        for symbol in self.symbols:
            # 1. Fetch OHLCV data (100 candles)
            ohlcv = self._fetch_market_data(symbol)
            
            # 2. Feature engineering
            features = self.feature_engineer.compute_features(ohlcv)
            
            # 3. Ensemble prediction
            action, confidence, meta = self.ensemble.predict(symbol, features)
            
            # 4. Regime detection
            regime = self.regime_detector.detect(ohlcv)
            
            # 5. Dynamic TP/SL calculation (confidence-based)
            tp_pct, sl_pct = self._calculate_dynamic_tpsl(confidence, regime)
            
            signals.append(Signal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                regime=regime
            ))
        
        return signals
```

### Position Monitor

**Fil:** `backend/services/position_monitor.py`

Overvåker alle åpne posisjoner og sikrer TP/SL beskyttelse:

```python
class PositionMonitor:
    """
    Runs independently every 10 seconds to:
    - Detect positions without TP/SL
    - Set hybrid TP/SL strategy (partial + trailing)
    - Re-evaluate AI sentiment (warn if changed)
    - Monitor PnL and trigger alerts
    """
    
    async def monitor_loop(self):
        while True:
            # 1. Fetch all open positions from Binance
            positions = self.client.futures_position_information()
            
            for position in positions:
                if float(position['positionAmt']) == 0:
                    continue  # Skip closed positions
                
                # 2. Check if position has TP/SL protection
                open_orders = self.client.futures_get_open_orders(
                    symbol=position['symbol']
                )
                
                has_tp = any(o['type'] == 'TAKE_PROFIT_MARKET' for o in open_orders)
                has_sl = any(o['type'] == 'STOP_MARKET' for o in open_orders)
                
                # 3. If unprotected, set TP/SL immediately
                if not has_tp or not has_sl:
                    await self._set_tpsl_protection(position)
                
                # 4. Re-evaluate AI sentiment
                current_signal = self.ai_engine.predict(position['symbol'])
                if current_signal.action != position['side']:
                    logger.warning(f"⚠️ {position['symbol']}: AI changed sentiment!")
            
            await asyncio.sleep(10)
```

**Hybrid TP/SL Strategy:**
1. **Partial Take Profit** (50-80% av position ved TP1)
2. **Trailing Stop** (resterende 20-50% med trailing stop)
3. **Stop Loss** (beskyttelse mot store tap)

---

## 🛡️ RISIKOSTYRING

Quantum Trader har **6 lag** av risikostyring:

### 1. Safety Governor

**Fil:** `backend/services/safety_governor.py`

Øverste lag - kan stoppe ALT:

```python
class SafetyGovernor:
    """
    Supreme risk controller - can halt trading immediately
    
    Monitors:
    - Daily drawdown (max -$1,500)
    - Total exposure (max $45,000 notional)
    - Losing streak (max 5 consecutive losses)
    - System health
    """
    
    def compute_directives(self) -> SafetyDirectives:
        # Check all risk limits
        if daily_loss > DAILY_LOSS_LIMIT:
            return SafetyDirectives(
                allow_trades=False,
                allow_new_positions=False,
                reason="Daily loss limit exceeded"
            )
        
        # Normal operation
        return SafetyDirectives(allow_trades=True, ...)
```

### 2. Orchestrator Policy

**Fil:** `backend/services/orchestrator_policy.py`

Central "Conductor" som lager unified policy:

```python
class OrchestratorPolicy:
    """
    Unifies all subsystem outputs into ONE policy
    
    Inputs:
    - Regime (TRENDING/RANGING/NORMAL)
    - Risk state (exposure, drawdown, streak)
    - Symbol performance (win rates)
    - Cost metrics (spread, slippage)
    - Volatility (ATR levels)
    
    Outputs:
    - min_confidence: Dynamic threshold (0.32-0.40)
    - max_risk_pct: Position sizing multiplier
    - allow_trades: Boolean permission
    """
    
    def create_policy(self) -> Policy:
        # 1. Detect market regime
        regime = self.regime_detector.get_current_regime()
        
        # 2. Get base thresholds
        if regime == "TRENDING":
            min_conf = 0.32  # Lower threshold in trends
        elif regime == "RANGING":
            min_conf = 0.40  # Higher in ranges
        else:
            min_conf = 0.38  # Normal market
        
        # 3. Adjust for volatility
        if self.volatility > 0.06:  # Extreme
            min_conf += 0.07
        
        # 4. Adjust for cost
        if self.spread_bps > 10:
            min_conf += 0.05
        
        # 5. Create unified policy
        return Policy(
            min_confidence=min_conf,
            max_risk_pct=1.5,
            allow_trades=True
        )
```

**Regime-Based Thresholds:**
```
TRENDING regime → min_confidence = 0.32 (lavere = mer trades)
RANGING regime  → min_confidence = 0.40 (høyere = færre trades)
NORMAL regime   → min_confidence = 0.38 (balansert)
```

### 3. Risk Management (Position Sizing)

**Katalog:** `backend/services/risk_management/`

ATR-basert dynamisk position sizing med AI multipliers:

```python
def calculate_position_size(
    equity: float,           # $8,930
    base_risk_pct: float,    # 1.5%
    policy_multiplier: float, # Regime-based (0.8-1.5x)
    confidence: float,       # AI confidence (0.0-1.0)
    atr: float,             # Volatility measure
    current_price: float
) -> float:
    """
    AI-Driven Position Sizing Formula
    
    Example (BTCUSDT, conf=0.83):
    equity = $8,930
    base_risk = 1.5% = $133.95
    policy_mult = 1.2 (TRENDING regime)
    confidence_mult = 2.0 (0.83 >= 0.85 → high confidence)
    atr = $1,200
    
    risk_usd = $133.95 × 1.2 × 2.0 = $321.48
    sl_distance_pct = (1.5 × $1,200) / $87,400 = 2.06%
    notional = $321.48 / 0.0206 = $15,606
    notional = clamp($15,606, min=$20, max=$5,000) = $5,000
    quantity = $5,000 / $87,400 = 0.0572 BTC
    margin @ 30x = $5,000 / 30 = $166.67
    """
    
    # 1. Calculate base risk in USD
    risk_usd = equity * base_risk_pct * policy_multiplier
    
    # 2. AI Confidence Multipliers
    if confidence >= 0.85:
        confidence_mult = 2.0    # High confidence → 2x size
    elif confidence >= 0.70:
        confidence_mult = 1.5    # Medium-high → 1.5x
    elif confidence >= 0.60:
        confidence_mult = 1.0    # Medium → 1x
    else:
        confidence_mult = 0.3    # Low → 0.3x (defensive)
    
    risk_usd *= confidence_mult
    
    # 3. Calculate stop loss distance (ATR-based)
    sl_distance = atr * 1.5
    sl_distance_pct = sl_distance / current_price
    
    # 4. Calculate notional position size
    notional = risk_usd / sl_distance_pct
    
    # 5. Apply limits
    notional = max(notional, 20)      # Min $20
    notional = min(notional, 5000)    # Max $5,000
    
    # 6. Convert to quantity
    quantity = notional / current_price
    
    return quantity
```

**Position Size Examples:**

Med $8,930 equity:

| Confidence | Risk  | Policy | AI Mult | Final Risk | ATR-based Notional | Margin @ 30x |
|-----------|-------|--------|---------|------------|-------------------|--------------|
| 0.50      | 1.5%  | 1.0x   | 0.3x    | $40        | $40               | $1.33        |
| 0.70      | 1.5%  | 1.0x   | 1.5x    | $201       | $201              | $6.70        |
| 0.85      | 1.5%  | 1.2x   | 2.0x    | $321       | $321-$5,000       | $10-$166     |
| 0.90+     | 1.5%  | 1.5x   | 2.0x    | $402       | $402-$5,000       | $13-$166     |

### 4. Trade Lifecycle Manager

Håndterer hele trade-syklusen:
- Pre-trade validation
- Signal quality assessment
- Market conditions check
- Consensus building
- Post-trade tracking

### 5. Cost Model

**Fil:** `backend/services/cost_model.py`

Estimerer trading costs:
- Spread (bid-ask difference)
- Slippage (price impact)
- Funding rate
- Exchange fees

### 6. Funding Rate Filter

**Fil:** `backend/services/funding_rate_filter.py`

Blokkerer trades med høye funding rates:
```python
if funding_rate > 0.01:  # 1% per 8h
    return "BLOCKED: High funding rate"
```

---

## 🤖 AI HEDGE FUND OS

**Fil:** `backend/services/ai_hedgefund_os.py`

Meta-intelligence layer som koordinerer ALT:

```python
class AIHedgeFundOS:
    """
    Supreme meta-intelligence overseer
    
    Coordinates 8 major subsystems:
    1. Universe OS (symbol selection)
    2. Risk OS (limits, exposure)
    3. Execution Layer (order routing)
    4. Position Intelligence (PIL)
    5. Portfolio Balancer (PBA)
    6. Model Supervisor (AI health)
    7. Profit Amplification (PAL)
    8. Self-Healing System
    """
    
    def coordination_cycle(self):
        # 1. Collect state from all subsystems
        universe_state = self.universe_os.get_state()
        risk_state = self.risk_os.get_state()
        execution_state = self.execution_layer.get_state()
        position_state = self.position_intelligence.get_state()
        portfolio_state = self.portfolio_balancer.get_state()
        model_state = self.model_supervisor.get_state()
        profit_state = self.profit_amplification.get_state()
        healing_state = self.self_healing.get_state()
        
        # 2. Detect conflicts and issues
        conflicts = self._detect_conflicts(all_states)
        
        # 3. Determine global risk mode
        risk_mode = self._compute_risk_mode(all_states)
        
        # 4. Create global directives
        directives = GlobalDirectives(
            allow_new_trades=True,
            allow_new_positions=True,
            scale_position_sizes=1.0,
            universe_mode="NORMAL",
            execution_preference="SMART",
            portfolio_action="HOLD"
        )
        
        # 5. Apply directives to all subsystems
        self._apply_directives(directives)
        
        # 6. Log coordination report
        self._save_coordination_report()
```

**Coordination Report Eksempel:**
```json
{
  "timestamp": "2025-11-26T00:00:00Z",
  "system_risk_mode": "NORMAL",
  "system_health": "HEALTHY",
  "subsystems": {
    "universe_os": {"status": "ACTIVE", "health": 95},
    "risk_os": {"status": "ACTIVE", "health": 100},
    "execution_layer": {"status": "ACTIVE", "health": 90}
  },
  "conflicts": [],
  "emergency_actions": [],
  "global_directives": {
    "allow_new_trades": true,
    "scale_position_sizes": 1.0
  }
}
```

---

## 📊 DATAFLYT

### Complete Trading Flow

```
1. MARKET DATA INGESTION (Hver 10. sekund)
   ↓
   Binance WebSocket → OHLCV data (100 candles per symbol)
   ↓
   
2. FEATURE ENGINEERING
   ↓
   feature_engineer.py → 50+ tekniske indikatorer
   ↓
   
3. AI PREDICTION (4-Model Ensemble)
   ↓
   XGBoost → BUY/SELL/HOLD + confidence
   LightGBM → BUY/SELL/HOLD + confidence
   N-HiTS → BUY/SELL/HOLD + confidence
   PatchTST → BUY/SELL/HOLD + confidence
   ↓
   ensemble_manager.py → Weighted voting + consensus
   ↓
   
4. REGIME DETECTION
   ↓
   regime_detector.py → TRENDING/RANGING/NORMAL
   ↓
   
5. ORCHESTRATOR POLICY
   ↓
   orchestrator_policy.py → Dynamic confidence threshold
   ↓
   Filter: confidence >= min_confidence (0.32-0.40)
   ↓
   
6. RISK MANAGEMENT
   ↓
   A. Safety Governor → Daily loss, exposure checks
   B. Position Sizing → ATR-based with AI multipliers
   C. Portfolio Limits → Max 10 positions, exposure limits
   ↓
   
7. EXECUTION
   ↓
   A. Check cooldown (120s)
   B. Place MARKET order
   C. Set hybrid TP/SL (partial + trailing)
   D. Log trade entry
   ↓
   
8. POSITION MONITORING (Parallel loop, hver 10s)
   ↓
   A. Fetch open positions
   B. Check TP/SL protection
   C. Re-evaluate AI sentiment
   D. Monitor PnL
   E. Trigger alerts if needed
   ↓
   
9. CONTINUOUS LEARNING (Background)
   ↓
   A. Collect trade results
   B. Retrain models (50+ samples or 24h)
   C. Backtest new models
   D. Deploy if improved
```

### Inter-Module Communication

**AI Engine → Trading System:**
```python
signal = {
    "symbol": "BTCUSDT",
    "action": "SELL",
    "confidence": 0.83,
    "regime": "TRENDING",
    "tp_pct": 0.083,  # 8.3% (confidence-based)
    "sl_pct": 0.072,  # 7.2%
    "metadata": {
        "xgb": {"action": "SELL", "conf": 0.67},
        "lgbm": {"action": "BUY", "conf": 0.89},
        "nhits": {"action": "HOLD", "conf": 0.50},
        "patchtst": {"action": "SELL", "conf": 0.99}
    }
}
```

**Orchestrator Policy → Executor:**
```python
policy = {
    "min_confidence": 0.32,    # TRENDING regime
    "max_risk_pct": 1.2,       # 20% boost
    "allow_trades": True,
    "max_open_positions": 10,
    "regime": "TRENDING",
    "volatility_adj": -0.02    # Lower threshold in low vol
}
```

**Safety Governor → All Systems:**
```python
directives = {
    "allow_trades": True,
    "allow_new_positions": True,
    "leverage_multiplier": 1.0,
    "size_multiplier": 1.0,
    "risk_level": "NORMAL"
}
```

---

## ⚙️ KONFIGURASJONER

### Environment Variables (.env)

**Critical Trading Parameters:**
```bash
# 🤖 AI Model Selection
AI_MODEL=hybrid                      # TFT + XGBoost ensemble

# 🎯 Event-Driven Trading
QT_EVENT_DRIVEN_MODE=true
QT_CONFIDENCE_THRESHOLD=0.45         # Ensemble threshold
QT_CHECK_INTERVAL=10                 # Check every 10s
QT_COOLDOWN_SECONDS=120              # 2 min cooldown

# 💰 Position Sizing (AI-Driven)
RM_MAX_POSITION_USD=5000             # Max $5k notional
RM_MIN_POSITION_USD=20               # Min $20 notional
RM_RISK_PER_TRADE_PCT=0.015          # 1.5% base risk
RM_MAX_RISK_PCT=0.03                 # 3% max risk
RM_HIGH_CONF_MULT=2.0                # 2x size @ conf>=0.85
RM_LOW_CONF_MULT=0.3                 # 0.3x size @ conf<0.60
RM_MAX_CONCURRENT_TRADES=10          # Max 10 positions

# 🎯 Leverage & Limits
QT_DEFAULT_LEVERAGE=30               # 30x leverage
QT_MAX_POSITIONS=10                  # Max 10 concurrent
QT_MAX_DAILY_LOSS=1500               # $1,500 daily limit

# 📊 TP/SL Configuration
QT_TP_PCT=0.06                       # 6% base TP
QT_SL_PCT=0.08                       # 8% base SL
QT_TRAIL_PCT=0.02                    # 2% trailing
QT_PARTIAL_TP=0.5                    # 50% partial exit

# 🔐 API Credentials
BINANCE_TESTNET_API_KEY=your_key
BINANCE_TESTNET_SECRET_KEY=your_secret
USE_BINANCE_TESTNET=true

# 🤖 AI Features
QT_CONTINUOUS_LEARNING=true          # Auto-retrain
QT_MIN_SAMPLES_FOR_RETRAIN=50        # Retrain after 50 trades
QT_RETRAIN_INTERVAL_HOURS=24         # Auto-retrain daily

# 🔍 Monitoring
QT_MODEL_SUPERVISOR_INTERVAL=300     # Check AI bias every 5 min
QT_PORTFOLIO_BALANCER_INTERVAL=60    # Balance every 1 min
```

### Orchestrator Profiles

**File:** `backend/services/orchestrator_config.py`

**SAFE Profile:**
```python
SAFE = {
    "base_confidence": 0.55,          # Higher threshold
    "base_risk_pct": 0.8,             # Lower risk
    "max_open_positions": 6,          # Fewer positions
    "total_exposure_limit": 10.0,     # Conservative
    "daily_dd_limit": 2.0             # Tight stop
}
```

**AGGRESSIVE Profile:**
```python
AGGRESSIVE = {
    "base_confidence": 0.45,          # Lower threshold
    "base_risk_pct": 1.5,             # Higher risk
    "max_open_positions": 10,         # More positions
    "total_exposure_limit": 20.0,     # More exposure
    "daily_dd_limit": 5.0             # Wider stop
}
```

### Docker Compose Configuration

**File:** `docker-compose.yml`

```yaml
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    container_name: quantum_backend
    restart: unless-stopped
    env_file: .env
    environment:
      - PYTHONPATH=/app
      - AI_MODEL=hybrid
      - QT_EVENT_DRIVEN_MODE=true
      # ... (alle env vars fra .env)
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
      - ./ai_engine:/app/ai_engine
      - ./models:/app/models
      - ./database:/app/database
    networks:
      - quantum_trader

networks:
  quantum_trader:
    driver: bridge
```

---

## 🚀 DEPLOYMENT

### Lokal Utvikling

**1. Installer avhengigheter:**
```bash
# Python backend
cd backend
pip install -r requirements.txt

# Frontend (optional)
cd frontend
npm install
```

**2. Konfigurer .env:**
```bash
cp .env.example .env
# Rediger .env med dine API keys
```

**3. Start med Docker:**
```bash
docker-compose up --build
```

**4. Verifiser:**
```bash
# Backend health
curl http://localhost:8000/health

# AI status
curl http://localhost:8000/api/ai/status
```

### Production Deployment

**1. VPS Setup (Ubuntu 22.04):**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin

# Clone repository
git clone https://github.com/yourusername/quantum_trader.git
cd quantum_trader
```

**2. Production Configuration:**
```bash
# Create production .env
cp .env.example .env.production

# Edit with production values
nano .env.production

# Set production profile
export ORCHESTRATOR_PROFILE=SAFE
```

**3. Start Production:**
```bash
# Build and start
docker-compose --profile live up -d

# View logs
docker-compose logs -f backend

# Monitor
docker-compose ps
```

**4. Monitoring:**
```bash
# Watch logs in real-time
docker logs quantum_backend -f --tail 100

# Check AI status
docker exec quantum_backend python -c "from backend.services.ai_trading_engine import AITradingEngine; print('AI OK')"

# Check positions
docker exec quantum_backend python check_current_positions.py
```

---

## 🔍 FEILSØKING

### Vanlige Problemer

#### 1. AI Modeller Ikke Lastet

**Symptom:**
```
ERROR: FileNotFoundError: models/xgb_model.pkl not found
```

**Løsning:**
```bash
# Tren modeller
cd ai_engine
python train_and_save.py

# Verifiser
ls -la ../models/
```

#### 2. Ingen Trades Utføres

**Symptom:** System kjører, men ingen trades

**Sjekkliste:**
```bash
# 1. Sjekk cooldown
docker logs quantum_backend --tail 50 | grep "cooldown"

# 2. Sjekk AI confidence
docker logs quantum_backend --tail 100 | grep "ENSEMBLE"

# 3. Sjekk orchestrator threshold
docker logs quantum_backend --tail 100 | grep "min_confidence"

# 4. Sjekk risk limits
docker exec quantum_backend python -c "
from backend.services.safety_governor import SafetyGovernor
gov = SafetyGovernor()
print(gov.compute_directives())
"
```

**Vanlige årsaker:**
- Cooldown aktiv (120s)
- AI confidence under threshold (< 0.32-0.40)
- Daily loss limit nådd
- Max positions nådd (10/10)

#### 3. Position Monitor Feil

**Symptom:**
```
ERROR: Order would immediately trigger (-2021)
```

**Forklaring:** Stop loss price er allerede nådd

**Løsning:**
- Position monitor justerer automatisk
- Eller steng position manuelt hvis AI sentiment endret

#### 4. API Rate Limits

**Symptom:**
```
ERROR: APIError(code=-1003): Too many requests
```

**Løsning:**
```python
# Øk intervals i .env
QT_CHECK_INTERVAL=15  # Fra 10 til 15s
QT_COOLDOWN_SECONDS=180  # Fra 120 til 180s
```

#### 5. N-HiTS/PatchTST Warmup

**Symptom:**
```
INFO: N-HiTS: Not enough history (25/120)
```

**Forklaring:** Modellen trenger mer data

**Timeline:**
- N-HiTS: 120 ticks × 10s = 20 minutter
- PatchTST: 30 ticks × 10s = 5 minutter

**Status:**
```bash
docker logs quantum_backend --tail 50 | grep "Not enough history"
```

---

## 📈 PERFORMANCE MONITORING

### Real-Time Metrics

**1. AI Signal Quality:**
```bash
docker logs quantum_backend --tail 100 | grep "ENSEMBLE"
```

Output:
```
[ENSEMBLE] BTCUSDT: SELL 83.18% | XGB:SELL/0.67 LGBM:BUY/0.89 PT:SELL/0.99
conf avg=0.83 max=0.99
```

**2. Position Status:**
```bash
docker exec quantum_backend python check_current_positions.py
```

**3. Daily PnL:**
```bash
docker exec quantum_backend python check_daily_pnl.py
```

**4. System Health:**
```bash
curl http://localhost:8000/health | jq
```

Output:
```json
{
  "status": "healthy",
  "uptime": 3600,
  "active_positions": 5,
  "daily_pnl": -450.50,
  "ai_status": {
    "xgboost": "active",
    "lightgbm": "active",
    "nhits": "warmup",
    "patchtst": "active"
  }
}
```

### Log Analysis

**Find specific events:**
```bash
# Trades executed
docker logs quantum_backend | grep "ORDER PLACED"

# TP/SL hits
docker logs quantum_backend | grep "TAKE_PROFIT\|STOP_LOSS"

# AI warnings
docker logs quantum_backend | grep "WARNING.*AI"

# Risk violations
docker logs quantum_backend | grep "BLOCKED\|REJECTED"
```

---

## 🎓 APPENDIX

### Glossary

**AI Terms:**
- **Ensemble:** Kombinasjon av flere modeller
- **Confidence:** AI's sikkerhet i prediction (0-1)
- **Feature Engineering:** Lage tekniske indikatorer fra pris-data
- **Warmup:** Initial data collection period for neural models

**Trading Terms:**
- **Notional:** Total posisjonsstørrelse før leverage
- **Margin:** Faktisk kapital brukt (notional / leverage)
- **TP (Take Profit):** Automatisk salg ved profitt-target
- **SL (Stop Loss):** Automatisk salg ved tap-grense
- **ATR (Average True Range):** Volatility measure
- **Funding Rate:** Kostnad for å holde perpetual futures

**Risk Terms:**
- **Drawdown:** Peak-to-trough decline
- **Exposure:** Total capital at risk
- **Win Rate:** Percentage of profitable trades
- **Sharpe Ratio:** Risk-adjusted return

### API Reference

**Health Check:**
```bash
GET /health
Response: {"status": "healthy", "uptime": 3600}
```

**AI Signals:**
```bash
GET /api/signals/live
Response: [{"symbol": "BTCUSDT", "action": "SELL", "confidence": 0.83}]
```

**Active Positions:**
```bash
GET /api/positions
Response: [{"symbol": "BTCUSDT", "side": "LONG", "pnl": -3.14}]
```

**Trade History:**
```bash
GET /api/trades?limit=50
Response: [{"symbol": "BTCUSDT", "action": "BUY", "price": 87500}]
```

### Code Examples

**Manual Trade Execution:**
```python
from backend.services.execution import ExecutionAdapter
from backend.config.execution import ExecutionConfig

config = ExecutionConfig.from_env()
adapter = ExecutionAdapter(config)

# Place market order
result = adapter.place_order(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.01,
    order_type="MARKET"
)
```

**Get AI Prediction:**
```python
from backend.services.ai_trading_engine import AITradingEngine

engine = AITradingEngine()
signal = engine.generate_signal("BTCUSDT")

print(f"Action: {signal.action}")
print(f"Confidence: {signal.confidence}")
print(f"TP: {signal.tp_pct}%")
print(f"SL: {signal.sl_pct}%")
```

**Check System Health:**
```python
from backend.services.self_healing import SelfHealingSystem

healer = SelfHealingSystem()
health = healer.comprehensive_health_check()

print(f"Status: {health['status']}")
print(f"Healthy: {health['healthy_count']}")
print(f"Issues: {health['issues']}")
```

---

## 📞 SUPPORT

### Kontakt

- **GitHub Issues:** https://github.com/quantum_trader/issues
- **Discord:** quantum-trader.dev/discord
- **Dokumentasjon:** quantum-trader.dev/docs

### Bidrag

Vi tar gjerne imot bidrag! Se [CONTRIBUTING.md](CONTRIBUTING.md)

### Lisens

MIT License - Se [LICENSE](LICENSE)

---

**Sist oppdatert:** 26. november 2025  
**Versjon:** 3.0  
**Forfatter:** Quantum Trader Team

---

## 🔐 SIKKERHET

**ADVARSEL:** Dette systemet handler ekte penger (eller testnet). Alltid:
- Test grundig på testnet først
- Bruk SAFE profile i produksjon
- Overvåk kontinuerlig
- Sett strenge risk limits
- Ha backup plan
- Aldri del API keys

**Best Practices:**
1. Start med lav leverage (5-10x)
2. Bruk SAFE orchestrator profile
3. Sett konservative daily loss limits
4. Overvåk første 24-48 timer kontinuerlig
5. Gjør regelmessig backtesting
6. Hold modeller oppdatert (continuous learning)

---

**END OF DOCUMENTATION**
