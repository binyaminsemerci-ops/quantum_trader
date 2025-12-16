# Risk & Trade Management System

**ATR-Based, Profit-Optimized Risk Management Layer**

## 🎯 Overview

Complete risk management framework designed to achieve stable equity curves with controlled drawdowns and high risk:reward ratios. Built on top of the 4-model AI ensemble, this system filters trades, sizes positions, manages exits, and enforces portfolio-level risk controls.

## 📊 Architecture

### 5 Core Components

```
┌─────────────────────────────────────────────────────────┐
│          TradeLifecycleManager (Orchestrator)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────┐  ┌──────────────────────────┐  │
│  │ TradeOpportunity  │  │ GlobalRiskController     │  │
│  │ Filter            │  │ - Daily/Weekly DD limits  │  │
│  │ - Consensus check │  │ - Max concurrent trades   │  │
│  │ - Confidence ≥70% │  │ - Portfolio exposure      │  │
│  │ - Trend alignment │  │ - Losing streak protection│  │
│  │ - Volatility gates│  │ - Circuit breaker         │  │
│  └───────────────────┘  └──────────────────────────┘  │
│                                                         │
│  ┌───────────────────┐  ┌──────────────────────────┐  │
│  │ RiskManager       │  │ ExitPolicyEngine         │  │
│  │ - ATR position    │  │ - ATR-based SL/TP        │  │
│  │   sizing          │  │ - Breakeven at +1R       │  │
│  │ - Risk 0.5-1.5%   │  │ - Partial TP at +2R      │  │
│  │ - Signal quality  │  │ - Trailing stops         │  │
│  │   adjustment      │  │ - Time-based exits       │  │
│  └───────────────────┘  └──────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

All settings are environment variables with sensible defaults.

### Trade Filter Settings

```bash
# Consensus requirements
RM_MIN_CONSENSUS_TYPES="UNANIMOUS,STRONG"  # Accept UNANIMOUS or STRONG consensus
RM_MIN_CONFIDENCE=0.70                      # 70% confidence minimum

# Trend alignment
RM_REQUIRE_TREND=true                       # Must align with 200 EMA
RM_EMA_PERIOD=200

# Volatility gates
RM_VOLATILITY_GATE=true                     # Enable volatility filtering
RM_MAX_ATR_RATIO=0.05                       # Max 5% ATR/price ratio
RM_HIGH_VOL_CONFIDENCE=0.80                 # Require 80% confidence if volatile

# Volume/spread filters
RM_MIN_VOLUME_24H=1000000                   # Min $1M 24h volume
RM_MAX_SPREAD_BPS=50                        # Max 50 basis points spread
```

### Position Sizing Settings

```bash
# Risk per trade
RM_RISK_PER_TRADE_PCT=0.01                  # 1% risk per trade
RM_MIN_RISK_PCT=0.005                       # Min 0.5%
RM_MAX_RISK_PCT=0.015                       # Max 1.5%

# Signal quality adjustment
RM_SIGNAL_QUALITY_ADJ=true                  # Adjust size based on confidence
RM_HIGH_CONF_MULT=1.5                       # 1.5x size for confidence ≥85%
RM_LOW_CONF_MULT=0.5                        # 0.5x size for confidence <60%

# ATR settings
RM_ATR_PERIOD=14                            # 14-period ATR
RM_ATR_MULT_SL=1.5                          # k1 = 1.5 (stop loss distance)

# Position constraints
RM_MAX_LEVERAGE=30.0                        # Max 30x leverage
RM_MIN_POSITION_USD=5.0                     # Min $5 position
RM_MAX_POSITION_USD=500.0                   # Max $500 position
```

### Exit Policy Settings

```bash
# ATR multipliers
RM_EXIT_ATR_PERIOD=14
RM_SL_MULTIPLIER=1.5                        # SL at 1.5x ATR
RM_TP_MULTIPLIER=3.75                       # TP at 3.75x ATR (2.5:1 R:R)
RM_TARGET_RR=2.5

# Partial exits
RM_ENABLE_PARTIAL_TP=true
RM_PARTIAL_TP_AT_R=2.0                      # Partial TP at +2R
RM_PARTIAL_TP_PERCENT=0.5                   # Close 50% of position

# Breakeven
RM_ENABLE_BREAKEVEN=true
RM_BREAKEVEN_AT_R=1.0                       # Move SL to BE at +1R
RM_BREAKEVEN_OFFSET_PCT=0.001               # 0.1% above entry

# Trailing stop
RM_ENABLE_TRAILING=true
RM_TRAILING_START_R=2.0                     # Start trailing at +2R
RM_TRAILING_DISTANCE_ATR=1.0                # Trail at 1 ATR from peak

# Time-based exit
RM_ENABLE_TIME_EXIT=true
RM_MAX_HOURS_NO_PROGRESS=24                 # Exit if no progress after 24h
```

### Global Risk Settings

```bash
# Drawdown limits
RM_MAX_DAILY_DD_PCT=0.03                    # 3% max daily drawdown
RM_MAX_WEEKLY_DD_PCT=0.10                   # 10% max weekly drawdown

# Position limits
RM_MAX_CONCURRENT_TRADES=4                  # Max 4 open positions
RM_MAX_EXPOSURE_PCT=0.80                    # Max 80% of equity exposed
RM_MAX_CORRELATION=0.70                     # Max 70% correlation between positions

# Losing streak protection
RM_ENABLE_STREAK_PROTECTION=true
RM_LOSING_STREAK_THRESHOLD=3                # Reduce risk after 3 losses
RM_STREAK_RISK_REDUCTION=0.5                # Cut risk by 50%

# Recovery mode
RM_ENABLE_RECOVERY_MODE=true
RM_RECOVERY_THRESHOLD_PCT=0.02              # Enter recovery at 2% DD
RM_RECOVERY_RISK_MULT=0.5                   # Half size in recovery

# Circuit breaker
RM_ENABLE_CIRCUIT_BREAKER=true
RM_CIRCUIT_BREAKER_LOSS_PCT=0.05            # Trigger at 5% loss
RM_CIRCUIT_BREAKER_COOLDOWN_HOURS=4         # Pause for 4 hours
```

### Logging Settings

```bash
# What to log
RM_LOG_TRADE_DECISIONS=true                 # Log every trade decision
RM_LOG_FILTER_REJECTIONS=true               # Log why trades rejected
RM_LOG_POSITION_SIZING=true                 # Log size calculations
RM_LOG_EXIT_DECISIONS=true                  # Log exit reasons

# Detail level
RM_LOG_MARKET_DATA=true                     # Include price/ATR/EMA
RM_LOG_SIGNAL_BREAKDOWN=true                # Include model votes
RM_LOG_RISK_METRICS=true                    # Include R-multiple, MFE/MAE

# Storage
RM_LOG_TO_DATABASE=true                     # Store in PostgreSQL
RM_LOG_TO_FILE=true                         # Write to trade_decisions.log
```

## 🚀 Usage

### Basic Usage

```python
from backend.config.risk_management import load_risk_management_config
from backend.services.risk_management import TradeLifecycleManager, SignalQuality, MarketConditions

# Load configuration
config = load_risk_management_config()

# Initialize manager
manager = TradeLifecycleManager(config)

# Evaluate a new signal
signal_quality = SignalQuality(
    consensus_type="STRONG",
    confidence=0.75,
    model_votes={"XGBoost": "LONG", "LightGBM": "LONG", "N-HiTS": "LONG", "PatchTST": "HOLD"},
    signal_strength=0.82
)

market_conditions = MarketConditions(
    price=50000.0,
    atr=1500.0,
    ema_200=48000.0,
    volume_24h=5_000_000_000,
    spread_bps=10,
    timestamp=datetime.now(timezone.utc)
)

decision = manager.evaluate_new_signal(
    symbol="BTCUSDT",
    action="LONG",
    signal_quality=signal_quality,
    market_conditions=market_conditions,
    current_equity=10000.0
)

if decision.approved:
    print(f"✅ Trade approved: {decision.quantity} @ ${decision.entry_price}")
    print(f"   SL: ${decision.stop_loss}, TP: ${decision.take_profit}")
else:
    print(f"❌ Trade rejected: {decision.rejection_reason}")
```

### Complete Trade Lifecycle

```python
# 1. Evaluate signal
decision = manager.evaluate_new_signal(...)

if not decision.approved:
    return

# 2. Execute order (your execution adapter)
order = execute_order(symbol, action, decision.quantity)

# 3. Record trade as opened
trade = manager.open_trade(
    trade_id=order.order_id,
    decision=decision,
    signal_quality=signal_quality,
    market_conditions=market_conditions,
    actual_entry_price=order.fill_price
)

# 4. Monitor position (in your position monitor loop)
exit_decision = manager.update_trade(
    trade_id=trade.trade_id,
    current_price=current_market_price
)

if exit_decision and exit_decision.should_exit:
    # Execute exit order
    close_order = execute_close(
        symbol=trade.symbol,
        quantity=exit_decision.exit_quantity or trade.current_quantity
    )
    
    # Record closure
    manager.close_trade(
        trade_id=trade.trade_id,
        exit_decision=exit_decision,
        actual_exit_price=close_order.fill_price
    )
```

## 📈 Trade State Machine

```
NEW (Signal received)
  │
  ├─[Filters]─> REJECTED (Filter rejection)
  │
  └─[Approved]─> APPROVED (Ready to execute)
                   │
                   └─[Execution]─> OPEN (Position opened)
                                     │
                                     ├─[+2R]─> PARTIAL_TP (50% closed)
                                     │           │
                                     │           └─> TRAILING (Trail from peak)
                                     │
                                     ├─[+1R]─> BREAKEVEN (SL to entry)
                                     │
                                     ├─[TP hit]─> CLOSED_TP (Full exit at TP)
                                     │
                                     ├─[SL hit]─> CLOSED_SL (Full exit at SL)
                                     │
                                     └─[Time]─> CLOSED_TIME (Time-based exit)
```

## 🔍 Key Features

### 1. Trade Opportunity Filter

**Quality-based filtering:**
- ✅ Consensus: Only UNANIMOUS or STRONG (3/4+ models agree)
- ✅ Confidence: Minimum 70% (80% if high volatility)
- ✅ Trend: Must align with 200 EMA (LONG above, SHORT below)
- ✅ Volatility: Reject if ATR ratio >5% (too chaotic)
- ✅ Volume: Minimum $1M 24h volume
- ✅ Spread: Maximum 50 basis points

### 2. ATR-Based Position Sizing

**Formula:**
```
sl_distance = ATR × 1.5
risk_amount = equity × risk_pct
position_size = risk_amount / sl_distance
```

**Adjustments:**
- High confidence (≥85%): 1.5x size
- Low confidence (<60%): 0.5x size
- Recovery mode: 0.5x size
- Losing streak: 0.5x size

### 3. Exit Policy Engine

**ATR-based levels:**
- **Stop Loss**: Entry ± (1.5 × ATR)
- **Take Profit**: Entry ± (3.75 × ATR) = 2.5:1 R:R

**Dynamic exits:**
- **Breakeven** at +1R: Move SL to entry + 0.1%
- **Partial TP** at +2R: Close 50% of position
- **Trailing** after +2R: Trail at 1 ATR from peak
- **Time exit** at 24h: If no progress, close

### 4. Global Risk Controller

**Portfolio-level protection:**
- Max 3% daily drawdown
- Max 10% weekly drawdown
- Max 4 concurrent trades
- Max 80% portfolio exposure
- Losing streak protection (3 losses → 50% risk)
- Recovery mode (2% DD → 50% risk)
- Circuit breaker (5% loss → 4h pause)

### 5. Comprehensive Logging

**Auto-training data collection:**
- Trade decisions with full context
- Filter rejections with reasons
- Position sizing calculations
- Exit decisions with MFE/MAE
- R-multiples and performance metrics

## 📊 Performance Expectations

**Realistic targets:**
- Win rate: 40-50% (quality over quantity)
- Average R: 2.0-2.5 (target 2.5:1 R:R)
- Max daily DD: <3%
- Max weekly DD: <10%
- Profit factor: >1.5
- Sharpe ratio: >1.0

**Goal:**
Stable equity curve with controlled drawdowns and high risk:reward ratios.

## 🛠 Integration

To integrate with `event_driven_executor.py`:

```python
from backend.services.risk_management import TradeLifecycleManager
from backend.config.risk_management import load_risk_management_config

# In EventDrivenExecutor.__init__
self.rm_config = load_risk_management_config()
self.trade_manager = TradeLifecycleManager(self.rm_config)

# In signal processing loop
for signal in signals:
    # Build SignalQuality and MarketConditions objects
    signal_quality = SignalQuality(...)
    market_conditions = MarketConditions(...)
    
    # Evaluate through risk management layer
    decision = self.trade_manager.evaluate_new_signal(
        symbol=signal['symbol'],
        action=signal['action'],
        signal_quality=signal_quality,
        market_conditions=market_conditions,
        current_equity=self.get_equity()
    )
    
    if decision.approved:
        # Execute trade with approved size/levels
        self._execute_trade(decision)
```

## 📝 Notes

- All percentages are expressed as decimals (0.01 = 1%)
- ATR multipliers are configurable but defaults are battle-tested
- Position sizing uses dynamic ATR for adaptive stops
- Global risk checks run before every trade
- Circuit breaker is fail-safe protection
- Recovery mode prevents digging deeper holes
- Logging is designed for ML model retraining

## 🎯 Philosophy

**"Risk management is not about avoiding losses - it's about controlling them."**

This system is designed to:
1. Filter for high-quality setups (quality > quantity)
2. Size positions based on volatility (smaller in chaos, larger in calm)
3. Let winners run with trailing stops
4. Cut losses quickly at predefined levels
5. Protect capital at portfolio level (daily DD, max exposure, circuit breaker)

The goal is **not** 99% win rate (impossible). The goal is:
- **Stable equity curve** (no wild swings)
- **Controlled drawdowns** (max 3% daily, 10% weekly)
- **High R:R ratio** (2.5:1 target, let winners run)
- **Consistent profitability** (profit factor >1.5)

## 📚 References

- ATR: J. Welles Wilder, "New Concepts in Technical Trading Systems"
- R-multiples: Van Tharp, "Trade Your Way to Financial Freedom"
- Position sizing: Ralph Vince, "The Mathematics of Money Management"
- Risk management: Jack Schwager, "Market Wizards" series
