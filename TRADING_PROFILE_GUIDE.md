# 🎯 Trading Profile System - Complete Guide

**Quantum Trader AI Hedge Fund OS**  
*Liquidity Filtering • Position Sizing • Dynamic TP/SL • Funding Protection*

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Integration Points](#integration-points)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

### The Problem

The trading system was experiencing systematic losses due to:

- **Poor Symbol Selection**: Trading illiquid altcoins (TAO, PUNDIX, AAVE, ZEC)
- **Spread/Fee Erosion**: High spreads eating profits on small positions
- **Funding Rate Penalties**: Entering positions just before unfavorable funding
- **Fixed TP/SL**: Not adapting to market volatility
- **Insufficient Position Sizes**: Mathematically impossible to overcome costs

### The Solution

**Trading Profile System** - A comprehensive layer that ensures AI trades only on:

✅ **High liquidity symbols** (volume, spread, depth filtering)  
✅ **Optimal position sizes** (AI-driven risk factors with safety bounds)  
✅ **Dynamic TP/SL levels** (ATR-based, multi-targets, trailing stops)  
✅ **Funding-safe timing** (avoid funding windows + rate filtering)  
✅ **Controlled leverage** (30x on Binance, 8-15x effective in code)

### Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| **Universe Size** | ~200 symbols | Top 20 liquid |
| **Min Volume** | Any | $5M+ / 24h |
| **Max Spread** | Any | 0.03% (3 bps) |
| **TP/SL Type** | Fixed | Dynamic ATR |
| **Risk/Reward** | ~1:1 | 1:1.5 → 1:2.5+ |
| **Funding Protection** | None | ±40/20min window |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADING PROFILE SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  LIQUIDITY &   │  │  POSITION      │  │  DYNAMIC TP/SL   │  │
│  │  UNIVERSE      │  │  SIZING        │  │  ENGINE          │  │
│  │  FILTER        │  │                │  │                  │  │
│  ├────────────────┤  ├────────────────┤  ├──────────────────┤  │
│  │• 24h Volume    │  │• AI Risk Factor│  │• ATR-based       │  │
│  │• Bid/Ask Spread│  │• Min/Max       │  │• Multi-targets   │  │
│  │• Orderbook     │  │  Margin        │  │• Break-even      │  │
│  │  Depth         │  │• Leverage by   │  │• Trailing Stop   │  │
│  │• Universe Tier │  │  Tier          │  │• Partial Closes  │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
│                                                                   │
│  ┌────────────────┐  ┌────────────────────────────────────────┐ │
│  │  FUNDING       │  │  BINANCE MARKET DATA                   │ │
│  │  PROTECTION    │  │                                        │ │
│  ├────────────────┤  ├────────────────────────────────────────┤ │
│  │• Time Windows  │  │• Futures API Integration              │ │
│  │• Rate Filter   │  │• 24h Tickers                          │ │
│  │• Pre: -40min   │  │• Orderbook Fetcher                    │ │
│  │• Post: +20min  │  │• Funding Rate & Timing                │ │
│  └────────────────┘  │• ATR Calculator (14-period, 15m)      │ │
│                      └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │
                               │ Integration Layer
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐     ┌──────────────┐      ┌──────────────┐
│ ORCHESTRATOR │     │  EXECUTION   │      │  RISK OS     │
│   POLICY     │     │    ENGINE    │      │              │
└──────────────┘     └──────────────┘      └──────────────┘
```

---

## 🧩 Components

### 1. **trading_profile.py** (Core Module)

**Purpose**: All filtering, sizing, TP/SL, funding logic

**Key Classes**:

```python
@dataclass
class SymbolMetrics:
    """Complete market data for a symbol"""
    symbol: str
    quote_volume_24h: float        # USDT volume
    bid: float                     # Best bid
    ask: float                     # Best ask
    depth_notional_5bps: float     # Depth within ±0.5%
    funding_rate: float            # Current funding
    next_funding_time: datetime    # Next funding event
    mark_price: float
    index_price: float
    open_interest: float
    universe_tier: UniverseTier    # MAIN, L1, L2, etc.

@dataclass
class LiquidityConfig:
    min_quote_volume_24h: float = 5_000_000  # $5M
    max_spread_bps: float = 3.0              # 0.03%
    min_depth_notional: float = 200_000      # $200k
    allowed_tiers: List[UniverseTier]        # MAIN, L1, L2

@dataclass
class RiskConfig:
    base_risk_frac: float = 0.01             # 1% per trade
    max_risk_frac: float = 0.03              # 3% max
    min_margin: float = 10.0                 # $10 min
    max_margin: float = 1000.0               # $1k max
    effective_leverage_main: float = 15.0    # BTC/ETH: 15x
    effective_leverage_l1: float = 12.0      # L1s: 12x
    effective_leverage_l2: float = 10.0      # L2s: 10x

@dataclass
class TpslConfig:
    atr_mult_sl: float = 1.0                 # SL at 1R
    atr_mult_tp1: float = 1.5                # TP1 at 1.5R
    atr_mult_tp2: float = 2.5                # TP2 at 2.5R (trailing)
    trail_dist_mult: float = 0.8             # Trail 0.8R below price
    partial_close_tp1: float = 0.5           # Close 50% at TP1
    partial_close_tp2: float = 0.3           # Close 30% at TP2

@dataclass
class FundingConfig:
    pre_window_minutes: int = 40             # Block -40min
    post_window_minutes: int = 20            # Block +20min
    min_long_funding: float = -0.0003        # Don't LONG if < -0.03%
    max_short_funding: float = 0.0003        # Don't SHORT if > +0.03%

@dataclass
class DynamicTpslLevels:
    sl_init: float                           # Initial SL
    tp1: float                               # First target
    tp2: float                               # Second target / trailing start
    be_trigger: float                        # Break-even trigger
    be_price: float                          # Break-even price
    trail_activation: float                  # Trailing activation level
    trail_distance: float                    # Trailing distance
```

**Key Functions**:

```python
# Liquidity
compute_spread(symbol: SymbolMetrics) -> float
compute_liquidity_score(symbol: SymbolMetrics, cfg: LiquidityConfig) -> float
is_symbol_tradeable(symbol: SymbolMetrics, cfg: LiquidityConfig) -> (bool, reason)
filter_and_rank_universe(symbols: List[SymbolMetrics], cfg) -> List[SymbolMetrics]

# Position Sizing
compute_position_margin(equity: float, ai_risk_factor: float, cfg: RiskConfig) -> float
compute_effective_leverage(symbol: SymbolMetrics, cfg: RiskConfig) -> float
compute_position_size(margin: float, leverage: float, entry_price: float) -> float

# TP/SL
compute_dynamic_tpsl_long(entry: float, atr: float, cfg: TpslConfig) -> DynamicTpslLevels
compute_dynamic_tpsl_short(entry: float, atr: float, cfg: TpslConfig) -> DynamicTpslLevels

# Funding
is_funding_window_blocked(now: datetime, symbol: SymbolMetrics, cfg) -> (bool, reason)
is_funding_rate_unfavourable(side: str, symbol: SymbolMetrics, cfg) -> (bool, reason)
check_funding_protection(side: str, symbol: SymbolMetrics, cfg) -> (bool, reason)

# Integrated Validation
validate_trade(symbol, side, liquidity_cfg, funding_cfg) -> (bool, reason)

# Universe Classification
classify_symbol_tier(symbol: str) -> UniverseTier  # BTC/ETH → MAIN, SOL/BNB → L1, etc.
```

---

### 2. **binance_market_data.py** (Data Fetcher)

**Purpose**: Fetch real-time market data from Binance Futures API

**Key Class**:

```python
class BinanceMarketDataFetcher:
    def __init__(self, api_key, api_secret, testnet=False)
    
    # Ticker data
    def get_24h_ticker(symbol: str) -> Dict
    
    # Orderbook
    def get_orderbook_depth(symbol: str, depth_pct=0.005) -> (bid, ask, depth_notional)
    
    # Funding
    def get_funding_info(symbol: str) -> (funding_rate, next_funding_time)
    
    # Prices
    def get_mark_price(symbol: str) -> (mark_price, index_price)
    def get_open_interest(symbol: str) -> float
    
    # Complete metrics
    def fetch_symbol_metrics(symbol: str) -> SymbolMetrics
    
    # Bulk fetch
    def fetch_all_futures_symbols() -> List[str]
    def fetch_universe_metrics(symbols: List[str], max_symbols=100) -> List[SymbolMetrics]
    
    # Cache management
    def clear_cache()

# Standalone functions
calculate_atr(symbol: str, period=14, timeframe='15m') -> float
calculate_atr_percentage(symbol: str, period=14, timeframe='15m') -> float
```

**Usage Example**:

```python
# Initialize fetcher
fetcher = create_market_data_fetcher()  # Uses env vars

# Fetch single symbol
metrics = fetcher.fetch_symbol_metrics('BTCUSDT')
print(f"Volume: ${metrics.quote_volume_24h/1e6:.1f}M")
print(f"Spread: {compute_spread(metrics)*10000:.2f} bps")
print(f"Funding: {metrics.funding_rate*100:.3f}%")

# Fetch entire universe
all_metrics = fetcher.fetch_universe_metrics()

# Calculate ATR
atr = calculate_atr('BTCUSDT', period=14, timeframe='15m')
print(f"BTC ATR: ${atr:.2f}")
```

---

### 3. **trading_profile.py** (Config)

**Purpose**: Configuration management + env var loading

**Key Class**:

```python
@dataclass
class TradingProfileConfig:
    liquidity: LiquidityConfig
    risk: RiskConfig
    tpsl: TpslConfig
    funding: FundingConfig
    enabled: bool = True
    auto_universe_update_seconds: int = 300  # 5min
    
    @classmethod
    def from_env() -> TradingProfileConfig
    
    @classmethod
    def from_json_file(path: Path) -> TradingProfileConfig
    
    def to_dict() -> dict
    def save_to_json(path: Path)

# Global access
load_trading_profile_config(from_env=True) -> TradingProfileConfig
get_trading_profile_config() -> TradingProfileConfig
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Global Control
TP_ENABLED=true
TP_AUTO_UPDATE_SECONDS=300                    # Update universe every 5min

# Liquidity & Universe
TP_MIN_VOLUME_24H=5000000                     # $5M minimum
TP_MAX_SPREAD_BPS=3.0                         # 0.03% max spread
TP_MIN_DEPTH=200000                           # $200k depth
TP_MAX_UNIVERSE_SIZE=20                       # Top 20 symbols
TP_ALLOWED_TIERS=MAIN,L1,L2                   # Universe tiers

# Risk & Position Sizing
TP_BASE_RISK_FRAC=0.01                        # 1% base risk
TP_MAX_RISK_FRAC=0.03                         # 3% max risk
TP_MIN_MARGIN=10.0                            # $10 min
TP_MAX_MARGIN=1000.0                          # $1000 max
TP_MAX_TOTAL_RISK=0.15                        # 15% total exposure
TP_MAX_POSITIONS=8                            # Max 8 positions
TP_MIN_AI_RISK_FACTOR=0.5                     # Conservative AI
TP_MAX_AI_RISK_FACTOR=1.5                     # Aggressive AI

# Leverage (Effective, not Binance setting)
TP_DEFAULT_LEVERAGE=30                        # Binance account setting
TP_LEVERAGE_MAIN=15.0                         # BTC/ETH: 15x effective
TP_LEVERAGE_L1=12.0                           # L1s: 12x effective
TP_LEVERAGE_L2=10.0                           # L2s: 10x effective
TP_LEVERAGE_MIN=8.0                           # Minimum: 8x

# Dynamic TP/SL
TP_ATR_MULT_SL=1.0                            # SL at 1R
TP_ATR_MULT_TP1=1.5                           # TP1 at 1.5R
TP_ATR_MULT_TP2=2.5                           # TP2 at 2.5R (trailing)
TP_ATR_MULT_BE=1.0                            # Break-even at 1R
TP_TRAIL_DIST_MULT=0.8                        # Trail 0.8R below price
TP_PARTIAL_CLOSE_TP1=0.5                      # Close 50% at TP1
TP_PARTIAL_CLOSE_TP2=0.3                      # Close 30% at TP2
TP_ATR_PERIOD=14                              # ATR period
TP_ATR_TIMEFRAME=15m                          # ATR timeframe

# Funding Protection
TP_FUNDING_PRE_WINDOW=40                      # Block -40min before
TP_FUNDING_POST_WINDOW=20                     # Block +20min after
TP_MIN_LONG_FUNDING=-0.0003                   # Don't LONG if < -0.03%
TP_MAX_SHORT_FUNDING=0.0003                   # Don't SHORT if > +0.03%
```

### Universe Tiers

```python
class UniverseTier(Enum):
    MAIN = "main"              # BTC, ETH
    L1 = "l1"                  # SOL, BNB, ADA, AVAX, DOT, ATOM, NEAR, FTM, etc.
    L2 = "l2"                  # ARB, OP, MATIC, UNI, AAVE, LINK, MKR, CRV, etc.
    DEFI = "defi"              # COMP, SUSHI, YFI, BAL, PERP, GMX, GNS, etc.
    INFRASTRUCTURE = "infrastructure"  # GRT, FIL, AR, BAND, API3, OCEAN
    MEME = "meme"              # DOGE, SHIB, PEPE, WIF (usually excluded)
    EXCLUDED = "excluded"      # Known bad symbols (TAO, PUNDIX, ZEC, etc.)
```

**Predefined Classifications**:
- **MAIN**: BTC, ETH (highest liquidity, 15x leverage)
- **L1**: SOL, BNB, ADA, AVAX, DOT, ATOM, NEAR, FTM, ALGO, XRP, TRX, TON, APT, SUI, SEI, INJ
- **L2**: ARB, OP, MATIC, IMX, METIS, MANTA, UNI, AAVE, LINK, MKR, CRV, LDO, SNX
- **EXCLUDED**: TAO, PUNDIX, ZEC, JUP, DYM, IO, NOT, LISTA, ZRO, OMNI, SAGA, BB, REZ, AEVO, PORTAL, PIXEL, STRK

---

## 🔗 Integration Points

### 1. **Orchestrator Policy Integration**

**File**: `backend/services/orchestrator_policy.py`

**What to Add**:

```python
from backend.services.ai.trading_profile import validate_trade
from backend.services.binance_market_data import create_market_data_fetcher, calculate_atr
from backend.config.trading_profile import get_trading_profile_config

class OrchestratorPolicy:
    def __init__(self):
        self.tp_config = get_trading_profile_config()
        self.market_data = create_market_data_fetcher()
    
    def can_trade_symbol(self, symbol: str, side: str) -> Tuple[bool, str]:
        """Check if symbol passes all trading profile filters."""
        if not self.tp_config.enabled:
            return True, "Trading profile disabled"
        
        # Fetch market data
        metrics = self.market_data.fetch_symbol_metrics(symbol)
        if not metrics:
            return False, "Failed to fetch market data"
        
        # Validate trade
        valid, reason = validate_trade(
            metrics,
            side,
            self.tp_config.liquidity,
            self.tp_config.funding
        )
        
        if not valid:
            log.info(f"Trade rejected: {symbol} {side} - {reason}")
            return False, reason
        
        return True, "Passed all filters"
```

**Integration Flow**:

```
AI Signal → Orchestrator → Trading Profile Validation → Execution
                ↓
            ✅ PASS: Continue to execution
            ❌ FAIL: Log reason, skip trade
```

---

### 2. **Execution Layer Integration**

**File**: `backend/services/execution.py`

**What to Add**:

```python
from backend.services.ai.trading_profile import compute_dynamic_tpsl_long, compute_dynamic_tpsl_short
from backend.services.binance_market_data import calculate_atr
from backend.config.trading_profile import get_trading_profile_config

class BinanceFuturesExecutionAdapter:
    def __init__(self):
        self.tp_config = get_trading_profile_config()
    
    async def submit_order_with_dynamic_tpsl(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        entry_price: float
    ):
        """Submit order with ATR-based TP/SL levels."""
        
        # Calculate ATR
        atr = calculate_atr(
            symbol,
            period=self.tp_config.tpsl.atr_period,
            timeframe=self.tp_config.tpsl.atr_timeframe
        )
        
        if not atr:
            log.error(f"Failed to calculate ATR for {symbol}")
            return None
        
        # Calculate TP/SL levels
        if side == 'BUY':
            levels = compute_dynamic_tpsl_long(
                entry_price,
                atr,
                self.tp_config.tpsl
            )
        else:
            levels = compute_dynamic_tpsl_short(
                entry_price,
                atr,
                self.tp_config.tpsl
            )
        
        log.info(f"Dynamic TP/SL for {symbol}: {levels.to_dict()}")
        
        # Submit entry order
        entry_order = await self.submit_market_order(symbol, side, quantity)
        
        # Submit TP1 order (partial close)
        tp1_qty = quantity * levels.partial_close_frac_tp1
        await self.submit_take_profit(
            symbol,
            opposite_side(side),
            tp1_qty,
            levels.tp1
        )
        
        # Submit TP2 order (partial close, activate trailing)
        tp2_qty = quantity * levels.partial_close_frac_tp2
        await self.submit_take_profit(
            symbol,
            opposite_side(side),
            tp2_qty,
            levels.tp2
        )
        
        # Submit stop loss
        await self.submit_stop_loss(
            symbol,
            opposite_side(side),
            quantity,
            levels.sl_init
        )
        
        return {
            'entry_order': entry_order,
            'tpsl_levels': levels.to_dict()
        }
```

**Position Management Flow**:

```
1. Entry → Place market order
2. TP1 → Partial close (50%) at 1.5R
3. Break-even → Move SL to entry when price hits 1R
4. TP2 → Partial close (30%) at 2.5R, activate trailing
5. Trailing → Follow price with 0.8R distance (remaining 20%)
```

---

### 3. **Position Sizing Integration**

**File**: `backend/services/risk_manager.py`

**What to Add**:

```python
from backend.services.ai.trading_profile import compute_position_margin, compute_effective_leverage, compute_position_size
from backend.config.trading_profile import get_trading_profile_config
from backend.services.binance_market_data import create_market_data_fetcher

class RiskManager:
    def __init__(self):
        self.tp_config = get_trading_profile_config()
        self.market_data = create_market_data_fetcher()
    
    def calculate_position_size(
        self,
        symbol: str,
        equity: float,
        ai_conviction: float,  # 0.0 to 1.0
        entry_price: float
    ) -> Dict:
        """Calculate optimal position size using trading profile."""
        
        # Convert AI conviction to risk factor (0.5 to 1.5)
        ai_risk_factor = 0.5 + (ai_conviction * 1.0)
        
        # Calculate margin
        margin = compute_position_margin(
            equity,
            ai_risk_factor,
            self.tp_config.risk
        )
        
        # Get symbol metrics for leverage
        metrics = self.market_data.fetch_symbol_metrics(symbol)
        if not metrics:
            log.error(f"Failed to fetch metrics for {symbol}")
            return None
        
        # Calculate effective leverage
        effective_leverage = compute_effective_leverage(
            metrics,
            self.tp_config.risk
        )
        
        # Calculate position size
        quantity = compute_position_size(
            margin,
            effective_leverage,
            entry_price
        )
        
        notional = quantity * entry_price
        
        return {
            'quantity': quantity,
            'margin': margin,
            'notional': notional,
            'effective_leverage': effective_leverage,
            'ai_risk_factor': ai_risk_factor,
            'risk_pct': (margin / equity) * 100
        }
```

---

## 📚 Usage Examples

### Example 1: Validate Symbol Before Trading

```python
from backend.services.ai.trading_profile import validate_trade
from backend.services.binance_market_data import create_market_data_fetcher
from backend.config.trading_profile import get_trading_profile_config

# Setup
config = get_trading_profile_config()
fetcher = create_market_data_fetcher()

# Check BTCUSDT LONG
metrics = fetcher.fetch_symbol_metrics('BTCUSDT')
valid, reason = validate_trade(
    metrics,
    'LONG',
    config.liquidity,
    config.funding
)

if valid:
    print("✅ BTC LONG allowed")
else:
    print(f"❌ BTC LONG rejected: {reason}")
```

### Example 2: Calculate Dynamic TP/SL

```python
from backend.services.ai.trading_profile import compute_dynamic_tpsl_long
from backend.services.binance_market_data import calculate_atr
from backend.config.trading_profile import get_trading_profile_config

config = get_trading_profile_config()

# BTC trade
entry_price = 43500.00
atr = calculate_atr('BTCUSDT', period=14, timeframe='15m')  # e.g., $650

levels = compute_dynamic_tpsl_long(entry_price, atr, config.tpsl)

print(f"Entry: ${entry_price:,.2f}")
print(f"ATR: ${atr:,.2f}")
print(f"SL: ${levels.sl_init:,.2f} (risk: ${entry_price - levels.sl_init:,.2f})")
print(f"TP1: ${levels.tp1:,.2f} (reward: ${levels.tp1 - entry_price:,.2f}) [Close 50%]")
print(f"TP2: ${levels.tp2:,.2f} (reward: ${levels.tp2 - entry_price:,.2f}) [Close 30%, start trail]")
print(f"Trail: ${levels.trail_distance:,.2f} below price")
```

**Output**:
```
Entry: $43,500.00
ATR: $650.00
SL: $42,850.00 (risk: $650.00)        # 1R
TP1: $44,475.00 (reward: $975.00) [Close 50%]    # 1.5R
TP2: $45,125.00 (reward: $1,625.00) [Close 30%, start trail]  # 2.5R
Trail: $520.00 below price            # 0.8R
```

### Example 3: Filter Universe

```python
from backend.services.ai.trading_profile import filter_and_rank_universe
from backend.services.binance_market_data import create_market_data_fetcher
from backend.config.trading_profile import get_trading_profile_config

config = get_trading_profile_config()
fetcher = create_market_data_fetcher()

# Fetch all symbols
all_metrics = fetcher.fetch_universe_metrics(max_symbols=200)

# Filter & rank
tradeable = filter_and_rank_universe(all_metrics, config.liquidity)

print(f"Top {len(tradeable)} tradeable symbols:")
for i, symbol in enumerate(tradeable, 1):
    print(f"{i}. {symbol.symbol} - ${symbol.quote_volume_24h/1e6:.1f}M volume")
```

**Output**:
```
Top 20 tradeable symbols:
1. BTCUSDT - $1,234.5M volume
2. ETHUSDT - $987.3M volume
3. SOLUSDT - $456.2M volume
4. BNBUSDT - $321.8M volume
...
```

### Example 4: Position Sizing

```python
from backend.services.ai.trading_profile import compute_position_margin, compute_effective_leverage, compute_position_size
from backend.services.binance_market_data import create_market_data_fetcher
from backend.config.trading_profile import get_trading_profile_config

config = get_trading_profile_config()
fetcher = create_market_data_fetcher()

# Account state
equity = 10000.00  # $10k account

# AI signal
symbol = 'BTCUSDT'
ai_conviction = 0.85  # High conviction
entry_price = 43500.00

# Calculate margin
ai_risk_factor = 0.5 + (ai_conviction * 1.0)  # 0.85 → 1.35
margin = compute_position_margin(equity, ai_risk_factor, config.risk)

# Get leverage
metrics = fetcher.fetch_symbol_metrics(symbol)
effective_leverage = compute_effective_leverage(metrics, config.risk)

# Calculate size
quantity = compute_position_size(margin, effective_leverage, entry_price)
notional = quantity * entry_price

print(f"Equity: ${equity:,.2f}")
print(f"AI Conviction: {ai_conviction:.2f} → Risk Factor: {ai_risk_factor:.2f}")
print(f"Margin: ${margin:,.2f} ({(margin/equity)*100:.2f}%)")
print(f"Effective Leverage: {effective_leverage:.1f}x")
print(f"Position Size: {quantity:.4f} BTC")
print(f"Notional: ${notional:,.2f}")
```

**Output**:
```
Equity: $10,000.00
AI Conviction: 0.85 → Risk Factor: 1.35
Margin: $135.00 (1.35%)
Effective Leverage: 15.0x
Position Size: 0.0465 BTC
Notional: $2,025.00
```

---

## ✅ Best Practices

### 1. **Universe Management**

✅ **Update universe regularly** (every 5min default)  
✅ **Monitor excluded symbols** - log why they're rejected  
✅ **Adjust tiers dynamically** - promote/demote based on liquidity changes  
❌ **Don't hardcode symbol lists** - use classification logic

### 2. **Position Sizing**

✅ **Respect min/max margins** - never go below $10 or above $1000  
✅ **Monitor total exposure** - sum all positions ≤ 15% of equity  
✅ **Let AI drive conviction** - but within 0.5-1.5x bounds  
❌ **Don't use full leverage** - effective leverage (8-15x) ≠ Binance setting (30x)

### 3. **TP/SL Management**

✅ **Calculate ATR on entry** - don't use stale values  
✅ **Move to break-even quickly** - protect capital at 1R  
✅ **Let winners run** - trailing stop captures extended moves  
✅ **Partial closes reduce stress** - lock profits incrementally  
❌ **Don't adjust TP/SL manually** - trust the ATR-based system

### 4. **Funding Protection**

✅ **Check funding before every entry** - not just long-term positions  
✅ **Monitor rate extremes** - funding >0.1% is dangerous  
✅ **Respect time windows** - 40min before, 20min after  
❌ **Don't enter during funding** - even if signal is strong

### 5. **Configuration Tuning**

✅ **Start conservative** - min volumes $5M+, max spread 3bps  
✅ **Backtest parameter changes** - don't tune on live data  
✅ **Monitor rejection rates** - if >80% trades rejected, relax filters  
✅ **Adjust leverage by tier** - MAIN (15x) > L1 (12x) > L2 (10x)  
❌ **Don't disable filters** - better to reduce position count than trade bad symbols

---

## 🐛 Troubleshooting

### Issue: "All symbols rejected"

**Symptoms**: Universe size = 0, no trades allowed

**Causes**:
1. TP_MIN_VOLUME_24H too high
2. TP_MAX_SPREAD_BPS too low
3. TP_ALLOWED_TIERS excludes all symbols

**Fix**:
```bash
# Relax volume requirement
TP_MIN_VOLUME_24H=2000000  # From $5M to $2M

# Relax spread requirement
TP_MAX_SPREAD_BPS=5.0  # From 3bps to 5bps

# Allow more tiers
TP_ALLOWED_TIERS=MAIN,L1,L2,DEFI
```

---

### Issue: "Position sizes too small"

**Symptoms**: Notional < $20, "min notional not met" errors

**Causes**:
1. TP_BASE_RISK_FRAC too low
2. TP_MIN_MARGIN too low
3. Effective leverage too low

**Fix**:
```bash
# Increase base risk
TP_BASE_RISK_FRAC=0.02  # From 1% to 2%

# Increase min margin
TP_MIN_MARGIN=20.0  # From $10 to $20

# Increase effective leverage
TP_LEVERAGE_L1=15.0  # From 12x to 15x
```

---

### Issue: "TP/SL too tight"

**Symptoms**: Stops hit immediately, no time to develop

**Causes**:
1. TP_ATR_MULT_SL too small
2. ATR period too short (sensitive to noise)
3. Wrong timeframe (1m too noisy, 4h too slow)

**Fix**:
```bash
# Widen stop loss
TP_ATR_MULT_SL=1.5  # From 1R to 1.5R

# Increase ATR period
TP_ATR_PERIOD=21  # From 14 to 21

# Use longer timeframe
TP_ATR_TIMEFRAME=30m  # From 15m to 30m
```

---

### Issue: "Missing funding events"

**Symptoms**: Trades opened just before funding, losses from rates

**Causes**:
1. TP_FUNDING_PRE_WINDOW too short
2. Server time not UTC
3. Binance API returning wrong funding time

**Fix**:
```bash
# Expand pre-window
TP_FUNDING_PRE_WINDOW=60  # From 40min to 60min

# Check server timezone
docker exec quantum_backend python -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc))"

# Verify Binance data
docker exec quantum_backend python -c "from backend.services.binance_market_data import *; f = create_market_data_fetcher(); rate, time = f.get_funding_info('BTCUSDT'); print(f'Rate: {rate}, Next: {time}')"
```

---

### Issue: "ATR calculation failing"

**Symptoms**: "Failed to calculate ATR" errors, None returned

**Causes**:
1. Not enough historical klines
2. Symbol not trading
3. Binance API rate limits

**Fix**:
```python
# Test ATR manually
from backend.services.binance_market_data import calculate_atr

atr = calculate_atr('BTCUSDT', period=14, timeframe='15m')
if atr:
    print(f"✅ ATR: {atr}")
else:
    print("❌ ATR calculation failed - check logs")
```

---

## 📊 Performance Metrics

Monitor these metrics to validate the Trading Profile System:

```python
# Rejection Rate
rejection_rate = rejected_signals / total_signals
# Target: 60-80% (aggressive filtering is good!)

# Universe Quality
avg_volume = sum(s.quote_volume_24h for s in universe) / len(universe)
avg_spread_bps = sum(compute_spread_bps(s) for s in universe) / len(universe)
# Target: avg_volume > $50M, avg_spread < 2bps

# Position Sizing
avg_margin_pct = sum(position.margin / equity for position in positions) / len(positions)
# Target: 1-2% per position

# TP/SL Hit Rates
tp1_hit_rate = tp1_hits / total_trades
tp2_hit_rate = tp2_hits / total_trades
sl_hit_rate = sl_hits / total_trades
# Target: TP1 >60%, TP2 >30%, SL <40%

# Funding Impact
avg_funding_cost = sum(funding_paid) / total_trades
# Target: <0.01% per trade (minimal)
```

---

## 🚀 Next Steps

1. ✅ **Module Complete** - All components implemented
2. ⏳ **Integration** - Connect to Orchestrator & Execution
3. ⏳ **Testing** - Unit tests + integration tests
4. ⏳ **Monitoring** - Dashboard for universe, rejections, TP/SL performance
5. ⏳ **Backtesting** - Validate on historical data
6. ⏳ **Live Testing** - Paper trading → small position sizes → full scale

---

## 📚 Additional Resources

- **Binance Futures API**: https://binance-docs.github.io/apidocs/futures/en/
- **ATR Indicator**: https://www.investopedia.com/terms/a/atr.asp
- **Position Sizing**: https://www.investopedia.com/terms/p/positionsizing.asp
- **Funding Rates**: https://www.binance.com/en/futures/funding-history

---

**Built with ❤️ by Quantum Trader Team**  
*Last Updated: 2025-11-26*

