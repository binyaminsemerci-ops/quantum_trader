# Strategy Runtime Engine - Integration Guide

## Overview

The **Strategy Runtime Engine** is the bridge between AI-generated strategies (from SG AI) and the live trading execution pipeline. It evaluates LIVE strategies against real-time market data and produces standardized `TradeDecision` objects that flow through the existing risk management and execution layers.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Strategy Generator AI                        │
│  (Generates, backtests, evolves strategies)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │ Produces StrategyConfig
                         │ Stores in Repository
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Strategy Runtime Engine                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Load LIVE  │→ │   Evaluate   │→ │   Generate   │         │
│  │  Strategies  │  │  Conditions  │  │   Signals    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────┬────────────────────────────────────────┘
                         │ Produces TradeDecision
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              Existing Execution Pipeline                         │
│                                                                  │
│  Orchestrator → RiskGuard → PortfolioBalancer →                │
│  SafetyGovernor → Executor → PositionMonitor                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. StrategyConfig (Data Model)

Defines a strategy's entry rules, risk parameters, and constraints:

```python
@dataclass
class StrategyConfig:
    strategy_id: str                      # Unique identifier
    name: str                             # Human-readable name
    status: Literal["CANDIDATE", "SHADOW", "LIVE", "DISABLED"]
    
    # Entry conditions
    entry_indicators: List[Dict]          # e.g., [{"name": "RSI", "operator": "<", "value": 30}]
    entry_logic: Literal["ALL", "ANY"]    # ALL=AND, ANY=OR
    
    # Position sizing
    base_size_usd: float                  # Base position size
    confidence_scaling: bool              # Scale size by confidence?
    
    # Risk management
    stop_loss_pct: float                  # Stop loss as % of entry
    take_profit_pct: float                # Take profit as % of entry
    
    # Filters
    allowed_regimes: List[str]            # e.g., ["TRENDING", "NORMAL"]
    min_confidence: float                 # Min confidence threshold
    max_positions: int                    # Max concurrent positions
    
    # Performance
    fitness_score: Optional[float]        # Historical performance score
```

### 2. TradeDecision (Output Model)

Standardized trading signal consumed by the execution pipeline:

```python
@dataclass
class TradeDecision:
    symbol: str                           # Trading symbol
    side: Literal["LONG", "SHORT"]        # Trade direction
    size_usd: float                       # Position size in USD
    confidence: float                     # 0.0 to 1.0
    strategy_id: str                      # Strategy that generated this
    
    # Price levels
    entry_price: Optional[float]
    take_profit: Optional[float]
    stop_loss: Optional[float]
    
    # Context
    timestamp: datetime
    regime: Optional[str]
    reasoning: Optional[str]
    metadata: Dict                        # Additional context
```

### 3. StrategyEvaluator

Pure logic component that evaluates strategy conditions:

```python
class StrategyEvaluator:
    def evaluate(
        self,
        strategy: StrategyConfig,
        symbol: str,
        market_data: pd.DataFrame,
        indicators: Dict[str, float],
        current_regime: Optional[str] = None
    ) -> Optional[StrategySignal]:
        """
        Evaluates strategy conditions against market state.
        Returns StrategySignal if conditions met, None otherwise.
        """
```

### 4. StrategyRuntimeEngine

Main engine that orchestrates everything:

```python
class StrategyRuntimeEngine:
    def __init__(
        self,
        strategy_repository: StrategyRepository,
        market_data_client: MarketDataClient,
        policy_store: PolicyStore,
        evaluator: Optional[StrategyEvaluator] = None
    ):
        """Initialize with required dependencies"""
    
    def refresh_strategies(self) -> None:
        """Reload LIVE strategies from repository"""
    
    def generate_signals(
        self,
        symbols: List[str],
        current_regime: Optional[str] = None
    ) -> List[TradeDecision]:
        """
        Main method: Generate trading signals for given symbols
        by evaluating all LIVE strategies.
        """
```

## Integration Points

### 1. With Strategy Generator AI

**SG AI produces strategies → Runtime Engine consumes them**

```python
# SG AI creates a new strategy
strategy = StrategyConfig(
    strategy_id="rsi_mean_revert_042",
    name="RSI Mean Reversion",
    status="LIVE",  # Promoted from SHADOW
    entry_indicators=[
        {"name": "RSI", "operator": "<", "value": 30},
        {"name": "SMA_CROSS", "operator": ">", "value": 0}
    ],
    entry_logic="ALL",
    base_size_usd=2000.0,
    stop_loss_pct=0.02,
    take_profit_pct=0.05,
    allowed_regimes=["TRENDING"],
    min_confidence=0.6,
    fitness_score=0.78
)

# Store in repository
strategy_repository.save(strategy)

# Runtime Engine automatically picks it up on next refresh
engine.refresh_strategies()
```

### 2. With Event-Driven Executor

**Runtime Engine plugs into the existing event loop:**

```python
# In your event-driven executor loop
class QuantumExecutor:
    def __init__(self):
        self.strategy_runtime = StrategyRuntimeEngine(
            strategy_repository=self.strategy_repo,
            market_data_client=self.market_data,
            policy_store=self.policy_store
        )
        # ... existing components
    
    async def run_cycle(self):
        """Main execution cycle"""
        # 1. Get current regime
        regime = self.regime_detector.get_current_regime()
        
        # 2. Get top symbols from Opportunity Ranker
        top_symbols = self.opportunity_ranker.get_top_symbols(limit=10)
        
        # 3. Generate signals from strategies
        strategy_decisions = self.strategy_runtime.generate_signals(
            symbols=top_symbols,
            current_regime=regime
        )
        
        # 4. Also get AI model predictions (existing flow)
        ai_decisions = self.get_ai_predictions(top_symbols)
        
        # 5. Combine all decisions
        all_decisions = strategy_decisions + ai_decisions
        
        # 6. Process through existing pipeline
        for decision in all_decisions:
            # Orchestrator check
            if not self.orchestrator.is_trade_allowed(decision):
                continue
            
            # Risk Guard validation
            if not self.risk_guard.validate(decision):
                continue
            
            # Portfolio Balancer check
            if not self.portfolio_balancer.can_add_position(decision):
                continue
            
            # Safety Governor check
            if not self.safety_governor.is_safe_to_trade():
                continue
            
            # Execute
            self.executor.place_trade(decision)
            
            # Track with strategy_id
            self.position_monitor.track_position(
                decision.symbol,
                strategy_id=decision.strategy_id
            )
```

### 3. With Meta Strategy Controller

**MSC AI controls which strategies can trade:**

```python
# MSC AI updates policy
policy_store.set_risk_mode("DEFENSIVE")
policy_store.set_global_min_confidence(0.7)
policy_store.set_allowed_strategies([
    "high_fitness_001",
    "high_fitness_002"
])

# Runtime Engine respects these policies
engine.refresh_strategies()  # Only loads allowed strategies
decisions = engine.generate_signals(symbols)  # Applies min confidence
```

### 4. With Performance Tracking

**Each trade is tagged with strategy_id:**

```python
# After trade execution
class PositionMonitor:
    def on_position_closed(self, position):
        """Called when position closes"""
        # Calculate PnL
        pnl = position.exit_price - position.entry_price
        pnl_pct = pnl / position.entry_price
        
        # Update strategy performance
        if position.strategy_id:
            self.update_strategy_metrics(
                strategy_id=position.strategy_id,
                pnl=pnl,
                pnl_pct=pnl_pct,
                hold_time=position.hold_time
            )
            
            # SG AI can use this for fitness calculation
            self.notify_sg_ai(position.strategy_id, pnl_pct)
```

## Configuration

### Repository Setup

Implement the `StrategyRepository` protocol:

```python
class PostgresStrategyRepository:
    def get_by_status(self, status: str) -> List[StrategyConfig]:
        """Query strategies by status"""
        query = """
            SELECT * FROM sg_strategies 
            WHERE status = %s
            ORDER BY fitness_score DESC
        """
        rows = self.execute(query, (status,))
        return [self._row_to_strategy(r) for r in rows]
    
    def update_last_execution(self, strategy_id: str, timestamp: datetime):
        """Update last execution timestamp"""
        query = """
            UPDATE sg_strategies 
            SET last_executed_at = %s
            WHERE strategy_id = %s
        """
        self.execute(query, (timestamp, strategy_id))
```

### Market Data Client

Implement the `MarketDataClient` protocol:

```python
class BinanceMarketDataClient:
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def get_latest_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get recent OHLCV bars"""
        klines = self.client.get_klines(
            symbol=symbol,
            interval=timeframe,
            limit=limit
        )
        return self._klines_to_dataframe(klines)
    
    def get_indicators(self, symbol: str, indicators: List[str]) -> Dict[str, float]:
        """Calculate or fetch indicators"""
        df = self.get_latest_bars(symbol, "1h", 100)
        
        result = {}
        if "RSI" in indicators:
            result["RSI"] = self._calculate_rsi(df['close'])
        if "MACD" in indicators:
            macd, signal = self._calculate_macd(df['close'])
            result["MACD"] = macd
            result["MACD_SIGNAL"] = signal
        
        return result
```

### Policy Store

Implement the `PolicyStore` protocol:

```python
class RedisPolicyStore:
    def get_risk_mode(self) -> str:
        """Get current risk mode"""
        return self.redis.get("global:risk_mode") or "NORMAL"
    
    def get_global_min_confidence(self) -> float:
        """Get global minimum confidence"""
        value = self.redis.get("global:min_confidence")
        return float(value) if value else 0.5
    
    def is_strategy_allowed(self, strategy_id: str) -> bool:
        """Check if strategy is allowed"""
        allowed = self.redis.smembers("global:allowed_strategies")
        # Empty set = all allowed
        return not allowed or strategy_id in allowed
```

## Usage Examples

### Basic Usage

```python
# Initialize
engine = StrategyRuntimeEngine(
    strategy_repository=PostgresStrategyRepository(),
    market_data_client=BinanceMarketDataClient(),
    policy_store=RedisPolicyStore()
)

# Generate signals
decisions = engine.generate_signals(
    symbols=["BTCUSDT", "ETHUSDT"],
    current_regime="TRENDING"
)

# Process decisions
for decision in decisions:
    print(f"{decision.strategy_id}: {decision.side} {decision.symbol} @ {decision.confidence:.1%}")
```

### Advanced: Custom Evaluator

```python
class CustomStrategyEvaluator(StrategyEvaluator):
    """Custom evaluator with additional logic"""
    
    def evaluate(self, strategy, symbol, market_data, indicators, current_regime):
        # Call parent evaluation
        signal = super().evaluate(strategy, symbol, market_data, indicators, current_regime)
        
        if not signal:
            return None
        
        # Add custom filtering
        if self._is_news_event_pending(symbol):
            logger.info(f"Filtering signal due to pending news event")
            return None
        
        return signal

# Use custom evaluator
engine = StrategyRuntimeEngine(
    strategy_repository=repo,
    market_data_client=client,
    policy_store=store,
    evaluator=CustomStrategyEvaluator()
)
```

## Performance Considerations

### Caching

```python
# Cache market data to avoid redundant API calls
class CachedMarketDataClient:
    def __init__(self, base_client, cache_ttl=60):
        self.base_client = base_client
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def get_latest_bars(self, symbol, timeframe, limit):
        cache_key = f"{symbol}:{timeframe}:{limit}"
        
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if (datetime.utcnow() - timestamp).seconds < self.cache_ttl:
                return data
        
        data = self.base_client.get_latest_bars(symbol, timeframe, limit)
        self.cache[cache_key] = (data, datetime.utcnow())
        return data
```

### Parallel Evaluation

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelStrategyRuntimeEngine(StrategyRuntimeEngine):
    def generate_signals(self, symbols, current_regime):
        # Refresh strategies
        self.refresh_strategies()
        
        # Evaluate strategies in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for strategy in self.active_strategies.values():
                for symbol in symbols:
                    future = executor.submit(
                        self._evaluate_strategy_for_symbol,
                        strategy, symbol, current_regime,
                        self.policy_store.get_global_min_confidence(),
                        self.policy_store.get_risk_mode()
                    )
                    futures.append(future)
            
            decisions = [f.result() for f in futures if f.result()]
        
        return decisions
```

## Monitoring & Debugging

### Logging

```python
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging for troubleshooting
logging.getLogger("backend.services.strategy_runtime_engine").setLevel(logging.DEBUG)
```

### Metrics

```python
from prometheus_client import Counter, Histogram

# Track signal generation
signals_generated = Counter(
    'strategy_runtime_signals_generated_total',
    'Total signals generated',
    ['strategy_id', 'symbol', 'side']
)

signal_confidence = Histogram(
    'strategy_runtime_signal_confidence',
    'Signal confidence distribution',
    ['strategy_id']
)

# In engine
def _signal_to_decision(self, signal, strategy, ...):
    decision = ...  # create decision
    
    signals_generated.labels(
        strategy_id=strategy.strategy_id,
        symbol=decision.symbol,
        side=decision.side
    ).inc()
    
    signal_confidence.labels(strategy_id=strategy.strategy_id).observe(decision.confidence)
    
    return decision
```

## Testing

Run the examples:

```bash
python backend/services/strategy_runtime_engine_examples.py
```

Expected output:
- Example 1: Single strategy evaluation
- Example 2: Multiple strategies generating signals
- Example 3: Full pipeline integration flow
- Example 4: Performance tracking demonstration

## Next Steps

1. **Implement Repository**: Create `PostgresStrategyRepository` with real DB access
2. **Connect Market Data**: Wire up `BinanceMarketDataClient` or your market data source
3. **Setup Policy Store**: Implement `RedisPolicyStore` or alternative
4. **Integrate with Executor**: Add strategy runtime to your event-driven loop
5. **Add Monitoring**: Instrument with Prometheus metrics
6. **Test with Shadow Strategies**: Run with SHADOW strategies before going LIVE

## Summary

The Strategy Runtime Engine provides:
- ✅ Clean separation between strategy generation (SG AI) and execution
- ✅ Standardized `TradeDecision` interface for existing pipeline
- ✅ Policy-driven control (MSC AI can enable/disable strategies)
- ✅ Per-strategy performance tracking
- ✅ Testable, modular design
- ✅ Production-ready with caching, parallel evaluation, monitoring

This completes the bridge from AI-generated strategies to live trading! 🚀
