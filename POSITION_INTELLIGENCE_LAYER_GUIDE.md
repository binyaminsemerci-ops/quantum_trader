# POSITION INTELLIGENCE LAYER (PIL) — Complete Operating Manual

**Version:** 1.0  
**Status:** ✅ DEPLOYED & OPERATIONAL  
**Last Updated:** 2025-11-22  
**System Type:** Per-Position Lifecycle Management & Optimization

---

## EXECUTIVE SUMMARY

The **Position Intelligence Layer (PIL)** is the autonomous system responsible for analyzing and managing **each open position** individually over its entire lifecycle. Unlike signal generation systems, PIL focuses exclusively on **maximizing per-position outcomes** through dynamic hold time management, sizing adjustments (scale-in/out), and exit tactics.

### Key Capabilities

- **30+ Metrics Per Position** - Comprehensive performance, risk, and context analysis
- **5-Tier Classification System** - STRONG_TREND / SLOW_GRINDER / STALLING / REVERSAL_RISK / TOXIC
- **Scale-In/Out Advisory** - Position size optimization based on performance
- **Exit Optimization** - Strategic exit timing with rationale
- **Priority Scoring** - 0-10 urgency scoring for each recommendation
- **AELM Integration** - Structured outputs for Autonomous Execution Layer Manager

---

## SYSTEM ARCHITECTURE

### Operating Modes

```
ADVISORY MODE (Current):
├─ Analyze positions
├─ Generate recommendations
├─ NO automatic execution
└─ Output JSON for AELM integration

AUTONOMOUS MODE (Future):
├─ All ADVISORY actions
├─ Automatic scale-in/out execution
├─ Automatic exit triggers
└─ Full AELM integration
```

### 6-Phase Execution Pipeline

```
PHASE 1: DATA INGESTION
├─ Load open positions
├─ Load trade history (for expectations)
├─ Load signal history (for confidence)
├─ Load universe data (for symbol classification)
└─ Load orchestrator state (for market context)

PHASE 2: POSITION METRICS COMPUTATION
├─ 30+ metrics per position
├─ Performance: current_R, peak_R, unrealized_pnl
├─ Time: time_in_trade_hours
├─ Momentum: momentum_score, trend_strength
├─ Volatility: volatility_change_factor
├─ Risk: risk_state (CALM/STRESSED/CRITICAL)
└─ Context: regime, spread, slippage

PHASE 3: POSITION CLASSIFICATION
├─ STRONG_TREND (R≥1.0, momentum≥0.6)
├─ SLOW_GRINDER (positive R, low momentum)
├─ STALLING (low momentum after >1h)
├─ REVERSAL_RISK (dropped >0.5R from peak)
└─ TOXIC (R < -0.5 after >30min)

PHASE 4: INTELLIGENCE GENERATION
├─ Action recommendations per classification
├─ Scale-in/out suggestions
├─ Exit recommendations with rationale
├─ Priority scoring (0-10)
└─ Urgency levels (NORMAL/ELEVATED/URGENT/CRITICAL)

PHASE 5: SUMMARY GENERATION
├─ Classification breakdown
├─ Risk state distribution
├─ Aggregate metrics
└─ Recommendations by category

PHASE 6: OUTPUT GENERATION
├─ position_intelligence.json (full details)
├─ position_intelligence_summary.json (summary)
└─ position_recommendations.json (AELM-ready)
```

---

## POSITION CLASSIFICATION SYSTEM

### 1. STRONG_TREND 🚀

**Criteria:**
- `current_R >= 1.0` (position in profit ≥1× risk)
- `momentum_score >= 0.6` (strong upward trajectory)
- Time in trade: Any duration

**Recommended Actions:**
- **HOLD_LONGER** - Let winners run
- **LOOSEN_TP** - Extend profit targets
- **ENABLE_TRAILING** - Lock in gains dynamically
- **SCALE_IN** (if CORE symbol only) - Pyramid winners

**Scale-In Logic:**
- Only for CORE symbols (from Universe Control Center)
- `current_R >= 0.5` (position already profitable)
- Volatility acceptable (`volatility_change_factor <= 1.5`)
- Suggested size: **+50%** of current position

**Example:**
```
Symbol: BTCUSDT (CORE)
current_R: 1.5
momentum_score: 0.75
Classification: STRONG_TREND
→ Recommendation: SCALE_IN +50%, LOOSEN_TP to 3.0R
```

---

### 2. SLOW_GRINDER 🐌

**Criteria:**
- `current_R > 0` (in profit)
- `momentum_score < 0.6` (slow progress)
- No reversal signals

**Recommended Actions:**
- **HOLD** - Wait for acceleration
- **MONITOR** - Watch for momentum increase or stalling
- **NO SCALE** - Do not add to slow positions

**Example:**
```
Symbol: ETHUSDT
current_R: 0.3
momentum_score: 0.35
time_in_trade: 2.5h
Classification: SLOW_GRINDER
→ Recommendation: HOLD, monitor for momentum shift
```

---

### 3. STALLING ⚠️

**Criteria:**
- `time_in_trade > 1h` (sufficient time elapsed)
- `momentum_score < 0.3` (losing momentum)
- Not yet in reversal territory

**Recommended Actions:**
- **PARTIAL_TP** - Take 50% off the table
- **TIGHTEN_SL** - Protect remaining position
- **SCALE_OUT** - Reduce exposure

**Scale-Out Logic:**
- Suggested size: **-50%** of current position
- Reasoning: Lock in partial profits before stall becomes reversal

**Example:**
```
Symbol: SOLUSDT
current_R: 0.4
momentum_score: 0.15
time_in_trade: 1.5h
Classification: STALLING
→ Recommendation: SCALE_OUT 50%, tighten SL to breakeven
```

---

### 4. REVERSAL_RISK 🔴

**Criteria:**
- `peak_R - current_R >= 0.5` (dropped significantly from peak)
- OR `volatility_change_factor > 2.0` (volatility spike against position)

**Recommended Actions:**
- **EXIT_SOON** - Close within next evaluation cycle
- **SCALE_OUT** - Aggressive reduction
- **TIGHTEN_SL** - Minimize further damage

**Scale-Out Logic:**
- If `current_R > 0.5`: SCALE_OUT **75%** (lock in profits on remainder)
- If `current_R <= 0.5`: EXIT **100%** (prevent loss)

**Example:**
```
Symbol: ADAUSDT
peak_R: 1.2
current_R: 0.6
momentum_score: -0.2
Classification: REVERSAL_RISK
→ Recommendation: SCALE_OUT 75%, EXIT remaining if drops below 0.5R
```

---

### 5. TOXIC ☠️

**Criteria:**
- `current_R < -0.5` (loss exceeds 50% of risk capital)
- `time_in_trade > 30min` (not a momentary dip)

**Recommended Actions:**
- **EXIT_IMMEDIATELY** - Emergency exit
- **OVERRIDE_MODE** - Use AGGRESSIVE exit settings
- **REDUCE_RISK** - Lower risk parameters for new positions

**Override Settings:**
```python
exit_mode: "AGGRESSIVE"
max_slippage: 0.3%  # Accept higher slippage to exit fast
risk_override: -50%  # Reduce risk on new positions
alert_level: "CRITICAL"
```

**Example:**
```
Symbol: DOGEUSDT
current_R: -0.7
time_in_trade: 45min
momentum_score: -0.6
Classification: TOXIC
→ Recommendation: EXIT_IMMEDIATELY, use AGGRESSIVE mode
```

---

## POSITION METRICS REFERENCE

### Performance Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| `unrealized_pnl` | Current profit/loss in USDT | -∞ to +∞ | Raw P&L |
| `current_R` | Current P&L in R-multiples | -∞ to +∞ | Performance relative to risk |
| `peak_R` | Highest R achieved | 0 to +∞ | Best performance |
| `trough_R` | Lowest R since entry | -∞ to 0 | Worst drawdown |
| `R_range` | `peak_R - trough_R` | 0 to +∞ | Volatility of position |

### Time Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `time_in_trade_minutes` | Minutes since entry | Absolute time |
| `time_in_trade_hours` | Hours since entry | Human-readable |
| `expected_hold_time_minutes` | Historical average for symbol | Benchmark |

### Momentum & Volatility

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| `momentum_score` | `current_R / (time_in_trade_hours + 0.1)` | -∞ to +∞ | R per hour |
| `volatility_change_factor` | `current_volatility / entry_volatility` | 0 to +∞ | Volatility increase |
| `trend_strength` | From universe data | 0 to 1 | Symbol trend quality |

### Risk Assessment

| Metric | Threshold | State | Action |
|--------|-----------|-------|--------|
| `volatility_change_factor` | ≤ 1.2 | CALM | Normal operations |
| `volatility_change_factor` | 1.2-2.0 | STRESSED | Tighten stops |
| `volatility_change_factor` | > 2.0 | CRITICAL | Consider exit |

---

## SCALE-IN/OUT DECISION MATRIX

### Scale-In Triggers

```
✅ ALL must be true:
├─ Classification: STRONG_TREND
├─ Symbol category: CORE only (from Universe Control Center)
├─ current_R >= 0.5 (position already profitable)
├─ volatility_change_factor <= 1.5 (volatility acceptable)
└─ Suggested size: +50% of current position

❌ DO NOT scale-in if:
├─ Symbol is EXPLORATORY or VOLATILE
├─ Position is SLOW_GRINDER, STALLING, or worse
├─ current_R < 0.5 (not yet sufficiently profitable)
└─ volatility_change_factor > 1.5 (too volatile)
```

### Scale-Out Triggers

```
PARTIAL SCALE-OUT (-50%):
├─ Classification: STALLING
├─ momentum_score < 0.3
├─ time_in_trade > 1h
└─ Reasoning: Lock partial profits before stall → reversal

AGGRESSIVE SCALE-OUT (-75%):
├─ Classification: REVERSAL_RISK
├─ peak_R - current_R >= 0.5
├─ current_R > 0.5 (still in profit)
└─ Reasoning: Protect profits from reversal

FULL EXIT (100%):
├─ Classification: TOXIC
├─ current_R < -0.5
└─ Reasoning: Emergency loss prevention
```

---

## OUTPUT FILES

### 1. position_intelligence.json

**Purpose:** Full per-position analysis  
**Size:** ~2KB per position  
**Update Frequency:** Every 60s (configurable)

**Structure:**
```json
{
  "generated_at": "2025-11-22T23:49:45.332753+00:00",
  "mode": "ADVISORY",
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      
      // Performance Metrics
      "unrealized_pnl": 125.50,
      "current_R": 1.5,
      "peak_R": 1.8,
      "trough_R": -0.1,
      
      // Time Metrics
      "time_in_trade_hours": 2.5,
      "expected_hold_time_minutes": 180,
      
      // Momentum & Volatility
      "momentum_score": 0.75,
      "volatility_change_factor": 1.1,
      "trend_strength": 0.85,
      
      // Risk Assessment
      "risk_state": "CALM",
      
      // Classification
      "classification": "STRONG_TREND",
      
      // Recommendations
      "hold_recommendation": "HOLD_LONGER",
      "scale_suggestion": "SCALE_IN",
      "scale_rationale": "CORE symbol, R≥0.5, strong momentum",
      "suggested_size_delta": 0.5,
      "exit_suggestion": "NO_EXIT",
      "exit_rationale": "Position performing well",
      "suggested_exit_percentage": 0.0,
      
      // Priority
      "priority_score": 8.5,
      "urgency": "NORMAL",
      
      // AELM Integration
      "aelm_hints": {
        "loosen_tp": true,
        "enable_trailing": true,
        "consider_scale_in": true
      }
    }
  ]
}
```

### 2. position_intelligence_summary.json

**Purpose:** Portfolio-level overview  
**Size:** ~2KB total  
**Update Frequency:** Every 60s

**Structure:**
```json
{
  "timestamp": "2025-11-22T23:49:45.332753+00:00",
  "total_positions": 5,
  
  // Classification Breakdown
  "strong_trend_count": 2,
  "slow_grinder_count": 1,
  "stalling_count": 1,
  "reversal_risk_count": 1,
  "toxic_count": 0,
  
  // Risk State Distribution
  "calm_count": 3,
  "stressed_count": 1,
  "critical_count": 1,
  
  // Aggregate Metrics
  "total_unrealized_pnl": 450.25,
  "total_current_R": 3.5,
  "avg_time_in_trade_hours": 2.1,
  "avg_momentum_score": 0.42,
  
  // Positions Requiring Action
  "positions_needing_attention": ["ADAUSDT", "DOGEUSDT"],
  "positions_to_scale_in": ["BTCUSDT"],
  "positions_to_scale_out": ["SOLUSDT"],
  "positions_to_exit": ["ADAUSDT"],
  
  // Focus Areas
  "focus_risk_reduction": ["ADAUSDT"],
  "focus_profit_maximization": ["BTCUSDT", "ETHUSDT"]
}
```

### 3. position_recommendations.json

**Purpose:** AELM-ready action list  
**Size:** ~1-2KB  
**Update Frequency:** Every 60s

**Structure:**
```json
{
  "generated_at": "2025-11-22T23:49:45.336888+00:00",
  "mode": "ADVISORY",
  
  "immediate_actions": [
    {
      "symbol": "ADAUSDT",
      "action": "EXIT",
      "urgency": "URGENT",
      "priority": 9.5,
      "rationale": "REVERSAL_RISK: dropped 0.6R from peak",
      "percentage": 100.0
    }
  ],
  
  "scale_recommendations": [
    {
      "symbol": "BTCUSDT",
      "action": "SCALE_IN",
      "urgency": "NORMAL",
      "priority": 7.0,
      "size_delta": 0.5,
      "rationale": "STRONG_TREND: CORE symbol, R≥0.5, strong momentum"
    },
    {
      "symbol": "SOLUSDT",
      "action": "SCALE_OUT",
      "urgency": "ELEVATED",
      "priority": 6.5,
      "size_delta": -0.5,
      "rationale": "STALLING: momentum fading after 1.5h"
    }
  ],
  
  "exit_recommendations": [
    {
      "symbol": "ADAUSDT",
      "urgency": "URGENT",
      "priority": 9.5,
      "percentage": 100.0,
      "rationale": "REVERSAL_RISK: dropped 0.6R from peak"
    }
  ]
}
```

---

## OPERATIONAL PROCEDURES

### Daily Operations

#### 1. Morning Startup Check
```bash
# Verify PIL is running
docker exec quantum_backend ps aux | grep position_intelligence

# Check for open positions
docker exec quantum_backend cat /app/data/open_positions.json | jq '.positions | length'

# Run initial analysis
docker exec quantum_backend python /app/position_intelligence_layer.py
```

#### 2. Review Summary
```bash
# Copy summary locally
docker cp quantum_backend:/app/data/position_intelligence_summary.json ./

# Check key metrics
cat position_intelligence_summary.json | jq '{
  total_positions,
  strong_trend_count,
  toxic_count,
  positions_needing_attention
}'
```

#### 3. Check Urgent Actions
```bash
# Copy recommendations
docker cp quantum_backend:/app/data/position_recommendations.json ./

# List immediate actions
cat position_recommendations.json | jq '.immediate_actions[] | {symbol, action, urgency, rationale}'
```

### Intraday Monitoring

#### Hourly Health Check
```bash
# Run PIL analysis
docker exec quantum_backend python /app/position_intelligence_layer.py

# Check for elevated urgency
cat position_recommendations.json | jq '[.immediate_actions[], .scale_recommendations[], .exit_recommendations[]] | map(select(.urgency == "URGENT" or .urgency == "CRITICAL"))'
```

#### Real-Time Alerts (Future with AELM)
```python
# Monitor for TOXIC positions
while True:
    summary = load_json("position_intelligence_summary.json")
    if summary["toxic_count"] > 0:
        alert("TOXIC POSITION DETECTED", level="CRITICAL")
    time.sleep(60)
```

### Position Lifecycle Example

```
t=0: Position Opens (LONG BTCUSDT @ $50,000)
├─ Initial R: 0.0
├─ Classification: (wait for 5min minimum)
└─ PIL: Monitoring...

t=30min: Early Performance
├─ current_R: 0.3
├─ momentum_score: 0.6
├─ Classification: SLOW_GRINDER
└─ Recommendation: HOLD, monitor

t=1h: Strong Momentum
├─ current_R: 1.2
├─ momentum_score: 1.2
├─ Classification: STRONG_TREND
└─ Recommendation: SCALE_IN +50% (if CORE)

t=2h: Peak Performance
├─ current_R: 1.8 (peak_R set)
├─ momentum_score: 0.9
├─ Classification: STRONG_TREND
└─ Recommendation: HOLD_LONGER, ENABLE_TRAILING

t=3h: Momentum Fading
├─ current_R: 1.5 (dropped 0.3R from peak)
├─ momentum_score: 0.5
├─ Classification: SLOW_GRINDER
└─ Recommendation: HOLD, watch for reversal

t=4h: Reversal Warning
├─ current_R: 1.0 (dropped 0.8R from peak)
├─ momentum_score: 0.25
├─ Classification: REVERSAL_RISK
└─ Recommendation: SCALE_OUT 75%

t=5h: Exit Executed
├─ 75% scaled out at 1.0R
├─ Remaining 25% at 0.8R
└─ Final outcome: Captured most of 1.8R peak
```

---

## INTEGRATION WITH OTHER SYSTEMS

### Universe Control Center Integration

**PIL Consumes:**
- Symbol classifications (CORE/EXPLORATORY/VOLATILE/TOXIC)
- Symbol stability scores
- Universe-level risk states

**Used For:**
- Scale-in decisions (only CORE symbols)
- Exit urgency (TOXIC symbols → immediate exit)
- Priority scoring (CORE symbols get higher priority)

**Example:**
```python
# universe_classification.json
{
  "BTCUSDT": {
    "category": "CORE",
    "stability_score": 0.95
  }
}

# Used in PIL scale-in logic:
if (position.classification == "STRONG_TREND" 
    and symbol_category == "CORE"
    and current_R >= 0.5):
    recommend_scale_in(size_delta=0.5)
```

### AELM Integration (Future)

**PIL Outputs →**
- `position_recommendations.json` (structured actions)

**AELM Consumes:**
- Immediate actions (TOXIC exits)
- Scale recommendations
- Exit recommendations

**AELM Executes:**
- Place scale-in orders
- Execute partial exits
- Adjust stop-loss/take-profit
- Enable trailing stops

**Workflow:**
```
PIL (every 60s)
  ↓ generates recommendations
position_recommendations.json
  ↓ read by
AELM (every 60s)
  ↓ executes actions
Exchange Orders
  ↓ updates
open_positions.json
  ↓ read by
PIL (next cycle)
```

---

## CONFIGURATION

### Environment Variables

```bash
# Operating Mode
PIL_MODE=ADVISORY  # or AUTONOMOUS

# Update Interval
PIL_UPDATE_INTERVAL_SECONDS=60

# Classification Thresholds
PIL_STRONG_TREND_R_THRESHOLD=1.0
PIL_STRONG_TREND_MOMENTUM_THRESHOLD=0.6
PIL_TOXIC_R_THRESHOLD=-0.5
PIL_TOXIC_MIN_TIME_MINUTES=30
PIL_REVERSAL_R_DROP_THRESHOLD=0.5

# Scale Thresholds
PIL_SCALE_IN_MIN_R=0.5
PIL_SCALE_IN_SIZE_DELTA=0.5
PIL_SCALE_OUT_STALLING_SIZE_DELTA=-0.5
PIL_SCALE_OUT_REVERSAL_SIZE_DELTA=-0.75

# Risk Thresholds
PIL_RISK_CALM_VOLATILITY_THRESHOLD=1.2
PIL_RISK_STRESSED_VOLATILITY_THRESHOLD=2.0
```

### File Paths

```
Input Files:
├─ /app/data/open_positions.json
├─ /app/data/trade_history.json
├─ /app/data/signal_history.json
├─ /app/data/universe_classification.json
└─ /app/data/orchestrator_state.json

Output Files:
├─ /app/data/position_intelligence.json
├─ /app/data/position_intelligence_summary.json
└─ /app/data/position_recommendations.json
```

---

## TESTING & VALIDATION

### Unit Tests (Future)

```python
# test_classification.py
def test_strong_trend_classification():
    metrics = PositionMetrics(current_R=1.5, momentum_score=0.75, ...)
    classification = classify_position(metrics)
    assert classification == "STRONG_TREND"

def test_toxic_classification():
    metrics = PositionMetrics(current_R=-0.7, time_in_trade_minutes=45, ...)
    classification = classify_position(metrics)
    assert classification == "TOXIC"
```

### Integration Tests

```bash
# Test with mock position
echo '{"positions": [{"symbol": "BTCUSDT", "side": "LONG", "entry_price": 50000, ...}]}' \
  > /app/data/open_positions.json

# Run PIL
docker exec quantum_backend python /app/position_intelligence_layer.py

# Verify output
docker exec quantum_backend cat /app/data/position_intelligence.json | jq '.positions[0].classification'
# Expected: "STRONG_TREND" or similar
```

### Performance Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Analysis time per position | < 100ms | TBD |
| Total execution time (10 positions) | < 2s | TBD |
| Memory usage | < 100MB | TBD |
| Classification accuracy | > 90% | TBD |

---

## TROUBLESHOOTING

### Issue: "No open positions found"

**Symptom:** PIL reports 0 positions  
**Causes:**
1. No active trades (expected)
2. `open_positions.json` missing or empty
3. Container file path incorrect

**Resolution:**
```bash
# Check for open positions file
docker exec quantum_backend ls -l /app/data/open_positions.json

# Check file contents
docker exec quantum_backend cat /app/data/open_positions.json | jq

# If missing, wait for trades to open or check signal generator
```

### Issue: "TypeError: non-default argument follows default argument"

**Symptom:** PIL crashes on startup  
**Cause:** Dataclass field ordering issue

**Resolution:**
Ensure all fields with default values come AFTER fields without defaults:
```python
@dataclass
class PositionIntelligence:
    # Required fields (no defaults)
    symbol: str
    side: str
    hold_recommendation: str
    scale_suggestion: str
    scale_rationale: str
    exit_suggestion: str
    exit_rationale: str
    
    # Optional fields (with defaults) - MUST BE AT END
    suggested_size_delta: float = 0.0
    suggested_exit_percentage: float = 0.0
```

### Issue: Incorrect Classifications

**Symptom:** Position classified as TOXIC when performing well  
**Debug Steps:**
```bash
# Extract position metrics
cat position_intelligence.json | jq '.positions[0] | {
  symbol,
  current_R,
  peak_R,
  momentum_score,
  time_in_trade_hours,
  classification
}'

# Check classification logic matches criteria
# TOXIC requires: current_R < -0.5 AND time > 30min
```

---

## PERFORMANCE METRICS

### System Health Indicators

```json
{
  "pil_health": {
    "last_run": "2025-11-22T23:49:45Z",
    "execution_time_seconds": 1.2,
    "positions_analyzed": 5,
    "recommendations_generated": 3,
    "errors": 0,
    "status": "HEALTHY"
  }
}
```

### Classification Distribution (Expected)

```
Healthy Portfolio:
├─ STRONG_TREND: 20-30%
├─ SLOW_GRINDER: 30-40%
├─ STALLING: 10-20%
├─ REVERSAL_RISK: 5-10%
└─ TOXIC: <5%

Warning Signs:
├─ TOXIC > 10%: Risk management failure
├─ REVERSAL_RISK > 20%: Market turning
└─ STALLING > 30%: Poor signal quality
```

---

## FUTURE ENHANCEMENTS

### Phase 2: AUTONOMOUS Mode
- Automatic scale-in/out execution
- Automatic exit triggers
- Stop-loss/take-profit adjustments
- Integration with AELM

### Phase 3: Machine Learning
- Predict optimal hold time per symbol
- Learn optimal scale-in/out sizes
- Adaptive classification thresholds
- Symbol-specific momentum models

### Phase 4: Multi-Timeframe Analysis
- Analyze position performance across multiple timeframes
- Detect divergences between timeframes
- Optimize exit timing based on timeframe confluence

### Phase 5: Correlation Analysis
- Detect correlated positions
- Recommend portfolio-level hedges
- Optimize overall portfolio risk

---

## APPENDIX

### Glossary

| Term | Definition |
|------|------------|
| **R** | Risk multiple - position P&L relative to initial risk capital |
| **peak_R** | Highest R-multiple achieved during position lifetime |
| **momentum_score** | R per hour - measures rate of profit generation |
| **CORE symbol** | High-quality, stable symbol (from Universe Control Center) |
| **TOXIC position** | Position with R < -0.5 requiring immediate exit |
| **AELM** | Autonomous Execution Layer Manager (future system) |

### Contact & Support

- **Documentation:** This file (POSITION_INTELLIGENCE_LAYER_GUIDE.md)
- **Source Code:** position_intelligence_layer.py
- **Related Systems:** Risk & Universe Control Center OS

---

**END OF GUIDE**
