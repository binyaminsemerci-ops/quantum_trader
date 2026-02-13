# Event Flow Architecture – Quantum Trader

> **Infrastructure**: Redis Streams  
> **Pattern**: Event-driven microservices  
> **Policy Reference**: constitution/GOVERNANCE.md

---

## 1. Event-Driven Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EVENT FLOW ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│    │  Market  │     │  Regime  │     │  Signal  │     │  Risk    │            │
│    │   Data   │────▶│ Detector │────▶│ Advisory │────▶│  Kernel  │            │
│    └──────────┘     └──────────┘     └──────────┘     └────┬─────┘            │
│                                                             │                   │
│                                                             │ APPROVE/VETO      │
│                                                             ▼                   │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│    │ Position │◀────│ Execution│◀────│ Position │◀────│  Trade   │            │
│    │  State   │     │  Engine  │     │  Sizer   │     │ Request  │            │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘            │
│                           │                                                     │
│                           │                                                     │
│                           ▼                                                     │
│    ┌──────────┐     ┌──────────┐                                               │
│    │   Exit   │◀────│ Position │                                               │
│    │  Brain   │     │ Monitor  │                                               │
│    └──────────┘     └──────────┘                                               │
│                                                                                 │
│    ════════════════════════════════════════════════════════════════════════    │
│                           REDIS STREAMS                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Redis Streams

### Stream Naming Convention

```
<domain>.<entity>.<action>

Examples:
market.candle.new          → New price candle
regime.classification.update → Regime changed
signals.advisory.new       → New advisory signal
risk.evaluation.complete   → Risk kernel decision
trades.order.filled        → Order executed
position.state.update      → Position changed
system.kill_switch.activated → Kill switch triggered
```

### Stream Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| Market Data | `market.*` | Price, volume, orderbook |
| Regime | `regime.*` | Market regime classification |
| Signals | `signals.*` | Advisory signals |
| Risk | `risk.*` | Risk evaluations, VETO decisions |
| Trades | `trades.*` | Order lifecycle |
| Position | `position.*` | Position state changes |
| Exit | `exit.*` | Exit decisions |
| System | `system.*` | Kill-switch, alerts, health |
| Audit | `audit.*` | Audit trail for compliance |

---

## 3. Stream Definitions

### Market Data Streams

```yaml
market.candle.new:
  description: New price candle received
  publisher: data_collector
  payload:
    symbol: string          # e.g., "BTCUSDT"
    timeframe: string       # e.g., "1m", "5m", "1h"
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
  consumers:
    - regime_detector
    - signal_advisory
    - exit_brain
    - dashboard
  retention: 24h

market.tick.price:
  description: Real-time price update
  publisher: data_collector
  payload:
    symbol: string
    price: float
    timestamp: datetime
  consumers:
    - exit_brain (for stop monitoring)
    - dashboard
  retention: 1h
  rate_limit: 10/second
```

### Regime Streams

```yaml
regime.classification.update:
  description: Market regime has been classified/changed
  publisher: regime_detector
  payload:
    symbol: string
    regime: TRENDING | RANGING | CHAOTIC
    confidence: float       # 0-1
    previous_regime: string
    changed: bool          # Did regime change?
    timestamp: datetime
  consumers:
    - signal_advisory (confidence adjustment)
    - exit_brain (regime exit check)
    - dashboard
  retention: 7d

regime.volatility.spike:
  description: Volatility spike detected
  publisher: regime_detector
  payload:
    symbol: string
    current_volatility: float
    baseline_volatility: float
    spike_ratio: float      # e.g., 2.5x baseline
    timestamp: datetime
  consumers:
    - risk_kernel (may trigger limits)
    - exit_brain (volatility exit)
    - alerting
  retention: 7d
```

### Signal Streams

```yaml
signals.advisory.new:
  description: New advisory signal from Signal Advisory
  publisher: signal_advisory
  payload:
    signal_id: string
    symbol: string
    timestamp: datetime
    edge_score: int         # 0-100
    confidence: float       # 0-1
    regime_fit: bool
    regime: string
    recommended_action: LONG | SHORT | NO_SIGNAL
    reasoning: string
    model_version: string
    execution_power: "NONE" # Always NONE
  consumers:
    - risk_kernel
    - logger
    - dashboard
  retention: 30d

signals.advisory.no_signal:
  description: Signal Advisory declined to emit signal
  publisher: signal_advisory
  payload:
    timestamp: datetime
    reason: string
    gate_failed: string     # Which gate failed
    inputs_summary: object
  consumers:
    - logger
    - dashboard
  retention: 7d
```

### Risk Streams

```yaml
risk.evaluation.complete:
  description: Risk Kernel completed evaluation
  publisher: risk_kernel
  payload:
    evaluation_id: string
    signal_id: string       # Linked to advisory signal
    timestamp: datetime
    decision: APPROVED | VETOED | MODIFIED
    veto_reason: string     # If vetoed
    modifications: object   # If modified (e.g., reduced size)
    risk_dimensions:
      trade_risk_passed: bool
      daily_risk_passed: bool
      drawdown_passed: bool
      system_integrity_passed: bool
    position_allowed:
      symbol: string
      direction: LONG | SHORT
      max_size: float       # Position size limit
      max_risk_usd: float   # Dollar risk limit
  consumers:
    - position_sizer (if approved)
    - logger
    - dashboard
  retention: 90d

risk.kill_switch.activated:
  description: Kill-switch has been triggered
  publisher: risk_kernel | kill_switch_service
  payload:
    timestamp: datetime
    trigger: string         # What caused it
    trigger_value: float    # e.g., -5.2% daily loss
    threshold: float        # e.g., -5% limit
    severity: WARNING | CRITICAL | EMERGENCY
    action_taken: string    # What system did
    human_override_required: bool
  consumers:
    - ALL services (broadcast)
    - alerting (immediate)
    - human_override_lock
  retention: permanent

risk.daily_status.update:
  description: Daily risk metrics update
  publisher: risk_kernel
  payload:
    timestamp: datetime
    daily_pnl_percent: float
    daily_pnl_usd: float
    open_positions: int
    total_exposure_percent: float
    drawdown_from_hwm: float
    risk_budget_remaining: float
  consumers:
    - dashboard
    - alerting
  retention: 365d
```

### Trade Streams

```yaml
trades.request.new:
  description: New trade request created
  publisher: position_sizer
  payload:
    request_id: string
    evaluation_id: string   # Links to risk evaluation
    symbol: string
    direction: LONG | SHORT
    size: float             # Position size
    entry_price: float      # Target entry
    stop_loss: float        # Required stop
    risk_amount: float      # Dollar risk
    risk_percent: float     # % of equity
    timestamp: datetime
  consumers:
    - execution_engine
    - logger
  retention: 90d

trades.order.submitted:
  description: Order submitted to exchange
  publisher: execution_engine
  payload:
    request_id: string
    order_id: string        # Exchange order ID
    symbol: string
    side: BUY | SELL
    type: MARKET | LIMIT
    size: float
    price: float            # Limit price or null
    timestamp: datetime
  consumers:
    - logger
    - dashboard
  retention: 90d

trades.order.filled:
  description: Order filled on exchange
  publisher: execution_engine
  payload:
    request_id: string
    order_id: string
    fill_id: string
    symbol: string
    fill_price: float
    fill_size: float
    fees: float
    timestamp: datetime
  consumers:
    - position_state
    - logger
    - dashboard
  retention: permanent

trades.order.rejected:
  description: Order rejected by exchange
  publisher: execution_engine
  payload:
    request_id: string
    order_id: string
    symbol: string
    reason: string
    timestamp: datetime
  consumers:
    - alerting
    - logger
  retention: 90d
```

### Position Streams

```yaml
position.state.update:
  description: Position state changed
  publisher: position_state_service
  payload:
    position_id: string
    symbol: string
    direction: LONG | SHORT
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    stop_loss: float
    age_seconds: int
    timestamp: datetime
  consumers:
    - exit_brain
    - risk_kernel
    - dashboard
  retention: 90d

position.opened:
  description: New position opened
  publisher: position_state_service
  payload:
    position_id: string
    symbol: string
    direction: LONG | SHORT
    size: float
    entry_price: float
    stop_loss: float
    risk_amount: float
    timestamp: datetime
  consumers:
    - exit_brain (start monitoring)
    - logger
    - dashboard
  retention: permanent

position.closed:
  description: Position fully closed
  publisher: position_state_service
  payload:
    position_id: string
    symbol: string
    direction: LONG | SHORT
    entry_price: float
    exit_price: float
    size: float
    realized_pnl: float
    realized_pnl_percent: float
    hold_duration_seconds: int
    exit_reason: string     # stop, target, time, regime, etc.
    timestamp: datetime
  consumers:
    - risk_kernel (update daily P&L)
    - logger
    - dashboard
  retention: permanent
```

### Exit Streams

```yaml
exit.decision.made:
  description: Exit Brain decided to exit/modify position
  publisher: exit_brain
  payload:
    position_id: string
    symbol: string
    decision: EXIT_FULL | EXIT_PARTIAL | TIGHTEN_STOP | HOLD
    reason: string          # time_exit, regime_exit, volatility_exit, etc.
    exit_percent: float     # If partial, what %
    new_stop: float         # If tightening
    priority: LOW | MEDIUM | HIGH | URGENT
    timestamp: datetime
  consumers:
    - execution_engine
    - logger
    - dashboard
  retention: 90d

exit.stop.triggered:
  description: Stop-loss was hit
  publisher: exit_brain | exchange_monitor
  payload:
    position_id: string
    symbol: string
    trigger_price: float
    stop_price: float
    timestamp: datetime
  consumers:
    - execution_engine (if not exchange-native stop)
    - logger
    - dashboard
  retention: permanent
```

### System Streams

```yaml
system.health.heartbeat:
  description: Service health heartbeat
  publisher: all_services
  payload:
    service: string
    status: HEALTHY | DEGRADED | UNHEALTHY
    uptime_seconds: int
    last_activity: datetime
    metrics: object
    timestamp: datetime
  consumers:
    - health_monitor
    - alerting
  retention: 24h
  interval: 30s

system.alert.triggered:
  description: System alert raised
  publisher: alerting_service
  payload:
    alert_id: string
    severity: INFO | WARNING | ERROR | CRITICAL
    source: string
    message: string
    details: object
    timestamp: datetime
  consumers:
    - dashboard
    - notification_service
  retention: 30d
```

### Audit Streams

```yaml
audit.decision.log:
  description: Audit log of all system decisions
  publisher: all_decision_services
  payload:
    decision_id: string
    service: string
    decision_type: string
    input_summary: object
    output: object
    reasoning: string
    policy_references: list
    timestamp: datetime
  consumers:
    - audit_storage
    - compliance_reports
  retention: permanent
  immutable: true
```

---

## 4. Consumer Groups

### Stream Consumer Groups

```yaml
consumer_groups:
  # Risk Kernel must process every signal
  risk_evaluation:
    stream: signals.advisory.new
    consumers:
      - risk_kernel_1
      - risk_kernel_2  # Failover
    ack_required: true
    retry_on_failure: true
    
  # Exit Brain monitors all positions
  exit_monitoring:
    stream: position.state.update
    consumers:
      - exit_brain_1
    ack_required: true
    
  # Execution must process all approved trades
  trade_execution:
    stream: risk.evaluation.complete
    filter: decision == "APPROVED"
    consumers:
      - execution_engine_1
    ack_required: true
    
  # Broadcast to all on kill-switch
  kill_switch_broadcast:
    stream: risk.kill_switch.activated
    consumers: ALL
    broadcast: true
```

---

## 5. Event Flow Examples

### Trade Lifecycle

```
1. market.candle.new
   └─▶ regime_detector processes

2. regime.classification.update
   └─▶ signal_advisory receives

3. signals.advisory.new
   └─▶ risk_kernel evaluates

4. risk.evaluation.complete (APPROVED)
   └─▶ position_sizer calculates size

5. trades.request.new
   └─▶ execution_engine submits

6. trades.order.submitted
   └─▶ waiting for fill

7. trades.order.filled
   └─▶ position_state updates

8. position.opened
   └─▶ exit_brain starts monitoring

9. [Time passes, price moves]

10. position.state.update
    └─▶ exit_brain evaluates

11. exit.decision.made (EXIT_FULL)
    └─▶ execution_engine closes

12. trades.order.filled (close order)
    └─▶ position_state updates

13. position.closed
    └─▶ risk_kernel updates daily P&L
```

### Kill-Switch Activation

```
1. position.closed (with large loss)
   └─▶ risk_kernel updates daily P&L

2. risk.daily_status.update
   daily_pnl_percent: -5.2%
   └─▶ risk_kernel checks threshold

3. risk.kill_switch.activated
   trigger: "daily_loss_exceeded"
   trigger_value: -5.2%
   threshold: -5.0%
   └─▶ BROADCAST to all services

4. All services receive kill-switch
   - execution_engine: Cancel all pending orders
   - exit_brain: Exit all positions
   - signal_advisory: Stop generating signals
   - dashboard: Display kill-switch status

5. system.alert.triggered
   severity: CRITICAL
   message: "Kill-switch activated: Daily loss -5.2%"
   └─▶ notification_service sends alerts
```

---

## 6. Redis Configuration

### Stream Settings

```yaml
redis_streams:
  # Default stream configuration
  defaults:
    maxlen: 10000           # Max entries per stream
    approximate: true       # Use ~ for performance
    
  # High-volume streams
  market.candle.new:
    maxlen: 50000
    
  # Long-retention streams
  audit.decision.log:
    maxlen: 0               # No trimming - permanent
    
  # Real-time streams
  market.tick.price:
    maxlen: 1000            # Short retention
```

### Consumer Configuration

```yaml
consumer_config:
  # Retry settings
  retry:
    max_attempts: 3
    backoff_ms: [100, 500, 2000]
    dead_letter_after: 3
    
  # Timeout settings
  read_timeout_ms: 5000
  processing_timeout_ms: 30000
  
  # Batch settings
  batch_size: 10
  batch_timeout_ms: 100
```

---

## 7. Monitoring

### Stream Health Metrics

```yaml
metrics:
  stream_length:
    description: Number of entries in stream
    alert_threshold: 50000
    
  consumer_lag:
    description: How far behind consumer is
    alert_threshold: 100 entries
    
  processing_rate:
    description: Events processed per second
    
  ack_rate:
    description: Acknowledgment rate
    alert_threshold: < 99%
    
  retry_rate:
    description: Events requiring retry
    alert_threshold: > 1%
```

---

*Events er systemets nervesystem. Healthy events = healthy system.*
