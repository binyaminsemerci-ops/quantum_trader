# State Management

## Authoritative State Sources

The system has three categories of authoritative state:

### 1. Positions (`state/positions/`)

- Source of truth: Exchange API (reconciled every 5 seconds)
- Local cache: Redis with TTL
- Backup: PostgreSQL

```python
class PositionState:
    position_id: str
    symbol: str
    side: Literal["LONG", "SHORT"]
    size: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal
    stop_loss: Decimal
    take_profit: Optional[Decimal]
    opened_at: datetime
    last_synced: datetime
```

### 2. Capital (`state/capital/`)

- Source of truth: Exchange wallet balance
- Reconciliation: Every 60 seconds
- Tracked values: Total equity, available margin, used margin

```python
class CapitalState:
    total_equity: Decimal
    available_balance: Decimal
    used_margin: Decimal
    unrealized_pnl: Decimal
    realized_pnl_today: Decimal
    drawdown_from_peak: Decimal
    scaling_level: int  # 1-4
    size_multiplier: Decimal  # Based on drawdown
```

### 3. System Health (`state/system_health/`)

- Aggregated from all services
- Updated: Every 10 seconds
- Critical for pre-flight and monitoring

```python
class SystemHealthState:
    all_services_healthy: bool
    data_quality_score: float
    api_connectivity: bool
    last_trade_time: Optional[datetime]
    active_positions_count: int
    daily_loss_pct: float
    current_drawdown_pct: float
    kill_switch_active: bool
    current_scaling_level: int
    loss_series_count: int
```

## State Persistence

All state changes are:
1. Validated before write
2. Written to primary storage
3. Replicated to backup
4. Logged to audit ledger

## State Recovery

On system restart:
1. Load from persistent storage
2. Reconcile with exchange
3. Resolve any discrepancies
4. Log reconciliation results
