# Exit Brain — Decision Flow

**Version**: 1.0  
**Type**: Continuous Monitoring  

---

## Decision Flowchart

```
         ┌──────────────────────────────────┐
         │    Every Position (Continuous)   │
         └───────────────┬──────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │     Kill-Switch Check            │
         │     (From Risk Kernel)           │
         └───────────────┬──────────────────┘
                         │
         ┌───────YES─────┴──────NO──────────┐
         ▼                                   ▼
  ┌──────────────┐                 ┌─────────────────────┐
  │ CLOSE ALL    │                 │ Check Capital Stress│
  │ (Immediate)  │                 └──────────┬──────────┘
  └──────────────┘                            │
                                   ┌──────────┴──────────┐
                              CRITICAL           NORMAL
                                   ▼                  ▼
                           ┌──────────────┐  ┌─────────────────┐
                           │ CLOSE/REDUCE │  │ Check Regime    │
                           └──────────────┘  └────────┬────────┘
                                                      │
                                          ┌───────────┴───────────┐
                                      CHANGED               SAME
                                          ▼                    ▼
                                   ┌──────────────┐  ┌─────────────────┐
                                   │ PARTIAL/FULL │  │ Check Volatility│
                                   │    EXIT      │  └────────┬────────┘
                                   └──────────────┘           │
                                                   ┌──────────┴──────────┐
                                               SPIKE              NORMAL
                                                   ▼                  ▼
                                            ┌──────────────┐  ┌─────────────────┐
                                            │ TIGHTEN/EXIT │  │ Check Time      │
                                            └──────────────┘  └────────┬────────┘
                                                                       │
                                                           ┌───────────┴───────────┐
                                                       EXCEEDED              WITHIN
                                                           ▼                    ▼
                                                    ┌──────────────┐  ┌─────────────────┐
                                                    │ TIME EXIT    │  │ Check Stop-Loss │
                                                    └──────────────┘  └────────┬────────┘
                                                                               │
                                                                    ┌──────────┴──────────┐
                                                                   HIT               OK
                                                                    ▼                  ▼
                                                             ┌──────────────┐  ┌───────────┐
                                                             │  STOP EXIT   │  │ CONTINUE  │
                                                             └──────────────┘  │ MONITORING│
                                                                               └───────────┘
```

---

## Core Logic

```python
class ExitBrain:
    def evaluate_position(self, position: Position) -> ExitDecision:
        """
        Continuous position monitoring.
        Runs every tick (or at minimum every 5 seconds).
        """
        
        # 1. Kill-switch check (highest priority)
        if self.is_kill_switch_active():
            return ExitDecision(
                action="CLOSE_FULL",
                urgency="IMMEDIATE",
                reason="Kill-switch active"
            )
        
        # 2. Capital stress
        stress = self.check_capital_stress(position)
        if stress and stress.urgency == "CRITICAL":
            return stress
        
        # 3. Regime change (most important for survival)
        regime = self.check_regime_exit(position)
        if regime:
            return regime
        
        # 4. Volatility spike
        vol = self.check_volatility_exit(position)
        if vol and vol.urgency in ["CRITICAL", "HIGH"]:
            return vol
        
        # 5. Time exit
        time = self.check_time_exit(position)
        if time:
            return time
        
        # 6. Stop-loss check
        stop = self.check_stop_loss(position)
        if stop:
            return stop
        
        # 7. Apply trailing stop update (if in profit)
        self.update_trailing_stop(position)
        
        # 8. Check take-profit levels
        tp = self.check_take_profit(position)
        if tp:
            return tp
        
        # 9. Handle minor stress/vol by tightening (non-exit)
        if stress:
            self.apply_stress_response(position, stress)
        if vol:
            self.apply_vol_response(position, vol)
        
        return ExitDecision(action="HOLD")
```

---

## Event Flow

### Input Events

```yaml
position.opened:
  # Start monitoring this position
  source: position_tracker
  payload:
    position_id: "uuid"
    symbol: "BTCUSDT"
    entry_price: 50000.0
    entry_regime: "TRENDING_UP"
    entry_volatility: 0.02

market.regime.updated:
  # Regime change detection
  source: market_regime
  payload:
    symbol: "BTCUSDT"
    previous_regime: "TRENDING_UP"
    current_regime: "RANGING"

volatility.updated:
  # Volatility change
  source: market_regime
  payload:
    symbol: "BTCUSDT"
    current_volatility: 0.045
    volatility_percentile: 85

capital.stress.updated:
  # Portfolio stress change
  source: risk_kernel
  payload:
    daily_pnl_pct: -0.035
    drawdown_pct: 0.09
```

### Output Events

```yaml
reduce.intent:
  # Partial exit
  target: execution
  payload:
    position_id: "uuid"
    reduce_percent: 0.50
    reason: "Regime changed"
    urgency: "HIGH"

close.intent:
  # Full exit
  target: execution
  payload:
    position_id: "uuid"
    reason: "Time limit exceeded"
    urgency: "HIGH"

exit.completed:
  # Acknowledgment after execution
  source: exit_brain
  payload:
    position_id: "uuid"
    exit_type: "REGIME"
    pnl: -150.00
```

---

## Trailing Stop Logic

```python
def update_trailing_stop(self, position: Position):
    """
    Updates trailing stop when position is in profit.
    Stop can only tighten, never widen.
    """
    
    # Calculate current profit
    if position.is_long:
        current_pnl = (current_price - position.entry_price) / position.entry_price
        high_water = position.high_water_mark or position.entry_price
    else:
        current_pnl = (position.entry_price - current_price) / position.entry_price
        high_water = position.low_water_mark or position.entry_price
    
    # Only trail when in profit
    if current_pnl <= 0:
        return
    
    # Update high water mark
    if position.is_long:
        if current_price > high_water:
            position.high_water_mark = current_price
            high_water = current_price
    else:
        if current_price < high_water:
            position.low_water_mark = current_price
            high_water = current_price
    
    # Calculate new stop
    trail_distance = self.get_trail_distance(current_pnl, position.regime)
    
    if position.is_long:
        new_stop = high_water * (1 - trail_distance)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
            self.emit_stop_updated(position)
    else:
        new_stop = high_water * (1 + trail_distance)
        if new_stop < position.stop_loss:
            position.stop_loss = new_stop
            self.emit_stop_updated(position)
```

---

## Monitoring Frequency

| Condition | Check Frequency |
|-----------|-----------------|
| Normal market | Every 5 seconds |
| High volatility | Every 1 second |
| Near stop-loss | Every 500ms |
| Kill-switch check | Every tick |

---

## Logging

Every decision logs:

```json
{
    "decision_id": "uuid",
    "position_id": "uuid",
    "timestamp": "ISO8601",
    "action": "CLOSE_PARTIAL",
    "reason": "Regime changed: TRENDING_UP → RANGING",
    "dimension": "REGIME",
    "urgency": "HIGH",
    "close_percent": 0.50,
    "position_pnl_before": 2.5,
    "metrics": {
        "time_in_trade_hours": 14.5,
        "entry_regime": "TRENDING_UP",
        "current_regime": "RANGING",
        "entry_volatility": 0.02,
        "current_volatility": 0.03
    }
}
```
