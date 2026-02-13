# Event System

## Event-Driven Architecture

All services communicate via events. No direct service-to-service calls.

## Event Channels

| Channel | Purpose | Producers | Consumers |
|---------|---------|-----------|-----------|
| `trade.proposed` | New trade request | signal_advisory | policy_engine |
| `policy.approved` | Trade passed policy | policy_engine | risk_kernel |
| `policy.rejected` | Trade rejected by policy | policy_engine | audit_ledger |
| `policy.veto` | Policy VETO | policy_engine | audit_ledger |
| `risk.approved` | Trade passed risk | risk_kernel | capital_allocation |
| `risk.veto` | Risk VETO | risk_kernel | audit_ledger |
| `risk.alert` | Risk warning | risk_kernel | ALL |
| `capital.allocated` | Capital assigned | capital_allocation | entry_gate |
| `entry.approved` | Entry allowed | entry_gate | execution |
| `entry.denied` | Entry blocked | entry_gate | audit_ledger |
| `exit.decision` | Exit triggered | exit_brain | execution |
| `order.submitted` | Order sent | execution | audit_ledger |
| `order.filled` | Order executed | execution | position_tracker |
| `position.opened` | New position | position_tracker | exit_brain |
| `position.closed` | Position closed | position_tracker | capital_allocation |
| `position.pnl.update` | P&L change | position_tracker | risk_kernel |
| `kill_switch.activated` | Emergency stop | human_override_lock | ALL |
| `regime.change` | Market regime shift | market_regime | exit_brain |
| `data.integrity.fail` | Data quality issue | data_integrity | kill_switch |

## Event Format

```json
{
    "event_id": "uuid",
    "event_type": "trade.proposed",
    "timestamp": "ISO8601",
    "source_service": "signal_advisory",
    "payload": {
        // Event-specific data
    },
    "correlation_id": "tracking-uuid"
}
```

## Event Guarantees

- At-least-once delivery
- Events are persisted before acknowledgment
- Out-of-order handling supported
- Idempotent consumers
