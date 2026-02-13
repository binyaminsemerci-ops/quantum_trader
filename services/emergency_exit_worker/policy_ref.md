# Emergency Exit Worker - Policy Reference

## Authority Chain

```
┌─────────────────────────────────────────────────────────────┐
│                 PANIC CLOSE AUTHORITY                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   AUTHORIZED PUBLISHERS (exhaustive list):                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 1. Risk Kernel        (automatic)                   │  │
│   │ 2. Exit Brain         (fatal health-failure only)   │  │
│   │ 3. Manual Ops Key     (rare, always logged)         │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   NO OTHER SERVICE MAY PUBLISH system.panic_close          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Policy Hierarchy

1. **Emergency Exit Worker** has ABSOLUTE authority to close ANY position
2. No other service can override EEW during panic_close execution
3. EEW does NOT require:
   - Risk Brain approval
   - Strategy Brain approval
   - CEO Brain approval
   - Position sizing check
   - Capital allocation check

## What EEW Ignores

| Input | EEW Response |
|-------|--------------|
| PnL | Ignored |
| Signals | Ignored |
| AI predictions | Ignored |
| Market regime | Ignored |
| Confidence scores | Ignored |
| Strategy state | Ignored |

## Authorization Verification

Before executing, EEW verifies:
1. Event came from authorized stream
2. Event has valid `source` field (risk_kernel / exit_brain / ops)
3. Event has `timestamp` within 60 seconds

## Post-Execution

1. All trading services HALT
2. No position opening allowed
3. Manual inspection required before restart
4. Audit log created with:
   - Trigger source
   - Timestamp
   - Positions closed
   - Positions failed
   - Total execution time

## References

- Quantum Trader Safety Policy v2.0
- Risk Kernel Specification
- System Recovery Procedures
