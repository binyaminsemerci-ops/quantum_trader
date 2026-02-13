# Exit / Harvest Brain — Policy Reference

**Service**: exit_brain  
**Port**: 8008  
**Authority**: Level 4 (Autonomous for Exits)  

---

## Constitutional Authority

This service derives authority from:

| Grunnlov | Requirement | Implementation |
|----------|-------------|----------------|
| §3 | Exit-dominant system | `exit_brain` owns ALL exits |
| §4 | Trailing stop logic | `update_trailing_stop()` |
| §5 | Mandatory stop-loss | `enforce_stop_loss()` |
| §14 | Regime dictates exit | `regime_exit_check()` |

---

## Mandate (Ufravikelig)

The Exit / Harvest Brain:

1. **OWNS** all position exits
2. **CAN override** entry, signal, and AI on exit decisions
3. **PRIORITIZES** capital health over maximum profit
4. **CANNOT** open new trades or increase position size

> **Entry gives you the right to lose. Exit determines HOW MUCH.**

---

## Policy Binding

```
/constitution/EXIT_POLICY.md    → This service enforces
/constitution/RISK_POLICY.md    → Drawdown responses
/constitution/GOVERNANCE.md     → §3, §4, §5, §14
```

---

## Decision Authority

| Decision | Exit Brain Power |
|----------|------------------|
| Close position | YES (autonomous) |
| Partial exit | YES (autonomous) |
| Tighten stop | YES (autonomous) |
| Override AI exit suggestion | YES |
| Open new position | NO |
| Widen stop | NO |
| Increase position | NO |

---

## Hierarchy Position

```
Risk Kernel (VETO)
      ↓
Exit Brain (AUTONOMOUS for exits)
      ↓
Entry Gate (NO influence on exits)
      ↓  
AI Advisory (suggestions only)
```

Exit Brain listens to Risk Kernel.  
Exit Brain ignores Entry Gate and AI for exit decisions.

---

## Exit vs Entry Priority

| Situation | Winner |
|-----------|--------|
| AI says hold, Exit Brain says close | EXIT BRAIN |
| Entry Gate approved, Exit Brain disagrees | EXIT BRAIN |
| Risk Kernel says close | RISK KERNEL (above Exit Brain) |
| Human override | HUMAN (via kill-switch) |
