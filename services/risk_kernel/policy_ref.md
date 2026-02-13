# Risk Kernel — Policy Reference

**Service**: risk_kernel  
**Port**: 8002  
**Authority**: Level 1 (VETO)  

---

## Constitutional Authority

This service derives authority from:

| Grunnlov | Requirement | Implementation |
|----------|-------------|----------------|
| §1 | Max 2% risk per trade | `validate_trade_risk()` |
| §2 | Max 5% daily loss | `check_daily_loss()` |
| §7 | Fail-Closed default | `fail_closed_handler()` |
| §10 | Kill-switch authority | `trigger_kill_switch()` |

---

## Mandate (Non-Negotiable)

The Risk Kernel:

1. **Approves or rejects** every trade
2. **Can halt** the entire system
3. **Can NEVER be overridden** by any other service
4. **When uncertain → NO**

---

## Policy Binding

```
/constitution/RISK_POLICY.md    → This service enforces
/constitution/GOVERNANCE.md     → §1, §2, §7, §10
/constitution/KILL_SWITCH_POLICY.md → Kill-switch execution
```

---

## Decision Authority

| Decision | Risk Kernel Power |
|----------|-------------------|
| Approve trade | YES |
| Reject trade | YES (final) |
| Request more info | NO (fail-closed) |
| Trigger kill-switch | YES (exclusive) |
| Override kill-switch | NO (human only) |

---

## Immutable Rules

1. Missing data = REJECT
2. Unknown state = REJECT
3. Calculation error = REJECT
4. Timeout = REJECT
5. Any doubt = REJECT

**There is no "proceed with caution" option.**
