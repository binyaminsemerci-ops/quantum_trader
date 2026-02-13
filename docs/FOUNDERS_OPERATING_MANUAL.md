# Founders Operating Manual

**Version**: 1.0  
**Audience**: Fund Founders  

---

## Part 1: Identity

### Who We Are

We are a systematic crypto futures trading fund operating with:
- Algorithmic signal generation (AI advisory)
- Rule-based risk management (policy enforcement)
- Automated execution (within human-defined bounds)
- Human oversight (ultimate control)

### Core Philosophy

> "Kapital bevares først. Profitt er et biprodukt."

We are capital preservers who happen to trade. Every decision prioritizes survival.

### Fund Principles

1. **Policy is Supreme** - Code follows policy, never the reverse
2. **AI is Advisory** - Signals inform, they don't decide
3. **Risk is Non-Negotiable** - Limits are hard, not soft
4. **Fail-Closed** - When uncertain, stop
5. **Document Everything** - If it's not logged, it didn't happen

---

## Part 2: Roles & Authority

### Decision Hierarchy

```
Level 0: Kill-Switch (SUPREME)
Level 1: Risk Kernel (VETO)
Level 2: Policy Engine (VETO)
Level 3: Capital Allocation
Level 4: Exit Brain
Level 5: Entry Gate
Level 6: AI Signals (Advisory only)
```

### Human Roles

| Role | Authority | Responsibility |
|------|-----------|----------------|
| **Founder** | Full system access, kill-switch | Final decisions, governance |
| **Operator** | Monitoring, alerts, limited override | Day-to-day oversight |
| **Observer** | Read-only dashboard access | Monitoring only |

### What Humans Decide

- Scaling levels (Level 1→4)
- Policy amendments
- Kill-switch reset
- Override decisions
- System restart authorization

### What Humans Don't Decide

- Individual trade entries (system decides)
- Position sizing (formula decides)
- Exit timing (ExitBrain decides)
- Stop-loss placement (policy decides)

---

## Part 3: Daily Operations

### Morning Checklist

```
□ Check overnight positions
□ Review daily P&L
□ Check drawdown status
□ Verify system health
□ Check no-trade calendar
□ Review any alerts
□ Confirm scaling level appropriate
```

### Pre-Flight (Automated)

The system runs pre-flight automatically before each trading session. Verify it passed.

### Monitoring During Trading

| Metric | Normal | Warning | Critical |
|--------|--------|---------|----------|
| Daily P&L | > -3% | -3% to -5% | < -5% |
| Drawdown | < 5% | 5-10% | > 10% |
| Open Positions | 0-3 | - | > 3 |
| Loss Series | 0-2 | 3-5 | > 5 |

### End of Day

```
□ Review all trades
□ Note any anomalies
□ Check next day's calendar
□ Verify positions appropriate for overnight
□ Confirm no pending issues
```

---

## Part 4: Intervention Protocol

### When to Intervene

**Intervene immediately if:**
- Kill-switch activated unexpectedly
- System unresponsive
- Exchange issues
- Unexpected behavior
- Any Grunnlov violation detected

**DO NOT intervene if:**
- Normal losses within limits
- AI signal disagreement
- "I feel" the market will reverse
- Boredom

### How to Intervene

1. **Assess**: Is this a real issue?
2. **Document**: Note what you observe
3. **Act**: Use appropriate control (monitoring → override → kill-switch)
4. **Review**: Post-intervention analysis

### Override Authority

| Action | Who Can | Requirements |
|--------|---------|--------------|
| View-only monitoring | Anyone | Dashboard access |
| Close single position | Founder | Documented reason |
| Kill-switch (manual) | Founder | Authentication |
| Resume after halt | Founder | Review complete |

---

## Part 5: Emergency Response

### Severity Classification

| Severity | Example | Response Time |
|----------|---------|---------------|
| **Critical** | Kill-switch, system down | < 5 minutes |
| **High** | Unusual losses, data issues | < 30 minutes |
| **Medium** | Warning alerts | < 2 hours |
| **Low** | Informational | End of day |

### Emergency Contacts

```
Primary: [Founder 1 contact]
Secondary: [Founder 2 contact]
Exchange Support: [Exchange contact]
```

### Black Swan Response

1. Kill-switch activates automatically
2. Human verifies all positions closed
3. No immediate action needed
4. Wait for assessment (24h minimum)
5. Follow recovery protocol

---

## Part 6: Day Lifecycle

### Phase 1: Before Market (Pre-Open)

```
Time: 30 minutes before trading
Tasks:
- Run pre-flight check
- Review overnight news
- Check no-trade calendar
- Verify system status
- Confirm capital available
```

### Phase 2: During Market (Active Trading)

```
Tasks:
- Monitor dashboard
- Respond to alerts
- Do NOT interfere with normal operations
- Document any observations
```

### Phase 3: After Market (Post-Close)

```
Tasks:
- Review day's performance
- Check all positions closed or managed
- Review any incidents
- Plan for next day
- Update any documentation
```

---

## Key Numbers to Remember

| Metric | Limit |
|--------|-------|
| Risk per trade | 2% max |
| Daily loss | 5% max |
| Consecutive losses (pause) | 5 |
| Consecutive losses (halt) | 10 |
| Max positions | 3 |
| Max hold time | 72 hours |

---

## Golden Rules

1. **Trust the system** - It's designed for survival
2. **Follow the policy** - No shortcuts
3. **Document everything** - Future you will thank you
4. **Intervene rarely** - But decisively when needed
5. **Survive first** - Profits come after

---

**END OF FOUNDERS OPERATING MANUAL**
