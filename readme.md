# Quantum Trader - Hedge Fund Trading System (Crypto Futures)

## Purpose

This repository contains a **policy-driven, risk-first, exit-dominant**
hedge fund trading system for crypto futures markets.

The system is designed to:
- Survive adverse market conditions
- Fail closed under uncertainty
- Prioritize exits over entries
- Prevent human emotional interference
- Scale only after proven stability

> **Profit is a byproduct, not a goal.**

---

## Core Principles

```
SURVIVAL > PROFIT
DISCIPLINE > OPPORTUNITY
EXIT > ENTRY
RISK > RETURN
SYSTEM > HUMAN
```

### Decision Hierarchy
```
Risk > Capital > Execution > Entry > AI
```

### Fundamental Truths
- Exit is more important than entry
- Flat is a valid position
- Cash is a strategy
- The system must stop itself

---

## Repository Structure

```
quantum_trader/
│
├── constitution/           # FUND POLICIES (SUPREME AUTHORITY)
│   ├── FUND_POLICY.md     # Master policy document
│   ├── GOVERNANCE.md      # Decision hierarchy & Grunnlover
│   ├── RISK_POLICY.md     # Risk limits & circuit breakers
│   ├── EXIT_POLICY.md     # Exit rules & formulas
│   ├── CAPITAL_POLICY.md  # Position sizing & scaling
│   ├── NO_TRADE_RULES.md  # When NOT to trade
│   └── KILL_SWITCH_POLICY.md # Emergency procedures
│
├── services/              # MICROSERVICES (1 service = 1 policy)
│   ├── policy_engine/     # Constitutional enforcement
│   ├── risk_kernel/       # Position safety (VETO power)
│   ├── market_regime/     # Regime detection
│   ├── data_integrity/    # Data validation
│   ├── capital_allocation/# Size & leverage
│   ├── signal_advisory/   # AI signals (ADVISORY ONLY)
│   ├── entry_gate/        # Entry qualification
│   ├── exit_brain/        # Exit execution
│   ├── execution/         # Order management
│   ├── audit_ledger/      # Immutable logging
│   └── human_override_lock/ # User protection
│
├── ops/                   # OPERATIONS (Keeps you alive)
│   ├── pre_flight/        # Pre-trading checklist
│   ├── no_trade_calendar/ # Forbidden trading days
│   ├── restart_protocol/  # Post-halt restart
│   └── black_swan_playbook/ # Extreme event response
│
├── tests/                 # TESTING (Proves system stops)
│   ├── failure_scenarios/ # 14 failure types
│   ├── stress_tests/      # 10-loss, 100-trade sims
│   ├── kill_switch_tests/ # Emergency halt validation
│   └── shadow_mode/       # Paper trading validation
│
├── events/                # EVENT SYSTEM (Nervous system)
│   ├── schemas/           # Event definitions
│   └── channels.md        # Pub/sub configuration
│
├── state/                 # AUTHORITATIVE STATE (Truth, not cache)
│   ├── positions/         # Current positions
│   ├── capital/           # Capital status
│   └── system_health/     # Health metrics
│
├── docs/                  # DOCUMENTATION (For humans)
│   ├── FOUNDERS_OPERATING_MANUAL.md
│   ├── DAY_0_HANDBOOK.md
│   └── RISK_DISCLOSURE.md
│
└── config/                # CONFIGURATION
    ├── limits.yaml        # Risk parameters
    ├── exchanges.yaml     # Exchange configuration
    └── scaling.yaml       # Scaling thresholds
```

---

## What This System IS

- A risk-controlled systematic framework
- A capital preservation system that may generate returns
- A rule-based execution engine
- An architecture designed to survive adverse conditions
- A fund that expects losses and is designed to endure them

## What This System Is NOT

- Not a discretionary trading system
- Not a prediction engine
- Not an indicator playground
- Not optimized for short-term PnL
- Not a "get rich quick" scheme

---

## Go-Live Philosophy

This system is considered **live-ready** only when:

1. All kill-switches are tested
2. Exits work without entries
3. 10 consecutive losses are survivable
4. No manual overrides are possible (except emergency)
5. Shadow mode shows positive expectancy
6. Pre-flight checklist passes 100%

---

## Key Documents

| Document | Location | Purpose |
|----------|----------|---------|
| Fund Policy | `/constitution/FUND_POLICY.md` | Supreme authority |
| 15 Grunnlover | `/constitution/GOVERNANCE.md` | Constitutional laws |
| Risk Limits | `/constitution/RISK_POLICY.md` | Risk parameters |
| Exit Rules | `/constitution/EXIT_POLICY.md` | Exit formulas |
| Operating Manual | `/docs/FOUNDERS_OPERATING_MANUAL.md` | How to run the fund |

---

## Correct Workflow

```
1. Change policy (rarely)     -> constitution/
2. Update ONE service         -> services/
3. Run failure tests          -> tests/
4. Shadow mode (7-30 days)    -> tests/shadow_mode/
5. Then and only then: LIVE
```

## WRONG Workflow (Avoid)

- Change code without policy update
- "Just a small adjustment"
- Optimize during drawdown
- Add features without tests
- Skip shadow mode

---

## Final Note

> *If the system ever feels "too conservative", it is likely working correctly.*

This system is designed to **survive first**. Returns are secondary.

The goal is not to maximize profit.
The goal is to **still be trading in 10 years**.

---

**Document Version**: 1.0
**Classification**: Fund-Grade
**Authority**: Derived from constitution/FUND_POLICY.md
