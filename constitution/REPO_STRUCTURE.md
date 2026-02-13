# 📁 REPOSITORY STRUCTURE (POLICY → CODE MAPPING)

**Document**: Policy-to-Code Architecture  
**Authority**: Derived from FUND_POLICY.md  
**Version**: 1.0  

---

> *"Hver mappe = én policy-artikkel i praksis"*

---

## STRUCTURE OVERVIEW

```
quantum_trader/
│
├── constitution/                    # 📜 POLICY (SUPREME AUTHORITY)
│   ├── FUND_POLICY.md              # Master policy document
│   ├── RISK_POLICY.md              # Detailed risk rules (optional expansion)
│   ├── EXIT_POLICY.md              # Detailed exit rules (optional expansion)
│   └── REPO_STRUCTURE.md           # This document
│
├── services/                        # 🔧 CORE SERVICES (1:1 with policy sections)
│   │
│   ├── policy_engine/              # Section 3: Governance
│   │   ├── grunnlover.py           # 15 constitutional laws
│   │   ├── decision_hierarchy.py   # VETO chain
│   │   └── enforcement.py          # Violation handling
│   │
│   ├── risk_kernel/                # Section 4: Risk Management
│   │   ├── position_limits.py      # Max risk per trade
│   │   ├── daily_limits.py         # Max daily loss
│   │   ├── circuit_breakers.py     # Drawdown stages
│   │   └── margin_safety.py        # Margin requirements
│   │
│   ├── market_regime/              # Section 5: Trading conditions
│   │   ├── regime_detector.py      # Trend/chop/volatile
│   │   ├── liquidity_monitor.py    # Liquidity assessment
│   │   └── funding_monitor.py      # Funding rate extremes
│   │
│   ├── data_integrity/             # Section 5: Data validation
│   │   ├── validator.py            # Data consistency checks
│   │   ├── reconciliation.py       # Position reconciliation
│   │   └── gap_detector.py         # Data gap handling
│   │
│   ├── capital_allocation/         # Section 7: Capital Policy
│   │   ├── position_sizing.py      # Size calculations
│   │   ├── leverage_policy.py      # Leverage limits
│   │   ├── scaling_levels.py       # 0-3 scaling tiers
│   │   └── auto_scale.py           # Up/down scaling logic
│   │
│   ├── entry_gate/                 # Section 5: Entry qualification
│   │   ├── qualification.py        # Entry requirements
│   │   ├── pre_conditions.py       # All conditions check
│   │   └── entry_blocker.py        # Block on violations
│   │
│   ├── exit_brain/                 # Section 6: Exit Policy
│   │   ├── exit_types.py           # 5 exit formulas
│   │   ├── priority_manager.py     # Exit priority (1-4)
│   │   ├── stop_loss.py            # Stop-loss management
│   │   └── partial_exit.py         # Partial exit logic
│   │
│   ├── execution/                  # Section 5: Execution rules
│   │   ├── order_manager.py        # Order handling
│   │   ├── slippage_monitor.py     # Slippage tracking
│   │   └── retry_policy.py         # Retry logic
│   │
│   ├── audit_ledger/               # Section 3: Audit requirements
│   │   ├── trade_log.py            # Trade logging
│   │   ├── decision_log.py         # Decision audit
│   │   └── immutable_store.py      # Append-only storage
│   │
│   ├── human_override_lock/        # Section 3: Override policy
│   │   ├── override_rules.py       # What's allowed
│   │   ├── protection.py           # User protection
│   │   └── emotion_detector.py     # Behavioral detection
│   │
│   └── signal_ai/                  # Section 3: AI (Advisory only)
│       ├── signal_generator.py     # Generates suggestions
│       ├── confidence_score.py     # Confidence levels
│       └── advisory_only.py        # NEVER executes
│
├── ops/                            # 🛠️ OPERATIONS (Sections 5, 8, 9)
│   │
│   ├── pre_flight/                 # Section 5.1: Pre-flight checklist
│   │   ├── checklist.py            # 22 checks
│   │   ├── system_health.py        # Health verification
│   │   └── go_no_go.py             # Final decision
│   │
│   ├── no_trade/                   # Section 5.2-5.4: No-trade days
│   │   ├── absolute.py             # Absolute no-trade
│   │   ├── conditional.py          # Observer-only
│   │   └── human_protection.py     # User protection
│   │
│   ├── kill_switch/                # Section 8.2: Kill-switch
│   │   ├── manual.py               # Human trigger
│   │   ├── automatic.py            # System trigger
│   │   └── effects.py              # What happens
│   │
│   ├── restart_protocol/           # Section 8.4: Restart
│   │   ├── phases.py               # 6 phases
│   │   ├── validation.py           # Pre-restart checks
│   │   └── graduated_start.py      # Slow restart
│   │
│   └── incident_response/          # Section 8: Incidents
│       ├── classification.py       # A-E severity
│       ├── black_swan.py           # Black swan playbook
│       └── documentation.py        # Incident logging
│
├── tests/                          # 🧪 TESTING (Section 9)
│   │
│   ├── failure_scenarios/          # Section 8: 14 scenarios
│   │   ├── test_class_a.py         # Critical
│   │   ├── test_class_b.py         # Severe
│   │   ├── test_class_c.py         # Major
│   │   ├── test_class_d.py         # Moderate
│   │   └── test_class_e.py         # Minor
│   │
│   ├── stress_tests/               # Section 8: Stress testing
│   │   ├── test_10_losses.py       # 10 consecutive losses
│   │   ├── test_100_trades.py      # Statistical simulation
│   │   └── test_black_swan.py      # Extreme scenarios
│   │
│   ├── shadow_mode/                # Section 9.2: Shadow testing
│   │   ├── shadow_runner.py        # Run without execution
│   │   ├── comparison.py           # Compare to live
│   │   └── validation.py           # Validate behavior
│   │
│   └── integration/                # Integration tests
│       ├── test_full_flow.py       # End-to-end
│       ├── test_veto_chain.py      # VETO hierarchy
│       └── test_kill_switch.py     # Emergency halt
│
├── config/                         # ⚙️ CONFIGURATION
│   ├── limits.yaml                 # Risk limits (from Section 4)
│   ├── exits.yaml                  # Exit parameters (from Section 6)
│   ├── scaling.yaml                # Scaling levels (from Section 7)
│   └── exchanges.yaml              # Exchange config (from Section 1.2)
│
├── monitoring/                     # 📊 MONITORING
│   ├── dashboards/                 # Visual dashboards
│   ├── alerts/                     # Alert configurations
│   └── metrics/                    # System metrics
│
└── docs/                           # 📚 DOCUMENTATION
    ├── policy_mapping.md           # Policy → Code mapping
    ├── decision_tree.md            # Decision flow diagrams
    └── runbooks/                   # Operational runbooks
```

---

## POLICY → SERVICE MAPPING

| Policy Section | Service | Key Files |
|----------------|---------|-----------|
| §1 Fund Mandate | config/ | limits.yaml, exchanges.yaml |
| §2 Investment Philosophy | policy_engine/ | grunnlover.py |
| §3 Governance | policy_engine/, audit_ledger/, human_override_lock/ | decision_hierarchy.py |
| §4 Risk Management | risk_kernel/ | circuit_breakers.py, margin_safety.py |
| §5 Trading & Execution | entry_gate/, execution/, ops/no_trade/ | qualification.py |
| §6 Exit Policy | exit_brain/ | exit_types.py, priority_manager.py |
| §7 Capital Allocation | capital_allocation/ | scaling_levels.py |
| §8 Incidents | ops/kill_switch/, ops/incident_response/ | black_swan.py |
| §9 Change Management | tests/shadow_mode/ | shadow_runner.py |

---

## SERVICE COMMUNICATION

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EVENT FLOW (REDIS STREAMS)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MARKET DATA → data_integrity → policy_engine → risk_kernel                 │
│                                       │              │                       │
│                                       ↓              ↓                       │
│  signal_ai (ADVISORY) ────────→ entry_gate ←─── capital_allocation          │
│                                       │                                      │
│                                       ↓                                      │
│                                  execution → exit_brain                      │
│                                       │           │                          │
│                                       ↓           ↓                          │
│                               audit_ledger ←─────┘                           │
│                                                                              │
│  VETO CHAIN: kill_switch → risk_kernel → policy_engine → human_override     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PORT ASSIGNMENTS

| Service | Port | Protocol |
|---------|------|----------|
| policy_engine | 8001 | FastAPI |
| risk_kernel | 8002 | FastAPI |
| market_regime | 8003 | FastAPI |
| data_integrity | 8004 | FastAPI |
| capital_allocation | 8005 | FastAPI |
| entry_gate | 8006 | FastAPI |
| exit_brain | 8007 | FastAPI |
| execution | 8008 | FastAPI |
| audit_ledger | 8009 | FastAPI |
| human_override_lock | 8010 | FastAPI |
| signal_ai | 8011 | FastAPI |

---

## GRUNNLOVER → CODE MAPPING

| Grunnlov # | Law | Implementation |
|------------|-----|----------------|
| §1 | Max risk per trade | risk_kernel/position_limits.py |
| §2 | Daily loss halt | risk_kernel/daily_limits.py |
| §3 | Never add to loser | entry_gate/entry_blocker.py |
| §4 | Emergency liquidation | risk_kernel/margin_safety.py |
| §5 | Override AI on violation | policy_engine/enforcement.py |
| §6 | Exit on data gap | data_integrity/gap_detector.py |
| §7 | Flat on extreme funding | market_regime/funding_monitor.py |
| §8 | Circuit breakers | risk_kernel/circuit_breakers.py |
| §9 | Pre-flight required | ops/pre_flight/go_no_go.py |
| §10 | Kill-switch always on | ops/kill_switch/manual.py |
| §11 | Exit never blocked | exit_brain/exit_types.py |
| §12 | Position = evidence | data_integrity/reconciliation.py |
| §13 | Slippage pause | execution/slippage_monitor.py |
| §14 | Exchange unstable = flat | market_regime/liquidity_monitor.py |
| §15 | Log everything | audit_ledger/immutable_store.py |

---

## DEVELOPMENT PRINCIPLES

### 1. Code Follows Policy
Every line of code must trace back to a policy section.  
If code cannot be traced, it should not exist.

### 2. Single Responsibility
Each service handles one policy concern.  
No service should span multiple policy sections.

### 3. VETO Must Flow Up
Lower services cannot override higher services.  
The hierarchy is enforced at code level.

### 4. Immutability
Audit logs are append-only.  
Configuration changes require restart.

### 5. Fail-Closed Default
All error handlers default to safety.  
Unknown states trigger halt.

---

## FILE NAMING CONVENTIONS

| Type | Pattern | Example |
|------|---------|---------|
| Service main | `{service_name}/main.py` | risk_kernel/main.py |
| Business logic | `{function}.py` | circuit_breakers.py |
| Tests | `test_{function}.py` | test_circuit_breakers.py |
| Config | `{domain}.yaml` | limits.yaml |
| Docs | `{topic}.md` | decision_tree.md |

---

**END OF REPOSITORY STRUCTURE DOCUMENT**

*This structure is derived from FUND_POLICY.md v1.0*  
*Any structural changes require policy review first*
