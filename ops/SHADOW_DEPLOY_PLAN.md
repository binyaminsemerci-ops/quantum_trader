# Shadow Deploy Plan – Quantum Trader

> **Purpose**: Safe deployment methodology from development to production  
> **Key Principle**: Prove safety before committing capital  
> **Policy Reference**: constitution/GOVERNANCE.md, constitution/RISK_POLICY.md

---

## 1. Shadow Deploy Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SHADOW DEPLOY PHILOSOPHY                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    "We don't trust code. We trust evidence."                                    │
│                                                                                 │
│    Every change that affects trading decisions must PROVE its safety            │
│    before being trusted with real capital.                                      │
│                                                                                 │
│    Shadow mode = Run real logic on real market data without real money          │
│                                                                                 │
│    Progression:                                                                 │
│    Phase 0: Observation Only (watch, don't decide)                              │
│    Phase 1: Dry-Run (decide, don't execute)                                     │
│    Phase 2: Micro-Capital (execute with minimal risk)                           │
│    Phase 3: Production (full deployment)                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deployment Phases

### Phase 0: Observation Mode

**Duration:** Minimum 24 hours (risk-critical: 48 hours)

**Purpose:** Verify the system can:
- Receive and process market data
- Communicate with all services
- Generate logs without errors
- NOT affect production in any way

**Configuration:**
```yaml
shadow_phase_0:
  mode: OBSERVATION
  
  enabled:
    - Market data ingestion
    - Regime detection
    - Signal generation (logged only)
    - Health monitoring
    
  disabled:
    - Risk evaluation (real decisions)
    - Position sizing
    - Order execution
    - Any production Redis streams
    
  isolation:
    network: shadow_network     # Separate Docker network
    redis: shadow_redis:6380    # Different Redis instance
    database: shadow_db         # Separate database
    
  outputs:
    - Shadow-only logs
    - Metrics to shadow Grafana
```

**Success Criteria:**
```
[ ] No errors in logs for 24h
[ ] All services healthy
[ ] Market data processing normally
[ ] Signal generation rate reasonable
[ ] No production impact detected
```

---

### Phase 1: Dry-Run Mode

**Duration:** Minimum 48 hours (risk-critical: 7 days)

**Purpose:** Verify trading logic works correctly:
- Full decision pipeline runs
- Risk Kernel makes decisions
- Positions are "paper traded"
- Performance can be measured

**Configuration:**
```yaml
shadow_phase_1:
  mode: DRY_RUN
  
  enabled:
    - All Phase 0 features
    - Risk Kernel (real decisions, no execution)
    - Position Sizer (calculates sizes)
    - Exit Brain (monitors paper positions)
    
  disabled:
    - Execution Engine (no real orders)
    - Real exchange API calls
    
  paper_trading:
    starting_capital: $100,000   # Simulated
    track_positions: true         # Maintain paper position state
    simulate_fills: true          # Assume market orders fill at current price
    simulate_slippage: 0.05%      # Add realistic slippage
    
  comparison:
    compare_to_production: true   # Compare decisions to live system
```

**Metrics Tracked:**
```yaml
dry_run_metrics:
  - signals_generated
  - risk_approvals
  - risk_vetoes
  - paper_trades_entered
  - paper_trades_exited
  - paper_pnl
  - paper_max_drawdown
  - paper_win_rate
  - decision_latency_p99
  - comparison_to_production  # Were decisions different?
```

**Success Criteria:**
```
[ ] Paper P&L is reasonable (not extreme losses)
[ ] Max drawdown < 20% (paper)
[ ] Risk Kernel veto rate > 0 (proves it's working)
[ ] No unexpected decision divergence from production
[ ] All services stable for 48h+
[ ] Decision latency < 100ms P99
```

---

### Phase 2: Micro-Capital Mode

**Duration:** Minimum 7 days (risk-critical: 14 days)

**Purpose:** Verify with real (but minimal) capital:
- Real orders execute correctly
- Slippage is as expected
- Exchange API integration works
- Real P&L tracking works

**Configuration:**
```yaml
shadow_phase_2:
  mode: MICRO_CAPITAL
  
  enabled:
    - All Phase 1 features
    - Execution Engine (REAL orders)
    - Real exchange API
    
  risk_limits:
    max_position_size: $100      # Micro positions
    max_total_exposure: $500     # Very limited
    max_daily_loss: $50          # Tight daily limit
    max_trades_per_day: 5        # Limited activity
    
  capital_allocation:
    capital_assigned: $1,000     # Small test capital
    isolated_from_main: true     # Separate sub-account
    
  monitoring:
    human_review_each_trade: true  # Manual review of every trade
    alert_on_any_loss: true        # Know immediately
```

**Metrics Tracked:**
```yaml
micro_capital_metrics:
  - real_trades_executed
  - real_pnl
  - real_slippage
  - order_fill_rate
  - api_error_rate
  - position_tracking_accuracy  # Does our state match exchange?
```

**Success Criteria:**
```
[ ] All trades execute correctly
[ ] Position state matches exchange
[ ] Slippage within expected range
[ ] No API errors
[ ] Real P&L matches expectations from Phase 1
[ ] No unexpected losses
[ ] Human review approves all trades
```

---

### Phase 3: Production Deployment

**Duration:** Ongoing

**Purpose:** Full production deployment with normal capital

**Configuration:**
```yaml
production:
  mode: PRODUCTION
  
  enabled:
    - All features
    - Full capital allocation
    - All risk limits per constitution
    
  gradual_rollout:
    day_1: 25% of normal capital
    day_3: 50% of normal capital
    day_7: 75% of normal capital
    day_14: 100% of normal capital
    
  monitoring:
    enhanced_first_30_days: true
    daily_human_review: true
```

---

## 3. Phase Transition Criteria

### Phase 0 → Phase 1

```yaml
phase_0_to_1_checklist:
  automated:
    - [ ] All services healthy for 24h
    - [ ] No ERROR level logs
    - [ ] Market data processing working
    - [ ] Signal generation active
    
  manual:
    - [ ] Logs reviewed by engineer
    - [ ] No unexpected behavior observed
    - [ ] Sign-off from team lead
```

### Phase 1 → Phase 2

```yaml
phase_1_to_2_checklist:
  automated:
    - [ ] All services healthy for 48h
    - [ ] Paper P&L positive OR explainable
    - [ ] Paper drawdown < 20%
    - [ ] Decision latency < 100ms
    - [ ] Risk Kernel functioning (has issued vetoes)
    
  manual:
    - [ ] Paper trades reviewed
    - [ ] Decision logic validated
    - [ ] Comparison to production acceptable
    - [ ] Sign-off from risk team
    - [ ] Sign-off from engineer
```

### Phase 2 → Phase 3

```yaml
phase_2_to_3_checklist:
  automated:
    - [ ] All services healthy for 7 days
    - [ ] All micro-trades executed correctly
    - [ ] Position state matches exchange
    - [ ] No API errors
    - [ ] Real slippage within expectations
    
  manual:
    - [ ] Every micro-trade reviewed
    - [ ] No unexpected behavior
    - [ ] Real P&L matches expectations
    - [ ] Sign-off from founders
    - [ ] Sign-off from risk team
    - [ ] Sign-off from engineering lead
```

---

## 4. Shadow Environment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SHADOW ENVIRONMENT TOPOLOGY                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PRODUCTION CLUSTER                      SHADOW CLUSTER                        │
│   ═══════════════════                     ═══════════════                       │
│                                                                                 │
│   ┌─────────────┐                         ┌─────────────┐                       │
│   │   Redis     │                         │   Redis     │                       │
│   │   :6379     │                         │   :6380     │                       │
│   └─────────────┘                         └─────────────┘                       │
│         │                                       │                               │
│         │                                       │                               │
│   ┌─────┴─────┐                           ┌─────┴─────┐                         │
│   │ Services  │                           │ Services  │                         │
│   │ (prod)    │                           │ (shadow)  │                         │
│   └───────────┘                           └───────────┘                         │
│         │                                       │                               │
│         │                                       │                               │
│   ┌─────────────┐     ┌───────────┐      ┌─────────────┐                       │
│   │  Exchange   │◀────│  Market   │─────▶│   Paper     │                       │
│   │    API      │     │   Data    │      │  Exchange   │ (Phase 0-1)           │
│   └─────────────┘     │  Mirror   │      └─────────────┘                       │
│                       └───────────┘                                             │
│                                                │                               │
│                                                │ (Phase 2)                      │
│                                                ▼                               │
│                                          ┌─────────────┐                       │
│                                          │  Exchange   │                       │
│                                          │  Sub-Acct   │                       │
│                                          └─────────────┘                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Network Isolation

```yaml
# docker-compose.shadow.yml
networks:
  shadow_network:
    driver: bridge
    internal: false  # Can access internet for market data
    
services:
  shadow_redis:
    networks:
      - shadow_network
    # NOT connected to production network
    
  shadow_risk_kernel:
    networks:
      - shadow_network
    environment:
      - PRODUCTION_ACCESS=false
      - SHADOW_MODE=true
```

---

## 5. Rollback Procedures

### Immediate Rollback

If ANY phase shows problems:

```bash
# Phase 2 rollback (micro-capital)
./scripts/shadow_rollback.sh phase2
# Actions:
# 1. Cancel all open orders
# 2. Close all micro positions
# 3. Stop all shadow services
# 4. Alert team

# Phase 1 rollback (dry-run)
./scripts/shadow_rollback.sh phase1
# Actions:
# 1. Stop shadow services
# 2. Archive logs for analysis
# 3. Alert team

# Phase 0 rollback (observation)
./scripts/shadow_rollback.sh phase0
# Actions:
# 1. Stop shadow services
# 2. Archive logs
```

### Rollback Triggers

```yaml
automatic_rollback_triggers:
  phase_2:
    - micro_capital_loss > $50
    - api_error_rate > 5%
    - position_state_mismatch
    - any_unexpected_behavior
    
  phase_1:
    - paper_drawdown > 25%
    - decision_latency_p99 > 500ms
    - service_crash
    
  phase_0:
    - service_crash
    - error_rate > 1%
```

---

## 6. Shadow Testing Reports

### Daily Shadow Report Template

```markdown
# Shadow Report: [DATE]

## Phase: [0/1/2]
## Branch: shadow/[branch-name]
## Day: [X of required Y]

## Service Health
| Service | Status | Uptime | Notes |
|---------|--------|--------|-------|
| risk_kernel | ✅ | 100% | |
| exit_brain | ✅ | 100% | |
| signal_advisory | ✅ | 100% | |

## Metrics (24h)
- Signals generated: X
- Risk approvals: X
- Risk vetoes: X
- Paper/Micro trades: X
- Paper/Micro P&L: $X.XX

## Anomalies
<!-- Any unexpected behavior -->

## Comparison to Production
<!-- If applicable -->

## Recommendation
[ ] Continue shadow testing
[ ] Proceed to next phase
[ ] Rollback and investigate

## Reviewer: [NAME]
## Date: [DATE]
```

---

## 7. Timeline Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SHADOW DEPLOY TIMELINE                                │
│                         (Risk-Critical Change)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Day 0-1:   Feature branch development + unit tests                             │
│  Day 2:     Code review, merge to shadow/xxx branch                             │
│  Day 2-4:   Phase 0: Observation (48h minimum)                                  │
│  Day 4:     Review Phase 0, transition checklist                                │
│  Day 4-11:  Phase 1: Dry-Run (7 days)                                           │
│  Day 11:    Review Phase 1, transition checklist                                │
│  Day 11-25: Phase 2: Micro-Capital (14 days)                                    │
│  Day 25:    Review Phase 2, all sign-offs                                       │
│  Day 26:    Merge to main, gradual production rollout starts                    │
│  Day 26-40: Gradual capital increase (25% → 50% → 75% → 100%)                   │
│  Day 40+:   Full production deployment                                          │
│                                                                                 │
│  TOTAL: ~40 days from feature complete to full production                       │
│                                                                                 │
│  "If this feels slow → that's the point."                                       │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Commands Reference

```bash
# Start shadow environment
make shadow-up PHASE=0

# Transition phases
make shadow-transition PHASE=1
make shadow-transition PHASE=2

# View shadow metrics
make shadow-metrics

# Generate shadow report
make shadow-report

# Rollback
make shadow-rollback

# Compare shadow to production
make shadow-compare

# Promote to production (requires approvals)
make shadow-promote
```

---

## 9. Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SHADOW DEPLOY SUMMARY                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Phase 0: OBSERVATION                                                           │
│    Duration: 24-48h                                                             │
│    Purpose: Verify basic functionality                                          │
│    Risk: None (no decisions, no capital)                                        │
│                                                                                 │
│  Phase 1: DRY-RUN                                                               │
│    Duration: 48h-7d                                                             │
│    Purpose: Verify trading logic                                                │
│    Risk: None (paper trading only)                                              │
│                                                                                 │
│  Phase 2: MICRO-CAPITAL                                                         │
│    Duration: 7-14d                                                              │
│    Purpose: Verify real execution                                               │
│    Risk: Minimal ($50 max daily loss)                                           │
│                                                                                 │
│  Phase 3: PRODUCTION                                                            │
│    Duration: Ongoing                                                            │
│    Purpose: Full deployment                                                     │
│    Risk: Managed by Risk Kernel                                                 │
│                                                                                 │
│  KEY PRINCIPLE:                                                                 │
│  Evidence > Trust                                                               │
│  Prove safety, then deploy.                                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Vi stoler ikke på kode. Vi stoler på bevis. Shadow mode produserer bevisene.*
