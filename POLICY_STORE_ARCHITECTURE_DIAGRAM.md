# PolicyStore Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QUANTUM TRADER AI SYSTEM                      │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    PolicyStore (Central Hub)                 │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │              GlobalPolicy (State)                    │    │   │
│  │  │                                                       │    │   │
│  │  │  • risk_mode: "AGGRESSIVE" | "NORMAL" | "DEFENSIVE" │    │   │
│  │  │  • allowed_strategies: ["STRAT_1", ...]             │    │   │
│  │  │  • allowed_symbols: ["BTCUSDT", ...]                │    │   │
│  │  │  • max_risk_per_trade: 0.01                         │    │   │
│  │  │  • max_positions: 10                                │    │   │
│  │  │  • global_min_confidence: 0.65                      │    │   │
│  │  │  • opp_rankings: {"BTCUSDT": 0.92, ...}            │    │   │
│  │  │  • model_versions: {"xgboost": "v14", ...}         │    │   │
│  │  │  • last_updated: "2025-11-30T..."                  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                               │   │
│  │  Operations: get() | update() | patch() | reset()           │   │
│  │  Thread-Safe: ✓   Atomic: ✓   Validated: ✓                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│                              ↕                                        │
│                    (reads & writes)                                   │
│                              ↕                                        │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   POLICY WRITERS (Controllers)               │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                               │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │   MSC AI             │  │  Opportunity Ranker  │         │   │
│  │  │  (Meta Strategy)     │  │     (OppRank)        │         │   │
│  │  ├──────────────────────┤  ├──────────────────────┤         │   │
│  │  │ Writes:              │  │ Writes:              │         │   │
│  │  │ • risk_mode          │  │ • opp_rankings       │         │   │
│  │  │ • allowed_strategies │  │ • allowed_symbols    │         │   │
│  │  │ • max_risk_per_trade │  │                      │         │   │
│  │  │ • max_positions      │  │ Frequency: Hourly    │         │   │
│  │  │ • min_confidence     │  │                      │         │   │
│  │  │                      │  │                      │         │   │
│  │  │ Frequency:           │  │                      │         │   │
│  │  │   Regime changes     │  │                      │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  │                                                               │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │   CLM                │  │  Strategy Generator  │         │   │
│  │  │ (Continuous Learn)   │  │      (SG AI)         │         │   │
│  │  ├──────────────────────┤  ├──────────────────────┤         │   │
│  │  │ Writes:              │  │ Writes:              │         │   │
│  │  │ • model_versions     │  │ • allowed_strategies │         │   │
│  │  │                      │  │   (promote/demote)   │         │   │
│  │  │ Frequency:           │  │                      │         │   │
│  │  │   After retraining   │  │ Frequency:           │         │   │
│  │  │   (days-weeks)       │  │   After evaluation   │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  │                                                               │   │
│  │  ┌──────────────────────┐                                    │   │
│  │  │  Safety Governor     │                                    │   │
│  │  ├──────────────────────┤                                    │   │
│  │  │ Writes:              │                                    │   │
│  │  │ • Emergency override │                                    │   │
│  │  │ • Circuit breaker    │                                    │   │
│  │  │                      │                                    │   │
│  │  │ Frequency: Rare      │                                    │   │
│  │  └──────────────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│                              ↑                                        │
│                    (reads only)                                       │
│                              ↑                                        │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   POLICY READERS (Executors)                 │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                               │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │  Orchestrator Policy │  │    RiskGuard         │         │   │
│  │  ├──────────────────────┤  ├──────────────────────┤         │   │
│  │  │ Reads:               │  │ Reads:               │         │   │
│  │  │ • allowed_strategies │  │ • max_risk_per_trade │         │   │
│  │  │ • allowed_symbols    │  │                      │         │   │
│  │  │ • min_confidence     │  │ Checks:              │         │   │
│  │  │ • opp_rankings       │  │   Risk limits        │         │   │
│  │  │                      │  │                      │         │   │
│  │  │ Checks:              │  │ Frequency:           │         │   │
│  │  │   Signal approval    │  │   Per trade          │         │   │
│  │  │                      │  │                      │         │   │
│  │  │ Frequency:           │  │                      │         │   │
│  │  │   Per signal         │  │                      │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  │                                                               │   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐         │   │
│  │  │ Portfolio Balancer   │  │  Ensemble Manager    │         │   │
│  │  ├──────────────────────┤  ├──────────────────────┤         │   │
│  │  │ Reads:               │  │ Reads:               │         │   │
│  │  │ • max_positions      │  │ • model_versions     │         │   │
│  │  │                      │  │                      │         │   │
│  │  │ Checks:              │  │ Uses:                │         │   │
│  │  │   Position capacity  │  │   Active models      │         │   │
│  │  │                      │  │   for predictions    │         │   │
│  │  │ Frequency:           │  │                      │         │   │
│  │  │   Per trade          │  │ Frequency:           │         │   │
│  │  │                      │  │   Per prediction     │         │   │
│  │  └──────────────────────┘  └──────────────────────┘         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
STARTUP:
   ┌─────────────┐
   │   Config    │
   │    File     │
   └──────┬──────┘
          │
          ↓
   ┌─────────────┐
   │  MSC AI     │ Initial policy setup
   │ (Bootstrap) │
   └──────┬──────┘
          │
          ↓ store.update()
   ┌────────────────────────┐
   │     PolicyStore        │
   └────────────────────────┘


NORMAL OPERATION:

┌──────────────────────────────────────────────────────────────┐
│  MSC AI Regime Detection                                     │
│  (Every 15 min)                                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             ↓ store.patch({"risk_mode": "AGGRESSIVE", ...})
      ┌────────────────────────┐
      │     PolicyStore        │
      └────────────────────────┘
             ↑
             │ store.patch({"opp_rankings": {...}, ...})
             │
┌────────────┴─────────────────────────────────────────────────┐
│  OpportunityRanker                                           │
│  (Hourly)                                                    │
└──────────────────────────────────────────────────────────────┘


SIGNAL PROCESSING:

┌──────────────┐
│   Signal     │
│  (from AI)   │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│  Orchestrator    │ policy = store.get()
├──────────────────┤ if signal.strategy in policy['allowed_strategies']
│ Reads:           │    and signal.confidence >= policy['min_confidence']:
│ • strategies     │      return APPROVE
│ • symbols        │
│ • confidence     │
└──────┬───────────┘
       │ APPROVED
       ↓
┌──────────────────┐
│   RiskGuard      │ policy = store.get()
├──────────────────┤ if risk <= policy['max_risk_per_trade']:
│ Reads:           │      return PASS
│ • max_risk       │
└──────┬───────────┘
       │ PASS
       ↓
┌──────────────────┐
│   Portfolio      │ policy = store.get()
│   Balancer       │ if len(positions) < policy['max_positions']:
├──────────────────┤      return OK
│ Reads:           │
│ • max_positions  │
└──────┬───────────┘
       │ OK
       ↓
┌──────────────────┐
│    EXECUTE       │
│     TRADE        │
└──────────────────┘


MODEL RETRAINING:

┌──────────────────┐
│   CLM            │
│ (Retraining)     │
└────────┬─────────┘
         │
         ↓ Train new version
┌──────────────────┐
│  Shadow Eval     │
│   (1 week)       │
└────────┬─────────┘
         │
         ↓ If better than prod
         │ store.patch({"model_versions": {"xgb": "v15"}})
         │
      ┌──┴──────────────────┐
      │    PolicyStore      │
      └─────────────────────┘
         │
         ↑ policy = store.get()
         │ active_version = policy['model_versions']['xgb']
         │
┌────────┴─────────┐
│  Ensemble Mgr    │ Use new model for predictions
└──────────────────┘


EMERGENCY:

┌──────────────────┐
│ Circuit Breaker  │ Drawdown > 10%
│    Triggered     │
└────────┬─────────┘
         │
         ↓ store.patch({"risk_mode": "DEFENSIVE",
         │              "max_positions": 3,
         │              "allowed_strategies": []})
         │
      ┌──┴──────────────────┐
      │    PolicyStore      │
      └─────────────────────┘
         │
         ↑ All components immediately see new policy
         │
┌────────┴─────────┐
│  All Components  │ Stop new trades, reduce risk
└──────────────────┘
```

## Component Interaction Matrix

```
                      │ Read │ Write │ Frequency      │ Criticality
──────────────────────┼──────┼───────┼────────────────┼──────────────
MSC AI                │  ✓   │   ✓   │ 15min-1hr      │ HIGH
Opportunity Ranker    │  ✓   │   ✓   │ 1hr            │ HIGH
CLM                   │  ✓   │   ✓   │ Days-weeks     │ MEDIUM
Strategy Generator    │  ✓   │   ✓   │ Days-weeks     │ MEDIUM
Safety Governor       │  ✓   │   ✓   │ Rare (emergency)│ CRITICAL
Orchestrator Policy   │  ✓   │   ✗   │ Per signal     │ CRITICAL
RiskGuard             │  ✓   │   ✗   │ Per trade      │ CRITICAL
Portfolio Balancer    │  ✓   │   ✗   │ Per trade      │ HIGH
Ensemble Manager      │  ✓   │   ✗   │ Per prediction │ HIGH
Executor              │  ✓   │   ✗   │ Continuous     │ CRITICAL
```

## State Transitions

```
RISK MODE TRANSITIONS:

                    ┌────────────────┐
                    │     NORMAL     │
                    │  (Balanced)    │
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │                           │
    Strong    │                           │  High volatility
    trend     ↓                           ↓  or choppy
              │                           │
     ┌────────┴────────┐         ┌───────┴────────┐
     │   AGGRESSIVE    │         │   DEFENSIVE    │
     │  (High risk)    │         │   (Low risk)   │
     └────────┬────────┘         └───────┬────────┘
              │                           │
              │  Volatility               │  Trend
              │  increase                 │  emerges
              └─────────────┬─────────────┘
                            │
                            ↓
                    ┌────────────────┐
                    │     NORMAL     │
                    └────────────────┘

CIRCUIT BREAKER:

  ┌────────────────┐
  │  Any Mode      │
  └───────┬────────┘
          │
          │ Drawdown > threshold
          │ (Emergency condition)
          ↓
  ┌────────────────┐
  │   DEFENSIVE    │ ← Force defensive
  │  (Emergency)   │   All new trades stopped
  └───────┬────────┘
          │
          │ Manual restore OR
          │ Conditions improve
          ↓
  ┌────────────────┐
  │  Previous Mode │
  └────────────────┘
```

## Storage Backend Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PolicyStore (Interface)                    │
│                                                          │
│  Methods:                                                │
│    • get() → dict                                        │
│    • update(dict) → None                                 │
│    • patch(dict) → None                                  │
│    • reset() → None                                      │
│    • get_policy_object() → GlobalPolicy                  │
└─────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ↓                ↓                ↓
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│  InMemoryStore   │ │ PostgresStore│ │  RedisStore  │
│                  │ │              │ │              │
│ Storage:         │ │ Storage:     │ │ Storage:     │
│   Python dict    │ │   JSONB col  │ │   JSON str   │
│                  │ │              │ │              │
│ Concurrency:     │ │ Concurrency: │ │ Concurrency: │
│   RLock          │ │   Row lock   │ │   WATCH/EXEC │
│                  │ │              │ │              │
│ Use case:        │ │ Use case:    │ │ Use case:    │
│   Dev/test       │ │   Production │ │   High perf  │
│   Single proc    │ │   Multi proc │ │   Multi proc │
└──────────────────┘ └──────────────┘ └──────────────┘
```

## Thread Safety Model

```
CONCURRENT READS (Non-blocking):

Thread 1: store.get() ────────────────────► dict copy
Thread 2: store.get() ────────────────────► dict copy
Thread 3: store.get() ────────────────────► dict copy

All return independent copies (safe)


CONCURRENT WRITES (Serialized):

Thread 1: store.patch({"risk_mode": "A"}) ─┐
                                           │  Lock acquired
Thread 2: store.patch({"max_pos": 12}) ────┼─► Waits
                                           │
Thread 3: store.patch({"confidence": 0.7}) ┼─► Waits
                                           │
                                           └─► Release
                                               ↓
                                               Lock acquired
                                               ↓
                                               Release
                                               ↓
                                               Lock acquired


READ DURING WRITE (Consistent):

Thread 1 (write): patch() ─┬─► Lock ─► Modify ─► Release
                           │
Thread 2 (read):  get() ───┴────► Wait ─────────► Read new value

Reader sees either old OR new value (never partial update)
```

## Validation Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Component calls store.update() or store.patch()             │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ↓
            ┌──────────────────────┐
            │  PolicyValidator     │
            │  validate(policy)    │
            └──────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │                           │
    VALID│                           │INVALID
         ↓                           ↓
┌────────────────────┐    ┌────────────────────────┐
│  Continue update   │    │ Raise                  │
│                    │    │ PolicyValidationError  │
└────────┬───────────┘    └────────────────────────┘
         │                           │
         ↓                           ↓
┌────────────────────┐    ┌────────────────────────┐
│  Merge/Update      │    │  No changes applied    │
│  Add timestamp     │    │  Store unchanged       │
└────────┬───────────┘    └────────────────────────┘
         │
         ↓
┌────────────────────┐
│  Store updated     │
│  atomically        │
└────────────────────┘
```

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   QUANTUM TRADER SYSTEM                      │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │              PolicyStore (Singleton)               │     │
│  └──────────────────────┬─────────────────────────────┘     │
│                         │ (injected into all components)     │
│                         │                                    │
│  ┌──────────────────────┴─────────────────────────────┐     │
│  │                                                      │     │
│  ↓                                                      ↓     │
│  ┌─────────────────┐                         ┌─────────────┐│
│  │  MSC AI         │                         │  OppRank    ││
│  │  (risk_mode)    │                         │  (rankings) ││
│  └────────┬────────┘                         └──────┬──────┘│
│           │                                         │        │
│           ↓ writes                         writes  ↓        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PolicyStore                            │   │
│  └─────────────────────────────────────────────────────┘   │
│           │                                                  │
│           ↓ reads                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Executor (main loop)                                │  │
│  │  ├─ Orchestrator (signal approval)                   │  │
│  │  ├─ RiskGuard (risk validation)                      │  │
│  │  ├─ PortfolioBalancer (capacity)                     │  │
│  │  ├─ EnsembleManager (active models)                  │  │
│  │  └─ SafetyGovernor (monitoring)                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘

DEPENDENCY INJECTION:

# At startup
policy_store = InMemoryPolicyStore()

# Initialize all components with shared store
msc_ai = MSCAIController(policy_store)
opp_rank = OpportunityRanker(policy_store)
orchestrator = OrchestratorPolicy(policy_store)
risk_guard = RiskGuard(policy_store, account_balance)
portfolio = PortfolioBalancer(policy_store)
ensemble = EnsembleManager(policy_store)
safety = SafetyGovernor(policy_store)

executor = Executor(
    policy_store=policy_store,
    orchestrator=orchestrator,
    risk_guard=risk_guard,
    portfolio=portfolio,
    ensemble=ensemble,
    safety=safety,
)
```

## Legend

```
┌────────┐
│  Box   │  = Component or Module
└────────┘

   ↓       = Data flow (direction)

  ───►     = Operation or Method call

   ↕       = Bidirectional interaction

  ┌───┐
  │ ✓ │    = Implemented / Available
  └───┘

  ┌───┐
  │ ✗ │    = Not implemented / Not available
  └───┘
```

---

**Note**: This is a conceptual diagram. Actual implementation may vary based on specific deployment requirements.
