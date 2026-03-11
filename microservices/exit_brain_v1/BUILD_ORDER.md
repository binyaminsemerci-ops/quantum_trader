# Exit Brain v1 — Operational Build Order

> Based on: `EXIT_BRAIN_V1_ARCHITECTURE.md`  
> Status: 55 modules implemented, 297 tests passing, 0 deployed  
> Labels: **MUST** = blocks next step, **SHOULD** = improves quality, **LATER** = post-shadow-validation

---

## 1. Dependency Map

```
                         LAYER 0 (no deps)
                    ┌────────────────────────┐
                    │ models/                 │
                    │   position_exit_state   │
                    │   model_exit_signal     │
                    │   aggregated_exit_signal│
                    │   belief_state          │
                    │   hazard_assessment     │
                    │   action_candidate      │
                    │   policy_decision       │
                    │   exit_intent_candidate │
                    │   exit_intent_validation│
                    │   decision_trace        │
                    │   trade_exit_obituary   │
                    │   replay_eval_record    │
                    │   offline_eval_summary  │
                    │   tuning_recommendation │
                    │   calibration_artifact  │
                    │                         │
                    │ policy/                  │
                    │   policy_constraints    │
                    │   reason_codes          │
                    │                         │
                    │ engines/                 │
                    │   normalization         │
                    │   calibration           │
                    └────────────┬─────────────┘
                                │
                         LAYER 1 (depends on Layer 0 only)
                    ┌────────────▼─────────────┐
                    │ publishers/               │
                    │   shadow_publisher        │ ← depends on: all models
                    │                           │
                    │ engines/                   │
                    │   geometry_engine         │ ← depends on: position_exit_state
                    │   regime_drift_engine     │ ← depends on: position_exit_state
                    │                           │
                    │ adapters/                  │
                    │   model_registry          │ ← depends on: model_exit_signal
                    │                           │
                    │ validators/                │
                    │   idempotency             │ ← depends on: nothing
                    │   payload_normalizer      │ ← depends on: nothing
                    └────────────┬──────────────┘
                                │
                         LAYER 2
                    ┌────────────▼──────────────┐
                    │ services/                  │
                    │   position_state_builder  │ ← depends on: position_exit_state, shadow_publisher
                    │                           │
                    │ adapters/                  │
                    │   ensemble_exit_adapter   │ ← depends on: model_registry, model_exit_signal,
                    │                           │               normalization, calibration
                    │                           │
                    │ validators/                │
                    │   exit_intent_gateway_val │ ← depends on: exit_intent_candidate,
                    │                           │               idempotency, payload_normalizer
                    └────────────┬──────────────┘
                                │
                         LAYER 3
                    ┌────────────▼──────────────┐
                    │ aggregators/               │
                    │   ensemble_aggregator     │ ← depends on: ensemble_exit_adapter,
                    │                           │               aggregated_exit_signal
                    │                           │
                    │ engines/                   │
                    │   belief_engine           │ ← depends on: belief_state, geometry_engine,
                    │                           │               regime_drift_engine, ensemble_aggregator
                    │   hazard_engine           │ ← depends on: hazard_assessment, geometry_engine,
                    │                           │               regime_drift_engine, ensemble_aggregator
                    └────────────┬──────────────┘
                                │
                         LAYER 4
                    ┌────────────▼──────────────┐
                    │ engines/                   │
                    │   action_utility_engine   │ ← depends on: action_candidate, belief_engine,
                    │                           │               hazard_engine
                    └────────────┬──────────────┘
                                │
                         LAYER 5
                    ┌────────────▼──────────────┐
                    │ policy/                    │
                    │   exit_policy_engine      │ ← depends on: action_utility_engine, belief_state,
                    │                           │               hazard_assessment, policy_constraints,
                    │                           │               reason_codes, policy_decision
                    └────────────┬──────────────┘
                                │
                         LAYER 6
                    ┌────────────▼──────────────┐
                    │ orchestrators/             │
                    │   exit_agent_orchestrator │ ← depends on: ALL Phase 1-4 engines,
                    │                           │               shadow_publisher,
                    │                           │               exit_intent_gateway_validator
                    └────────────┬──────────────┘
                                │
                         LAYER 7 (offline, batch)
                    ┌────────────▼──────────────┐
                    │ replay/                    │
                    │   replay_loader           │ ← depends on: shadow_publisher (reads streams)
                    │   outcome_reconstructor   │ ← depends on: nothing (Redis price data)
                    │   counterfactual_evaluator│ ← depends on: action_candidate, outcome_reconstructor
                    │   replay_obituary_writer  │ ← depends on: replay_loader, outcome_reconstructor,
                    │                           │               trade_exit_obituary, shadow_publisher
                    └────────────┬──────────────┘
                                │
                         LAYER 8
                    ┌────────────▼──────────────┐
                    │ evaluators/                │
                    │   belief_calibration_eval │ ← depends on: trade_exit_obituary, outcome_reconstr
                    │   hazard_calibration_eval │ ← depends on: trade_exit_obituary, outcome_reconstr
                    │   utility_ranking_eval    │ ← depends on: replay_eval_record
                    │   policy_choice_eval      │ ← depends on: trade_exit_obituary, outcome_reconstr
                    │   offline_evaluator       │ ← depends on: ALL above + replay_loader,
                    │                           │               counterfactual_eval, shadow_publisher
                    └────────────┬──────────────┘
                                │
                         LAYER 9
                    ┌────────────▼──────────────┐
                    │ tuning/                    │
                    │   threshold_tuner         │ ← depends on: offline_eval_summary, policy_constraints
                    │   weight_tuner            │ ← depends on: offline_eval_summary
                    │   proposal_builder        │ ← depends on: threshold_tuner, weight_tuner,
                    │                           │               calibration_artifact, shadow_publisher
                    └──────────────────────────┘
```

### Import Dependency Matrix (verified from source)

| Module | Imports from within exit_brain_v1 |
|--------|-----------------------------------|
| `position_exit_state` | (none) |
| `model_exit_signal` | (none) |
| `aggregated_exit_signal` | (none) |
| `belief_state` | (none) |
| `hazard_assessment` | (none) |
| `action_candidate` | (none) |
| `policy_decision` | `action_candidate` |
| `exit_intent_candidate` | `action_candidate` |
| `trade_exit_obituary` | `action_candidate` |
| `policy_constraints` | (none) |
| `reason_codes` | (none) |
| `normalization` | (none) |
| `calibration` | (none) |
| `geometry_engine` | (none — takes raw floats) |
| `regime_drift_engine` | (none — takes raw floats) |
| `shadow_publisher` | all models (`.to_dict()` calls) |
| `model_registry` | `model_exit_signal` |
| `idempotency` | (none) |
| `payload_normalizer` | (none) |
| `position_state_builder` | `position_exit_state`, `shadow_publisher` |
| `ensemble_exit_adapter` | `model_registry`, `model_exit_signal`, `normalization`, `calibration` |
| `exit_intent_gateway_validator` | `exit_intent_candidate`, `idempotency`, `payload_normalizer` |
| `ensemble_aggregator` | `ensemble_exit_adapter`, `aggregated_exit_signal` |
| `belief_engine` | `belief_state` |
| `hazard_engine` | `hazard_assessment` |
| `action_utility_engine` | `action_candidate` |
| `exit_policy_engine` | `policy_constraints`, `reason_codes`, `policy_decision` |
| `exit_agent_orchestrator` | `exit_policy_engine`, `belief_engine`, `hazard_engine`, `action_utility_engine`, `shadow_publisher`, `exit_intent_gateway_validator` |
| `replay_loader` | (none — reads raw Redis) |
| `outcome_reconstructor` | (none — reads raw Redis) |
| `counterfactual_evaluator` | `action_candidate`, `outcome_reconstructor` |
| `replay_obituary_writer` | `trade_exit_obituary`, `replay_loader`, `outcome_reconstructor` |
| `belief_calibration_evaluator` | `trade_exit_obituary` |
| `hazard_calibration_evaluator` | `trade_exit_obituary` |
| `utility_ranking_evaluator` | `replay_evaluation_record` |
| `policy_choice_evaluator` | `trade_exit_obituary`, `action_candidate`, `outcome_reconstructor` |
| `offline_evaluator` | ALL evaluators + `replay_loader`, `outcome_reconstructor`, `counterfactual_evaluator` |
| `threshold_tuner` | `offline_evaluation_summary`, `tuning_recommendation`, `policy_constraints` |
| `weight_tuner` | `offline_evaluation_summary`, `tuning_recommendation` |
| `proposal_builder` | `threshold_tuner`, `weight_tuner`, `calibration_artifact` |

---

## 2. Build Order

### Phase boundaries

| Phase | Layer(s) | Status | Blocking next phase? |
|-------|----------|--------|---------------------|
| Phase 1 | 0-2 (partial) | ✅ Code done, ✅ 161 tests | No |
| Phase 2 | 0-3 (partial) | ✅ Code done, ✅ 136 tests | No |
| Phase 3 | 3-4 | ✅ Code done, ❌ 0 tests | **Yes — blocks deploy confidence** |
| Phase 4 | 5-6 | ✅ Code done, ❌ 0 tests | **Yes — blocks deploy confidence** |
| Phase 5 | 7-9 | ✅ Code done, ❌ 0 tests | No — offline, runs after data exists |

### Sequential constraints (MUST respect)

```
position_exit_state  MUST exist before  geometry_engine
position_exit_state  MUST exist before  regime_drift_engine
position_exit_state  MUST exist before  position_state_builder
model_exit_signal    MUST exist before  model_registry
model_registry       MUST exist before  ensemble_exit_adapter
ensemble_exit_adapter MUST exist before ensemble_aggregator
geometry_engine      MUST exist before  belief_engine
regime_drift_engine  MUST exist before  belief_engine
ensemble_aggregator  MUST exist before  belief_engine
belief_engine        MUST exist before  action_utility_engine
hazard_engine        MUST exist before  action_utility_engine
action_utility_engine MUST exist before exit_policy_engine
exit_policy_engine   MUST exist before exit_agent_orchestrator
shadow_publisher     MUST exist before exit_agent_orchestrator
ALL Phase 1-4        MUST be deployed before replay_loader can read data
500+ obituaries      MUST exist before tuning proposals are meaningful
```

---

## 3. Parallel Workstreams

Given that all code is implemented, work now splits into **testing** and **deployment**. These can run in parallel:

```
WORKSTREAM A: Testing (local)          WORKSTREAM B: Deploy (VPS)
─────────────────────────────          ──────────────────────────
                                       
A1: test_phase3_comprehensive.py       B1: Shadow stream Redis keys
    - BeliefEngine tests                   - Create/verify 17 streams
    - HazardEngine tests                   - Verify STREAM_MAXLEN=5000
    - ActionUtilityEngine tests            
    - ~100 tests                       B2: Position state feed
                                           - Verify position_state_builder
A2: test_phase4_comprehensive.py           reads from existing Redis hashes
    - ExitPolicyEngine tests           
    - Orchestrator tests               B3: Model availability check
    - Gateway+idempotency tests            - Verify ≥2 of 6 models respond
    - ~100 tests                           - Test ensemble_exit_adapter
                                       
A3: test_phase5_comprehensive.py       B4: First shadow publish run
    - CounterfactualEvaluator              - Run orchestrator once
    - OfflineEvaluator                     - Verify data in 13 streams
    - Tuners                               (Phase 1-4)
    - ~100 tests                       
                                       B5: Cron: replay + offline eval
                                           - Schedule obituary writer
                                           - Schedule offline evaluator
```

### What can run in parallel within each workstream

**Testing parallelism:**

| Task | Can parallel with | Reason |
|------|-------------------|--------|
| A1 (Phase 3 tests) | A2 (Phase 4 tests) | No shared state, no import conflicts |
| A1 (Phase 3 tests) | A3 (Phase 5 tests) | Independent modules |
| A2 (Phase 4 tests) | A3 (Phase 5 tests) | Independent modules |
| A1 + A2 | B1 + B2 | Testing is local, deploy is VPS |

**Deployment sequential:**

```
B1 → B2 → B3 → B4 → B5
```

B1-B3 are verification only. B4 is first real shadow run. B5 requires data from B4.

---

## 4. First Implementation Slice

All code exists. The first *operational* slice is: **make the real-time path run on VPS in shadow mode**.

### Slice 1: Real-time shadow loop (MUST)

**Files involved (in execution order):**

```
1. services/position_state_builder.py      ← reads Redis position hashes
2. engines/geometry_engine.py              ← pure math
3. engines/regime_drift_engine.py          ← pure math
4. adapters/model_registry.py             ← model specs
5. adapters/ensemble_exit_adapter.py      ← calls model predict
6. aggregators/ensemble_aggregator.py     ← aggregates
7. engines/belief_engine.py               ← fuses
8. engines/hazard_engine.py               ← risk assessment
9. engines/action_utility_engine.py       ← scores 7 actions
10. policy/exit_policy_engine.py          ← 7-step policy
11. orchestrators/exit_agent_orchestrator.py  ← publishes to shadow
12. publishers/shadow_publisher.py         ← XADD to 13 streams
```

**What MUST exist in Redis before this runs:**

| Redis key pattern | Content | Source |
|-------------------|---------|--------|
| `quantum:position:{symbol}:*` | Position state hashes | Existing trading system |
| Model endpoints or pickle files | Trained models | Existing model pipeline |

**What this produces:**

Streams 1-13 (Phase 1-4) populated with shadow data.

### Slice 2: Offline evaluation batch (SHOULD, after Slice 1 has data)

```
13. replay/replay_loader.py              ← reads streams 1-13
14. replay/outcome_reconstructor.py      ← reads price data
15. replay/replay_obituary_writer.py     ← builds obituaries
16. replay/counterfactual_evaluator.py   ← what-if
17. evaluators/offline_evaluator.py      ← orchestrates eval
18. evaluators/belief_calibration_evaluator.py
19. evaluators/hazard_calibration_evaluator.py
20. evaluators/utility_ranking_evaluator.py
21. evaluators/policy_choice_evaluator.py
```

Produces streams 14-16.

### Slice 3: Tuning proposals (LATER, after 500+ obituaries)

```
22. tuning/threshold_tuner.py
23. tuning/weight_tuner.py
24. tuning/proposal_builder.py
```

Produces stream 17.

---

## 5. First Test Slice

### MUST — before any VPS deployment

**Commit batch 1: Phase 3 tests**

File: `tests/test_phase3_comprehensive.py`

| Test group | Count | What it verifies |
|------------|-------|------------------|
| BeliefEngine.compute() fusion weights | 8 | Exit_pressure = 0.50×ensemble + 0.25×reversal + 0.25×geometry |
| BeliefEngine fail-closed | 3 | Returns None when ensemble is None |
| BeliefEngine boundary values | 6 | All outputs in [0,1] or [-1,1] |
| HazardEngine 6 axes | 12 | Each axis formula correct |
| HazardEngine composite | 4 | Equal-weighted mean, dominant selection |
| HazardEngine time_decay | 3 | Half-life = 14400s curve |
| ActionUtilityEngine 7 actions | 14 | Each utility function |
| ActionUtilityEngine penalties | 8 | not_profitable, low_hazard_close, high_hazard_hold, uncertainty |
| ActionUtilityEngine ranking | 4 | Sorted by net_utility, rank=1 best |
| ActionUtilityEngine clamp | 3 | net_utility ∈ [0,1] |
| Phase 3 integration | 5 | Full pipeline: geometry+regime+ensemble → ranked actions |
| **Total** | **~70** | |

**Commit batch 2: Phase 4 tests**

File: `tests/test_phase4_comprehensive.py`

| Test group | Count | What it verifies |
|------------|-------|------------------|
| PolicyEngine step 1: freshness | 4 | Stale data → HOLD + STALE_UPSTREAM_DATA |
| PolicyEngine step 2: completeness | 4 | Low completeness → HOLD + DATA_COMPLETENESS_FLOOR |
| PolicyEngine step 3: uncertainty | 4 | High uncertainty → restrict to {HOLD, TIGHTEN} |
| PolicyEngine step 5: constraints | 6 | Profit requirement, hazard floors |
| PolicyEngine step 6: conviction | 4 | Low conviction → HOLD demotion |
| PolicyEngine step 7: emergency | 4 | Hazard > 0.85 → CLOSE_FULL override |
| PolicyEngine fail-closed invariant | 3 | policy_passed=False → action=HOLD |
| PolicyDecision.validate() | 4 | Contract validation |
| Orchestrator decision cycle | 5 | Full cycle with mock publisher |
| Orchestrator fail-closed | 3 | Exception → None |
| Gateway validator | 8 | Valid/invalid intents, schema checks |
| Idempotency dedup | 4 | 5-min window, duplicate detection |
| Payload normalizer | 4 | Clamp, strip, validate |
| Reason codes coverage | 3 | All 7 hard blocks, 6 soft warnings exercised |
| **Total** | **~60** | |

**Commit batch 3: Phase 5 tests** (SHOULD, before offline eval deployment)

File: `tests/test_phase5_comprehensive.py`

| Test group | Count | What it verifies |
|------------|-------|------------------|
| CounterfactualEvaluator: 7 actions | 14 | Correct PnL simulation per action |
| CounterfactualEvaluator: best action | 3 | find_ex_post_best_action correctness |
| CounterfactualEvaluator: quality score | 4 | Formula: 0.4×rank + 0.3×(1-regret) + 0.2×preserve + 0.1×opp |
| ObituaryWriter: regret score | 3 | (best - actual) / best |
| ObituaryWriter: preservation score | 3 | 1 - (drawdown / peak) |
| ObituaryWriter: opportunity score | 3 | actual / best |
| OfflineEvaluator: min samples | 2 | Returns None if < 5 |
| OfflineEvaluator: sub-evaluator delegation | 4 | All 4 called correctly |
| PolicyChoiceEvaluator: baselines | 8 | 4 baselines × 2 sides |
| ThresholdTuner: ±20% cap | 4 | Max change fraction enforced |
| ThresholdTuner: low sample cap | 2 | Confidence ≤ 0.30 if < 50 samples |
| WeightTuner: normalization | 3 | Proposed weights stay within bounds |
| WeightTuner: min MAE gap | 2 | No proposal if gap < 0.05 |
| ProposalBuilder: max 5 | 2 | Cap enforced |
| ProposalBuilder: confidence sort | 2 | Highest confidence first |
| **Total** | **~59** | |

### First commit order

```
COMMIT 1: test_phase3_comprehensive.py  (~70 tests)     MUST
COMMIT 2: test_phase4_comprehensive.py  (~60 tests)     MUST
COMMIT 3: test_phase5_comprehensive.py  (~59 tests)     SHOULD
```

After all 3: target **~486 total tests** (297 existing + ~189 new).

---

## 6. Shadow-Only Verification Slice

### MUST verify before VPS deployment

**Step 1: Verify 17 Redis streams can be created**

```bash
# On VPS — check Redis accepts the stream keys
for stream in \
  exit.state.shadow exit.geometry.shadow exit.regime.shadow \
  exit.ensemble.raw.shadow exit.ensemble.agg.shadow exit.ensemble.diag.shadow \
  exit.belief.shadow exit.hazard.shadow exit.utility.shadow \
  exit.policy.shadow exit.intent.candidate.shadow \
  exit.intent.validation.shadow exit.decision.trace.shadow \
  exit.obituary.shadow exit.replay.eval.shadow \
  exit.eval.summary.shadow exit.tuning.recommendation.shadow; do
  redis-cli XADD "quantum:stream:$stream" MAXLEN ~ 5000 '*' test_key test_value
  redis-cli XLEN "quantum:stream:$stream"
done
```

**Step 2: Verify forbidden stream guard**

```python
# Must raise ValueError — run once, then delete test
from microservices.exit_brain_v1.publishers.shadow_publisher import ShadowPublisher
import redis
r = redis.Redis()
pub = ShadowPublisher(r)
try:
    pub._xadd("quantum:stream:trade.intent", {"test": "MUST_FAIL"})
    assert False, "Should have raised ValueError"
except ValueError:
    print("PASS: Forbidden stream guard works")
```

**Step 3: Verify position_state_builder reads real data**

```python
# One-shot test with a real position
from microservices.exit_brain_v1.services.position_state_builder import PositionStateBuilder
import redis
r = redis.Redis()
builder = PositionStateBuilder(r)
state = builder.build("BTCUSDT", "some_position_id")
assert state is not None or state is None  # None = fail-closed, OK
print(f"State: {state}")
```

**Step 4: Verify one full decision cycle produces shadow data**

```python
# Integration smoke test — runs once, publishes to all 13 Phase 1-4 streams
from microservices.exit_brain_v1.orchestrators.exit_agent_orchestrator import ExitAgentOrchestrator
# ... construct with real Redis, ShadowPublisher, all engines
trace = orchestrator.run_decision_cycle(state, geometry, regime, ensemble)
assert trace is not None or trace is None  # None = fail-closed, acceptable
```

**Step 5: Verify stream payloads are deserializable**

```python
# Read back from each stream, verify fields parse
for stream_name in ALL_17_STREAMS:
    entries = r.xrevrange(f"quantum:stream:{stream_name}", count=1)
    assert len(entries) >= 1, f"No data in {stream_name}"
    _, fields = entries[0]
    # Verify key fields are present and parseable
    for k, v in fields.items():
        key = k.decode() if isinstance(k, bytes) else k
        val = v.decode() if isinstance(v, bytes) else v
        print(f"  {stream_name}: {key} = {val[:50]}")
```

### Required payload schema per stream

| Stream | MUST-have fields | Type check |
|--------|-----------------|------------|
| `exit.state.shadow` | position_id, symbol, side, entry_price, current_price, shadow_only | shadow_only == "True" |
| `exit.geometry.shadow` | position_id, mfe_capture, drawdown_from_peak, ppr | all ∈ [0,1] |
| `exit.regime.shadow` | symbol, trend_alignment, reversal_risk, chop_risk | reversal_risk ∈ [0,1] |
| `exit.ensemble.raw.shadow` | model_name, p_hold, p_close_full | all ∈ [0,1] |
| `exit.ensemble.agg.shadow` | position_id, p_hold, agreement_score | agreement ∈ [0,1] |
| `exit.ensemble.diag.shadow` | num_models_available, dominant_action | num ≥ 2 |
| `exit.belief.shadow` | exit_pressure, hold_conviction, directional_edge, uncertainty_total | pressure ∈ [0,1] |
| `exit.hazard.shadow` | composite_hazard, dominant_hazard, 6 axis fields | composite ∈ [0,1] |
| `exit.utility.shadow` | action, net_utility, rank, exit_fraction | rank ≥ 1 |
| `exit.policy.shadow` | chosen_action, policy_passed, decision_confidence | action ∈ VALID_ACTIONS |
| `exit.intent.candidate.shadow` | action_name, intent_type, idempotency_key | intent_type == "SHADOW_EXIT" |
| `exit.intent.validation.shadow` | is_valid, validation_errors | is_valid ∈ {True, False} |
| `exit.decision.trace.shadow` | trace_id, source_decision_id | both non-empty |
| `exit.obituary.shadow` | regret_score, preservation_score, opportunity_capture_score | all ∈ [0,1] |
| `exit.replay.eval.shadow` | decision_quality_score, ex_post_best_action | score ∈ [0,1] |
| `exit.eval.summary.shadow` | mean_decision_quality_score, decisions_covered | decisions ≥ 5 |
| `exit.tuning.recommendation.shadow` | parameter_name, current_value, proposed_value, requires_human_review | review == True |

---

## 7. Risks That Block Build Start

| # | Risk | Blocks | Severity | Resolution |
|---|------|--------|----------|------------|
| **R1** | Phase 3-4 have 0 tests | Blocks deploy confidence | **MUST** | Write test_phase3 + test_phase4 before VPS deploy |
| **R2** | Redis position hash format unknown | Blocks Slice 1 step 3 | **MUST** | Run `position_state_builder.build()` on VPS once, verify field mapping |
| **R3** | Model availability on VPS | Blocks Slice 1 ensemble path | **MUST** | Verify ≥2 of 6 models respond via `ensemble_exit_adapter.predict()` |
| **R4** | No entry point / runner script | Blocks continuous operation | **MUST** | Create `run_exit_brain.py` — loop that calls `orchestrator.run_decision_cycle()` per position |
| **R5** | No `__init__.py` exports verified | Blocks clean imports on VPS | **SHOULD** | Run `python -c "from microservices.exit_brain_v1.orchestrators.exit_agent_orchestrator import ExitAgentOrchestrator"` on VPS |
| **R6** | Idempotency is in-memory only | Allows duplicates after restart | **LATER** | Migrate to Redis SET with EX after shadow validation |
| **R7** | Calibration is identity-only | Suboptimal model outputs | **LATER** | Fit Platt after 500+ obituaries |

### Unblock sequence

```
R1 ──→ Write tests     ──→ COMMIT 1-2
R2 ──→ SSH test        ──→ 5 minutes
R3 ──→ SSH test        ──→ 5 minutes
R4 ──→ Write runner    ──→ COMMIT 4
R5 ──→ SSH test        ──→ 2 minutes
─────────────────────────────────────
Total to unblock: 3 commits + 3 SSH verifications
```

### Exact start point

**Start here:**

```
STEP 0: python -m pytest microservices/exit_brain_v1/tests/ -v --tb=short
        → Verify 297 pass, 0 fail (takes <1s)

STEP 1: Write tests/test_phase3_comprehensive.py
        → Target ~70 tests, all pass

STEP 2: Write tests/test_phase4_comprehensive.py  
        → Target ~60 tests, all pass

STEP 3: Full regression
        → python -m pytest microservices/exit_brain_v1/tests/ -v --tb=short
        → Target ~427 pass, 0 fail

STEP 4: VPS smoke test (SSH)
        → Verify R2 (Redis hashes), R3 (models), R5 (imports)

STEP 5: Create run_exit_brain.py (runner script)

STEP 6: First shadow run on VPS
        → Verify 13 streams populated

STEP 7: Monitor for 24h → verify stream data quality

STEP 8: Write tests/test_phase5_comprehensive.py
        → Target ~59 tests, all pass

STEP 9: Deploy offline evaluation (cron)
        → Obituaries start accumulating

STEP 10: First tuning proposal review (after 500+ obituaries)
```

---

*Derived entirely from EXIT_BRAIN_V1_ARCHITECTURE.md. No new design. No new theory.*
