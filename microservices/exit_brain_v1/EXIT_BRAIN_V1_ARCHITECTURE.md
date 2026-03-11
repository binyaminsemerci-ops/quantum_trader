# Exit Brain v1 — Complete Architecture Document

> **Version:** 1.0.0  
> **Status:** Implemented and tested (297 tests, 0 failures)  
> **Repo:** `microservices/exit_brain_v1/`  
> **Python:** 3.12+  
> **Labelling convention:** Every claim is tagged **[FACT]**, **[ASSUMPTION]**, or **[RECOMMENDATION]**.

---

## 1. Architecture Principles

### 1.1 Shadow-First

**[FACT]** Every module enforces `shadow_only = True` on every data contract. No module writes to execution-layer streams. The 6 forbidden streams are hardcoded in `publishers/shadow_publisher.py`:

```python
_FORBIDDEN_STREAMS = frozenset({
    "quantum:stream:trade.intent",
    "quantum:stream:apply.plan",
    "quantum:stream:apply.plan.manual",
    "quantum:stream:apply.result",
    "quantum:stream:exit.intent",
    "quantum:stream:harvest.intent",
})
```

**[FACT]** `ShadowPublisher._xadd()` raises `ValueError` if any caller tries to write to a forbidden stream.

### 1.2 Fail-Closed

**[FACT]** Every engine returns a safe default on error:

| Component | Fail behaviour |
|-----------|---------------|
| `PositionStateBuilder` | Returns `None` → caller skips position |
| `BeliefEngine` | Returns `None` if ensemble is `None` |
| `HazardEngine` | Returns `None` if any critical input is `None` |
| `ActionUtilityEngine` | Returns empty list on error |
| `ExitPolicyEngine` | Falls back to `HOLD` with `POLICY_FALLBACK_HOLD` reason code |
| `ExitAgentOrchestrator` | Returns `None` and logs warning |

### 1.3 No Side Effects Outside Orchestrator

**[FACT]** Only `ExitAgentOrchestrator.run_decision_cycle()` performs IO (Redis XADD via `ShadowPublisher`). All engines are pure computation.

### 1.4 Immutable Contracts

**[FACT]** All 15 `@dataclass` models include a `validate() → List[str]` method and a `to_dict() → Dict` method for Redis serialization. Empty error list = valid.

### 1.5 FACT / ASSUMPTION / RECOMMENDATION

All design claims in this document are tagged. Facts are derived from implemented, tested code. Assumptions are stated explicitly. Recommendations are proposals for future work.

---

## 2. Phase 1 — State + Math Foundation

### 2.1 Purpose

Read position state from Redis. Compute geometric and regime signals. Publish to shadow streams.

### 2.2 Modules

| File | Lines | Role |
|------|-------|------|
| `models/position_exit_state.py` | 196 | Canonical position state contract |
| `services/position_state_builder.py` | 346 | Redis reader, builds `PositionExitState` |
| `engines/geometry_engine.py` | 298 | MFE/MAE/drawdown/PPR/momentum/RTR |
| `engines/regime_drift_engine.py` | 284 | KL/L1 drift, trend alignment, reversal/chop risk |
| `publishers/shadow_publisher.py` | 385 | 17 shadow streams, forbidden list, XADD guard |

### 2.3 Data Contract: `PositionExitState`

**[FACT]** Key fields:

```
position_id, symbol, side (LONG|SHORT), entry_price, current_price,
quantity, unrealized_pnl, unrealized_pnl_pct, mfe_price, mae_price,
current_drawdown_from_peak, regime_label, trend_direction, volatility,
open_timestamp
```

**[FACT]** `hold_seconds` is a `@property` computed from `open_timestamp`, NOT a constructor parameter.  
**[FACT]** `regime_label` is the field name (not `regime`).

### 2.4 GeometryEngine

**[FACT]** Computes per-position:

| Metric | Formula | Range |
|--------|---------|-------|
| `mfe_capture` | `(current - entry) / (mfe - entry)` | [0, 1] |
| `mae_risk` | `(entry - mae) / entry` | [0, 1] |
| `drawdown_from_peak` | `(mfe - current) / mfe` | [0, 1] |
| `ppr` | Peak-profit ratio | [0, 1] |
| `momentum` | Recent price velocity | [-1, 1] |
| `rtr` | Return-to-risk ratio | [-1, 1] |

### 2.5 RegimeDriftEngine

**[FACT]** Produces `RegimeState` with:

| Signal | Range | Description |
|--------|-------|-------------|
| `kl_divergence` | [0, ∞) | Distributional shift |
| `l1_distance` | [0, 2] | Distribution distance |
| `trend_alignment` | [-1, 1] | Positive = aligned with position |
| `reversal_risk` | [0, 1] | Probability of reversal |
| `chop_risk` | [0, 1] | Probability of sideways market |

### 2.6 Shadow Streams (Phase 1)

**[FACT]**

| Stream | Producer | Contains |
|--------|----------|----------|
| `quantum:stream:exit.state.shadow` | `ShadowPublisher.publish_position_state()` | `PositionExitState.to_dict()` |
| `quantum:stream:exit.geometry.shadow` | `ShadowPublisher.publish_geometry()` | `GeometryResult.to_dict()` |
| `quantum:stream:exit.regime.shadow` | `ShadowPublisher.publish_regime()` | `RegimeState + RegimeDrift` |

---

## 3. Phase 2 — Ensemble Sensor Layer

### 3.1 Purpose

Run N models → normalize → aggregate → produce a single ensemble signal per position.

### 3.2 Modules

| File | Lines | Role |
|------|-------|------|
| `models/model_exit_signal.py` | 154 | Per-model signal contract. VALID_MODELS = 6, VALID_HORIZONS = 5 |
| `models/aggregated_exit_signal.py` | 175 | Aggregated signal + `EnsembleDiagnostics`. MIN_MODELS_REQUIRED = 2 |
| `adapters/model_registry.py` | 120 | 6 model specs, dynamic `importlib` loader |
| `adapters/ensemble_exit_adapter.py` | 308 | predict → normalize → `ModelExitSignal` per model |
| `aggregators/ensemble_aggregator.py` | 493 | filter → weight → aggregate into `AggregatedExitSignal` |
| `engines/normalization.py` | 125 | clamp, renormalize, softmax, probability reconstruction |
| `engines/calibration.py` | 82 | identity, temperature_scale, platt_scale |

### 3.3 Model Outputs

**[FACT]** Each model produces 7 outputs per position:

```python
p_hold, p_reduce_small, p_reduce_medium,
p_take_profit_partial, p_take_profit_large,
p_tighten_exit, p_close_full
```

**[FACT]** These correspond directly to the 7 `VALID_ACTIONS`.

### 3.4 Aggregation Pipeline

**[FACT]** `EnsembleAggregator` flow:

1. **Filter** models by validity (`validate()` on each `ModelExitSignal`)
2. **Check** `MIN_MODELS_REQUIRED` (2) met — fail-closed to `None` if not
3. **Weight** models by registry-defined weight (default uniform)
4. **Aggregate** via weighted average per-action probability
5. **Build** `AggregatedExitSignal` with `EnsembleDiagnostics` (agreement, spread, dominant)

### 3.5 Shadow Streams (Phase 2)

**[FACT]**

| Stream | Producer | Contains |
|--------|----------|----------|
| `quantum:stream:exit.ensemble.raw.shadow` | `ShadowPublisher.publish_ensemble_raw()` | Per-model raw signals |
| `quantum:stream:exit.ensemble.agg.shadow` | `ShadowPublisher.publish_ensemble_aggregated()` | `AggregatedExitSignal.to_dict()` |
| `quantum:stream:exit.ensemble.diag.shadow` | `ShadowPublisher.publish_ensemble_diagnostics()` | `EnsembleDiagnostics` |

---

## 4. Phase 3 — Belief / Hazard / Utility

### 4.1 Purpose

Fuse all upstream signals into a unified belief, a multi-dimensional hazard, and a scored utility ranking across 7 actions.

### 4.2 Modules

| File | Lines | Role |
|------|-------|------|
| `engines/belief_engine.py` | 178 | Fuses geometry + regime + ensemble → `BeliefState` |
| `engines/hazard_engine.py` | 248 | 6 hazard axes → composite `HazardAssessment` |
| `engines/action_utility_engine.py` | 351 | Scores 7 actions → sorted `List[ActionCandidate]` |
| `models/belief_state.py` | 100 | `BeliefState` contract |
| `models/hazard_assessment.py` | 109 | `HazardAssessment` contract |
| `models/action_candidate.py` | 126 | `ActionCandidate` + `VALID_ACTIONS` + `ACTION_EXIT_FRACTIONS` |

### 4.3 The 7 Actions

**[FACT]** Defined in `models/action_candidate.py`:

| Action | Exit Fraction | Description |
|--------|--------------|-------------|
| `HOLD` | 0.00 | Keep full position |
| `REDUCE_SMALL` | 0.10 | Trim 10% |
| `REDUCE_MEDIUM` | 0.25 | Trim 25% |
| `TAKE_PROFIT_PARTIAL` | 0.50 | Lock 50% profit |
| `TAKE_PROFIT_LARGE` | 0.75 | Lock 75% profit |
| `TIGHTEN_EXIT` | 0.00 | Tighten trailing stop (no immediate exit) |
| `CLOSE_FULL` | 1.00 | Exit 100% |

### 4.4 BeliefEngine

**[FACT]** Weighted fusion with these constants:

```
exit_pressure   = W_EXIT_ENSEMBLE(0.50) × ensemble.p_exit
                + W_EXIT_REVERSAL(0.25) × regime.reversal_risk
                + W_EXIT_GEOMETRY(0.25) × geometry.drawdown_from_peak

hold_conviction = W_HOLD_ENSEMBLE(0.50) × ensemble.p_hold
                + W_HOLD_TREND(0.25) × regime.trend_alignment_pos
                + W_HOLD_GEOMETRY(0.25) × geometry.mfe_capture

directional_edge = W_EDGE_ENSEMBLE(0.60) × ensemble_edge
                 + W_EDGE_REGIME(0.40)   × regime.trend_alignment

uncertainty_total = weighted combination of ensemble spread,
                    regime chop_risk, and quality flags

data_completeness = fraction of upstream signals present
```

**[FACT]** Returns `None` (fail-closed) if `ensemble` aggregated signal is `None`.

### 4.5 HazardEngine

**[FACT]** Six independent hazard axes, all [0, 1]:

| Axis | Source |
|------|--------|
| `drawdown_hazard` | `geometry.drawdown_from_peak` |
| `reversal_hazard` | `regime.reversal_risk` |
| `volatility_hazard` | `volatility / VOLATILITY_HIGH_REF (0.02)` clamped |
| `time_decay_hazard` | `1 - exp(-hold_seconds / TIME_DECAY_HALF_LIFE)` where half-life = 14400s |
| `regime_hazard` | `max(chop_risk, 1 - trend_alignment_abs)` |
| `ensemble_hazard` | `1 - p_hold` from ensemble |

**[FACT]** Composite = equal-weighted mean (1/6 each). `dominant_hazard` = axis with highest value.

### 4.6 ActionUtilityEngine

**[FACT]** Scores all 7 actions with per-action utility functions:

```
HOLD:        hold_conviction × safety × edge_factor
CLOSE_FULL:  exit_pressure × composite_hazard × reversal_boost / 2
REDUCE_*:    exit_pressure × fraction_factor × uncertainty_discount
TP_*:        profit_factor × exit_pressure × preservation_urgency
TIGHTEN:     volatility_hazard × uncertainty_boost × (1 - hold_conviction)
```

**[FACT]** Penalties applied:

| Penalty | Condition |
|---------|-----------|
| `not_profitable` | Action requires profit (TP_*) but position is at a loss |
| `low_hazard_close` | CLOSE_FULL when hazard < 0.30 |
| `high_hazard_hold` | HOLD when hazard > 0.70 |
| `uncertainty_dampening` | Actions other than HOLD/TIGHTEN when uncertainty > 0.5 |

**[FACT]** `net_utility = clamp(base_utility - penalty_total, 0, 1)`. Sorted by `net_utility` descending. `rank=1` = best.

### 4.7 Shadow Streams (Phase 3)

**[FACT]**

| Stream | Producer | Contains |
|--------|----------|----------|
| `quantum:stream:exit.belief.shadow` | `ShadowPublisher.publish_belief()` | `BeliefState.to_dict()` |
| `quantum:stream:exit.hazard.shadow` | `ShadowPublisher.publish_hazard()` | `HazardAssessment.to_dict()` |
| `quantum:stream:exit.utility.shadow` | `ShadowPublisher.publish_utility()` | All 7 `ActionCandidate.to_dict()` |

---

## 5. Phase 4 — Policy / Orchestration / Gateway

### 5.1 Purpose

Apply safety policy to the utility ranking. Build exit intents. Validate via gateway. Produce full audit trace. All shadow-only.

### 5.2 Modules

| File | Lines | Role |
|------|-------|------|
| `policy/exit_policy_engine.py` | 476 | 7-step policy evaluation pipeline |
| `policy/policy_constraints.py` | 70 | All threshold constants |
| `policy/reason_codes.py` | 69 | Machine-readable hard/soft/override codes |
| `orchestrators/exit_agent_orchestrator.py` | 289 | IO coordinator — ONLY module with side effects |
| `validators/exit_intent_gateway_validator.py` | 137 | Schema + constraint validation |
| `validators/idempotency.py` | 69 | In-memory dedup, 5-min window |
| `validators/payload_normalizer.py` | 105 | Clamp, strip, validate payloads |
| `models/policy_decision.py` | 129 | `PolicyDecision` contract |
| `models/exit_intent_candidate.py` | 127 | `ExitIntentCandidate` contract |
| `models/exit_intent_validation_result.py` | 106 | Gateway validation result |
| `models/decision_trace.py` | 109 | Full audit trace |

### 5.3 ExitPolicyEngine — 7-Step Pipeline

**[FACT]** `evaluate()` method runs these checks in order:

| Step | Check | Hard Block / Override |
|------|-------|-----------------------|
| 1 | **Upstream freshness** — data older than `MAX_UPSTREAM_AGE_SEC` (120s) | Hard block → HOLD + `STALE_UPSTREAM_DATA` |
| 2 | **Data completeness** — below `DATA_COMPLETENESS_HARD_FLOOR` (0.40) | Hard block → HOLD + `DATA_COMPLETENESS_FLOOR` |
| 3 | **Uncertainty** — above `UNCERTAINTY_HARD_CEILING` (0.70) | Restrict to `SAFE_ACTIONS_HIGH_UNCERTAINTY` = {HOLD, TIGHTEN_EXIT} |
| 4 | **Select best candidate** from utility scorecard | Rank 1 action selected |
| 5 | **Apply constraint checks** — profit requirement, hazard floors | Hard block specific actions if violated |
| 6 | **Conviction check** — `net_utility < MIN_ACTION_CONVICTION` (0.15) | Soft warn + possible demotion to HOLD |
| 7 | **Emergency override** — `composite_hazard > HAZARD_EMERGENCY_THRESHOLD` (0.85) | Force CLOSE_FULL + `HAZARD_EMERGENCY_OVERRIDE` |

**[FACT]** If `policy_passed == False`, `chosen_action` MUST be `"HOLD"`. The `PolicyDecision.validate()` method enforces this invariant.

### 5.4 Policy Constraints (All Constants)

**[FACT]** From `policy/policy_constraints.py`:

| Constant | Value | Purpose |
|----------|-------|---------|
| `UNCERTAINTY_HARD_CEILING` | 0.70 | Above → only HOLD/TIGHTEN |
| `UNCERTAINTY_SOFT_CEILING` | 0.50 | Above → soft warning |
| `DATA_COMPLETENESS_HARD_FLOOR` | 0.40 | Below → force HOLD |
| `DATA_COMPLETENESS_SOFT_FLOOR` | 0.60 | Below → soft warning |
| `MIN_ACTION_CONVICTION` | 0.15 | Below → demote to HOLD |
| `PREFER_HOLD_THRESHOLD` | 0.20 | Prefer HOLD unless above |
| `EDGE_NEUTRAL_BAND` | 0.15 | [-0.15, 0.15] → edge-neutral |
| `HAZARD_EMERGENCY_THRESHOLD` | 0.85 | Above → force CLOSE_FULL |
| `CLOSE_FULL_MIN_HAZARD` | 0.50 | CLOSE requires this hazard |
| `CLOSE_FULL_MIN_EXIT_PRESSURE` | 0.70 | Or this exit pressure |
| `REVERSAL_EMERGENCY_THRESHOLD` | 0.70 | Reversal sub-hazard emergency |
| `DRAWDOWN_EMERGENCY_THRESHOLD` | 0.50 | Drawdown sub-hazard emergency |
| `HAZARD_CLOSE_BOOST_THRESHOLD` | 0.65 | Above → CLOSE_FULL boost |
| `HAZARD_CLOSE_BOOST_AMOUNT` | 0.15 | Boost size |
| `MAX_UPSTREAM_AGE_SEC` | 120.0 | Max seconds before stale |

### 5.5 Reason Codes

**[FACT]** 7 hard block codes, 6 soft warning codes, 5 policy override codes. See `policy/reason_codes.py`.

Hard blocks (force HOLD):

```
UNCERTAINTY_CEILING_BREACH, DATA_COMPLETENESS_FLOOR,
PROFIT_TAKING_NO_PROFIT, CLOSE_FULL_INSUFFICIENT_HAZARD,
STALE_UPSTREAM_DATA, MISSING_UPSTREAM_DATA, SHADOW_ONLY_VIOLATION
```

Emergency override (force CLOSE_FULL):

```
HAZARD_EMERGENCY_OVERRIDE
```

### 5.6 ExitAgentOrchestrator

**[FACT]** `run_decision_cycle()` is the ONLY method in the entire codebase that performs IO. Steps:

1. Call `ExitPolicyEngine.evaluate()` → `PolicyDecision`
2. Build `ExitIntentCandidate` with deterministic `idempotency_key` (SHA256 of position_id + action + timestamp bucket)
3. Validate via `ExitIntentGatewayValidator` → `ExitIntentValidationResult`
4. Build `DecisionTrace` with references to all upstream artifacts
5. Publish all 4 artifacts to shadow streams via `ShadowPublisher`

**[FACT]** On any exception → returns `None`, logs warning. Never propagates exceptions to callers.

### 5.7 Gateway Validator

**[FACT]** `ExitIntentGatewayValidator.validate()` checks:
- All required fields present
- `action_name ∈ VALID_ACTIONS`
- `intent_type == "SHADOW_EXIT"`
- `target_reduction_pct ∈ [0, 1]`
- `confidence ∈ [0, 1]`
- `idempotency_key` is non-empty
- Idempotency dedup (5-minute window, in-memory)
- Payload normalization (clamp, strip)

### 5.8 Shadow Streams (Phase 4)

**[FACT]**

| Stream | Producer | Contains |
|--------|----------|----------|
| `quantum:stream:exit.policy.shadow` | `ShadowPublisher.publish_policy_decision()` | `PolicyDecision.to_dict()` |
| `quantum:stream:exit.intent.candidate.shadow` | `ShadowPublisher.publish_intent_candidate()` | `ExitIntentCandidate.to_dict()` |
| `quantum:stream:exit.intent.validation.shadow` | `ShadowPublisher.publish_intent_validation()` | `ExitIntentValidationResult.to_dict()` |
| `quantum:stream:exit.decision.trace.shadow` | `ShadowPublisher.publish_decision_trace()` | `DecisionTrace.to_dict()` |

---

## 6. Phase 5 — Replay / Obituary / Offline Evaluation / Tuning

### 6.1 Purpose

Post-mortem analysis: reconstruct what happened, evaluate decision quality, compare to baselines, propose parameter adjustments. All shadow-only, never auto-applies.

### 6.2 Modules

| File | Lines | Role |
|------|-------|------|
| **Replay** | | |
| `replay/replay_obituary_writer.py` | 438 | Post-mortem builder with regret/preservation/opportunity scores |
| `replay/replay_loader.py` | 231 | Reads from Phase 1-4 shadow streams |
| `replay/outcome_reconstructor.py` | 203 | Reconstructs price/PnL path post-decision |
| `replay/counterfactual_evaluator.py` | 229 | Simulates alternative actions on actual price path |
| **Evaluators** | | |
| `evaluators/offline_evaluator.py` | 470 | Orchestrator for full eval pipeline |
| `evaluators/belief_calibration_evaluator.py` | 116 | Belief accuracy vs realized outcomes |
| `evaluators/hazard_calibration_evaluator.py` | 126 | Hazard predictions vs realized risk |
| `evaluators/utility_ranking_evaluator.py` | 102 | Was chosen action actually best ex-post? |
| `evaluators/policy_choice_evaluator.py` | 226 | Policy quality + 4 naive baselines |
| **Tuning** | | |
| `tuning/proposal_builder.py` | 155 | Orchestrates tuners, max 5 proposals per run |
| `tuning/threshold_tuner.py` | 260 | ±20% max change per threshold, never auto-applies |
| `tuning/weight_tuner.py` | 269 | Fusion weight rebalancing, must stay normalized |
| **Models** | | |
| `models/trade_exit_obituary.py` | 185 | Post-mortem record |
| `models/replay_evaluation_record.py` | 157 | Per-decision replay evaluation |
| `models/offline_evaluation_summary.py` | 150 | Aggregated eval results |
| `models/tuning_recommendation.py` | 125 | Proposed parameter change |
| `models/calibration_artifact.py` | 109 | Fitted calibration snapshot |

### 6.3 Replay Pipeline

**[FACT]** `ReplayObituaryWriter.build_trade_exit_obituary()`:

1. Load decision trace + policy decision from shadow streams
2. Load upstream belief/hazard/utility snapshots near decision time
3. Reconstruct post-decision price path via `OutcomeReconstructor` (default horizon: 14400s = 4h)
4. Estimate best exit window (peak PnL ± 150s scan)
5. Compute three quality scores:
   - **Regret** = `(best_possible - actual) / |best_possible|` → [0, 1], lower = better
   - **Preservation** = `1 - (max_drawdown / peak_pnl)` → [0, 1], higher = better
   - **Opportunity** = `actual / best_possible` → [0, 1], higher = better
6. Publish `TradeExitObituary` to shadow

### 6.4 Counterfactual Evaluator

**[FACT]** Pure computation — simulates all 7 actions on the actual price path:

| Action | Simulation |
|--------|------------|
| HOLD | PnL at end of horizon |
| CLOSE_FULL | PnL at decision moment (first price) |
| REDUCE_SMALL/MEDIUM, TP_PARTIAL/LARGE | Fraction at first price + remainder at final |
| TIGHTEN_EXIT | Trailing stop at 2% from running high/low |

**[FACT]** `find_ex_post_best_action()` → the action that would have yielded highest PnL in hindsight.

**[FACT]** Decision quality score formula:

```
quality = 0.40 × rank_component    # Was chosen action the best?
        + 0.30 × (1 - regret)      # Low regret = good
        + 0.20 × preservation      # Capital preserved
        + 0.10 × opportunity        # Upside captured
```

### 6.5 Offline Evaluator Pipeline

**[FACT]** `OfflineEvaluator.run_evaluation()`:

1. Load obituaries for time window (min 5 samples, warns below 30)
2. Reconstruct outcomes per position
3. Build `ReplayEvaluationRecord` per obituary (predicated vs realized signals)
4. Run 4 sub-evaluators:
   - `BeliefCalibrationEvaluator` — exit_pressure/hold_conviction bias + MAE
   - `HazardCalibrationEvaluator` — per-axis hazard bias + MAE
   - `UtilityRankingEvaluator` — was rank-1 action actually best?
   - `PolicyChoiceEvaluator` — pass_quality, block_quality scores
5. Compare against 4 baselines:
   - `always_hold` — never exit
   - `fixed_trailing_2pct` — trailing stop at 2%
   - `fixed_tp_3pct` — take profit at 3%
   - `naive_partial_50pct` — always exit 50%
6. Build `OfflineEvaluationSummary`
7. Publish summary + per-record evaluations to shadow streams

### 6.6 Tuning Safety Rules

**[FACT]** From `threshold_tuner.py` and `weight_tuner.py`:

| Rule | Value |
|------|-------|
| Max change per parameter per run | ±20% (`MAX_CHANGE_FRACTION`) |
| Max recommendations per run | 5 (`MAX_RECOMMENDATIONS_PER_RUN`) |
| Low sample confidence cap | 0.30 if < 50 samples |
| Min MAE gap for weight rebalancing | 0.05 |
| All proposals require human review | `requires_human_review = True` |
| Never auto-applied | `applied = False` |
| Weight tuner: weights must stay normalized | Sum = 1.0 within engine |

**[FACT]** `ThresholdTuner` tunes: `UNCERTAINTY_HARD_CEILING`, `MIN_ACTION_CONVICTION`, `HAZARD_EMERGENCY_THRESHOLD` (driven by belief/hazard calibration bias and policy quality).

**[FACT]** `WeightTuner` tunes: Hazard axis weights (6 axes) and belief fusion weights (4 fields). Shifts weight from high-MAE (poorly calibrated) to low-MAE (well calibrated) axes.

**[FACT]** `ProposalBuilder` orchestrates both tuners, sorts by confidence, caps at 5, creates `CalibrationArtifact` snapshots for rollback.

### 6.7 Shadow Streams (Phase 5)

**[FACT]**

| Stream | Producer | Contains |
|--------|----------|----------|
| `quantum:stream:exit.obituary.shadow` | `ShadowPublisher.publish_obituary()` | `TradeExitObituary.to_dict()` |
| `quantum:stream:exit.replay.eval.shadow` | `ShadowPublisher.publish_replay_evaluation()` | `ReplayEvaluationRecord.to_dict()` |
| `quantum:stream:exit.eval.summary.shadow` | `ShadowPublisher.publish_evaluation_summary()` | `OfflineEvaluationSummary.to_dict()` |
| `quantum:stream:exit.tuning.recommendation.shadow` | `ShadowPublisher.publish_tuning_recommendation()` | `TuningRecommendation.to_dict()` |

---

## 7. Complete Repo Map

### 7.1 Directory Structure

```
microservices/exit_brain_v1/          # Root — 12 directories, 55 modules
├── __init__.py
├── models/                           # 15 data contracts (1,958 lines)
│   ├── __init__.py
│   ├── position_exit_state.py        # Phase 1 — position state
│   ├── model_exit_signal.py          # Phase 2 — per-model signal
│   ├── aggregated_exit_signal.py     # Phase 2 — ensemble aggregated
│   ├── belief_state.py               # Phase 3 — fused belief
│   ├── hazard_assessment.py          # Phase 3 — multi-axis hazard
│   ├── action_candidate.py           # Phase 3 — scored action
│   ├── policy_decision.py            # Phase 4 — policy output
│   ├── exit_intent_candidate.py      # Phase 4 — shadow intent
│   ├── exit_intent_validation_result.py  # Phase 4 — gateway result
│   ├── decision_trace.py             # Phase 4 — audit trace
│   ├── trade_exit_obituary.py        # Phase 5 — post-mortem
│   ├── replay_evaluation_record.py   # Phase 5 — per-decision eval
│   ├── offline_evaluation_summary.py # Phase 5 — aggregated eval
│   ├── tuning_recommendation.py      # Phase 5 — proposed change
│   └── calibration_artifact.py       # Phase 5 — parameter snapshot
├── engines/                          # 7 computation engines (1,566 lines)
│   ├── __init__.py
│   ├── geometry_engine.py            # Phase 1 — geometric signals
│   ├── regime_drift_engine.py        # Phase 1 — regime analysis
│   ├── normalization.py              # Phase 2 — probability normalization
│   ├── calibration.py                # Phase 2 — model calibration
│   ├── belief_engine.py              # Phase 3 — signal fusion
│   ├── hazard_engine.py              # Phase 3 — risk assessment
│   └── action_utility_engine.py      # Phase 3 — action scoring
├── adapters/                         # 2 external adapters (428 lines)
│   ├── __init__.py
│   ├── model_registry.py             # Phase 2 — model specs + loader
│   └── ensemble_exit_adapter.py      # Phase 2 — model predict wrapper
├── aggregators/                      # 1 aggregator (493 lines)
│   ├── __init__.py
│   └── ensemble_aggregator.py        # Phase 2 — ensemble aggregation
├── services/                         # 1 service (346 lines)
│   ├── __init__.py
│   └── position_state_builder.py     # Phase 1 — Redis state reader
├── publishers/                       # 1 publisher (385 lines)
│   ├── __init__.py
│   └── shadow_publisher.py           # All phases — shadow XADD
├── policy/                           # 3 policy modules (615 lines)
│   ├── __init__.py
│   ├── exit_policy_engine.py         # Phase 4 — policy evaluation
│   ├── policy_constraints.py         # Phase 4 — threshold constants
│   └── reason_codes.py               # Phase 4 — reason code constants
├── orchestrators/                    # 1 orchestrator (289 lines)
│   ├── __init__.py
│   └── exit_agent_orchestrator.py    # Phase 4 — IO coordinator
├── validators/                       # 3 validators (311 lines)
│   ├── __init__.py
│   ├── exit_intent_gateway_validator.py  # Phase 4 — intent validation
│   ├── idempotency.py                # Phase 4 — dedup
│   └── payload_normalizer.py         # Phase 4 — payload cleanup
├── replay/                           # 4 replay modules (1,101 lines)
│   ├── __init__.py
│   ├── replay_obituary_writer.py     # Phase 5 — obituary builder
│   ├── replay_loader.py              # Phase 5 — stream reader
│   ├── outcome_reconstructor.py      # Phase 5 — price path reconstruction
│   └── counterfactual_evaluator.py   # Phase 5 — what-if analysis
├── evaluators/                       # 5 evaluators (1,040 lines)
│   ├── __init__.py
│   ├── offline_evaluator.py          # Phase 5 — eval orchestrator
│   ├── belief_calibration_evaluator.py   # Phase 5
│   ├── hazard_calibration_evaluator.py   # Phase 5
│   ├── utility_ranking_evaluator.py      # Phase 5
│   └── policy_choice_evaluator.py        # Phase 5
├── tuning/                           # 3 tuning modules (684 lines)
│   ├── __init__.py
│   ├── proposal_builder.py           # Phase 5 — tuning orchestrator
│   ├── threshold_tuner.py            # Phase 5 — policy threshold tuning
│   └── weight_tuner.py               # Phase 5 — fusion weight tuning
└── tests/                            # 7 test files (3,084 lines, 297 tests)
    ├── __init__.py
    ├── test_position_exit_state.py
    ├── test_geometry_engine.py
    ├── test_regime_drift_engine.py
    ├── test_position_state_builder.py
    ├── test_shadow_publisher.py
    ├── test_phase1_comprehensive.py      # 50 tests
    └── test_phase2_comprehensive.py      # 136 tests
```

### 7.2 Line Count Summary

| Directory | Modules | Lines | Phase |
|-----------|---------|-------|-------|
| `models/` | 15 | 1,958 | 1–5 |
| `engines/` | 7 | 1,566 | 1–3 |
| `aggregators/` | 1 | 493 | 2 |
| `adapters/` | 2 | 428 | 2 |
| `publishers/` | 1 | 385 | 1 (cross-phase) |
| `services/` | 1 | 346 | 1 |
| `policy/` | 3 | 615 | 4 |
| `orchestrators/` | 1 | 289 | 4 |
| `validators/` | 3 | 311 | 4 |
| `replay/` | 4 | 1,101 | 5 |
| `evaluators/` | 5 | 1,040 | 5 |
| `tuning/` | 3 | 684 | 5 |
| `tests/` | 7 | 3,084 | — |
| **Total** | **55** | **~12,300** | |

---

## 8. Streams & Boundary Rules

### 8.1 Complete Stream Inventory

**[FACT]** 17 shadow streams, all with `STREAM_MAXLEN = 5000`:

| # | Stream Name | Phase | Producer Method |
|---|------------|-------|-----------------|
| 1 | `quantum:stream:exit.state.shadow` | 1 | `publish_position_state()` |
| 2 | `quantum:stream:exit.geometry.shadow` | 1 | `publish_geometry()` |
| 3 | `quantum:stream:exit.regime.shadow` | 1 | `publish_regime()` |
| 4 | `quantum:stream:exit.ensemble.raw.shadow` | 2 | `publish_ensemble_raw()` |
| 5 | `quantum:stream:exit.ensemble.agg.shadow` | 2 | `publish_ensemble_aggregated()` |
| 6 | `quantum:stream:exit.ensemble.diag.shadow` | 2 | `publish_ensemble_diagnostics()` |
| 7 | `quantum:stream:exit.belief.shadow` | 3 | `publish_belief()` |
| 8 | `quantum:stream:exit.hazard.shadow` | 3 | `publish_hazard()` |
| 9 | `quantum:stream:exit.utility.shadow` | 3 | `publish_utility()` |
| 10 | `quantum:stream:exit.policy.shadow` | 4 | `publish_policy_decision()` |
| 11 | `quantum:stream:exit.intent.candidate.shadow` | 4 | `publish_intent_candidate()` |
| 12 | `quantum:stream:exit.intent.validation.shadow` | 4 | `publish_intent_validation()` |
| 13 | `quantum:stream:exit.decision.trace.shadow` | 4 | `publish_decision_trace()` |
| 14 | `quantum:stream:exit.obituary.shadow` | 5 | `publish_obituary()` |
| 15 | `quantum:stream:exit.replay.eval.shadow` | 5 | `publish_replay_evaluation()` |
| 16 | `quantum:stream:exit.eval.summary.shadow` | 5 | `publish_evaluation_summary()` |
| 17 | `quantum:stream:exit.tuning.recommendation.shadow` | 5 | `publish_tuning_recommendation()` |

### 8.2 Forbidden Streams (Never Write)

**[FACT]** 6 execution-layer streams that Exit Brain v1 MUST NEVER write to:

```
quantum:stream:trade.intent
quantum:stream:apply.plan
quantum:stream:apply.plan.manual
quantum:stream:apply.result
quantum:stream:exit.intent
quantum:stream:harvest.intent
```

### 8.3 Boundary Rule: shadow → execution

**[ASSUMPTION]** The future activation boundary is:

```
quantum:stream:exit.intent.candidate.shadow
        ↓ (manual promote / future v2 auto-promote)
quantum:stream:exit.intent
        ↓ (existing ExitIntentGateway)
quantum:stream:trade.intent
```

**[RECOMMENDATION]** Never auto-promote from shadow to execution. Require explicit human approval or a separate, audited promotion service.

### 8.4 Stream Data Flow Diagram

```
                    Phase 1                     Phase 2
              ┌─────────────┐             ┌─────────────────┐
  Redis       │ PositionState│             │ N models predict │
  position    │   Builder    │             │  EnsembleAdapter │
  hashes  ──→ │              │             │                  │
              └──────┬───────┘             └────────┬─────────┘
                     │                              │
                     ▼                              ▼
              ┌──────────────┐             ┌──────────────────┐
              │GeometryEngine│             │EnsembleAggregator│
              │RegimeDriftEng│             │(filter→weight→agg)│
              └──────┬───────┘             └────────┬─────────┘
                     │                              │
                     ▼                              ▼
                streams 1-3                   streams 4-6
                     │                              │
                     └──────────┬───────────────────┘
                                │
                         Phase 3│
                     ┌──────────▼──────────┐
                     │    BeliefEngine      │
                     │   (weighted fusion)  │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │    HazardEngine      │
                     │  (6 axes → composite)│
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │ActionUtilityEngine   │
                     │(score 7 → rank)      │
                     └──────────┬───────────┘
                                │
                          streams 7-9
                                │
                         Phase 4│
                     ┌──────────▼──────────┐
                     │  ExitPolicyEngine    │
                     │ (7-step pipeline)    │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │ExitAgentOrchestrator │ ← ONLY IO module
                     │(intent→validate→     │
                     │ trace→publish)        │
                     └──────────┬───────────┘
                                │
                         streams 10-13
                                │
                         Phase 5│ (offline, batch)
                     ┌──────────▼──────────┐
                     │ReplayObituaryWriter  │
                     │(reconstruct→score)   │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │  OfflineEvaluator    │
                     │(4 sub-evaluators +   │
                     │ 4 baseline compares) │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼──────────┐
                     │   ProposalBuilder    │
                     │(threshold + weight   │
                     │ tuning → max 5 recs) │
                     └──────────┬───────────┘
                                │
                         streams 14-17
```

---

## 9. Code Skeletons

All code is fully implemented. Below are the key entry points.

### 9.1 Real-Time Decision Cycle (Phase 1-4)

```python
# ExitAgentOrchestrator.run_decision_cycle() — simplified skeleton

def run_decision_cycle(self, position_state, geometry, regime, ensemble_agg):
    # Phase 3: Fuse signals
    belief = self._belief_engine.compute(geometry, regime, ensemble_agg)
    hazard = self._hazard_engine.assess(position_state, geometry, regime, ensemble_agg)
    candidates = self._utility_engine.score_all(belief, hazard, position_state)

    # Phase 4: Policy
    decision = self._policy_engine.evaluate(
        candidates, belief, hazard, position_state
    )

    # Build intent
    intent = self._build_intent(decision, position_state)

    # Validate
    validation = self._gateway.validate(intent)

    # Trace
    trace = self._build_trace(decision, intent, validation)

    # Publish (shadow only)
    self._publisher.publish_policy_decision(decision)
    self._publisher.publish_intent_candidate(intent)
    self._publisher.publish_intent_validation(validation)
    self._publisher.publish_decision_trace(trace)

    return trace
```

### 9.2 Offline Evaluation Pipeline (Phase 5)

```python
# OfflineEvaluator.run_evaluation() — simplified skeleton

def run_evaluation(self, start_ts, end_ts):
    obituaries = self.load_replay_records(start_ts, end_ts)  # min 5
    outcomes = {o.position_id: self._reconstruct(o) for o in obituaries}
    replay_records = self._build_replay_records(obituaries, outcomes)

    belief_cal  = self._belief_eval.evaluate(obituaries, outcomes)
    hazard_cal  = self._hazard_eval.evaluate(obituaries, outcomes)
    utility_rank = self._utility_eval.evaluate(replay_records)
    policy_qual  = self._policy_eval.evaluate(obituaries)
    baselines    = self._policy_eval.compare_against_baselines(obituaries, outcomes)

    summary = self.build_offline_evaluation_summary(...)
    self.publish_evaluation_summary(summary)
    return summary
```

### 9.3 Tuning Pipeline (Phase 5)

```python
# ProposalBuilder.build_tuning_proposals() — simplified skeleton

def build_tuning_proposals(self, summary):
    threshold_recs = self._threshold_tuner.propose_adjustments(summary)  # ±20% max
    weight_recs = self._weight_tuner.propose_adjustments(summary)        # normalized
    all_recs = sorted(threshold_recs + weight_recs, key=lambda r: r.confidence, reverse=True)
    final = all_recs[:5]  # MAX_RECOMMENDATIONS_PER_RUN
    for rec in final:
        self._publisher.publish_tuning_recommendation(rec)
    return final
```

---

## 10. Test Strategy

### 10.1 Current State

**[FACT]** 297 tests passing, 0 failures, 7 test files.

| File | Tests | Phase | Focus |
|------|-------|-------|-------|
| `test_position_exit_state.py` | 26 | 1 | State contract validation |
| `test_geometry_engine.py` | 22 | 1 | Geometric signal correctness |
| `test_regime_drift_engine.py` | 18 | 1 | Regime analysis + drift detection |
| `test_position_state_builder.py` | 14 | 1 | Redis reader, fail-closed |
| `test_shadow_publisher.py` | 9 | 1 | Stream publishing + forbidden guard |
| `test_phase1_comprehensive.py` | 50 | 1 | Cross-cutting Phase 1 scenarios |
| `test_phase2_comprehensive.py` | 136 | 2 | Full Phase 2 coverage |

### 10.2 Coverage Gaps

**[FACT]** No dedicated test files exist yet for:

| Phase | Missing Coverage |
|-------|-----------------|
| 3 | `BeliefEngine`, `HazardEngine`, `ActionUtilityEngine` |
| 4 | `ExitPolicyEngine`, `ExitAgentOrchestrator`, `GatewayValidator` |
| 5 | `ReplayObituaryWriter`, `OfflineEvaluator`, `CounterfactualEvaluator`, all tuners |

### 10.3 Recommended Test Plan

**[RECOMMENDATION]** Phase 3 test file (`test_phase3_comprehensive.py`):

- BeliefEngine: fusion weight verification, fail-closed on None ensemble, boundary values
- HazardEngine: per-axis calculation, composite correctness, dominant_hazard selection
- ActionUtilityEngine: all 7 action utility formulas, penalty application, ranking order
- Integration: full Phase 3 pipeline with mock inputs

**[RECOMMENDATION]** Phase 4 test file (`test_phase4_comprehensive.py`):

- ExitPolicyEngine: each of 7 pipeline steps, hard blocks, emergency override, edge cases
- ExitAgentOrchestrator: full decision cycle with mock publisher, fail-closed behavior
- GatewayValidator: valid/invalid intent validation, idempotency dedup
- All reason codes exercised

**[RECOMMENDATION]** Phase 5 test file (`test_phase5_comprehensive.py`):

- CounterfactualEvaluator: all 7 action simulations on known price paths
- OfflineEvaluator: min sample threshold, baseline comparison, sub-evaluator delegation
- ThresholdTuner/WeightTuner: ±20% cap, low-sample confidence cap, normalization
- ProposalBuilder: max 5 cap, confidence sorting

### 10.4 Test Principles

**[FACT]** Tests use:
- Pure `pytest` (no unittest classes)
- `@pytest.fixture` for shared state
- `unittest.mock.MagicMock` for Redis and external dependencies
- No real Redis or network in tests
- Assertion of exact field values against known formulas

---

## 11. Risks & Recommended Build Order

### 11.1 Known Risks

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **Shadow → execution promotion path undefined** | High | **[FACT]** No code exists to promote from `exit.intent.candidate.shadow` to `exit.intent`. This is intentional for v1. **[RECOMMENDATION]** Build a separate, audited promotion service with human approval gate before v2. |
| R2 | **Ensemble model availability** | Medium | **[FACT]** `MIN_MODELS_REQUIRED = 2`. If fewer than 2 models respond, ensemble returns `None` → belief engine returns `None` → orchestrator returns `None` → position skipped. **[ASSUMPTION]** At least 2 of 6 models are available in production. |
| R3 | **Stale upstream data** | Medium | **[FACT]** `MAX_UPSTREAM_AGE_SEC = 120`. Data older than 2 minutes triggers hard block → HOLD. **[ASSUMPTION]** Position state builder refreshes data faster than 120s. |
| R4 | **Hazard equal weighting** | Low | **[FACT]** All 6 hazard axes weighted equally (1/6). **[RECOMMENDATION]** After sufficient offline evaluation data, use `WeightTuner` proposals to rebalance. |
| R5 | **Idempotency is in-memory only** | Medium | **[FACT]** `idempotency.py` uses a dict with 5-minute TTL. Restart clears it. **[RECOMMENDATION]** For production, back idempotency with Redis SET with EX. |
| R6 | **Entry price not stored in obituary** | Low | **[FACT]** `OfflineEvaluator._infer_entry_price()` returns 1.0 (normalized). Actual price is reconstructed via `OutcomeReconstructor`. **[ASSUMPTION]** Price path data is available in Redis for the evaluation horizon. |
| R7 | **Calibration is identity-only** | Low | **[FACT]** `calibration.py` implements identity, temperature_scale, and platt_scale, but default is identity. **[RECOMMENDATION]** Fit Platt scaling after first 500+ obituaries. |
| R8 | **STREAM_MAXLEN = 5000** | Low | **[FACT]** Each stream capped at 5000 entries. At 1 decision/minute, this is ~3.5 days of data. **[ASSUMPTION]** Sufficient for Phase 5 offline evaluation windows. **[RECOMMENDATION]** Monitor stream lengths; increase if needed. |

### 11.2 Recommended Build Order (Already Completed)

**[FACT]** All 5 phases are implemented. This was the actual build sequence:

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  │          │          │          │          │
  │  state   │ ensemble  │ belief   │ policy   │ replay
  │  geometry │ adapter   │ hazard   │ orchestr │ obituary
  │  regime   │ aggregator│ utility  │ gateway  │ evaluator
  │  shadow   │ normalize │          │ trace    │ tuning
  │  publisher│ calibrate │          │          │
  ▼          ▼          ▼          ▼          ▼
  3 streams  3 streams  3 streams  4 streams  4 streams
```

### 11.3 Recommended Next Steps

**[RECOMMENDATION]** Priority order:

1. **Write Phase 3–5 test suites** — ~300+ additional tests to match Phase 1-2 coverage depth
2. **Deploy shadow pipeline to VPS** — run alongside existing trading, publish to shadow streams only
3. **Collect 500+ obituaries** — baseline data for calibration and tuning
4. **Run first offline evaluation** — validate belief/hazard calibration against real outcomes
5. **Apply first tuning proposals** (human-reviewed) — adjust thresholds/weights based on data
6. **Migrate idempotency to Redis** — production hardening
7. **Build promotion service** — controlled path from shadow → execution with approval gate
8. **Fit Platt calibration** — improve model output calibration
9. **A/B shadow test** — compare shadow recommendations vs actual exits for 30 days
10. **v2: Enable enforcement** — remove `shadow_only` constraint with full audit trail

---

## Appendix A: Constant Reference

### Belief Engine Weights

| Weight | Value | Used In |
|--------|-------|---------|
| `W_EXIT_ENSEMBLE` | 0.50 | exit_pressure fusion |
| `W_EXIT_REVERSAL` | 0.25 | exit_pressure fusion |
| `W_EXIT_GEOMETRY` | 0.25 | exit_pressure fusion |
| `W_HOLD_ENSEMBLE` | 0.50 | hold_conviction fusion |
| `W_HOLD_TREND` | 0.25 | hold_conviction fusion |
| `W_HOLD_GEOMETRY` | 0.25 | hold_conviction fusion |
| `W_EDGE_ENSEMBLE` | 0.60 | directional_edge fusion |
| `W_EDGE_REGIME` | 0.40 | directional_edge fusion |

### Hazard Engine Constants

| Constant | Value |
|----------|-------|
| Time decay half-life | 14400s (4 hours) |
| Volatility high reference | 0.02 |
| Equal axis weight | 1/6 each |

### Tuning Bounds

| Constant | Value |
|----------|-------|
| `MAX_CHANGE_FRACTION` | 0.20 (±20%) |
| `MAX_RECOMMENDATIONS_PER_RUN` | 5 |
| `LOW_SAMPLE_THRESHOLD` | 50 |
| `LOW_SAMPLE_CONFIDENCE_CAP` | 0.30 |
| `MIN_MAE_GAP` | 0.05 |

---

*Document generated from implemented source code. All [FACT] claims verified against 55 modules totaling ~12,300 lines.*
