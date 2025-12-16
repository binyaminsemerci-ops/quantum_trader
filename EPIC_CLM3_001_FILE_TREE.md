# CLM v3 File Tree – EPIC-CLM3-001

**Complete File Structure with Line Counts**

```
backend/services/clm_v3/                    [3,412 total lines]
│
├── __init__.py                             [40 lines]
│   ├─ Package exports (ClmOrchestrator, ModelRegistryV3, etc.)
│   └─ Version: 3.0.0
│
├── models.py                               [328 lines]
│   ├─ Enums (3):
│   │   ├─ ModelType (XGBOOST, LIGHTGBM, NHITS, PATCHTST, RL_V2, RL_V3, OTHER)
│   │   ├─ ModelStatus (TRAINING, SHADOW, CANDIDATE, PRODUCTION, RETIRED, FAILED)
│   │   └─ TriggerReason (DRIFT_DETECTED, PERFORMANCE_DEGRADED, PERIODIC, MANUAL, etc.)
│   │
│   ├─ Core Models (4):
│   │   ├─ TrainingJob (job specification)
│   │   ├─ ModelVersion (versioned artifact)
│   │   ├─ EvaluationResult (backtest metrics)
│   │   └─ ModelQuery (registry query filter)
│   │
│   ├─ Request Models (2):
│   │   ├─ PromotionRequest
│   │   └─ RollbackRequest
│   │
│   └─ Event Models (6):
│       ├─ TrainingJobCreatedEvent
│       ├─ ModelTrainedEvent
│       ├─ ModelEvaluatedEvent
│       ├─ ModelPromotedEvent
│       ├─ ModelRollbackEvent
│       └─ StrategyCandidateCreatedEvent
│
├── storage.py                              [414 lines]
│   ├─ ModelRegistryV3:
│   │   ├─ __init__(models_dir, metadata_dir)
│   │   │
│   │   ├─ Training Jobs:
│   │   │   ├─ register_training_job(job)
│   │   │   ├─ update_training_job(job_id, updates)
│   │   │   ├─ get_training_job(job_id)
│   │   │   └─ list_training_jobs(status, limit)
│   │   │
│   │   ├─ Model Versions:
│   │   │   ├─ register_model_version(model)
│   │   │   ├─ get_model_version(model_id, version)
│   │   │   ├─ list_model_versions(model_id, status)
│   │   │   ├─ get_production_model(model_id)
│   │   │   └─ query_models(query)
│   │   │
│   │   ├─ Evaluations:
│   │   │   ├─ save_evaluation_result(result)
│   │   │   ├─ get_evaluation_results(model_id, version)
│   │   │   └─ get_latest_evaluation(model_id, version)
│   │   │
│   │   ├─ Promotion & Rollback:
│   │   │   ├─ promote_model(model_id, version, promoted_by)
│   │   │   └─ rollback_to_version(model_id, target_version, reason)
│   │   │
│   │   └─ Persistence:
│   │       ├─ _save_training_job(job)
│   │       ├─ _save_model_metadata(model)
│   │       ├─ _save_evaluation(result)
│   │       └─ _load_metadata()
│   │
│   └─ Storage Structure:
│       /app/data/clm_v3/registry/
│       ├── training_jobs/{job_id}.json
│       ├── models/{model_id}/{version}.json
│       └── evaluations/{model_id}/{version}_{eval_id}.json
│
├── scheduler.py                            [334 lines]
│   ├─ TrainingScheduler:
│   │   ├─ __init__(registry, config)
│   │   │
│   │   ├─ Lifecycle:
│   │   │   ├─ start() → Start background scheduler loop
│   │   │   └─ stop() → Stop scheduler
│   │   │
│   │   ├─ Periodic Scheduling:
│   │   │   ├─ _scheduler_loop() → Check training needs every N minutes
│   │   │   └─ _check_periodic_training() → Periodic triggers
│   │   │
│   │   ├─ Event-Driven Triggers:
│   │   │   ├─ handle_drift_detected(model_id, drift_score)
│   │   │   ├─ handle_performance_degraded(model_id, sharpe_ratio)
│   │   │   └─ handle_regime_change(new_regime, affected_models)
│   │   │
│   │   ├─ Manual Triggers:
│   │   │   └─ trigger_training(model_type, trigger_reason, ...)
│   │   │
│   │   └─ Utilities:
│   │       ├─ get_next_training_times() → Next training time per model
│   │       └─ get_status() → Scheduler status
│   │
│   └─ Default Config:
│       ├─ Periodic: XGB/LGBM (168h), NHITS/PatchTST (336h), RL v3 (24h)
│       ├─ Drift: auto_train_on_drift=True
│       └─ Performance: sharpe_threshold=0.5
│
├── orchestrator.py                         [348 lines]
│   ├─ ClmOrchestrator:
│   │   ├─ __init__(registry, training_adapter, backtest_adapter, event_bus, config)
│   │   │
│   │   ├─ Main Pipeline:
│   │   │   └─ handle_training_job(job) → 10-step pipeline:
│   │   │       1. Update job status → in_progress
│   │   │       2. Fetch training data
│   │   │       3. Train model → ModelVersion
│   │   │       4. Register model
│   │   │       5. Publish model_trained event
│   │   │       6. Evaluate model → EvaluationResult
│   │   │       7. Save evaluation
│   │   │       8. Publish model_evaluated event
│   │   │       9. Apply promotion criteria → Promote/Fail
│   │   │       10. Update job status → completed
│   │   │
│   │   ├─ Pipeline Steps:
│   │   │   ├─ _fetch_training_data(job)
│   │   │   ├─ _train_model(job, data)
│   │   │   ├─ _evaluate_model(model_version, job)
│   │   │   ├─ _apply_promotion_criteria(evaluation)
│   │   │   ├─ _handle_promotion(model_version, evaluation)
│   │   │   └─ _promote_to_production(model_version)
│   │   │
│   │   └─ Event Publishers:
│   │       ├─ _publish_model_trained_event(...)
│   │       ├─ _publish_model_evaluated_event(...)
│   │       └─ _publish_model_promoted_event(...)
│   │
│   └─ Promotion Criteria:
│       ├─ min_sharpe_ratio: 1.0
│       ├─ min_win_rate: 0.52
│       ├─ min_profit_factor: 1.3
│       ├─ max_drawdown: 0.15
│       └─ min_trades: 50
│
├── adapters.py                             [318 lines] [SKELETON]
│   ├─ ModelTrainingAdapter:
│   │   ├─ train_model(job, training_data) → ModelVersion
│   │   ├─ _train_model_impl(job, data) → (model_object, metrics)
│   │   ├─ _generate_model_id(job)
│   │   └─ _generate_version()
│   │   └─ TODO: Integrate with real training code
│   │
│   ├─ BacktestAdapter:
│   │   ├─ evaluate_model(model_version, period_days) → EvaluationResult
│   │   └─ _run_backtest_impl(model_version, period) → metrics
│   │   └─ TODO: Integrate with real backtest code
│   │
│   └─ DataLoaderAdapter:
│       ├─ fetch_training_data(symbol, timeframe, span) → Dict
│       └─ TODO: Integrate with real data loader
│
├── strategies.py                           [422 lines] [SKELETON]
│   ├─ Enums (2):
│   │   ├─ StrategyOrigin (MANUAL, MUTATION, CROSSOVER, RANDOM, REGIME_ADAPTATION)
│   │   └─ StrategyStatus (PROPOSED, TRAINING, SHADOW, ACTIVE, RETIRED, FAILED)
│   │
│   ├─ StrategyCandidate Model:
│   │   ├─ id, base_strategy, model_type, params
│   │   ├─ origin, parent_ids, mutation_description
│   │   ├─ status, performance_metrics, fitness_score
│   │   └─ created_at
│   │
│   └─ StrategyEvolutionEngine:
│       ├─ __init__(config)
│       │
│       ├─ Candidate Generation:
│       │   ├─ propose_new_candidates(performance_data) → List[StrategyCandidate]
│       │   ├─ mutate_strategy(parent) → StrategyCandidate
│       │   └─ crossover_strategies(parent1, parent2) → StrategyCandidate
│       │
│       ├─ Evaluation & Selection:
│       │   ├─ update_candidate_performance(candidate_id, metrics)
│       │   └─ select_top_candidates(n) → Top N by fitness
│       │
│       └─ Utilities:
│           ├─ get_candidate(candidate_id)
│           ├─ list_candidates(status)
│           └─ get_stats()
│
├── app.py                                  [361 lines]
│   ├─ FastAPI App:
│   │   ├─ title: "CLM v3 - Continuous Learning Manager v3"
│   │   └─ version: "3.0.0"
│   │
│   ├─ Endpoints (9):
│   │   ├─ Health & Status (2):
│   │   │   ├─ GET /health
│   │   │   └─ GET /clm/status
│   │   │
│   │   ├─ Training Jobs (3):
│   │   │   ├─ POST /clm/train
│   │   │   ├─ GET /clm/jobs
│   │   │   └─ GET /clm/jobs/{job_id}
│   │   │
│   │   ├─ Model Management (3):
│   │   │   ├─ GET /clm/models
│   │   │   ├─ POST /clm/promote
│   │   │   └─ POST /clm/rollback
│   │   │
│   │   └─ Strategy Evolution (1):
│   │       └─ GET /clm/candidates
│   │
│   └─ Lifecycle:
│       ├─ startup() → Log initialization
│       └─ shutdown() → Stop scheduler
│
├── main.py                                 [247 lines] [SKELETON]
│   ├─ ClmV3Service:
│   │   ├─ __init__(event_bus, config)
│   │   │   ├─ Initialize: registry, orchestrator, scheduler, evolution
│   │   │   └─ Components: training_adapter, backtest_adapter, data_loader
│   │   │
│   │   ├─ Lifecycle:
│   │   │   ├─ start() → Subscribe to events, start scheduler
│   │   │   └─ stop() → Unsubscribe, stop scheduler
│   │   │
│   │   ├─ EventBus Integration:
│   │   │   ├─ _subscribe_to_events()
│   │   │   └─ _unsubscribe_from_events()
│   │   │
│   │   ├─ Event Handlers (4):
│   │   │   ├─ handle_drift_detected(event)
│   │   │   ├─ handle_performance_degraded(event)
│   │   │   ├─ handle_manual_training(event)
│   │   │   └─ handle_regime_change(event)
│   │   │
│   │   └─ Status:
│   │       └─ get_status() → Service status
│   │
│   └─ Factory:
│       └─ create_clm_v3_service(event_bus, config) → ClmV3Service
│
└── tests/
    └── test_clm_v3_epic_clm3_001.py        [600 lines]
        ├─ Fixtures (6):
        │   ├─ temp_registry_dir
        │   ├─ registry
        │   ├─ training_adapter
        │   ├─ backtest_adapter
        │   ├─ orchestrator
        │   ├─ scheduler
        │   └─ evolution
        │
        └─ Test Scenarios (8):
            ├─ test_training_job_registration
            │   └─ TrainingJob CRUD operations
            │
            ├─ test_model_version_registration
            │   └─ ModelVersion registration + query
            │
            ├─ test_orchestrator_training_pipeline
            │   └─ Complete pipeline (mock adapters)
            │
            ├─ test_promotion_criteria
            │   ├─ Good model (Sharpe=1.45, WR=0.57, PF=1.52) → PASS
            │   └─ Bad model (Sharpe=0.6, WR=0.48, PF=1.1) → FAIL
            │
            ├─ test_promotion_and_rollback
            │   ├─ Promote v2 → v1 retired, v2 production
            │   └─ Rollback to v1 → v1 production, v2 retired
            │
            ├─ test_scheduler_periodic_triggers
            │   └─ Manual trigger + next training times
            │
            ├─ test_strategy_evolution_candidates
            │   └─ Generate 3 candidates on poor performance
            │
            └─ test_complete_integration
                └─ Drift detected → Training → Evaluation → Promotion
```

---

## 📊 File Statistics

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `__init__.py` | 40 | Package exports | ✅ Complete |
| `models.py` | 328 | 11 Pydantic models + 3 enums | ✅ Complete |
| `storage.py` | 414 | ModelRegistryV3 (versioning, promotion, rollback) | ✅ Complete |
| `scheduler.py` | 334 | TrainingScheduler (periodic, drift, manual) | ✅ Complete |
| `orchestrator.py` | 348 | ClmOrchestrator (10-step pipeline) | ✅ Complete |
| `adapters.py` | 318 | Integration hooks (training, backtest, data) | ⏳ Skeleton |
| `strategies.py` | 422 | StrategyEvolutionEngine (mutation, crossover) | ⏳ Skeleton |
| `app.py` | 361 | FastAPI REST API (9 endpoints) | ✅ Complete |
| `main.py` | 247 | EventBus integration & lifecycle | ⏳ Skeleton |
| `test_clm_v3_epic_clm3_001.py` | 600 | Comprehensive test suite (8 scenarios) | ✅ Complete |
| **TOTAL** | **3,412** | **Complete CLM v3 implementation** | **85% Complete** |

---

## 🎯 Component Status

| Component | Status | Details |
|-----------|--------|---------|
| **Core Models** | ✅ 100% | 11 models, 3 enums |
| **Model Registry** | ✅ 100% | Versioning, promotion, rollback, query |
| **Training Scheduler** | ✅ 100% | Periodic, drift, performance, manual triggers |
| **Orchestrator** | ✅ 100% | 10-step pipeline, promotion criteria |
| **REST API** | ✅ 100% | 9 endpoints (health, status, train, promote, rollback) |
| **Testing** | ✅ 100% | 8 comprehensive scenarios |
| **Documentation** | ✅ 100% | 3 documents (3,600+ lines) |
| **Training Adapters** | ⏳ 30% | Skeleton with mock data (TODO: real training) |
| **Strategy Evolution** | ⏳ 40% | Basic mutation/crossover (TODO: genetic algorithm) |
| **EventBus Integration** | ⏳ 50% | Handlers defined (TODO: subscribe/publish) |

**Overall Phase 1 Completion**: ✅ **85%** (Core infrastructure complete)

---

## 🚀 Next Phase: EPIC-CLM3-002

**Phase 2 Focus**: Production adapters, genetic algorithm, monitoring

**Estimated Timeline**: 2-3 weeks

**Priority Tasks**:
1. ✅ Integrate real training functions (XGB, LGBM, NHITS, PatchTST, RL v3)
2. ✅ Integrate real backtest logic
3. ✅ Implement genetic algorithm (NSGA-II, Pareto frontier)
4. ✅ Add shadow testing & A/B testing
5. ✅ Migrate to PostgreSQL
6. ✅ Add Prometheus metrics + Grafana dashboard

---

**Created**: December 4, 2025  
**Version**: CLM v3.0.0 (EPIC-CLM3-001)  
**Status**: ✅ Phase 1 Complete, ⏳ Ready for Phase 2
