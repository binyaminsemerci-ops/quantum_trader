"""
CLM v3 Tests - Comprehensive test suite for EPIC-CLM3-001.

Tests:
1. TrainingJob registration and lifecycle
2. ModelVersion registration and query
3. Orchestrator training pipeline (mock)
4. Promotion criteria evaluation
5. Rollback workflow
6. Scheduler periodic triggers
7. Strategy Evolution candidate generation
8. EventBus integration (mock)
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from backend.services.clm_v3.models import (
    ModelType,
    ModelStatus,
    TrainingJob,
    ModelVersion,
    EvaluationResult,
    TriggerReason,
)
from backend.services.clm_v3.storage import ModelRegistryV3
from backend.services.clm_v3.scheduler import TrainingScheduler
from backend.services.clm_v3.orchestrator import ClmOrchestrator
from backend.services.clm_v3.adapters import (
    ModelTrainingAdapter,
    BacktestAdapter,
)
from backend.services.clm_v3.strategies import (
    StrategyEvolutionEngine,
    StrategyCandidate,
    StrategyOrigin,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_registry_dir(tmp_path):
    """Temporary directory for registry."""
    return tmp_path / "clm_v3_test"


@pytest.fixture
def registry(temp_registry_dir):
    """ModelRegistryV3 instance."""
    return ModelRegistryV3(
        models_dir=str(temp_registry_dir / "models"),
        metadata_dir=str(temp_registry_dir / "metadata"),
    )


@pytest.fixture
def training_adapter(temp_registry_dir):
    """ModelTrainingAdapter instance."""
    return ModelTrainingAdapter(
        models_dir=str(temp_registry_dir / "models"),
    )


@pytest.fixture
def backtest_adapter():
    """BacktestAdapter instance."""
    return BacktestAdapter()


@pytest.fixture
def orchestrator(registry, training_adapter, backtest_adapter):
    """ClmOrchestrator instance."""
    return ClmOrchestrator(
        registry=registry,
        training_adapter=training_adapter,
        backtest_adapter=backtest_adapter,
    )


@pytest.fixture
def scheduler(registry):
    """TrainingScheduler instance."""
    return TrainingScheduler(registry=registry)


@pytest.fixture
def evolution():
    """StrategyEvolutionEngine instance."""
    return StrategyEvolutionEngine()


# ============================================================================
# Test 1: TrainingJob Registration
# ============================================================================

@pytest.mark.asyncio
async def test_training_job_registration(registry):
    """Test TrainingJob registration in registry."""
    # Create training job
    job = TrainingJob(
        model_type=ModelType.XGBOOST,
        symbol="BTCUSDT",
        timeframe="1h",
        dataset_span_days=90,
        trigger_reason=TriggerReason.MANUAL,
        triggered_by="test_user",
    )
    
    # Register
    registered_job = registry.register_training_job(job)
    
    # Verify
    assert registered_job.id == job.id
    assert registered_job.model_type == ModelType.XGBOOST
    assert registered_job.status == "pending"
    
    # Retrieve
    retrieved_job = registry.get_training_job(job.id)
    assert retrieved_job is not None
    assert retrieved_job.id == job.id
    
    print(f"✅ Test 1 PASSED: TrainingJob registered successfully (id={job.id})")


# ============================================================================
# Test 2: ModelVersion Registration & Query
# ============================================================================

@pytest.mark.asyncio
async def test_model_version_registration(registry):
    """Test ModelVersion registration and querying."""
    # Create model version
    model = ModelVersion(
        model_id="xgboost_btcusdt_1h",
        version="v20251204_120000",
        model_type=ModelType.XGBOOST,
        status=ModelStatus.CANDIDATE,
        model_path="/app/models/test_model.pkl",
        model_size_bytes=1024000,
        training_data_range={
            "start": "2024-01-01T00:00:00",
            "end": "2024-03-01T00:00:00",
        },
        feature_count=50,
        training_params={"max_depth": 5, "learning_rate": 0.01},
        train_metrics={"train_loss": 0.042, "train_sharpe": 1.45},
        validation_metrics={"val_loss": 0.05, "val_sharpe": 1.32},
    )
    
    # Register
    registered_model = registry.register_model_version(model)
    
    # Verify
    assert registered_model.model_id == "xgboost_btcusdt_1h"
    assert registered_model.status == ModelStatus.CANDIDATE
    
    # Query
    from backend.services.clm_v3.models import ModelQuery
    query = ModelQuery(model_type=ModelType.XGBOOST, status=ModelStatus.CANDIDATE)
    results = registry.query_models(query)
    
    assert len(results) == 1
    assert results[0].model_id == "xgboost_btcusdt_1h"
    
    print(f"✅ Test 2 PASSED: ModelVersion registered and queried successfully")


# ============================================================================
# Test 3: Orchestrator Training Pipeline
# ============================================================================

@pytest.mark.asyncio
async def test_orchestrator_training_pipeline(orchestrator, registry):
    """Test complete orchestrator training pipeline (with mock adapters)."""
    # Create training job
    job = TrainingJob(
        model_type=ModelType.LIGHTGBM,
        symbol="ETHUSDT",
        timeframe="1h",
        dataset_span_days=90,
        trigger_reason=TriggerReason.DRIFT_DETECTED,
        triggered_by="drift_detector",
    )
    
    # Register job
    registry.register_training_job(job)
    
    # Run training pipeline
    model_version = await orchestrator.handle_training_job(job)
    
    # Verify
    assert model_version is not None
    assert model_version.model_type == ModelType.LIGHTGBM
    assert model_version.status in [ModelStatus.CANDIDATE, ModelStatus.SHADOW]
    
    # Check job status
    updated_job = registry.get_training_job(job.id)
    assert updated_job.status == "completed"
    assert updated_job.completed_at is not None
    
    # Check evaluation
    evaluation = registry.get_latest_evaluation(model_version.model_id, model_version.version)
    assert evaluation is not None
    assert evaluation.sharpe_ratio > 0
    
    print(f"✅ Test 3 PASSED: Orchestrator pipeline completed successfully")
    print(f"   Model: {model_version.model_id} v{model_version.version}")
    print(f"   Status: {model_version.status.value}")
    print(f"   Evaluation: Sharpe={evaluation.sharpe_ratio:.3f}, WR={evaluation.win_rate:.3f}")


# ============================================================================
# Test 4: Promotion Criteria Evaluation
# ============================================================================

@pytest.mark.asyncio
async def test_promotion_criteria(orchestrator):
    """Test promotion criteria evaluation."""
    # Good evaluation (should pass)
    good_eval = EvaluationResult(
        model_id="xgboost_test",
        version="v1",
        evaluation_type="backtest",
        evaluation_period_days=30,
        total_trades=100,
        win_rate=0.58,
        profit_factor=1.65,
        sharpe_ratio=1.45,
        sortino_ratio=1.80,
        max_drawdown=0.08,
        avg_win=120.0,
        avg_loss=-75.0,
        total_pnl=4500.0,
        risk_adjusted_return=0.35,
        calmar_ratio=5.0,
        passed=False,  # Will be set by criteria
        promotion_score=0.0,
    )
    
    # Apply criteria
    good_eval = orchestrator._apply_promotion_criteria(good_eval)
    
    assert good_eval.passed == True
    assert good_eval.promotion_score > 50
    assert good_eval.failure_reason is None
    
    print(f"✅ Test 4a PASSED: Good model passed criteria (score={good_eval.promotion_score:.2f})")
    
    # Bad evaluation (should fail)
    bad_eval = EvaluationResult(
        model_id="xgboost_test",
        version="v2",
        evaluation_type="backtest",
        evaluation_period_days=30,
        total_trades=100,
        win_rate=0.48,  # Too low
        profit_factor=1.1,  # Too low
        sharpe_ratio=0.6,  # Too low
        sortino_ratio=0.8,
        max_drawdown=0.22,  # Too high
        avg_win=80.0,
        avg_loss=-90.0,
        total_pnl=-500.0,
        risk_adjusted_return=0.05,
        calmar_ratio=0.8,
        passed=False,
        promotion_score=0.0,
    )
    
    # Apply criteria
    bad_eval = orchestrator._apply_promotion_criteria(bad_eval)
    
    assert bad_eval.passed == False
    assert bad_eval.promotion_score < 50
    assert bad_eval.failure_reason is not None
    
    print(f"✅ Test 4b PASSED: Bad model failed criteria (reason: {bad_eval.failure_reason})")


# ============================================================================
# Test 5: Model Promotion & Rollback
# ============================================================================

@pytest.mark.asyncio
async def test_promotion_and_rollback(registry):
    """Test model promotion and rollback workflow."""
    # Create v1 (production)
    v1 = ModelVersion(
        model_id="xgboost_promo_test",
        version="v1",
        model_type=ModelType.XGBOOST,
        status=ModelStatus.PRODUCTION,
        model_path="/app/models/v1.pkl",
        model_size_bytes=1000000,
        training_data_range={"start": "2024-01-01", "end": "2024-02-01"},
        feature_count=50,
        training_params={},
        train_metrics={"sharpe": 1.2},
        validation_metrics={"sharpe": 1.1},
    )
    registry.register_model_version(v1)
    
    # Create v2 (candidate)
    v2 = ModelVersion(
        model_id="xgboost_promo_test",
        version="v2",
        model_type=ModelType.XGBOOST,
        status=ModelStatus.CANDIDATE,
        model_path="/app/models/v2.pkl",
        model_size_bytes=1100000,
        training_data_range={"start": "2024-02-01", "end": "2024-03-01"},
        feature_count=50,
        training_params={},
        train_metrics={"sharpe": 1.5},
        validation_metrics={"sharpe": 1.4},
    )
    registry.register_model_version(v2)
    
    # Promote v2
    success = registry.promote_model("xgboost_promo_test", "v2", promoted_by="test")
    assert success == True
    
    # Verify v1 retired, v2 production
    v1_updated = registry.get_model_version("xgboost_promo_test", "v1")
    v2_updated = registry.get_model_version("xgboost_promo_test", "v2")
    
    assert v1_updated.status == ModelStatus.RETIRED
    assert v2_updated.status == ModelStatus.PRODUCTION
    
    print(f"✅ Test 5a PASSED: v2 promoted to production")
    
    # Rollback to v1
    success = registry.rollback_to_version("xgboost_promo_test", "v1", reason="v2 unstable")
    assert success == True
    
    # Verify v1 restored
    v1_restored = registry.get_model_version("xgboost_promo_test", "v1")
    v2_rolled_back = registry.get_model_version("xgboost_promo_test", "v2")
    
    assert v1_restored.status == ModelStatus.PRODUCTION
    assert v2_rolled_back.status == ModelStatus.RETIRED
    
    print(f"✅ Test 5b PASSED: Rolled back to v1")


# ============================================================================
# Test 6: Scheduler Periodic Triggers
# ============================================================================

@pytest.mark.asyncio
async def test_scheduler_periodic_triggers(scheduler):
    """Test scheduler periodic training triggers."""
    # Configure short interval for testing
    scheduler.config["periodic_training"]["xgboost_interval_hours"] = 0.01  # ~36 seconds
    
    # Trigger manual training (to set last_training)
    job = await scheduler.trigger_training(
        model_type=ModelType.XGBOOST,
        trigger_reason=TriggerReason.MANUAL,
        triggered_by="test",
    )
    
    assert job.model_type == ModelType.XGBOOST
    assert job.trigger_reason == TriggerReason.MANUAL
    
    # Get next training times
    next_times = scheduler.get_next_training_times()
    assert "xgboost_main" in next_times
    
    print(f"✅ Test 6 PASSED: Scheduler triggers working")
    print(f"   Next XGBoost training: {next_times['xgboost_main'].isoformat()}")


# ============================================================================
# Test 7: Strategy Evolution Candidate Generation
# ============================================================================

@pytest.mark.asyncio
async def test_strategy_evolution_candidates(evolution):
    """Test StrategyEvolutionEngine candidate generation."""
    # Enable evolution for testing
    evolution.config["enabled"] = True
    
    # Poor performance data (should trigger candidate generation)
    poor_performance = {
        "sharpe_ratio": 0.3,  # Below threshold (0.5)
        "win_rate": 0.48,
        "profit_factor": 1.1,
    }
    
    # Generate candidates
    candidates = await evolution.propose_new_candidates(
        performance_data=poor_performance,
        base_strategy="trend_following",
        model_type=ModelType.XGBOOST,
    )
    
    # Verify
    assert len(candidates) > 0, "Should generate at least 1 candidate for poor performance"
    
    for candidate in candidates:
        assert candidate.base_strategy == "trend_following"
        assert candidate.model_type == ModelType.XGBOOST
        assert candidate.origin == StrategyOrigin.MUTATION
        assert candidate.params is not None
    
    print(f"✅ Test 7 PASSED: Generated {len(candidates)} strategy candidates")
    for i, c in enumerate(candidates):
        print(f"   Candidate {i+1}: {c.mutation_description}")


# ============================================================================
# Test 8: Complete Integration Test
# ============================================================================

@pytest.mark.asyncio
async def test_complete_integration(orchestrator, scheduler, evolution, registry):
    """Complete integration test: drift detection → training → evaluation → promotion."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Drift Detection → Training → Promotion")
    print("="*60)
    
    # Step 1: Drift detected (simulated)
    print("\n[Step 1] Simulating drift detection...")
    await scheduler.handle_drift_detected(
        model_id="xgboost_integration",
        model_type=ModelType.XGBOOST,
        drift_score=0.85,
    )
    
    # Step 2: Training job created
    jobs = registry.list_training_jobs(status="pending", limit=10)
    assert len(jobs) > 0, "Training job should be created"
    latest_job = jobs[0]
    
    print(f"[Step 2] Training job created: {latest_job.id}")
    print(f"         Trigger: {latest_job.trigger_reason.value}")
    
    # Step 3: Execute training pipeline
    print(f"[Step 3] Executing training pipeline...")
    model_version = await orchestrator.handle_training_job(latest_job)
    
    assert model_version is not None
    print(f"[Step 4] Model trained: {model_version.model_id} v{model_version.version}")
    print(f"         Status: {model_version.status.value}")
    
    # Step 5: Check evaluation
    evaluation = registry.get_latest_evaluation(model_version.model_id, model_version.version)
    assert evaluation is not None
    
    print(f"[Step 5] Evaluation complete:")
    print(f"         Sharpe: {evaluation.sharpe_ratio:.3f}")
    print(f"         Win Rate: {evaluation.win_rate:.3f}")
    print(f"         Profit Factor: {evaluation.profit_factor:.3f}")
    print(f"         Passed: {evaluation.passed}")
    print(f"         Promotion Score: {evaluation.promotion_score:.2f}")
    
    # Step 6: Check if promoted (auto-promotion to CANDIDATE)
    if evaluation.passed:
        assert model_version.status in [ModelStatus.CANDIDATE, ModelStatus.SHADOW]
        print(f"[Step 6] ✅ Model promoted to {model_version.status.value}")
    else:
        assert model_version.status == ModelStatus.FAILED
        print(f"[Step 6] ❌ Model FAILED evaluation (reason: {evaluation.failure_reason})")
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST PASSED")
    print("="*60)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CLM v3 Test Suite - EPIC-CLM3-001")
    print("="*80 + "\n")
    
    # Run all tests
    pytest.main([__file__, "-v", "-s"])
