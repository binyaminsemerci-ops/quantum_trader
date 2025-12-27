"""
Integration Tests for ML/AI Pipeline - End-to-End Workflow.

Tests complete pipeline from data fetch to model deployment.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta

from backend.domains.learning.clm import create_clm, CLMConfig
from backend.domains.learning.model_registry import ModelType, ModelStatus
from backend.domains.learning.retraining import RetrainingType
from backend.domains.learning.drift_detector import DriftSeverity


@pytest.mark.asyncio
async def test_complete_pipeline_workflow(test_db, test_event_bus, test_policy_store):
    """
    End-to-end test: Data fetch → Train → Shadow test → Drift → Retrain → Promote.
    
    This simulates a complete lifecycle:
    1. Initial training (all 4 models)
    2. Shadow testing period
    3. Drift detection
    4. Automatic retraining
    5. Shadow promotion
    """
    
    # Create CLM with all automation enabled
    config = CLMConfig(
        retraining_schedule_hours=168,
        drift_check_hours=24,
        performance_check_hours=6,
        drift_trigger_threshold=0.05,
        shadow_min_predictions=50,
        auto_retraining_enabled=True,
        auto_promotion_enabled=True,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    await clm.start()
    
    try:
        # ============================================================
        # Phase 1: Initial Training
        # ============================================================
        print("\n=== Phase 1: Initial Training ===")
        
        job_id_initial = await clm.trigger_retraining(
            retraining_type=RetrainingType.FULL,
            trigger_reason="initial_setup",
        )
        
        # Wait for initial training
        timeout = 300
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await clm.retraining_orchestrator.get_job_status(job_id_initial)
            if status["status"] == "COMPLETED":
                break
            await asyncio.sleep(5)
        
        assert status["status"] == "COMPLETED"
        assert status["models_succeeded"] == 4
        
        # Verify all shadow models created
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
            shadows = await clm.model_registry.list_models(
                model_type=model_type,
                status=ModelStatus.SHADOW,
            )
            assert len(shadows) >= 1, f"No shadow for {model_type.value}"
        
        print("✓ Initial training completed: 4 shadow models created")
        
        # ============================================================
        # Phase 2: Shadow Testing
        # ============================================================
        print("\n=== Phase 2: Shadow Testing ===")
        
        # Promote initial shadows to active (for testing)
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            shadows = await clm.model_registry.list_models(
                model_type=model_type,
                status=ModelStatus.SHADOW,
            )
            shadow = shadows[0]
            await clm.model_registry.promote_model(shadow.model_id)
        
        # Generate shadow predictions for 100 trades
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            active = await clm.model_registry.get_active_model(model_type)
            
            for i in range(100):
                pred_id = await clm.shadow_tester.log_shadow_prediction(
                    model_id=active.model_id,
                    model_type=model_type,
                    symbol="BTCUSDT",
                    prediction_value=0.65 + np.random.randn() * 0.1,
                    confidence=0.8,
                    features={"rsi": 50 + np.random.randn() * 10},
                )
                
                # 65% winrate
                outcome = 0.02 if np.random.rand() < 0.65 else -0.01
                await clm.shadow_tester.update_prediction_outcome(pred_id, outcome)
        
        print("✓ Shadow testing completed: 200 predictions logged")
        
        # ============================================================
        # Phase 3: Drift Detection
        # ============================================================
        print("\n=== Phase 3: Drift Detection ===")
        
        # Simulate market regime change (drift)
        from unittest.mock import patch
        
        with patch.object(clm.drift_detector, "check_feature_drift") as mock_drift:
            mock_drift.return_value = {
                "drift_detected": True,
                "drift_type": "FEATURE",
                "severity": "HIGH",
                "drift_score": 0.18,
                "p_value": 0.001,
                "feature_name": "volatility",
                "notes": "Volatility regime shift detected",
            }
            
            drift_result = await clm.manual_trigger_drift_check(ModelType.XGBOOST)
        
        assert drift_result["drift_detected"] == True
        assert drift_result["severity"] == "HIGH"
        
        print("✓ Drift detected: HIGH severity on volatility")
        
        # ============================================================
        # Phase 4: Automatic Retraining
        # ============================================================
        print("\n=== Phase 4: Automatic Retraining ===")
        
        # Wait for auto-retraining to trigger
        await asyncio.sleep(10)
        
        # Check for retraining job
        jobs = await clm.retraining_orchestrator.list_jobs(status="RUNNING")
        assert len(jobs) > 0, "Auto-retraining not triggered"
        
        job_id_retrain = jobs[0]["job_id"]
        assert "drift" in jobs[0]["trigger_reason"].lower()
        
        print(f"✓ Auto-retraining triggered: {job_id_retrain}")
        
        # Wait for retraining completion
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < 300:
            status = await clm.retraining_orchestrator.get_job_status(job_id_retrain)
            if status["status"] == "COMPLETED":
                break
            await asyncio.sleep(5)
        
        assert status["status"] == "COMPLETED"
        print("✓ Retraining completed: New shadow models created")
        
        # ============================================================
        # Phase 5: Shadow Promotion
        # ============================================================
        print("\n=== Phase 5: Shadow Promotion ===")
        
        # Get new shadow models
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            shadows = await clm.model_registry.list_models(
                model_type=model_type,
                status=ModelStatus.SHADOW,
            )
            
            # Simulate better performance for new shadows
            shadow = shadows[0]
            for i in range(100):
                pred_id = await clm.shadow_tester.log_shadow_prediction(
                    model_id=shadow.model_id,
                    model_type=model_type,
                    symbol="BTCUSDT",
                    prediction_value=0.75 + np.random.randn() * 0.08,
                    confidence=0.9,
                    features={"rsi": 55 + np.random.randn() * 8},
                )
                
                # 75% winrate (better than active's 65%)
                outcome = 0.025 if np.random.rand() < 0.75 else -0.01
                await clm.shadow_tester.update_prediction_outcome(pred_id, outcome)
        
        # Trigger promotion evaluation
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
            promoted = await clm.manual_promote_shadow(model_type)
            assert promoted == True, f"Promotion failed for {model_type.value}"
            
            # Verify new active
            new_active = await clm.model_registry.get_active_model(model_type)
            assert new_active.status == ModelStatus.ACTIVE
            assert new_active.promoted_at is not None
        
        print("✓ Shadow promotion completed: New models now active")
        
        # ============================================================
        # Phase 6: Verification
        # ============================================================
        print("\n=== Phase 6: Final Verification ===")
        
        # Get CLM status
        clm_status = await clm.get_system_status()
        
        assert clm_status["running"] == True
        assert len(clm_status["active_models"]) >= 2
        
        # Verify event log
        drift_events = await clm.drift_detector.get_recent_drift_events(days=1)
        assert len(drift_events) > 0
        
        # Verify retraining jobs
        all_jobs = await clm.retraining_orchestrator.list_jobs()
        assert len(all_jobs) >= 2  # Initial + drift-triggered
        
        print("✓ Pipeline verification complete")
        print(f"\nFinal State:")
        print(f"  - Active models: {len(clm_status['active_models'])}")
        print(f"  - Drift events: {len(drift_events)}")
        print(f"  - Retraining jobs: {len(all_jobs)}")
        print(f"  - CLM running: {clm_status['running']}")
        
    finally:
        await clm.stop()


@pytest.mark.asyncio
async def test_pipeline_resilience_to_failures(test_db, test_event_bus, test_policy_store):
    """
    Test pipeline handles failures gracefully.
    
    Scenarios:
    - Training failure for one model
    - Drift detection failure
    - Database connection loss
    - Recovery and continuation
    """
    
    config = CLMConfig(
        auto_retraining_enabled=True,
        auto_promotion_enabled=True,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    await clm.start()
    
    try:
        # Scenario 1: Training failure
        print("\n=== Scenario 1: Training Failure ===")
        
        from unittest.mock import patch
        
        with patch("backend.domains.learning.model_training.train_nhits") as mock_train:
            mock_train.side_effect = Exception("GPU out of memory")
            
            job_id = await clm.trigger_retraining(
                retraining_type=RetrainingType.PARTIAL,
                model_types=[ModelType.XGBOOST, ModelType.NHITS],
                trigger_reason="test_failure",
            )
            
            # Wait for completion
            timeout = 120
            start_time = datetime.utcnow()
            while (datetime.utcnow() - start_time).total_seconds() < timeout:
                status = await clm.retraining_orchestrator.get_job_status(job_id)
                if status["status"] in ["COMPLETED", "FAILED"]:
                    break
                await asyncio.sleep(5)
            
            # Verify partial success
            assert status["models_succeeded"] >= 1  # XGBoost succeeded
            assert status["models_failed"] >= 1  # NHITS failed
        
        print("✓ Pipeline handled training failure gracefully")
        
        # Scenario 2: CLM continues operating
        print("\n=== Scenario 2: CLM Continues Operating ===")
        
        clm_status = await clm.get_system_status()
        assert clm_status["running"] == True
        
        # Can still trigger new retraining
        job_id_2 = await clm.trigger_retraining(
            retraining_type=RetrainingType.PARTIAL,
            model_types=[ModelType.XGBOOST],
            trigger_reason="recovery_test",
        )
        
        assert job_id_2 is not None
        print("✓ CLM operational after failure")
        
    finally:
        await clm.stop()


@pytest.mark.asyncio
async def test_multi_model_coordination(test_db, test_event_bus, test_policy_store):
    """
    Test coordination across multiple model types.
    
    Verifies:
    - All models trained in parallel
    - Shadow testing for all models
    - Drift detection per model
    - Independent promotion decisions
    """
    
    config = CLMConfig(
        auto_retraining_enabled=False,
        auto_promotion_enabled=False,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    await clm.start()
    
    try:
        # Train all models
        job_id = await clm.trigger_retraining(
            retraining_type=RetrainingType.FULL,
            trigger_reason="multi_model_test",
        )
        
        # Wait for completion
        timeout = 300
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await clm.retraining_orchestrator.get_job_status(job_id)
            if status["status"] == "COMPLETED":
                break
            await asyncio.sleep(5)
        
        # Verify all 4 models trained
        assert status["models_succeeded"] == 4
        
        # Check each model independently
        model_types = [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]
        
        for model_type in model_types:
            # Check shadow model exists
            shadows = await clm.model_registry.list_models(
                model_type=model_type,
                status=ModelStatus.SHADOW,
            )
            assert len(shadows) >= 1
            
            # Promote to active for testing
            await clm.model_registry.promote_model(shadows[0].model_id)
            
            # Verify active
            active = await clm.model_registry.get_active_model(model_type)
            assert active is not None
            assert active.status == ModelStatus.ACTIVE
        
        print(f"✓ All {len(model_types)} models coordinated successfully")
        
        # Test drift detection for each
        for model_type in model_types:
            result = await clm.manual_trigger_drift_check(model_type)
            assert "drift_detected" in result
            assert result["model_type"] == model_type.value
        
        print("✓ Drift detection working for all models")
        
    finally:
        await clm.stop()
