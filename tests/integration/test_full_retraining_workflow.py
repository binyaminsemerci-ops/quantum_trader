"""
Integration Tests for ML/AI Pipeline - Full Retraining Workflow.

Tests the complete retraining workflow from trigger to shadow model registration.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from backend.domains.learning.clm import ContinuousLearningManager, CLMConfig, create_clm
from backend.domains.learning.retraining import RetrainingType
from backend.domains.learning.model_registry import ModelType, ModelStatus


@pytest.fixture
async def clm_instance(test_db, test_event_bus, test_policy_store):
    """Create CLM instance for testing."""
    config = CLMConfig(
        retraining_schedule_hours=168,
        drift_check_hours=24,
        performance_check_hours=6,
        drift_trigger_threshold=0.05,
        shadow_min_predictions=10,  # Lower threshold for tests
        auto_retraining_enabled=False,  # Manual control in tests
        auto_promotion_enabled=False,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    await clm.start()
    
    yield clm
    
    await clm.stop()


@pytest.mark.asyncio
async def test_full_retraining_workflow(clm_instance):
    """
    Test complete retraining workflow.
    
    Steps:
    1. Trigger full retraining
    2. Verify retraining job created
    3. Wait for job completion
    4. Verify all models trained
    5. Verify shadow models registered
    6. Verify database records created
    """
    clm = clm_instance
    
    # Step 1: Trigger retraining
    job_id = await clm.trigger_retraining(
        retraining_type=RetrainingType.FULL,
        model_types=None,  # All models
        trigger_reason="test_workflow",
    )
    
    assert job_id is not None
    assert job_id.startswith("retrain_")
    
    # Step 2: Verify job created
    job_status = await clm.retraining_orchestrator.get_job_status(job_id)
    assert job_status is not None
    assert job_status["status"] == "PENDING"
    assert job_status["retraining_type"] == "full"
    assert job_status["trigger_reason"] == "test_workflow"
    
    # Step 3: Wait for job completion (timeout 5 minutes)
    timeout = 300  # 5 minutes
    start_time = datetime.utcnow()
    
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        job_status = await clm.retraining_orchestrator.get_job_status(job_id)
        
        if job_status["status"] in ["COMPLETED", "FAILED"]:
            break
        
        await asyncio.sleep(5)
    
    # Step 4: Verify job completed successfully
    assert job_status["status"] == "COMPLETED", f"Job failed: {job_status.get('error_message')}"
    assert job_status["models_succeeded"] == 4, f"Expected 4 models, got {job_status['models_succeeded']}"
    assert job_status["models_failed"] == 0
    
    # Step 5: Verify shadow models registered
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
        shadow_models = await clm.model_registry.list_models(
            model_type=model_type,
            status=ModelStatus.SHADOW,
        )
        
        assert len(shadow_models) >= 1, f"No shadow model for {model_type.value}"
        
        # Verify model has metrics
        shadow_model = shadow_models[0]
        assert shadow_model.metrics is not None
        assert "train_loss" in shadow_model.metrics or "train_accuracy" in shadow_model.metrics
        assert shadow_model.feature_count > 0
        assert shadow_model.file_path is not None
    
    # Step 6: Verify database records
    # Check retraining_jobs table
    async with clm.model_registry.session_factory() as session:
        result = await session.execute(
            "SELECT * FROM retraining_jobs WHERE job_id = :job_id",
            {"job_id": job_id},
        )
        job_record = result.fetchone()
        assert job_record is not None
        assert job_record["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_partial_retraining_workflow(clm_instance):
    """
    Test partial retraining (only tree models).
    """
    clm = clm_instance
    
    # Trigger partial retraining
    job_id = await clm.trigger_retraining(
        retraining_type=RetrainingType.PARTIAL,
        model_types=[ModelType.XGBOOST, ModelType.LIGHTGBM],
        trigger_reason="test_partial",
    )
    
    # Wait for completion
    timeout = 180  # 3 minutes (faster than full)
    start_time = datetime.utcnow()
    
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        job_status = await clm.retraining_orchestrator.get_job_status(job_id)
        if job_status["status"] in ["COMPLETED", "FAILED"]:
            break
        await asyncio.sleep(5)
    
    # Verify only 2 models trained
    assert job_status["status"] == "COMPLETED"
    assert job_status["models_succeeded"] == 2
    
    # Verify only tree models have new shadow versions
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        shadows = await clm.model_registry.list_models(
            model_type=model_type,
            status=ModelStatus.SHADOW,
        )
        assert len(shadows) >= 1


@pytest.mark.asyncio
async def test_incremental_retraining_workflow(clm_instance):
    """
    Test incremental retraining (retrain only on new data).
    """
    clm = clm_instance
    
    # First, need active models
    job_id_full = await clm.trigger_retraining(
        retraining_type=RetrainingType.FULL,
        trigger_reason="test_setup",
    )
    
    # Wait for full retraining
    timeout = 300
    start_time = datetime.utcnow()
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        status = await clm.retraining_orchestrator.get_job_status(job_id_full)
        if status["status"] == "COMPLETED":
            break
        await asyncio.sleep(5)
    
    # Now trigger incremental
    job_id_inc = await clm.trigger_retraining(
        retraining_type=RetrainingType.INCREMENTAL,
        trigger_reason="test_incremental",
    )
    
    # Wait for completion
    start_time = datetime.utcnow()
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        status = await clm.retraining_orchestrator.get_job_status(job_id_inc)
        if status["status"] in ["COMPLETED", "FAILED"]:
            break
        await asyncio.sleep(5)
    
    # Verify incremental completed
    assert status["status"] == "COMPLETED"
    assert status["retraining_type"] == "incremental"


@pytest.mark.asyncio
async def test_retraining_failure_handling(clm_instance):
    """
    Test error handling during retraining.
    """
    clm = clm_instance
    
    # Mock training function to fail
    with patch("backend.domains.learning.model_training.train_xgboost") as mock_train:
        mock_train.side_effect = Exception("Training failed")
        
        job_id = await clm.trigger_retraining(
            retraining_type=RetrainingType.PARTIAL,
            model_types=[ModelType.XGBOOST],
            trigger_reason="test_failure",
        )
        
        # Wait for completion
        timeout = 60
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await clm.retraining_orchestrator.get_job_status(job_id)
            if status["status"] in ["COMPLETED", "FAILED"]:
                break
            await asyncio.sleep(2)
        
        # Verify job recorded failure
        assert status["models_failed"] >= 1
        assert status["error_message"] is not None


@pytest.mark.asyncio
async def test_concurrent_retraining_prevention(clm_instance):
    """
    Test that concurrent retraining jobs are prevented.
    """
    clm = clm_instance
    
    # Start first job
    job_id_1 = await clm.trigger_retraining(
        retraining_type=RetrainingType.FULL,
        trigger_reason="test_concurrent_1",
    )
    
    # Try to start second job immediately
    with pytest.raises(Exception) as exc_info:
        job_id_2 = await clm.trigger_retraining(
            retraining_type=RetrainingType.FULL,
            trigger_reason="test_concurrent_2",
        )
    
    assert "already running" in str(exc_info.value).lower()
    
    # Clean up: cancel first job
    await clm.retraining_orchestrator.cancel_job(job_id_1)


@pytest.mark.asyncio
async def test_retraining_data_validation(clm_instance):
    """
    Test that retraining validates data quality before training.
    """
    clm = clm_instance
    
    # Mock data fetcher to return insufficient data
    with patch.object(clm.data_fetcher, "fetch_historical_data") as mock_fetch:
        # Return only 10 days of data (insufficient)
        mock_fetch.return_value = []  # Empty dataset
        
        job_id = await clm.trigger_retraining(
            retraining_type=RetrainingType.FULL,
            trigger_reason="test_validation",
        )
        
        # Wait for completion
        timeout = 60
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            status = await clm.retraining_orchestrator.get_job_status(job_id)
            if status["status"] in ["COMPLETED", "FAILED"]:
                break
            await asyncio.sleep(2)
        
        # Should fail due to insufficient data
        assert status["status"] == "FAILED"
        assert "data" in status.get("error_message", "").lower()
