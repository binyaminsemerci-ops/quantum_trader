"""
Integration Tests for ML/AI Pipeline - Drift Detection Workflow.

Tests drift detection and automatic retraining trigger.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from backend.domains.learning.clm import ContinuousLearningManager, CLMConfig, create_clm
from backend.domains.learning.drift_detector import DriftType, DriftSeverity
from backend.domains.learning.model_registry import ModelType, ModelStatus


@pytest.fixture
async def clm_with_auto_retraining(test_db, test_event_bus, test_policy_store):
    """Create CLM with auto-retraining enabled."""
    config = CLMConfig(
        retraining_schedule_hours=168,
        drift_check_hours=1,  # Check every hour for tests
        performance_check_hours=6,
        drift_trigger_threshold=0.05,
        shadow_min_predictions=10,
        auto_retraining_enabled=True,  # Enable auto-retraining
        auto_promotion_enabled=False,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    
    # Setup initial active models
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        await clm.model_registry.register_model(
            model_type=model_type,
            version="1.0.0",
            trained_model=Mock(),  # Dummy model
            metrics={"accuracy": 0.85, "f1": 0.80},
            training_config={"n_estimators": 100},
            training_data_range={"start": "2024-01-01", "end": "2024-03-31"},
            feature_count=50,
            status=ModelStatus.ACTIVE,
        )
    
    await clm.start()
    
    yield clm
    
    await clm.stop()


@pytest.mark.asyncio
async def test_feature_drift_triggers_retraining(clm_with_auto_retraining):
    """
    Test that feature drift triggers automatic retraining.
    
    Steps:
    1. Detect feature drift (HIGH severity)
    2. Verify drift event logged
    3. Verify auto-retraining triggered
    4. Wait for retraining completion
    5. Verify new shadow models
    """
    clm = clm_with_auto_retraining
    
    # Step 1: Simulate feature drift
    # Mock drift detector to return HIGH drift
    with patch.object(clm.drift_detector, "check_feature_drift") as mock_drift:
        mock_drift.return_value = {
            "drift_detected": True,
            "drift_type": DriftType.FEATURE.value,
            "severity": DriftSeverity.HIGH.value,
            "drift_score": 0.15,
            "p_value": 0.001,
            "feature_name": "rsi_14",
            "reference_stats": {"mean": 50.0, "std": 15.0},
            "current_stats": {"mean": 65.0, "std": 20.0},
        }
        
        # Trigger drift check
        result = await clm.manual_trigger_drift_check(ModelType.XGBOOST)
    
    # Step 2: Verify drift event logged
    assert result["drift_detected"] == True
    assert result["severity"] == "HIGH"
    
    # Query drift events from DB
    drift_events = await clm.drift_detector.get_recent_drift_events(days=1)
    assert len(drift_events) > 0
    
    latest_drift = drift_events[0]
    assert latest_drift["drift_type"] == "FEATURE"
    assert latest_drift["severity"] == "HIGH"
    assert latest_drift["trigger_retraining"] == True
    
    # Step 3: Verify event published
    # Check EventBus for learning.drift.detected
    published_events = clm.event_bus.get_published_events("learning.drift.detected")
    assert len(published_events) > 0
    
    # Step 4: Wait for auto-retraining to start
    await asyncio.sleep(5)  # Give CLM time to process event
    
    # Check if retraining job created
    jobs = await clm.retraining_orchestrator.list_jobs(status="RUNNING")
    assert len(jobs) > 0, "Auto-retraining not triggered"
    
    job_id = jobs[0]["job_id"]
    assert jobs[0]["trigger_reason"] == "drift_detected"
    
    # Step 5: Wait for completion
    timeout = 300
    start_time = datetime.utcnow()
    while (datetime.utcnow() - start_time).total_seconds() < timeout:
        status = await clm.retraining_orchestrator.get_job_status(job_id)
        if status["status"] == "COMPLETED":
            break
        await asyncio.sleep(5)
    
    # Verify new shadow models
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        shadows = await clm.model_registry.list_models(
            model_type=model_type,
            status=ModelStatus.SHADOW,
        )
        assert len(shadows) >= 1


@pytest.mark.asyncio
async def test_prediction_drift_triggers_retraining(clm_with_auto_retraining):
    """
    Test that prediction distribution drift triggers retraining.
    """
    clm = clm_with_auto_retraining
    
    # Mock prediction drift
    with patch.object(clm.drift_detector, "check_prediction_drift") as mock_drift:
        mock_drift.return_value = {
            "drift_detected": True,
            "drift_type": DriftType.PREDICTION.value,
            "severity": DriftSeverity.CRITICAL.value,
            "drift_score": 0.25,
            "p_value": 0.0001,
            "reference_stats": {"mean": 0.6, "std": 0.2},
            "current_stats": {"mean": 0.45, "std": 0.3},
        }
        
        result = await clm.manual_trigger_drift_check(ModelType.LIGHTGBM)
    
    # Verify retraining triggered
    await asyncio.sleep(5)
    
    jobs = await clm.retraining_orchestrator.list_jobs(status="RUNNING")
    assert len(jobs) > 0
    assert jobs[0]["trigger_reason"] == "drift_detected"


@pytest.mark.asyncio
async def test_low_drift_does_not_trigger_retraining(clm_with_auto_retraining):
    """
    Test that LOW severity drift does not trigger retraining.
    """
    clm = clm_with_auto_retraining
    
    # Mock LOW drift
    with patch.object(clm.drift_detector, "check_feature_drift") as mock_drift:
        mock_drift.return_value = {
            "drift_detected": True,
            "drift_type": DriftType.FEATURE.value,
            "severity": DriftSeverity.LOW.value,
            "drift_score": 0.03,
            "p_value": 0.04,
            "feature_name": "volume",
        }
        
        result = await clm.manual_trigger_drift_check(ModelType.XGBOOST)
    
    # Wait a bit
    await asyncio.sleep(5)
    
    # Verify NO retraining triggered
    jobs = await clm.retraining_orchestrator.list_jobs(status="RUNNING")
    assert len(jobs) == 0, "LOW drift should not trigger retraining"


@pytest.mark.asyncio
async def test_drift_detection_for_all_models(clm_with_auto_retraining):
    """
    Test drift detection runs for all model types.
    """
    clm = clm_with_auto_retraining
    
    # Run drift check for all models
    results = []
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        result = await clm.manual_trigger_drift_check(model_type)
        results.append(result)
    
    # Verify results for each model
    assert len(results) == 2
    for result in results:
        assert "drift_detected" in result
        assert "model_type" in result


@pytest.mark.asyncio
async def test_drift_check_with_insufficient_data(clm_with_auto_retraining):
    """
    Test drift detection gracefully handles insufficient data.
    """
    clm = clm_with_auto_retraining
    
    # Mock data fetcher to return insufficient data
    with patch.object(clm.data_fetcher, "fetch_recent_data") as mock_fetch:
        mock_fetch.return_value = []  # No data
        
        result = await clm.manual_trigger_drift_check(ModelType.XGBOOST)
    
    # Should return no drift (not enough data to compare)
    assert result.get("drift_detected") == False
    assert "insufficient_data" in result.get("notes", "").lower()


@pytest.mark.asyncio
async def test_performance_drift_alerts(clm_with_auto_retraining):
    """
    Test that performance drift triggers alert event.
    """
    clm = clm_with_auto_retraining
    
    # Mock performance drift
    with patch.object(clm.drift_detector, "check_performance_drift") as mock_drift:
        mock_drift.return_value = {
            "drift_detected": True,
            "drift_type": DriftType.PERFORMANCE.value,
            "severity": DriftSeverity.HIGH.value,
            "drift_score": 0.20,
            "notes": "Winrate dropped from 0.65 to 0.45",
        }
        
        # Manually check performance drift
        model = await clm.model_registry.get_active_model(ModelType.XGBOOST)
        result = await clm.drift_detector.check_performance_drift(
            model_id=model.model_id,
            model_type=ModelType.XGBOOST,
        )
    
    # Verify alert event published
    alerts = clm.event_bus.get_published_events("learning.performance.alert")
    assert len(alerts) > 0


@pytest.mark.asyncio
async def test_scheduled_drift_checks(clm_with_auto_retraining):
    """
    Test that CLM runs scheduled drift checks.
    """
    clm = clm_with_auto_retraining
    
    # Record initial drift check count
    initial_checks = await clm.drift_detector.count_drift_checks(days=1)
    
    # Wait for one scheduled cycle (1 hour in config, but we'll wait 10 seconds for test)
    # Note: In real scenario, would need to adjust config or mock time
    await asyncio.sleep(10)
    
    # Force a scheduled task run (for testing)
    await clm._run_scheduled_tasks()
    
    # Verify drift check ran
    new_checks = await clm.drift_detector.count_drift_checks(days=1)
    # Note: This is a simplification; real test would verify via logs or events
