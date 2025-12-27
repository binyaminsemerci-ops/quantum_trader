"""
Test: RL Training / CLM / Shadow Models Service - Sprint 2 Service #5

Tests:
- Training cycle execution with fake dependencies
- Event publishing (model.training_started, model.training_completed)
- Shadow model registration, evaluation, promotion
- Drift detection and event triggering
- CLM full cycle execution
"""
import pytest
import asyncio
from unittest.mock import MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from microservices.rl_training.config import settings
from microservices.rl_training.training_daemon import RLTrainingDaemon
from microservices.rl_training.clm import ContinuousLearningManager
from microservices.rl_training.shadow_models import ShadowModelManager
from microservices.rl_training.drift_detection import DriftDetector
from microservices.rl_training.handlers import EventHandlers, setup_event_subscriptions
from microservices.rl_training.dependencies import create_fake_dependencies
from microservices.rl_training.models import (
    ModelType,
    TrainingTrigger,
    DriftSeverity,
    ManualRetrainRequestEvent,
)


@pytest.fixture
def fake_deps():
    """Create fake dependencies"""
    return create_fake_dependencies(settings)


@pytest.fixture
def training_daemon(fake_deps):
    """Create TrainingDaemon with fake dependencies"""
    policy_store, data_source, model_registry, event_bus = fake_deps
    
    return RLTrainingDaemon(
        policy_store=policy_store,
        data_source=data_source,
        model_registry=model_registry,
        event_bus=event_bus,
        config=settings
    )


@pytest.fixture
def clm(training_daemon):
    """Create ContinuousLearningManager"""
    return ContinuousLearningManager(
        training_daemon=training_daemon,
        config=settings
    )


@pytest.fixture
def shadow_manager(fake_deps):
    """Create ShadowModelManager"""
    _, _, model_registry, event_bus = fake_deps
    
    return ShadowModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
        config=settings
    )


@pytest.fixture
def drift_detector(fake_deps):
    """Create DriftDetector"""
    _, _, _, event_bus = fake_deps
    
    return DriftDetector(
        event_bus=event_bus,
        config=settings
    )


# ============================================================================
# TRAINING DAEMON TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_training_cycle_execution(training_daemon, fake_deps):
    """Test that run_training_cycle executes full workflow"""
    _, _, _, event_bus = fake_deps
    
    # Run training cycle
    result = await training_daemon.run_training_cycle(
        model_type=ModelType.XGBOOST,
        trigger=TrainingTrigger.MANUAL,
        reason="Test training cycle"
    )
    
    # Verify result
    assert result["status"] == "success"
    assert "job_id" in result
    assert "model_version" in result
    assert "metrics" in result
    assert result["metrics"]["sharpe_ratio"] > 0
    
    # Verify events were published
    published_events = event_bus.get_published_events()
    assert len(published_events) == 2  # training_started + training_completed
    
    # Check training_started event
    started_event = published_events[0]
    assert started_event["event_type"] == "model.training_started"
    assert started_event["event_data"]["model_type"] == "xgboost"
    assert started_event["event_data"]["trigger"] == "manual"
    
    # Check training_completed event
    completed_event = published_events[1]
    assert completed_event["event_type"] == "model.training_completed"
    assert completed_event["event_data"]["status"] == "success"


@pytest.mark.asyncio
async def test_training_cycle_insufficient_samples(fake_deps):
    """Test training cycle fails with insufficient samples"""
    policy_store, data_source, model_registry, event_bus = fake_deps
    
    # Create custom data source that returns insufficient samples
    class InsufficientDataSource:
        async def fetch_training_data(self, *args, **kwargs):
            return {
                "features": [],
                "labels": [],
                "sample_count": 50,  # Below threshold of 100
                "feature_names": ["rsi", "macd"],
                "lookback_days": 90
            }
    
    # Create training daemon with insufficient data source
    training_daemon = RLTrainingDaemon(
        policy_store=policy_store,
        data_source=InsufficientDataSource(),
        model_registry=model_registry,
        event_bus=event_bus,
        config=settings
    )
    
    # Run training cycle
    result = await training_daemon.run_training_cycle(
        model_type=ModelType.XGBOOST,
        trigger=TrainingTrigger.MANUAL,
        reason="Test insufficient samples"
    )
    
    # Should fail
    assert result["status"] == "error"
    assert "Insufficient samples" in result["error"]


@pytest.mark.asyncio
async def test_training_history(training_daemon):
    """Test training history tracking"""
    # Run a few training cycles
    for i in range(3):
        await training_daemon.run_training_cycle(
            model_type=ModelType.XGBOOST,
            trigger=TrainingTrigger.MANUAL,
            reason=f"Test {i}"
        )
    
    # Get history
    history = training_daemon.get_training_history(limit=10)
    
    assert len(history) == 3
    assert all(job["model_type"] == "xgboost" for job in history)
    assert all(job["status"] == "success" for job in history)


# ============================================================================
# CLM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_clm_check_if_retrain_needed(clm):
    """Test CLM checks if retraining is needed"""
    needs_retrain = await clm.check_if_retrain_needed()
    
    # Should need initial training (never trained before)
    assert ModelType.XGBOOST in needs_retrain
    assert needs_retrain[ModelType.XGBOOST] is True


@pytest.mark.asyncio
async def test_clm_trigger_retraining(clm):
    """Test CLM manual retraining trigger"""
    result = await clm.trigger_retraining(
        model_type=ModelType.XGBOOST,
        reason="Manual CLM trigger test"
    )
    
    assert result["status"] == "success"
    assert result["model_version"] is not None
    
    # Check last retrain time was updated
    last_times = clm.get_last_retrain_times()
    assert "xgboost" in last_times


@pytest.mark.asyncio
async def test_clm_full_cycle(clm):
    """Test CLM full cycle execution"""
    result = await clm.run_full_cycle()
    
    assert "cycle_completed_at" in result
    assert "models_retrained" in result
    assert result["models_retrained"] >= 2  # XGBOOST + LIGHTGBM


# ============================================================================
# SHADOW MODEL TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_shadow_model_registration(shadow_manager):
    """Test shadow model registration"""
    model_name = await shadow_manager.register_shadow_model(
        model_type=ModelType.XGBOOST,
        version="v20251204_test"
    )
    
    assert model_name == "xgboost_v20251204_test"
    
    # Verify it's in shadow models list
    shadow_models = shadow_manager.get_shadow_models()
    assert len(shadow_models) == 1
    assert shadow_models[0].model_name == model_name


@pytest.mark.asyncio
async def test_shadow_model_evaluation_insufficient_data(shadow_manager):
    """Test shadow evaluation with insufficient predictions"""
    # Register shadow
    model_name = await shadow_manager.register_shadow_model(
        model_type=ModelType.XGBOOST,
        version="v20251204"
    )
    
    # Evaluate with insufficient predictions (0 < 100)
    result = await shadow_manager.evaluate_shadow_model(
        model_name=model_name,
        shadow_metrics={"sharpe_ratio": 1.85, "win_rate": 0.58},
        champion_metrics={"sharpe_ratio": 1.72, "win_rate": 0.56}
    )
    
    assert result["status"] == "insufficient_data"
    assert result["ready_for_promotion"] is False


@pytest.mark.asyncio
async def test_shadow_model_evaluation_ready_for_promotion(shadow_manager, fake_deps):
    """Test shadow evaluation when ready for promotion"""
    _, _, _, event_bus = fake_deps
    
    # Register shadow
    model_name = await shadow_manager.register_shadow_model(
        model_type=ModelType.XGBOOST,
        version="v20251204"
    )
    
    # Simulate 100+ predictions
    shadow = shadow_manager._shadow_models[model_name]
    shadow.num_predictions = 150
    
    # Evaluate with better performance
    result = await shadow_manager.evaluate_shadow_model(
        model_name=model_name,
        shadow_metrics={"sharpe_ratio": 1.85, "win_rate": 0.58},
        champion_metrics={"sharpe_ratio": 1.72, "win_rate": 0.56}
    )
    
    # Check evaluation
    assert result["status"] == "evaluated"
    assert result["sharpe_diff"] > settings.MIN_IMPROVEMENT_FOR_PROMOTION
    assert result["ready_for_promotion"] is True
    assert result["recommendation"] == "promote"


@pytest.mark.asyncio
async def test_shadow_model_promotion(shadow_manager, fake_deps):
    """Test shadow model promotion to active champion"""
    _, _, _, event_bus = fake_deps
    
    # Register shadow
    model_name = await shadow_manager.register_shadow_model(
        model_type=ModelType.XGBOOST,
        version="v20251204"
    )
    
    # Simulate evaluation
    shadow = shadow_manager._shadow_models[model_name]
    shadow.num_predictions = 150
    shadow.performance_metrics = {"sharpe_ratio": 1.85, "win_rate": 0.58}
    shadow.champion_metrics = {"sharpe_ratio": 1.72, "win_rate": 0.56}
    
    # Clear previous events
    event_bus.clear_published_events()
    
    # Promote
    result = await shadow_manager.promote_shadow_to_active(
        model_name=model_name,
        reason="Better Sharpe ratio"
    )
    
    # Check result
    assert result["status"] == "promoted"
    assert result["model_name"] == model_name
    assert result["improvement_pct"] > 0
    
    # Verify model.promoted event was published
    promoted_events = event_bus.get_published_events_by_type("model.promoted")
    assert len(promoted_events) == 1
    assert promoted_events[0]["event_data"]["model_type"] == "xgboost"
    assert promoted_events[0]["event_data"]["new_version"] == "v20251204"
    
    # Verify shadow was removed (now active)
    shadow_models = shadow_manager.get_shadow_models()
    assert len(shadow_models) == 0
    
    # Verify champion was set
    champion = shadow_manager.get_champion_model()
    assert champion == model_name


# ============================================================================
# DRIFT DETECTION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_drift_detection_no_drift(drift_detector):
    """Test drift detection when no drift occurs"""
    # Set reference distribution
    drift_detector.set_reference_distribution(
        feature_name="rsi",
        distribution={
            "frequencies": [0.1, 0.2, 0.3, 0.2, 0.2],
            "mean": 50.0,
            "std": 15.0
        }
    )
    
    # Check with very similar distribution (low PSI)
    result = await drift_detector.check_feature_drift(
        feature_name="rsi",
        current_distribution={
            "frequencies": [0.11, 0.19, 0.31, 0.19, 0.20],
            "mean": 50.5,
            "std": 15.2
        }
    )
    
    assert result["status"] == "checked"
    assert result["drift_detected"] is False
    assert result["severity"] == "none"


@pytest.mark.asyncio
async def test_drift_detection_moderate_drift(drift_detector, fake_deps):
    """Test drift detection when moderate drift occurs"""
    _, _, _, event_bus = fake_deps
    
    # Set reference distribution
    drift_detector.set_reference_distribution(
        feature_name="rsi",
        distribution={
            "frequencies": [0.1, 0.2, 0.3, 0.2, 0.2],
            "mean": 50.0,
            "std": 15.0
        }
    )
    
    # Clear previous events
    event_bus.clear_published_events()
    
    # Check with very different distribution (high PSI)
    result = await drift_detector.check_feature_drift(
        feature_name="rsi",
        current_distribution={
            "frequencies": [0.3, 0.1, 0.1, 0.2, 0.3],
            "mean": 45.0,
            "std": 20.0
        }
    )
    
    # Should detect drift
    assert result["status"] == "checked"
    assert result["psi_score"] > settings.DRIFT_TRIGGER_THRESHOLD
    assert result["drift_detected"] is True
    
    # Verify drift event was published
    drift_events = event_bus.get_published_events_by_type("data.drift_detected")
    assert len(drift_events) > 0


@pytest.mark.asyncio
async def test_drift_detection_performance_degradation(drift_detector, fake_deps):
    """Test performance degradation detection"""
    _, _, _, event_bus = fake_deps
    
    # Clear previous events
    event_bus.clear_published_events()
    
    # Check with degraded performance
    result = await drift_detector.check_performance_degradation(
        model_name="xgboost_v1",
        current_metrics={"sharpe_ratio": 1.2, "win_rate": 0.48},
        baseline_metrics={"sharpe_ratio": 1.8, "win_rate": 0.58}
    )
    
    # Should detect degradation
    assert result["status"] == "checked"
    assert result["degraded"] is True
    assert result["sharpe_change"] < -settings.PERFORMANCE_DECAY_THRESHOLD
    
    # Verify drift event was published
    drift_events = event_bus.get_published_events_by_type("data.drift_detected")
    assert len(drift_events) > 0
    assert drift_events[0]["event_data"]["drift_type"] == "performance_drift"


# ============================================================================
# EVENT HANDLER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_event_handler_manual_retrain_request(training_daemon, clm, drift_detector, fake_deps):
    """Test event handler processes manual.retrain_request"""
    _, _, _, event_bus = fake_deps
    
    # Create event handlers
    handlers = EventHandlers(
        training_daemon=training_daemon,
        clm=clm,
        drift_detector=drift_detector,
        config=settings
    )
    
    # Setup subscriptions
    setup_event_subscriptions(event_bus, handlers)
    
    # Clear previous events
    event_bus.clear_published_events()
    
    # Publish manual retrain request
    await event_bus.publish(
        "manual.retrain_request",
        ManualRetrainRequestEvent(
            model_type=ModelType.XGBOOST,
            reason="Testing manual trigger",
            requested_by="test_user",
            timestamp="2025-12-04T12:00:00Z"
        ).model_dump()
    )
    
    # Wait for async processing (minimal delay)
    await asyncio.sleep(0.001)
    
    # Verify training was triggered
    published_events = event_bus.get_published_events()
    training_events = [
        e for e in published_events
        if e["event_type"] in ["model.training_started", "model.training_completed"]
    ]
    
    assert len(training_events) >= 2  # started + completed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
