"""
Tests for Continuous Learning Manager.
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from backend.services.eventbus import InMemoryEventBus
from backend.services.continuous_learning import (
    ContinuousLearningManager,
    ShadowEvaluator,
    ModelVersion,
    RetrainingConfig,
    ModelStage,
)


@pytest.fixture
def eventbus():
    return InMemoryEventBus()


@pytest.fixture
def evaluator():
    return ShadowEvaluator()


@pytest.fixture
def clm(eventbus, evaluator):
    return ContinuousLearningManager(eventbus, evaluator)


@pytest.mark.asyncio
async def test_register_model(clm):
    """Test registering a model."""
    model = ModelVersion(
        model_name="test_model",
        version="1.0",
        stage=ModelStage.LIVE,
    )
    
    config = RetrainingConfig(
        model_name="test_model",
        retrain_interval_hours=24.0,
    )
    
    clm.register_model("test_model", model, config)
    
    retrieved = clm.get_model("test_model")
    assert retrieved is not None
    assert retrieved.model_name == "test_model"
    assert retrieved.version == "1.0"


@pytest.mark.asyncio
async def test_retrain_model(clm):
    """Test model retraining."""
    # Register initial model
    model = ModelVersion(
        model_name="lstm_pred",
        version="1.0",
        stage=ModelStage.LIVE,
        model_type="lstm",
    )
    
    config = RetrainingConfig(model_name="lstm_pred")
    clm.register_model("lstm_pred", model, config)
    
    # Retrain
    new_model = await clm.retrain_model("lstm_pred")
    
    assert new_model.model_name == "lstm_pred"
    assert new_model.version == "1.1"
    assert new_model.stage == ModelStage.SHADOW


@pytest.mark.asyncio
async def test_shadow_evaluator_accuracy():
    """Test shadow evaluator accuracy calculation."""
    evaluator = ShadowEvaluator()
    
    # Perfect predictions
    predictions = [0.9, 0.8, 0.1, 0.2]
    actuals = [1.0, 1.0, 0.0, 0.0]
    
    accuracy = evaluator.calculate_accuracy(predictions, actuals, threshold=0.5)
    assert accuracy == 1.0


@pytest.mark.asyncio
async def test_shadow_evaluator_f1_score():
    """Test F1 score calculation."""
    evaluator = ShadowEvaluator()
    
    # TP=2, FP=1, FN=1
    predictions = [0.9, 0.8, 0.6, 0.2]  # 3 positive, 1 negative
    actuals = [1.0, 1.0, 0.0, 1.0]      # 3 positive, 1 negative
    
    f1 = evaluator.calculate_f1_score(predictions, actuals, threshold=0.5)
    
    # precision = 2/3, recall = 2/3, f1 = 2/3
    assert abs(f1 - 0.6667) < 0.01


@pytest.mark.asyncio
async def test_promote_model(clm, eventbus):
    """Test model promotion."""
    async def running_bus():
        task = asyncio.create_task(eventbus.run_forever())
        await asyncio.sleep(0.05)
        yield
        eventbus.stop()
        await asyncio.sleep(0.05)
    
    async for _ in running_bus():
        model = ModelVersion(
            model_name="test",
            version="2.0",
            stage=ModelStage.SHADOW,
        )
        
        clm._models["test"] = model
        
        await clm.promote_model(model)
        
        await asyncio.sleep(0.1)
        
        assert model.stage == ModelStage.LIVE
        assert model.promoted_to_live_at is not None


@pytest.mark.asyncio
async def test_should_retrain(clm):
    """Test retraining schedule check."""
    model = ModelVersion(
        model_name="test",
        version="1.0",
        stage=ModelStage.LIVE,
    )
    
    config = RetrainingConfig(
        model_name="test",
        retrain_interval_hours=1.0,  # 1 hour
    )
    
    clm.register_model("test", model, config)
    
    # Just registered, should not retrain
    assert not clm.should_retrain("test")
    
    # Simulate time passing
    clm._last_retrain["test"] = datetime.now() - timedelta(hours=2)
    
    # Now should retrain
    assert clm.should_retrain("test")


@pytest.mark.asyncio
async def test_evaluate_shadow_better_than_live(evaluator):
    """Test evaluation when shadow is better."""
    shadow = ModelVersion(
        model_name="test",
        version="2.0",
        stage=ModelStage.SHADOW,
        accuracy=0.85,
    )
    
    live = ModelVersion(
        model_name="test",
        version="1.0",
        stage=ModelStage.LIVE,
        accuracy=0.80,
    )
    
    # Record better predictions for shadow
    for i in range(200):
        # Shadow: 85% accurate
        shadow_pred = 0.9 if i % 100 < 85 else 0.1
        # Live: 80% accurate
        live_pred = 0.9 if i % 100 < 80 else 0.1
        # Actual
        actual = 1.0 if i % 100 < 90 else 0.0
        
        evaluator.record_shadow_prediction("test", shadow_pred, actual)
        evaluator.record_live_prediction("test", live_pred)
    
    result = await evaluator.evaluate(shadow, live, evaluation_hours=24.0, min_samples=100)
    
    assert result.should_promote == True
    assert result.shadow_accuracy > result.live_accuracy
