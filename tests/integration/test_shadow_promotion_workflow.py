"""
Integration Tests for ML/AI Pipeline - Shadow Testing and Auto-Promotion.

Tests shadow model evaluation and automatic promotion to active.
"""

import asyncio
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from backend.domains.learning.clm import ContinuousLearningManager, CLMConfig, create_clm
from backend.domains.learning.model_registry import ModelType, ModelStatus


@pytest.fixture
async def clm_with_shadow_models(test_db, test_event_bus, test_policy_store):
    """Create CLM with active + shadow models."""
    config = CLMConfig(
        shadow_min_predictions=50,  # Need 50 predictions before promotion
        auto_promotion_enabled=True,
        auto_retraining_enabled=False,
    )
    
    clm = await create_clm(test_db, test_event_bus, test_policy_store, config)
    
    # Setup active models
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        await clm.model_registry.register_model(
            model_type=model_type,
            version="1.0.0",
            trained_model=Mock(),
            metrics={"accuracy": 0.75, "f1": 0.70},
            training_config={},
            training_data_range={},
            feature_count=50,
            status=ModelStatus.ACTIVE,
        )
    
    # Setup shadow models (better metrics)
    for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM]:
        await clm.model_registry.register_model(
            model_type=model_type,
            version="2.0.0",
            trained_model=Mock(),
            metrics={"accuracy": 0.85, "f1": 0.82},
            training_config={},
            training_data_range={},
            feature_count=50,
            status=ModelStatus.SHADOW,
        )
    
    await clm.start()
    
    yield clm
    
    await clm.stop()


@pytest.mark.asyncio
async def test_shadow_model_evaluation(clm_with_shadow_models):
    """
    Test shadow model makes predictions and tracks outcomes.
    
    Steps:
    1. Get active and shadow models
    2. Send predictions through shadow tester
    3. Verify shadow predictions logged
    4. Update with outcomes
    5. Verify outcomes recorded
    """
    clm = clm_with_shadow_models
    
    # Step 1: Get models
    active_xgb = await clm.model_registry.get_active_model(ModelType.XGBOOST)
    shadow_models = await clm.model_registry.list_models(
        model_type=ModelType.XGBOOST,
        status=ModelStatus.SHADOW,
    )
    shadow_xgb = shadow_models[0]
    
    # Step 2: Generate predictions
    n_predictions = 100
    for i in range(n_predictions):
        features = {
            "rsi": 50.0 + np.random.randn() * 10,
            "macd": 0.5 + np.random.randn() * 0.2,
            "volume": 1000000 + np.random.randn() * 100000,
        }
        
        # Active model prediction
        active_pred = 0.6 + np.random.randn() * 0.1
        
        # Shadow model prediction (slightly better)
        shadow_pred = 0.65 + np.random.randn() * 0.08
        
        # Log shadow prediction
        await clm.shadow_tester.log_shadow_prediction(
            model_id=shadow_xgb.model_id,
            model_type=ModelType.XGBOOST,
            symbol="BTCUSDT",
            prediction_value=shadow_pred,
            confidence=0.8,
            features=features,
        )
    
    # Step 3: Verify predictions logged
    summary = await clm.shadow_tester.get_shadow_test_summary(
        model_type=ModelType.XGBOOST,
        days=1,
    )
    
    assert len(summary) > 0
    shadow_summary = summary[0]
    assert shadow_summary["model_id"] == shadow_xgb.model_id
    assert shadow_summary["total_predictions"] >= n_predictions
    
    # Step 4: Update with outcomes
    # Simulate outcomes after some time
    predictions = await clm.shadow_tester._get_predictions_without_outcomes(
        shadow_xgb.model_id,
    )
    
    for pred in predictions[:50]:  # Update first 50
        # Shadow model has better outcomes
        outcome = 0.02 if pred["prediction_value"] > 0.6 else -0.01
        
        await clm.shadow_tester.update_prediction_outcome(
            prediction_id=pred["prediction_id"],
            actual_outcome=outcome,
        )
    
    # Step 5: Verify outcomes recorded
    summary_updated = await clm.shadow_tester.get_shadow_test_summary(
        model_type=ModelType.XGBOOST,
        days=1,
    )
    
    shadow_summary_updated = summary_updated[0]
    assert shadow_summary_updated["predictions_with_outcomes"] >= 50


@pytest.mark.asyncio
async def test_auto_promotion_on_better_performance(clm_with_shadow_models):
    """
    Test automatic promotion when shadow outperforms active.
    
    Steps:
    1. Generate predictions for shadow model
    2. Update with better outcomes than active
    3. Trigger shadow promotion evaluation
    4. Verify shadow promoted to active
    5. Verify old active retired
    """
    clm = clm_with_shadow_models
    
    # Get models
    active_xgb = await clm.model_registry.get_active_model(ModelType.XGBOOST)
    shadow_models = await clm.model_registry.list_models(
        model_type=ModelType.XGBOOST,
        status=ModelStatus.SHADOW,
    )
    shadow_xgb = shadow_models[0]
    
    # Generate predictions with better outcomes
    n_predictions = 100
    for i in range(n_predictions):
        features = {"rsi": 50.0, "macd": 0.5}
        shadow_pred = 0.7  # Higher confidence
        
        pred_id = await clm.shadow_tester.log_shadow_prediction(
            model_id=shadow_xgb.model_id,
            model_type=ModelType.XGBOOST,
            symbol="BTCUSDT",
            prediction_value=shadow_pred,
            confidence=0.85,
            features=features,
        )
        
        # Better outcomes (70% winrate vs active's 60%)
        outcome = 0.03 if np.random.rand() < 0.70 else -0.01
        
        await clm.shadow_tester.update_prediction_outcome(
            prediction_id=pred_id,
            actual_outcome=outcome,
        )
    
    # Trigger promotion evaluation
    promoted = await clm.manual_promote_shadow(ModelType.XGBOOST)
    
    # Verify promotion
    assert promoted == True
    
    # Verify new active model
    new_active = await clm.model_registry.get_active_model(ModelType.XGBOOST)
    assert new_active.model_id == shadow_xgb.model_id
    assert new_active.status == ModelStatus.ACTIVE
    assert new_active.promoted_at is not None
    
    # Verify old active retired
    old_active_updated = await clm.model_registry.get_model(active_xgb.model_id)
    assert old_active_updated.status == ModelStatus.RETIRED


@pytest.mark.asyncio
async def test_no_promotion_on_insufficient_data(clm_with_shadow_models):
    """
    Test that promotion requires minimum predictions.
    """
    clm = clm_with_shadow_models
    
    # Get shadow model
    shadow_models = await clm.model_registry.list_models(
        model_type=ModelType.LIGHTGBM,
        status=ModelStatus.SHADOW,
    )
    shadow_lgb = shadow_models[0]
    
    # Log only 10 predictions (below threshold of 50)
    for i in range(10):
        await clm.shadow_tester.log_shadow_prediction(
            model_id=shadow_lgb.model_id,
            model_type=ModelType.LIGHTGBM,
            symbol="ETHUSDT",
            prediction_value=0.7,
            confidence=0.8,
            features={"rsi": 50},
        )
    
    # Try to promote (should fail)
    promoted = await clm.manual_promote_shadow(ModelType.LIGHTGBM)
    
    assert promoted == False


@pytest.mark.asyncio
async def test_no_promotion_on_worse_performance(clm_with_shadow_models):
    """
    Test that shadow is NOT promoted if performance is worse.
    """
    clm = clm_with_shadow_models
    
    shadow_models = await clm.model_registry.list_models(
        model_type=ModelType.XGBOOST,
        status=ModelStatus.SHADOW,
    )
    shadow_xgb = shadow_models[0]
    
    # Generate predictions with WORSE outcomes
    for i in range(100):
        pred_id = await clm.shadow_tester.log_shadow_prediction(
            model_id=shadow_xgb.model_id,
            model_type=ModelType.XGBOOST,
            symbol="BTCUSDT",
            prediction_value=0.5,
            confidence=0.6,
            features={"rsi": 50},
        )
        
        # Worse outcomes (40% winrate)
        outcome = 0.01 if np.random.rand() < 0.40 else -0.02
        
        await clm.shadow_tester.update_prediction_outcome(
            prediction_id=pred_id,
            actual_outcome=outcome,
        )
    
    # Try to promote
    promoted = await clm.manual_promote_shadow(ModelType.XGBOOST)
    
    # Should NOT promote
    assert promoted == False
    
    # Verify active model unchanged
    active = await clm.model_registry.get_active_model(ModelType.XGBOOST)
    assert active.version == "1.0.0"  # Original active


@pytest.mark.asyncio
async def test_scheduled_shadow_promotion_checks(clm_with_shadow_models):
    """
    Test that CLM runs scheduled shadow promotion checks.
    """
    clm = clm_with_shadow_models
    
    # Setup shadow with good performance
    shadow_models = await clm.model_registry.list_models(
        model_type=ModelType.LIGHTGBM,
        status=ModelStatus.SHADOW,
    )
    shadow = shadow_models[0]
    
    for i in range(100):
        pred_id = await clm.shadow_tester.log_shadow_prediction(
            model_id=shadow.model_id,
            model_type=ModelType.LIGHTGBM,
            symbol="BTCUSDT",
            prediction_value=0.75,
            confidence=0.85,
            features={"rsi": 60},
        )
        
        outcome = 0.03 if np.random.rand() < 0.75 else -0.01
        await clm.shadow_tester.update_prediction_outcome(pred_id, outcome)
    
    # Force scheduled task run
    await clm._check_shadow_promotions()
    
    # Verify promotion occurred
    new_active = await clm.model_registry.get_active_model(ModelType.LIGHTGBM)
    assert new_active.model_id == shadow.model_id


@pytest.mark.asyncio
async def test_multiple_shadow_models_best_promoted(clm_with_shadow_models):
    """
    Test that when multiple shadows exist, the best one is promoted.
    """
    clm = clm_with_shadow_models
    
    # Register additional shadow models
    shadow_2 = await clm.model_registry.register_model(
        model_type=ModelType.XGBOOST,
        version="2.1.0",
        trained_model=Mock(),
        metrics={"accuracy": 0.82, "f1": 0.79},
        training_config={},
        training_data_range={},
        feature_count=50,
        status=ModelStatus.SHADOW,
    )
    
    shadow_3 = await clm.model_registry.register_model(
        model_type=ModelType.XGBOOST,
        version="2.2.0",
        trained_model=Mock(),
        metrics={"accuracy": 0.88, "f1": 0.86},  # Best metrics
        training_config={},
        training_data_range={},
        feature_count=50,
        status=ModelStatus.SHADOW,
    )
    
    # Generate predictions for all shadows
    for shadow in [shadow_2, shadow_3]:
        for i in range(100):
            pred_id = await clm.shadow_tester.log_shadow_prediction(
                model_id=shadow,
                model_type=ModelType.XGBOOST,
                symbol="BTCUSDT",
                prediction_value=0.8,
                confidence=0.9,
                features={"rsi": 65},
            )
            
            # shadow_3 has best winrate (80%)
            winrate = 0.80 if shadow == shadow_3 else 0.72
            outcome = 0.03 if np.random.rand() < winrate else -0.01
            await clm.shadow_tester.update_prediction_outcome(pred_id, outcome)
    
    # Trigger promotion
    await clm.manual_promote_shadow(ModelType.XGBOOST)
    
    # Verify best shadow (shadow_3) was promoted
    active = await clm.model_registry.get_active_model(ModelType.XGBOOST)
    assert active.model_id == shadow_3
