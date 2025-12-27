"""
Unit Tests for Continuous Learning Manager

Tests core CLM functionality with mocked dependencies.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

import pandas as pd
import numpy as np

from backend.services.ai.continuous_learning_manager import (
    ContinuousLearningManager,
    RetrainTriggerDetector,
    ModelType,
    ModelStatus,
    ModelArtifact,
    EvaluationResult,
    ShadowTestResult,
    RetrainTrigger,
)


class TestModelArtifact(unittest.TestCase):
    """Test ModelArtifact data structure"""
    
    def test_artifact_creation(self):
        artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v1",
            trained_at=datetime.utcnow(),
            metrics={"rmse": 0.05},
            model_object={"dummy": "model"},
        )
        
        self.assertEqual(artifact.model_type, ModelType.XGBOOST)
        self.assertEqual(artifact.version, "v1")
        self.assertEqual(artifact.status, ModelStatus.CANDIDATE)
    
    def test_artifact_to_dict(self):
        now = datetime.utcnow()
        artifact = ModelArtifact(
            model_type=ModelType.LIGHTGBM,
            version="v2",
            trained_at=now,
            metrics={"mae": 0.03},
            model_object={},
            training_range=(now - timedelta(days=30), now),
        )
        
        data = artifact.to_dict()
        
        self.assertEqual(data["model_type"], "lightgbm")
        self.assertEqual(data["version"], "v2")
        self.assertIn("trained_at", data)
        self.assertIn("training_range", data)


class TestEvaluationResult(unittest.TestCase):
    """Test EvaluationResult logic"""
    
    def test_is_better_than_active_rmse_improvement(self):
        result = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v1",
            rmse=0.03,
            mae=0.025,
            error_std=0.01,
            directional_accuracy=0.55,
            hit_rate=0.52,
            vs_active_rmse_delta=-0.05,  # 5% improvement
        )
        
        self.assertTrue(result.is_better_than_active(threshold=0.02))
    
    def test_is_better_than_active_direction_improvement(self):
        result = EvaluationResult(
            model_type=ModelType.LIGHTGBM,
            version="v1",
            rmse=0.04,
            mae=0.03,
            error_std=0.01,
            directional_accuracy=0.60,
            hit_rate=0.57,
            vs_active_rmse_delta=0.01,  # Slightly worse RMSE
            vs_active_direction_delta=0.05,  # But much better direction
        )
        
        self.assertTrue(result.is_better_than_active(threshold=0.02))
    
    def test_is_not_better_than_active(self):
        result = EvaluationResult(
            model_type=ModelType.NHITS,
            version="v1",
            rmse=0.05,
            mae=0.04,
            error_std=0.015,
            directional_accuracy=0.52,
            hit_rate=0.50,
            vs_active_rmse_delta=0.001,  # Negligible difference
            vs_active_direction_delta=0.001,
        )
        
        self.assertFalse(result.is_better_than_active(threshold=0.02))


class TestRetrainTriggerDetector(unittest.TestCase):
    """Test trigger detection logic"""
    
    def setUp(self):
        self.mock_registry = Mock()
        self.mock_data_client = Mock()
        
        self.detector = RetrainTriggerDetector(
            registry=self.mock_registry,
            data_client=self.mock_data_client,
            time_threshold_days=7,
            data_threshold_points=10000,
        )
    
    def test_time_trigger_old_model(self):
        # Model trained 10 days ago
        old_artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v1",
            trained_at=datetime.utcnow() - timedelta(days=10),
            metrics={},
            model_object={},
        )
        
        self.mock_registry.get_active.return_value = old_artifact
        
        result = self.detector.check_time_trigger(ModelType.XGBOOST)
        self.assertTrue(result)
    
    def test_time_trigger_recent_model(self):
        # Model trained 3 days ago
        recent_artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v1",
            trained_at=datetime.utcnow() - timedelta(days=3),
            metrics={},
            model_object={},
        )
        
        self.mock_registry.get_active.return_value = recent_artifact
        
        result = self.detector.check_time_trigger(ModelType.XGBOOST)
        self.assertFalse(result)
    
    def test_time_trigger_no_active_model(self):
        self.mock_registry.get_active.return_value = None
        
        result = self.detector.check_time_trigger(ModelType.XGBOOST)
        self.assertTrue(result)  # Should trigger if no active model
    
    def test_data_volume_trigger(self):
        # Simulate large data volume
        large_df = pd.DataFrame({
            'close': np.random.randn(15000)
        })
        
        self.mock_data_client.load_recent_data.return_value = large_df
        self.mock_registry.get_active.return_value = Mock()
        
        result = self.detector.check_data_volume_trigger(ModelType.XGBOOST)
        self.assertTrue(result)
    
    def test_get_trigger_returns_first_match(self):
        # Mock time trigger
        old_artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v1",
            trained_at=datetime.utcnow() - timedelta(days=10),
            metrics={},
            model_object={},
        )
        
        self.mock_registry.get_active.return_value = old_artifact
        
        trigger = self.detector.get_trigger(ModelType.XGBOOST)
        
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger, RetrainTrigger.TIME_BASED)


class TestContinuousLearningManager(unittest.TestCase):
    """Test CLM core functionality"""
    
    def setUp(self):
        self.mock_data_client = Mock()
        self.mock_feature_engineer = Mock()
        self.mock_trainer = Mock()
        self.mock_evaluator = Mock()
        self.mock_shadow_tester = Mock()
        self.mock_registry = Mock()
        
        self.clm = ContinuousLearningManager(
            data_client=self.mock_data_client,
            feature_engineer=self.mock_feature_engineer,
            trainer=self.mock_trainer,
            evaluator=self.mock_evaluator,
            shadow_tester=self.mock_shadow_tester,
            registry=self.mock_registry,
            retrain_interval_days=7,
            shadow_test_hours=24,
        )
    
    def test_initialization(self):
        self.assertEqual(self.clm.retrain_interval_days, 7)
        self.assertEqual(self.clm.shadow_test_hours, 24)
        self.assertEqual(len(self.clm.model_types), 4)
    
    def test_load_training_data(self):
        # Mock data
        df = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100),
        })
        
        self.mock_data_client.load_training_data.return_value = df
        self.mock_feature_engineer.transform.return_value = df
        
        result = self.clm._load_training_data()
        
        self.assertIsInstance(result, pd.DataFrame)
        self.mock_data_client.load_training_data.assert_called_once()
        self.mock_feature_engineer.transform.assert_called_once()
    
    def test_train_single_model_xgboost(self):
        df = pd.DataFrame({'close': np.random.randn(100)})
        
        mock_model = {"type": "xgboost", "trained": True}
        self.mock_trainer.train_xgboost.return_value = mock_model
        
        artifact = self.clm._train_single_model(ModelType.XGBOOST, df)
        
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.model_type, ModelType.XGBOOST)
        self.assertEqual(artifact.status, ModelStatus.CANDIDATE)
        self.assertEqual(artifact.model_object, mock_model)
        self.mock_trainer.train_xgboost.assert_called_once()
    
    def test_train_single_model_failure(self):
        df = pd.DataFrame({'close': np.random.randn(100)})
        
        # Simulate training failure
        self.mock_trainer.train_xgboost.side_effect = Exception("Training failed")
        
        artifact = self.clm._train_single_model(ModelType.XGBOOST, df)
        
        self.assertIsNone(artifact)
    
    def test_retrain_all_success(self):
        df = pd.DataFrame({'close': np.random.randn(100)})
        
        self.mock_data_client.load_training_data.return_value = df
        self.mock_feature_engineer.transform.return_value = df
        
        # Mock all trainers
        self.mock_trainer.train_xgboost.return_value = {"type": "xgboost"}
        self.mock_trainer.train_lightgbm.return_value = {"type": "lightgbm"}
        self.mock_trainer.train_nhits.return_value = {"type": "nhits"}
        self.mock_trainer.train_patchtst.return_value = {"type": "patchtst"}
        
        artifacts = self.clm.retrain_all()
        
        self.assertEqual(len(artifacts), 4)
        self.assertIn(ModelType.XGBOOST, artifacts)
        self.assertIn(ModelType.LIGHTGBM, artifacts)
        self.assertIsNotNone(artifacts[ModelType.XGBOOST])
    
    def test_evaluate_models(self):
        # Create mock artifacts
        artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v1",
            trained_at=datetime.utcnow(),
            metrics={},
            model_object={"dummy": "model"},
        )
        
        artifacts = {ModelType.XGBOOST: artifact}
        
        # Mock data
        val_df = pd.DataFrame({'close': np.random.randn(100)})
        self.mock_data_client.load_validation_data.return_value = val_df
        self.mock_feature_engineer.transform.return_value = val_df
        
        # Mock evaluation
        eval_result = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v1",
            rmse=0.03,
            mae=0.025,
            error_std=0.01,
            directional_accuracy=0.60,
            hit_rate=0.57,
        )
        
        self.mock_evaluator.evaluate.return_value = eval_result
        self.mock_registry.get_active.return_value = None
        
        results = self.clm.evaluate_models(artifacts)
        
        self.assertIn(ModelType.XGBOOST, results)
        self.assertIsNotNone(results[ModelType.XGBOOST])
        self.assertEqual(results[ModelType.XGBOOST].rmse, 0.03)
    
    def test_promote_if_better_should_promote(self):
        # Create artifacts
        artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v_new",
            trained_at=datetime.utcnow(),
            metrics={},
            model_object={},
        )
        
        artifacts = {ModelType.XGBOOST: artifact}
        
        # Create evaluation showing improvement
        eval_result = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v_new",
            rmse=0.02,
            mae=0.015,
            error_std=0.01,
            directional_accuracy=0.65,
            hit_rate=0.62,
            vs_active_rmse_delta=-0.05,  # 5% improvement
        )
        
        evaluations = {ModelType.XGBOOST: eval_result}
        
        # Shadow recommends promotion
        shadow_result = ShadowTestResult(
            model_type=ModelType.XGBOOST,
            candidate_version="v_new",
            active_version="v_old",
            live_predictions=1000,
            candidate_mae=0.02,
            active_mae=0.03,
            candidate_direction_acc=0.65,
            active_direction_acc=0.58,
            error_ks_statistic=0.1,
            error_mean_diff=-0.01,
            error_std_diff=0.0,
            recommend_promotion=True,
            reason="Better performance",
        )
        
        shadows = {ModelType.XGBOOST: shadow_result}
        
        # Mock active model
        active_artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v_old",
            trained_at=datetime.utcnow() - timedelta(days=10),
            metrics={},
            model_object={},
        )
        
        self.mock_registry.get_active.return_value = active_artifact
        
        promoted = self.clm.promote_if_better(artifacts, evaluations, shadows)
        
        self.assertIn(ModelType.XGBOOST, promoted)
        self.mock_registry.save_model.assert_called()
        self.mock_registry.promote.assert_called_with(ModelType.XGBOOST, "v_new")
    
    def test_promote_if_better_should_not_promote(self):
        artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v_new",
            trained_at=datetime.utcnow(),
            metrics={},
            model_object={},
        )
        
        artifacts = {ModelType.XGBOOST: artifact}
        
        # No improvement
        eval_result = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v_new",
            rmse=0.04,
            mae=0.03,
            error_std=0.01,
            directional_accuracy=0.52,
            hit_rate=0.50,
            vs_active_rmse_delta=0.001,  # Negligible
        )
        
        evaluations = {ModelType.XGBOOST: eval_result}
        
        # Shadow doesn't recommend
        shadow_result = ShadowTestResult(
            model_type=ModelType.XGBOOST,
            candidate_version="v_new",
            active_version="v_old",
            live_predictions=1000,
            candidate_mae=0.04,
            active_mae=0.035,
            candidate_direction_acc=0.52,
            active_direction_acc=0.53,
            error_ks_statistic=0.05,
            error_mean_diff=0.005,
            error_std_diff=0.0,
            recommend_promotion=False,
            reason="No improvement",
        )
        
        shadows = {ModelType.XGBOOST: shadow_result}
        
        promoted = self.clm.promote_if_better(artifacts, evaluations, shadows)
        
        self.assertNotIn(ModelType.XGBOOST, promoted)
        self.mock_registry.promote.assert_not_called()
    
    def test_run_full_cycle_no_triggers(self):
        # Mock no triggers
        with patch.object(self.clm, 'check_if_retrain_needed') as mock_check:
            mock_check.return_value = {
                ModelType.XGBOOST: None,
                ModelType.LIGHTGBM: None,
                ModelType.NHITS: None,
                ModelType.PATCHTST: None,
            }
            
            report = self.clm.run_full_cycle(force=False)
            
            self.assertEqual(len(report.models_trained), 0)
            self.assertEqual(len(report.promoted_models), 0)
    
    def test_run_full_cycle_forced(self):
        df = pd.DataFrame({'close': np.random.randn(100)})
        
        # Mock data loading
        self.mock_data_client.load_training_data.return_value = df
        self.mock_data_client.load_validation_data.return_value = df
        self.mock_feature_engineer.transform.return_value = df
        
        # Mock training
        self.mock_trainer.train_xgboost.return_value = {"type": "xgboost"}
        
        # Mock evaluation
        eval_result = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v1",
            rmse=0.03,
            mae=0.02,
            error_std=0.01,
            directional_accuracy=0.60,
            hit_rate=0.57,
        )
        self.mock_evaluator.evaluate.return_value = eval_result
        
        # Mock shadow test
        shadow_result = ShadowTestResult(
            model_type=ModelType.XGBOOST,
            candidate_version="v1",
            active_version="v0",
            live_predictions=100,
            candidate_mae=0.02,
            active_mae=0.03,
            candidate_direction_acc=0.60,
            active_direction_acc=0.55,
            error_ks_statistic=0.1,
            error_mean_diff=-0.01,
            error_std_diff=0.0,
            recommend_promotion=True,
            reason="Better",
        )
        self.mock_shadow_tester.run_shadow_test.return_value = shadow_result
        
        # Mock registry
        self.mock_registry.get_active.return_value = None
        
        # Run with only XGBOOST
        report = self.clm.run_full_cycle(
            models=[ModelType.XGBOOST],
            force=True
        )
        
        self.assertIn(ModelType.XGBOOST, report.models_trained)
        self.assertGreater(report.total_duration_seconds, 0)


if __name__ == '__main__':
    unittest.main()
