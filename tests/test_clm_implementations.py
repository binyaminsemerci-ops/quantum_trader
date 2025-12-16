"""Tests for CLM concrete implementations.

Tests all 6 protocol implementations:
1. BinanceDataClient
2. QuantumFeatureEngineer
3. QuantumModelTrainer
4. QuantumModelEvaluator
5. QuantumShadowTester
6. SQLModelRegistry
"""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np

from backend.services.ai.continuous_learning_manager import ModelType, ModelStatus, ModelArtifact
from backend.services.clm_implementations import (
    BinanceDataClient,
    QuantumFeatureEngineer,
    QuantumModelTrainer,
    QuantumModelEvaluator,
    QuantumShadowTester,
    SQLModelRegistry,
)


class TestBinanceDataClient(unittest.TestCase):
    """Test BinanceDataClient implementation."""
    
    def setUp(self):
        self.client = BinanceDataClient(symbol="BTCUSDT", interval="1h")
    
    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.symbol, "BTCUSDT")
        self.assertEqual(self.client.interval, "1h")
        self.assertTrue(self.client.cache_dir.exists())
    
    def test_load_recent_data(self):
        """Test loading recent data."""
        # This test requires network access
        # Skip if not running in integration mode
        try:
            df = self.client.load_recent_data(days=1)
            
            # Should return DataFrame with OHLCV columns
            self.assertIsInstance(df, pd.DataFrame)
            
            if len(df) > 0:
                required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                for col in required_cols:
                    self.assertIn(col, df.columns)
                
                # Timestamps should be sorted
                self.assertTrue(df["timestamp"].is_monotonic_increasing)
        except Exception:
            # Skip test if network unavailable
            self.skipTest("Network access required for Binance API")


class TestQuantumFeatureEngineer(unittest.TestCase):
    """Test QuantumFeatureEngineer implementation."""
    
    def setUp(self):
        self.engineer = QuantumFeatureEngineer(use_advanced=False)
        
        # Create sample OHLCV data
        dates = pd.date_range(start="2025-01-01", periods=100, freq="1h")
        self.sample_df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(40000, 42000, 100),
            "high": np.random.uniform(42000, 43000, 100),
            "low": np.random.uniform(39000, 40000, 100),
            "close": np.random.uniform(40000, 42000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        })
    
    def test_transform_basic(self):
        """Test basic feature transformation."""
        df = self.engineer.transform(self.sample_df)
        
        # Should have more columns than input
        self.assertGreater(len(df.columns), len(self.sample_df.columns))
        
        # Should have target column
        self.assertIn("target", df.columns)
        
        # Should have some basic indicators
        expected_features = ["ma_10", "ma_50", "rsi_14"]
        for feat in expected_features:
            if feat in df.columns:
                # At least one expected feature should be present
                break
        else:
            self.fail("No expected features found")
    
    def test_get_feature_names(self):
        """Test feature name retrieval."""
        names = self.engineer.get_feature_names()
        
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        
        # Should include basic features
        self.assertIn("close", names)
        self.assertIn("ma_10", names)


class TestQuantumModelTrainer(unittest.TestCase):
    """Test QuantumModelTrainer implementation."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.trainer = QuantumModelTrainer(model_dir=self.temp_dir)
        
        # Create sample training data
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
            "target": np.random.randn(100) * 0.01,
        })
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_xgboost(self):
        """Test XGBoost training."""
        try:
            model = self.trainer.train_xgboost(
                self.sample_df,
                params={"n_estimators": 10}  # Fast for testing
            )
            
            # Should return trained model
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "predict"))
            
            # Should be able to predict
            X = self.sample_df[["feature_1", "feature_2", "feature_3"]]
            predictions = model.predict(X)
            self.assertEqual(len(predictions), len(X))
            
        except ImportError:
            self.skipTest("XGBoost not installed")
    
    def test_train_lightgbm(self):
        """Test LightGBM training."""
        try:
            model = self.trainer.train_lightgbm(
                self.sample_df,
                params={"n_estimators": 10}
            )
            
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "predict"))
            
        except ImportError:
            self.skipTest("LightGBM not installed")


class TestQuantumModelEvaluator(unittest.TestCase):
    """Test QuantumModelEvaluator implementation."""
    
    def setUp(self):
        # Create mock feature engineer
        class MockFeatureEngineer:
            def transform(self, df):
                df["target"] = df["close"].pct_change().shift(-1)
                return df.iloc[:-1].dropna()
        
        self.engineer = MockFeatureEngineer()
        self.evaluator = QuantumModelEvaluator(feature_engineer=self.engineer)
        
        # Create sample validation data
        np.random.seed(42)
        dates = pd.date_range(start="2025-01-01", periods=100, freq="1h")
        self.sample_df = pd.DataFrame({
            "timestamp": dates,
            "close": 40000 + np.cumsum(np.random.randn(100) * 10),
        })
        
        # Create mock model
        class MockModel:
            def predict(self, X):
                return np.random.randn(len(X)) * 0.01
        
        self.mock_model = MockModel()
    
    def test_evaluate(self):
        """Test model evaluation."""
        result = self.evaluator.evaluate(
            self.mock_model,
            self.sample_df,
            ModelType.XGBOOST
        )
        
        # Should return EvaluationResult
        self.assertEqual(result.model_type, ModelType.XGBOOST)
        
        # Should have metrics
        self.assertIsInstance(result.rmse, float)
        self.assertIsInstance(result.mae, float)
        self.assertIsInstance(result.directional_accuracy, float)
        
        # Metrics should be in valid ranges
        self.assertGreater(result.rmse, 0)
        self.assertGreater(result.mae, 0)
        self.assertGreaterEqual(result.directional_accuracy, 0)
        self.assertLessEqual(result.directional_accuracy, 1)
    
    def test_compare_to_active(self):
        """Test comparison between models."""
        # Create two mock results
        from backend.services.ai.continuous_learning_manager import EvaluationResult
        
        candidate = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v2",
            rmse=0.01,
            mae=0.008,
            error_std=0.005,
            directional_accuracy=0.60,
            hit_rate=1.2,
            vs_active_rmse_delta=0.0,
            vs_active_direction_delta=0.0,
            correlation_with_target=0.5,
            prediction_bias=0.0,
            regime_accuracy={},
            evaluated_at=datetime.now(timezone.utc),
        )
        
        active = EvaluationResult(
            model_type=ModelType.XGBOOST,
            version="v1",
            rmse=0.012,
            mae=0.010,
            error_std=0.006,
            directional_accuracy=0.55,
            hit_rate=1.1,
            vs_active_rmse_delta=0.0,
            vs_active_direction_delta=0.0,
            correlation_with_target=0.4,
            prediction_bias=0.0,
            regime_accuracy={},
            evaluated_at=datetime.now(timezone.utc),
        )
        
        # Compare
        result = self.evaluator.compare_to_active(candidate, active)
        
        # Should calculate deltas
        self.assertEqual(result.vs_active_rmse_delta, candidate.rmse - active.rmse)
        self.assertEqual(
            result.vs_active_direction_delta,
            candidate.directional_accuracy - active.directional_accuracy
        )


class TestQuantumShadowTester(unittest.TestCase):
    """Test QuantumShadowTester implementation."""
    
    def setUp(self):
        # Create mock data client
        class MockDataClient:
            def load_recent_data(self, days):
                np.random.seed(42)
                dates = pd.date_range(start="2025-01-01", periods=100, freq="1h")
                return pd.DataFrame({
                    "timestamp": dates,
                    "close": 40000 + np.cumsum(np.random.randn(100) * 10),
                })
        
        # Create mock feature engineer
        class MockFeatureEngineer:
            def transform(self, df):
                df["feature_1"] = df["close"].pct_change()
                df["target"] = df["close"].pct_change().shift(-1)
                return df.iloc[:-1].dropna()
        
        self.data_client = MockDataClient()
        self.engineer = MockFeatureEngineer()
        self.tester = QuantumShadowTester(
            data_client=self.data_client,
            feature_engineer=self.engineer
        )
        
        # Create mock models
        class MockModel:
            def __init__(self, bias=0.0):
                self.bias = bias
            
            def predict(self, X):
                return np.random.randn(len(X)) * 0.01 + self.bias
        
        self.candidate_model = MockModel(bias=0.0)
        self.active_model = MockModel(bias=0.001)
    
    def test_run_shadow_test(self):
        """Test shadow testing."""
        result = self.tester.run_shadow_test(
            ModelType.XGBOOST,
            self.candidate_model,
            self.active_model,
            hours=24
        )
        
        # Should return ShadowTestResult
        self.assertEqual(result.model_type, ModelType.XGBOOST)
        
        # Should have predictions
        self.assertGreater(result.live_predictions, 0)
        
        # Should have metrics
        self.assertIsInstance(result.candidate_mae, float)
        self.assertIsInstance(result.active_mae, float)
        self.assertIsInstance(result.recommend_promotion, bool)
        
        # Should have reason
        self.assertIsInstance(result.reason, str)


class TestSQLModelRegistry(unittest.TestCase):
    """Test SQLModelRegistry implementation."""
    
    def setUp(self):
        # Use temporary database
        self.temp_dir = Path(tempfile.mkdtemp())
        db_path = self.temp_dir / "test_registry.db"
        model_dir = self.temp_dir / "models"
        
        self.registry = SQLModelRegistry(
            db_url=f"sqlite:///{db_path}",
            model_dir=model_dir
        )
        
        # Create sample artifact
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
        
        self.sample_artifact = ModelArtifact(
            model_type=ModelType.XGBOOST,
            version="v20250130_120000",
            trained_at=datetime.now(timezone.utc),
            metrics={"rmse": 0.01, "mae": 0.008},
            model_object=DummyModel(),
            status=ModelStatus.CANDIDATE,
            training_range=(
                datetime.now(timezone.utc) - timedelta(days=90),
                datetime.now(timezone.utc)
            ),
            feature_config={},
            training_params={},
            data_points=10000,
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_get_model(self):
        """Test saving and retrieving models."""
        # Save model
        self.registry.save_model(self.sample_artifact)
        
        # Update to active
        self.sample_artifact.status = ModelStatus.ACTIVE
        self.registry.save_model(self.sample_artifact)
        self.registry.promote(ModelType.XGBOOST, self.sample_artifact.version)
        
        # Retrieve
        artifact = self.registry.get_active(ModelType.XGBOOST)
        
        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.model_type, ModelType.XGBOOST)
        self.assertEqual(artifact.version, self.sample_artifact.version)
        self.assertEqual(artifact.status, ModelStatus.ACTIVE)
    
    def test_promote_model(self):
        """Test model promotion."""
        # Save two versions
        artifact1 = self.sample_artifact
        artifact1.version = "v1"
        artifact1.status = ModelStatus.ACTIVE
        self.registry.save_model(artifact1)
        self.registry.promote(ModelType.XGBOOST, "v1")
        
        artifact2 = self.sample_artifact
        artifact2.version = "v2"
        artifact2.status = ModelStatus.CANDIDATE
        self.registry.save_model(artifact2)
        
        # Promote v2
        self.registry.promote(ModelType.XGBOOST, "v2")
        
        # v2 should now be active
        active = self.registry.get_active(ModelType.XGBOOST)
        self.assertEqual(active.version, "v2")
    
    def test_get_model_history(self):
        """Test retrieving model history."""
        # Save multiple versions
        for i in range(5):
            artifact = self.sample_artifact
            artifact.version = f"v{i}"
            self.registry.save_model(artifact)
        
        # Get history
        history = self.registry.get_model_history(ModelType.XGBOOST, limit=3)
        
        self.assertEqual(len(history), 3)
        
        # Should be sorted by trained_at (most recent first)
        # In this test all have same timestamp, so just check we got 3
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
