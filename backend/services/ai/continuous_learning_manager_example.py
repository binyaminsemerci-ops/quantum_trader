"""
Continuous Learning Manager - Example Usage

Demonstrates CLM with fake implementations for testing and development.
"""

from datetime import datetime, timedelta
from typing import Any
import random

import pandas as pd
import numpy as np

from backend.services.ai.continuous_learning_manager import (
    ContinuousLearningManager,
    ModelType,
    ModelArtifact,
    ModelStatus,
    EvaluationResult,
    ShadowTestResult,
)


# ============================================================================
# Fake Implementations (for testing/demo)
# ============================================================================

class FakeDataClient:
    """Fake data client that generates synthetic data"""
    
    def load_training_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        days = (end - start).days
        dates = pd.date_range(start, end, freq='1h')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(len(dates)).cumsum() + 100,
            'high': np.random.randn(len(dates)).cumsum() + 102,
            'low': np.random.randn(len(dates)).cumsum() + 98,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates)),
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def load_recent_data(self, days: int) -> pd.DataFrame:
        end = datetime.utcnow()
        start = end - timedelta(days=days)
        return self.load_training_data(start, end)
    
    def load_validation_data(self, days: int) -> pd.DataFrame:
        return self.load_recent_data(days)


class FakeFeatureEngineer:
    """Fake feature engineer that adds simple features"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Add simple features
        df['returns'] = df['close'].pct_change()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['volatility'] = df['returns'].rolling(20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Target (next hour return)
        df['target'] = df['close'].pct_change().shift(-1)
        
        df.dropna(inplace=True)
        
        return df
    
    def get_feature_names(self) -> list[str]:
        return ['returns', 'sma_10', 'sma_30', 'volatility', 'rsi']
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class FakeModelTrainer:
    """Fake trainer that returns dummy model objects"""
    
    def train_xgboost(self, df: pd.DataFrame, params: dict) -> Any:
        print(f"  [Trainer] Training XGBoost with {len(df)} samples...")
        return {"type": "xgboost", "trained": True, "n_samples": len(df)}
    
    def train_lightgbm(self, df: pd.DataFrame, params: dict) -> Any:
        print(f"  [Trainer] Training LightGBM with {len(df)} samples...")
        return {"type": "lightgbm", "trained": True, "n_samples": len(df)}
    
    def train_nhits(self, df: pd.DataFrame, params: dict) -> Any:
        print(f"  [Trainer] Training N-HiTS with {len(df)} samples...")
        return {"type": "nhits", "trained": True, "n_samples": len(df)}
    
    def train_patchtst(self, df: pd.DataFrame, params: dict) -> Any:
        print(f"  [Trainer] Training PatchTST with {len(df)} samples...")
        return {"type": "patchtst", "trained": True, "n_samples": len(df)}


class FakeModelEvaluator:
    """Fake evaluator that generates synthetic metrics"""
    
    def evaluate(
        self, 
        model: Any, 
        df: pd.DataFrame,
        model_type: ModelType
    ) -> EvaluationResult:
        # Generate synthetic metrics
        rmse = random.uniform(0.01, 0.05)
        mae = rmse * 0.8
        error_std = rmse * 0.3
        directional_accuracy = random.uniform(0.52, 0.65)
        hit_rate = directional_accuracy * 0.95
        
        return EvaluationResult(
            model_type=model_type,
            version="unknown",
            rmse=rmse,
            mae=mae,
            error_std=error_std,
            directional_accuracy=directional_accuracy,
            hit_rate=hit_rate,
            correlation_with_target=random.uniform(0.3, 0.7),
            prediction_bias=random.uniform(-0.001, 0.001),
        )
    
    def compare_to_active(
        self,
        candidate_result: EvaluationResult,
        active_result: EvaluationResult
    ) -> EvaluationResult:
        # Add comparison deltas
        candidate_result.vs_active_rmse_delta = (
            candidate_result.rmse - active_result.rmse
        )
        candidate_result.vs_active_direction_delta = (
            candidate_result.directional_accuracy - active_result.directional_accuracy
        )
        
        return candidate_result


class FakeShadowTester:
    """Fake shadow tester that simulates live testing"""
    
    def run_shadow_test(
        self,
        model_type: ModelType,
        candidate_model: Any,
        active_model: Any,
        hours: int = 24
    ) -> ShadowTestResult:
        print(f"  [Shadow] Testing {model_type.value} for {hours}h...")
        
        # Simulate predictions
        n_predictions = hours * 6  # 10-min intervals
        
        candidate_mae = random.uniform(0.015, 0.035)
        active_mae = random.uniform(0.020, 0.040)
        
        candidate_dir_acc = random.uniform(0.53, 0.62)
        active_dir_acc = random.uniform(0.50, 0.58)
        
        # Candidate is better if MAE is lower or direction accuracy higher
        recommend = (
            candidate_mae < active_mae * 0.95
            or candidate_dir_acc > active_dir_acc + 0.02
        )
        
        reason = ""
        if recommend:
            if candidate_mae < active_mae:
                reason = f"Lower MAE: {candidate_mae:.4f} vs {active_mae:.4f}"
            else:
                reason = f"Higher Dir Acc: {candidate_dir_acc:.2%} vs {active_dir_acc:.2%}"
        else:
            reason = "No significant improvement"
        
        return ShadowTestResult(
            model_type=model_type,
            candidate_version="candidate",
            active_version="active",
            live_predictions=n_predictions,
            candidate_mae=candidate_mae,
            active_mae=active_mae,
            candidate_direction_acc=candidate_dir_acc,
            active_direction_acc=active_dir_acc,
            error_ks_statistic=random.uniform(0.05, 0.15),
            error_mean_diff=candidate_mae - active_mae,
            error_std_diff=random.uniform(-0.001, 0.001),
            recommend_promotion=recommend,
            reason=reason,
            tested_hours=float(hours),
        )


class InMemoryModelRegistry:
    """Simple in-memory model registry for testing"""
    
    def __init__(self):
        self.models: dict[ModelType, list[ModelArtifact]] = {
            model_type: [] for model_type in ModelType
        }
        self.active: dict[ModelType, str] = {}
        
        # Seed with initial "active" models
        self._seed_initial_models()
    
    def _seed_initial_models(self):
        """Create initial active models"""
        for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, 
                           ModelType.NHITS, ModelType.PATCHTST]:
            artifact = ModelArtifact(
                model_type=model_type,
                version="v20250101_000000",
                trained_at=datetime.utcnow() - timedelta(days=10),
                metrics={
                    "rmse": 0.035,
                    "mae": 0.028,
                    "directional_accuracy": 0.55,
                    "hit_rate": 0.52,
                },
                model_object={"type": model_type.value, "initial": True},
                status=ModelStatus.ACTIVE,
                data_points=10000,
            )
            
            self.models[model_type].append(artifact)
            self.active[model_type] = artifact.version
    
    def get_active(self, model_type: ModelType) -> ModelArtifact | None:
        active_version = self.active.get(model_type)
        if not active_version:
            return None
        
        for artifact in self.models[model_type]:
            if artifact.version == active_version:
                return artifact
        
        return None
    
    def save_model(self, artifact: ModelArtifact) -> None:
        self.models[artifact.model_type].append(artifact)
    
    def promote(self, model_type: ModelType, new_version: str) -> None:
        self.active[model_type] = new_version
    
    def retire(self, model_type: ModelType, version: str) -> None:
        for artifact in self.models[model_type]:
            if artifact.version == version:
                artifact.status = ModelStatus.RETIRED
    
    def get_model_history(
        self, 
        model_type: ModelType, 
        limit: int = 10
    ) -> list[ModelArtifact]:
        models = self.models[model_type]
        return sorted(models, key=lambda x: x.trained_at, reverse=True)[:limit]


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Demonstrate CLM usage"""
    
    print("=" * 80)
    print("CONTINUOUS LEARNING MANAGER - DEMO")
    print("=" * 80)
    print()
    
    # Initialize CLM with fake dependencies
    clm = ContinuousLearningManager(
        data_client=FakeDataClient(),
        feature_engineer=FakeFeatureEngineer(),
        trainer=FakeModelTrainer(),
        evaluator=FakeModelEvaluator(),
        shadow_tester=FakeShadowTester(),
        registry=InMemoryModelRegistry(),
        retrain_interval_days=7,
        shadow_test_hours=2,  # Shorter for demo
        min_improvement_threshold=0.02,
        training_lookback_days=90,
    )
    
    print("CLM Initialized ✓\n")
    
    # Check current model status
    print("Current Model Status:")
    print("-" * 80)
    status = clm.get_model_status_summary()
    for model_type, info in status.items():
        print(f"  {model_type:12} | {info['active_version']:20} | "
              f"Trained: {info['active_trained_at'][:10] if info['active_trained_at'] else 'N/A'}")
    print()
    
    # Check retraining triggers
    print("Checking Retraining Triggers:")
    print("-" * 80)
    triggers = clm.check_if_retrain_needed()
    for model_type, trigger in triggers.items():
        if trigger:
            print(f"  ✅ {model_type.value:12} | Trigger: {trigger.value}")
        else:
            print(f"  ⏭️  {model_type.value:12} | No trigger")
    print()
    
    # Run full retraining cycle
    print("Running Full Retraining Cycle:")
    print("=" * 80)
    print()
    
    report = clm.run_full_cycle(force=True)
    
    print()
    print("=" * 80)
    print("RETRAINING REPORT")
    print("=" * 80)
    print(report.summary())
    print()
    
    print("Promoted Models:")
    for model_type in report.promoted_models:
        print(f"  ✅ {model_type.value}")
    
    print()
    print("Failed/Skipped Models:")
    for model_type in report.failed_models:
        print(f"  ❌ {model_type.value}")
    
    print()
    print("Evaluation Results:")
    print("-" * 80)
    for model_type, eval_result in report.models_evaluated.items():
        if eval_result:
            print(f"  {model_type.value:12} | "
                  f"RMSE: {eval_result.rmse:.4f} | "
                  f"Dir Acc: {eval_result.directional_accuracy:.2%} | "
                  f"Better: {eval_result.is_better_than_active()}")
    
    print()
    print("Shadow Test Results:")
    print("-" * 80)
    for model_type, shadow_result in report.shadow_results.items():
        if shadow_result:
            print(f"  {model_type.value:12} | "
                  f"Candidate MAE: {shadow_result.candidate_mae:.4f} | "
                  f"Active MAE: {shadow_result.active_mae:.4f} | "
                  f"Recommend: {shadow_result.recommend_promotion}")
            if shadow_result.reason:
                print(f"    └─ {shadow_result.reason}")
    
    print()
    print("Updated Model Status:")
    print("-" * 80)
    status = clm.get_model_status_summary()
    for model_type, info in status.items():
        print(f"  {model_type:12} | {info['active_version']:20} | "
              f"Metrics: {info['active_metrics']}")
    
    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
