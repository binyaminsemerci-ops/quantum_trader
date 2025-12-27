"""
Continuous Learning Manager (CLM)

Automatically retrains, evaluates, shadow-tests, and promotes ML models
to prevent drift and adapt to evolving market conditions.

Core responsibilities:
- Detect retraining triggers (time, data volume, performance decay)
- Train new model versions (XGBoost, LightGBM, N-HiTS, PatchTST)
- Evaluate candidates on validation sets
- Shadow-test in live mode (parallel with active models)
- Promote superior models to production
- Track model performance history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, Optional
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ModelType(str, Enum):
    """Supported model types"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "nhits"
    PATCHTST = "patchtst"
    RL_SIZING = "rl_sizing"


class ModelStatus(str, Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class RetrainTrigger(str, Enum):
    """Why retraining was triggered"""
    TIME_BASED = "time_based"
    DATA_VOLUME = "data_volume"
    PERFORMANCE_DECAY = "performance_decay"
    REGIME_SHIFT = "regime_shift"
    MANUAL = "manual"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ModelArtifact:
    """
    Represents a trained model version with metadata.
    """
    model_type: ModelType
    version: str
    trained_at: datetime
    metrics: dict[str, float]
    model_object: Any
    status: ModelStatus = ModelStatus.CANDIDATE
    training_range: Optional[tuple[datetime, datetime]] = None
    feature_config: Optional[dict] = None
    training_params: Optional[dict] = None
    data_points: int = 0
    
    def to_dict(self) -> dict:
        """Serialize for storage"""
        return {
            "model_type": self.model_type.value,
            "version": self.version,
            "trained_at": self.trained_at.isoformat(),
            "metrics": self.metrics,
            "status": self.status.value,
            "training_range": [
                self.training_range[0].isoformat(), 
                self.training_range[1].isoformat()
            ] if self.training_range else None,
            "feature_config": self.feature_config,
            "training_params": self.training_params,
            "data_points": self.data_points,
        }


@dataclass
class EvaluationResult:
    """Model evaluation metrics"""
    model_type: ModelType
    version: str
    
    # Regression metrics
    rmse: float
    mae: float
    error_std: float
    
    # Classification metrics
    directional_accuracy: float
    hit_rate: float
    
    # Regime-specific (optional)
    regime_accuracy: Optional[dict[str, float]] = None
    
    # Comparison to active
    vs_active_rmse_delta: Optional[float] = None
    vs_active_direction_delta: Optional[float] = None
    
    # Statistical
    correlation_with_target: float = 0.0
    prediction_bias: float = 0.0
    
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_better_than_active(self, threshold: float = 0.02) -> bool:
        """
        Check if candidate is meaningfully better than active.
        
        Args:
            threshold: Minimum improvement required (2% by default)
        """
        if self.vs_active_rmse_delta is None:
            return False
        
        # Better = lower RMSE and higher directional accuracy
        rmse_improved = self.vs_active_rmse_delta < -threshold
        direction_improved = (
            self.vs_active_direction_delta is not None 
            and self.vs_active_direction_delta > threshold
        )
        
        return rmse_improved or direction_improved


@dataclass
class ShadowTestResult:
    """Results from shadow testing in live mode"""
    model_type: ModelType
    candidate_version: str
    active_version: str
    
    # Live performance
    live_predictions: int
    candidate_mae: float
    active_mae: float
    
    candidate_direction_acc: float
    active_direction_acc: float
    
    # Distribution comparison
    error_ks_statistic: float  # Kolmogorov-Smirnov test
    error_mean_diff: float
    error_std_diff: float
    
    # Regime breakdown (optional)
    regime_performance: Optional[dict] = None
    
    # Recommendation
    recommend_promotion: bool = False
    reason: str = ""
    
    tested_from: datetime = field(default_factory=datetime.utcnow)
    tested_hours: float = 0.0


@dataclass
class RetrainReport:
    """Complete retraining cycle report"""
    trigger: RetrainTrigger
    triggered_at: datetime
    
    models_trained: list[ModelType]
    models_evaluated: dict[ModelType, EvaluationResult]
    shadow_results: dict[ModelType, ShadowTestResult]
    
    promoted_models: list[ModelType]
    failed_models: list[ModelType]
    
    total_duration_seconds: float
    
    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"CLM Report [{self.triggered_at.strftime('%Y-%m-%d %H:%M')}]\n"
            f"Trigger: {self.trigger.value}\n"
            f"Trained: {len(self.models_trained)} models\n"
            f"Promoted: {len(self.promoted_models)} models\n"
            f"Failed: {len(self.failed_models)} models\n"
            f"Duration: {self.total_duration_seconds:.1f}s"
        )


# ============================================================================
# Protocols (Dependency Interfaces)
# ============================================================================

class DataClient(Protocol):
    """Data fetching interface"""
    
    def load_training_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Load historical data for training"""
        ...
    
    def load_recent_data(self, days: int) -> pd.DataFrame:
        """Load recent data for evaluation"""
        ...
    
    def load_validation_data(self, days: int) -> pd.DataFrame:
        """Load validation set"""
        ...


class FeatureEngineer(Protocol):
    """Feature transformation interface"""
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering pipeline"""
        ...
    
    def get_feature_names(self) -> list[str]:
        """Return list of feature columns"""
        ...


class ModelTrainer(Protocol):
    """Model training interface"""
    
    def train_xgboost(self, df: pd.DataFrame, params: dict) -> Any:
        """Train XGBoost model"""
        ...
    
    def train_lightgbm(self, df: pd.DataFrame, params: dict) -> Any:
        """Train LightGBM model"""
        ...
    
    def train_nhits(self, df: pd.DataFrame, params: dict) -> Any:
        """Train N-HiTS model"""
        ...
    
    def train_patchtst(self, df: pd.DataFrame, params: dict) -> Any:
        """Train PatchTST model"""
        ...


class ModelEvaluator(Protocol):
    """Model evaluation interface"""
    
    def evaluate(
        self, 
        model: Any, 
        df: pd.DataFrame,
        model_type: ModelType
    ) -> EvaluationResult:
        """Evaluate model on validation data"""
        ...
    
    def compare_to_active(
        self,
        candidate_result: EvaluationResult,
        active_result: EvaluationResult
    ) -> EvaluationResult:
        """Add comparison metrics"""
        ...


class ShadowTester(Protocol):
    """Shadow testing interface"""
    
    def run_shadow_test(
        self,
        model_type: ModelType,
        candidate_model: Any,
        active_model: Any,
        hours: int = 24
    ) -> ShadowTestResult:
        """Run live shadow test"""
        ...


class ModelRegistry(Protocol):
    """Model storage and versioning interface"""
    
    def get_active(self, model_type: ModelType) -> Optional[ModelArtifact]:
        """Get currently active model"""
        ...
    
    def save_model(self, artifact: ModelArtifact) -> None:
        """Save model artifact"""
        ...
    
    def promote(self, model_type: ModelType, new_version: str) -> None:
        """Promote candidate to active"""
        ...
    
    def retire(self, model_type: ModelType, version: str) -> None:
        """Mark model as retired"""
        ...
    
    def get_model_history(
        self, 
        model_type: ModelType, 
        limit: int = 10
    ) -> list[ModelArtifact]:
        """Get model version history"""
        ...


# ============================================================================
# Retraining Trigger System
# ============================================================================

class RetrainTriggerDetector:
    """
    Detects when model retraining should be triggered.
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        data_client: DataClient,
        time_threshold_days: int = 7,
        data_threshold_points: int = 10000,
        performance_decay_threshold: float = 0.05,
    ):
        self.registry = registry
        self.data_client = data_client
        self.time_threshold_days = time_threshold_days
        self.data_threshold_points = data_threshold_points
        self.performance_decay_threshold = performance_decay_threshold
    
    def check_time_trigger(self, model_type: ModelType) -> bool:
        """Check if enough time has passed since last training"""
        active = self.registry.get_active(model_type)
        if not active:
            return True
        
        age = datetime.utcnow() - active.trained_at
        return age.days >= self.time_threshold_days
    
    def check_data_volume_trigger(self, model_type: ModelType) -> bool:
        """Check if enough new data points are available"""
        active = self.registry.get_active(model_type)
        if not active:
            return True
        
        # Check new data since training
        recent_df = self.data_client.load_recent_data(days=7)
        new_points = len(recent_df)
        
        return new_points >= self.data_threshold_points
    
    def check_performance_decay(self, model_type: ModelType) -> bool:
        """
        Check if model performance has degraded.
        
        This would compare recent predictions vs actuals
        to baseline performance metrics.
        """
        # Placeholder: In production, query recent prediction accuracy
        # from a performance tracking table
        return False
    
    def check_regime_shift(self) -> bool:
        """Check if market regime has shifted significantly"""
        # Placeholder: In production, integrate with RegimeDetector
        return False
    
    def get_trigger(self, model_type: ModelType) -> Optional[RetrainTrigger]:
        """
        Determine if and why retraining should be triggered.
        
        Returns:
            RetrainTrigger if retraining needed, None otherwise
        """
        if self.check_time_trigger(model_type):
            return RetrainTrigger.TIME_BASED
        
        if self.check_data_volume_trigger(model_type):
            return RetrainTrigger.DATA_VOLUME
        
        if self.check_performance_decay(model_type):
            return RetrainTrigger.PERFORMANCE_DECAY
        
        if self.check_regime_shift():
            return RetrainTrigger.REGIME_SHIFT
        
        return None


# ============================================================================
# Continuous Learning Manager (Main)
# ============================================================================

class ContinuousLearningManager:
    """
    Orchestrates the complete model lifecycle:
    - Trigger detection
    - Training
    - Evaluation
    - Shadow testing
    - Promotion
    
    Ensures models stay fresh and aligned with current market dynamics.
    """
    
    def __init__(
        self,
        data_client: DataClient,
        feature_engineer: FeatureEngineer,
        trainer: ModelTrainer,
        evaluator: ModelEvaluator,
        shadow_tester: ShadowTester,
        registry: ModelRegistry,
        retrain_interval_days: int = 7,
        shadow_test_hours: int = 24,
        min_improvement_threshold: float = 0.02,
        training_lookback_days: int = 90,
        policy_store = None,  # [NEW] PolicyStore for model version tracking
    ):
        """
        Initialize CLM.
        
        Args:
            data_client: Data fetching service
            feature_engineer: Feature transformation pipeline
            trainer: Model training service
            evaluator: Model evaluation service
            shadow_tester: Shadow testing service
            registry: Model storage and versioning
            retrain_interval_days: How often to retrain (time trigger)
            shadow_test_hours: How long to run shadow tests
            min_improvement_threshold: Min % improvement to promote
            training_lookback_days: Historical data window for training
            policy_store: PolicyStore for writing model versions
        """
        self.data_client = data_client
        self.feature_engineer = feature_engineer
        self.trainer = trainer
        self.evaluator = evaluator
        self.shadow_tester = shadow_tester
        self.registry = registry
        self.policy_store = policy_store  # [NEW] Store reference for version writes
        
        self.retrain_interval_days = retrain_interval_days
        self.shadow_test_hours = shadow_test_hours
        self.min_improvement_threshold = min_improvement_threshold
        self.training_lookback_days = training_lookback_days
        
        # Trigger detector
        self.trigger_detector = RetrainTriggerDetector(
            registry=registry,
            data_client=data_client,
            time_threshold_days=retrain_interval_days,
        )
        
        # Supported models
        self.model_types = [
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
            ModelType.NHITS,
            ModelType.PATCHTST,
        ]
        
        if self.policy_store:
            logger.info("✅ PolicyStore integration enabled in CLM (model version tracking)")
        
        logger.info(
            f"[CLM] Initialized: retrain_interval={retrain_interval_days}d, "
            f"shadow_hours={shadow_test_hours}h, "
            f"min_improvement={min_improvement_threshold*100}%"
        )
    
    # ========================================================================
    # Trigger Detection
    # ========================================================================
    
    def check_if_retrain_needed(
        self, 
        model_type: Optional[ModelType] = None
    ) -> dict[ModelType, Optional[RetrainTrigger]]:
        """
        Check which models need retraining.
        
        Args:
            model_type: Check specific model, or None for all
        
        Returns:
            Dict mapping model_type to trigger reason (or None)
        """
        models_to_check = (
            [model_type] if model_type else self.model_types
        )
        
        triggers = {}
        for mt in models_to_check:
            trigger = self.trigger_detector.get_trigger(mt)
            triggers[mt] = trigger
            
            if trigger:
                logger.info(f"[CLM] Retrain trigger for {mt.value}: {trigger.value}")
        
        return triggers
    
    # ========================================================================
    # Training Pipeline
    # ========================================================================
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load and prepare training dataset"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.training_lookback_days)
        
        logger.info(
            f"[CLM] Loading training data: {start_date.date()} to {end_date.date()}"
        )
        
        raw_df = self.data_client.load_training_data(start_date, end_date)
        
        # Apply feature engineering
        df = self.feature_engineer.transform(raw_df)
        
        logger.info(f"[CLM] Training data ready: {len(df)} rows, {len(df.columns)} features")
        
        return df
    
    def _train_single_model(
        self, 
        model_type: ModelType, 
        df: pd.DataFrame
    ) -> Optional[ModelArtifact]:
        """
        Train a single model type.
        
        Args:
            model_type: Model to train
            df: Training dataframe
        
        Returns:
            ModelArtifact or None if training failed
        """
        logger.info(f"[CLM] Training {model_type.value}...")
        
        try:
            # Get training params (could be loaded from config)
            params = self._get_training_params(model_type)
            
            # Train
            start_time = datetime.utcnow()
            
            if model_type == ModelType.XGBOOST:
                model_obj = self.trainer.train_xgboost(df, params)
            elif model_type == ModelType.LIGHTGBM:
                model_obj = self.trainer.train_lightgbm(df, params)
            elif model_type == ModelType.NHITS:
                model_obj = self.trainer.train_nhits(df, params)
            elif model_type == ModelType.PATCHTST:
                model_obj = self.trainer.train_patchtst(df, params)
            else:
                logger.error(f"[CLM] Unknown model type: {model_type}")
                return None
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create version string
            version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create artifact
            artifact = ModelArtifact(
                model_type=model_type,
                version=version,
                trained_at=start_time,
                metrics={},  # Will be filled by evaluation
                model_object=model_obj,
                status=ModelStatus.CANDIDATE,
                training_range=(df.index[0], df.index[-1]) if len(df) > 0 else None,
                training_params=params,
                data_points=len(df),
            )
            
            logger.info(
                f"[CLM] ✅ {model_type.value} trained: {version} "
                f"({duration:.1f}s, {len(df)} points)"
            )
            
            return artifact
            
        except Exception as e:
            logger.error(f"[CLM] ❌ Training failed for {model_type.value}: {e}")
            return None
    
    def retrain_all(
        self, 
        models: Optional[list[ModelType]] = None
    ) -> dict[ModelType, Optional[ModelArtifact]]:
        """
        Train new versions of all (or specified) models.
        
        Args:
            models: List of models to train, or None for all
        
        Returns:
            Dict mapping model_type to ModelArtifact (or None if failed)
        """
        models_to_train = models if models else self.model_types
        
        logger.info(f"[CLM] Starting training for {len(models_to_train)} models")
        
        # Load training data once
        df = self._load_training_data()
        
        # Train each model
        artifacts = {}
        for model_type in models_to_train:
            artifact = self._train_single_model(model_type, df)
            artifacts[model_type] = artifact
            
            # [PERMANENT FIX] Save model immediately after training
            if artifact is not None:
                try:
                    # Save as CANDIDATE first
                    artifact.status = ModelStatus.CANDIDATE
                    self.registry.save_model(artifact)
                    logger.info(f"[CLM] 💾 Saved {model_type.value} {artifact.version} as CANDIDATE")
                    
                    # Auto-promote if no active model exists (first-time setup)
                    active = self.registry.get_active(model_type)
                    if not active:
                        artifact.status = ModelStatus.ACTIVE
                        self.registry.promote(model_type, artifact.version)
                        logger.info(f"[CLM] ✅ Auto-promoted {model_type.value} {artifact.version} (no active model)")
                        
                        # Write to PolicyStore
                        if self.policy_store:
                            try:
                                model_versions = self.policy_store.get('model_versions', {})
                                model_versions[model_type.value] = artifact.version
                                self.policy_store.patch({'model_versions': model_versions})
                                logger.info(f"[CLM] 📝 Updated PolicyStore: {model_type.value}={artifact.version}")
                            except Exception as e:
                                logger.warning(f"[CLM] PolicyStore update failed: {e}")
                                
                except Exception as e:
                    logger.error(f"[CLM] ❌ Failed to save {model_type.value}: {e}", exc_info=True)
        
        successful = sum(1 for a in artifacts.values() if a is not None)
        logger.info(f"[CLM] Training complete: {successful}/{len(models_to_train)} successful")
        
        return artifacts
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    def evaluate_models(
        self, 
        artifacts: dict[ModelType, Optional[ModelArtifact]]
    ) -> dict[ModelType, Optional[EvaluationResult]]:
        """
        Evaluate candidate models on validation set.
        
        Args:
            artifacts: Trained model artifacts
        
        Returns:
            Dict mapping model_type to EvaluationResult
        """
        logger.info("[CLM] Starting model evaluation...")
        
        # Load validation data
        val_df = self.data_client.load_validation_data(days=30)
        val_df = self.feature_engineer.transform(val_df)
        
        results = {}
        
        for model_type, artifact in artifacts.items():
            if artifact is None:
                results[model_type] = None
                continue
            
            try:
                # Evaluate candidate
                result = self.evaluator.evaluate(
                    artifact.model_object,
                    val_df,
                    model_type
                )
                
                # Compare to active model
                active = self.registry.get_active(model_type)
                if active:
                    active_result = self.evaluator.evaluate(
                        active.model_object,
                        val_df,
                        model_type
                    )
                    result = self.evaluator.compare_to_active(result, active_result)
                
                # Store metrics in artifact
                artifact.metrics = {
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "directional_accuracy": result.directional_accuracy,
                    "hit_rate": result.hit_rate,
                }
                
                results[model_type] = result
                
                logger.info(
                    f"[CLM] {model_type.value} evaluation: "
                    f"RMSE={result.rmse:.4f}, "
                    f"Dir Acc={result.directional_accuracy:.2%}"
                )
                
            except Exception as e:
                logger.error(f"[CLM] Evaluation failed for {model_type.value}: {e}")
                results[model_type] = None
        
        return results
    
    # ========================================================================
    # Shadow Testing
    # ========================================================================
    
    def run_shadow_tests(
        self, 
        artifacts: dict[ModelType, Optional[ModelArtifact]]
    ) -> dict[ModelType, Optional[ShadowTestResult]]:
        """
        Run shadow tests for candidate models.
        
        Candidates run in parallel with active models in live mode.
        
        Args:
            artifacts: Candidate model artifacts
        
        Returns:
            Dict mapping model_type to ShadowTestResult
        """
        logger.info(f"[CLM] Starting shadow tests ({self.shadow_test_hours}h)...")
        
        results = {}
        
        for model_type, artifact in artifacts.items():
            if artifact is None:
                results[model_type] = None
                continue
            
            active = self.registry.get_active(model_type)
            if not active:
                logger.warning(f"[CLM] No active model for {model_type.value}, skipping shadow")
                results[model_type] = None
                continue
            
            try:
                shadow_result = self.shadow_tester.run_shadow_test(
                    model_type=model_type,
                    candidate_model=artifact.model_object,
                    active_model=active.model_object,
                    hours=self.shadow_test_hours,
                )
                
                results[model_type] = shadow_result
                
                logger.info(
                    f"[CLM] {model_type.value} shadow test: "
                    f"Candidate MAE={shadow_result.candidate_mae:.4f}, "
                    f"Active MAE={shadow_result.active_mae:.4f}, "
                    f"Recommend={shadow_result.recommend_promotion}"
                )
                
            except Exception as e:
                logger.error(f"[CLM] Shadow test failed for {model_type.value}: {e}")
                results[model_type] = None
        
        return results
    
    # ========================================================================
    # Promotion Logic
    # ========================================================================
    
    def promote_if_better(
        self,
        artifacts: dict[ModelType, Optional[ModelArtifact]],
        evaluations: dict[ModelType, Optional[EvaluationResult]],
        shadows: dict[ModelType, Optional[ShadowTestResult]],
    ) -> list[ModelType]:
        """
        Promote candidate models if they outperform active models.
        
        [FIX #3] Includes post-promotion circuit breaker with 24h monitoring.
        
        Promotion criteria:
        1. Evaluation shows improvement > threshold
        2. Shadow test recommends promotion
        3. No critical issues detected
        
        Args:
            artifacts: Candidate models
            evaluations: Evaluation results
            shadows: Shadow test results
        
        Returns:
            List of promoted model types
        """
        logger.info("[CLM] Evaluating promotion candidates...")
        
        promoted = []
        
        for model_type, artifact in artifacts.items():
            if artifact is None:
                continue
            
            eval_result = evaluations.get(model_type)
            shadow_result = shadows.get(model_type)
            
            # Decision criteria
            eval_better = (
                eval_result is not None 
                and eval_result.is_better_than_active(self.min_improvement_threshold)
            )
            
            shadow_recommends = (
                shadow_result is not None 
                and shadow_result.recommend_promotion
            )
            
            should_promote = eval_better or shadow_recommends
            
            if should_promote:
                try:
                    # Retire old active model first
                    active = self.registry.get_active(model_type)
                    if active and active.version != artifact.version:
                        self.registry.retire(model_type, active.version)
                        logger.info(f"[CLM] 🔄 Retired old version: {model_type.value} {active.version}")
                    
                    # Promote candidate to active (model already saved in retrain_all)
                    artifact.status = ModelStatus.ACTIVE
                    self.registry.promote(model_type, artifact.version)
                    
                    promoted.append(model_type)
                    
                    # [FIX #3] Initialize post-promotion circuit breaker
                    old_version_for_rollback = active.version if active else None
                    self._init_circuit_breaker(model_type, artifact.version, old_version_for_rollback)
                    
                    # Write version to PolicyStore
                    if self.policy_store:
                        try:
                            # Get all active model versions
                            model_versions = {}
                            for mt in self.model_types:
                                active = self.registry.get_active(mt)
                                if active:
                                    model_versions[mt.value] = active.version
                            
                            # Write to PolicyStore
                            self.policy_store.patch({'model_versions': model_versions})
                            logger.info(f"[CLM] ✅ Model versions written to PolicyStore: {model_versions}")
                        except Exception as e:
                            logger.error(f"[CLM] ❌ Failed to write to PolicyStore: {e}")
                    
                    logger.info(
                        f"[CLM] ✅ Promoted {model_type.value} {artifact.version} to ACTIVE"
                    )
                    logger.info(
                        f"[FIX #3] 🚪 Circuit breaker active for 24h - monitoring performance"
                    )
                    
                except Exception as e:
                    logger.error(f"[CLM] ❌ Promotion failed for {model_type.value}: {e}")
            else:
                logger.info(
                    f"[CLM] ⏭️ Skipping promotion for {model_type.value} "
                    f"(eval_better={eval_better}, shadow_recommends={shadow_recommends})"
                )
        
        return promoted
    
    # ========================================================================
    # Full Cycle
    # ========================================================================
    
    def run_full_cycle(
        self,
        models: Optional[list[ModelType]] = None,
        force: bool = False,
    ) -> RetrainReport:
        """
        Execute complete retraining cycle:
        1. Check triggers
        2. Train models
        3. Evaluate
        4. Shadow test
        5. Promote
        
        Args:
            models: Specific models to retrain, or None for all
            force: Skip trigger checks and force retraining
        
        Returns:
            RetrainReport with complete cycle results
        """
        start_time = datetime.utcnow()
        
        logger.info("=" * 80)
        logger.info("[CLM] 🚀 Starting full retraining cycle")
        logger.info("=" * 80)
        
        # Step 1: Check triggers
        if not force:
            triggers = self.check_if_retrain_needed()
            models_to_retrain = [mt for mt, trigger in triggers.items() if trigger]
            
            if not models_to_retrain:
                logger.info("[CLM] No retraining triggers detected")
                return RetrainReport(
                    trigger=RetrainTrigger.MANUAL,
                    triggered_at=start_time,
                    models_trained=[],
                    models_evaluated={},
                    shadow_results={},
                    promoted_models=[],
                    failed_models=[],
                    total_duration_seconds=0.0,
                )
            
            # Use first trigger as primary
            primary_trigger = next(iter([t for t in triggers.values() if t]))
        else:
            models_to_retrain = models if models else self.model_types
            primary_trigger = RetrainTrigger.MANUAL
        
        logger.info(f"[CLM] Models to retrain: {[m.value for m in models_to_retrain]}")
        
        # Step 2: Train
        artifacts = self.retrain_all(models_to_retrain)
        
        # Step 3: Evaluate
        evaluations = self.evaluate_models(artifacts)
        
        # Step 4: Shadow test
        shadows = self.run_shadow_tests(artifacts)
        
        # Step 5: Promote
        promoted = self.promote_if_better(artifacts, evaluations, shadows)
        
        # Determine failures
        failed = [
            mt for mt, artifact in artifacts.items()
            if artifact is None or mt not in promoted
        ]
        
        # Create report
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        report = RetrainReport(
            trigger=primary_trigger,
            triggered_at=start_time,
            models_trained=list(artifacts.keys()),
            models_evaluated=evaluations,
            shadow_results=shadows,
            promoted_models=promoted,
            failed_models=failed,
            total_duration_seconds=duration,
        )
        
        logger.info("=" * 80)
        logger.info("[CLM] ✅ Full cycle complete")
        logger.info(report.summary())
        logger.info("=" * 80)
        
        return report
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _get_training_params(self, model_type: ModelType) -> dict:
        """Get default training parameters for model type"""
        params = {
            ModelType.XGBOOST: {
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.01,
                "subsample": 0.8,
            },
            ModelType.LIGHTGBM: {
                "n_estimators": 500,
                "max_depth": 7,
                "learning_rate": 0.01,
                "subsample": 0.8,
            },
            ModelType.NHITS: {
                "input_size": 120,
                "h": 24,
                "n_blocks": 3,
            },
            ModelType.PATCHTST: {
                "input_size": 120,
                "h": 24,
                "patch_len": 16,
            },
        }
        
        return params.get(model_type, {})
    
    def get_model_status_summary(self) -> dict:
        """Get summary of all model statuses"""
        summary = {}
        
        for model_type in self.model_types:
            active = self.registry.get_active(model_type)
            history = self.registry.get_model_history(model_type, limit=5)
            
            summary[model_type.value] = {
                "active_version": active.version if active else None,
                "active_trained_at": active.trained_at.isoformat() if active else None,
                "active_metrics": active.metrics if active else {},
                "history_count": len(history),
            }
        
        return summary
    
    def _init_circuit_breaker(
        self, 
        model_type: ModelType, 
        new_version: str, 
        old_version: Optional[str]
    ) -> None:
        """[FIX #3] Initialize 24h circuit breaker for newly promoted model."""
        import json
        from pathlib import Path
        
        circuit_breaker_file = Path("/app/data") / "circuit_breakers.json"
        circuit_breaker_file.parent.mkdir(parents=True, exist_ok=True)
        
        breaker_data = {
            "model_type": model_type.value,
            "new_version": new_version,
            "old_version": old_version,
            "promoted_at": datetime.utcnow().isoformat(),
            "monitoring_until": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "trades_count": 0,
            "wins": 0,
            "losses": 0,
            "total_r": 0.0,
            "status": "MONITORING"
        }
        
        try:
            circuit_breakers = {}
            if circuit_breaker_file.exists():
                with open(circuit_breaker_file, 'r') as f:
                    circuit_breakers = json.load(f)
            
            circuit_breakers[model_type.value] = breaker_data
            
            with open(circuit_breaker_file, 'w') as f:
                json.dump(circuit_breakers, f, indent=2)
            
            logger.info(f"[FIX #3] Circuit breaker initialized for {model_type.value}")
        except Exception as e:
            logger.error(f"[FIX #3] Failed to init circuit breaker: {e}")
    
    def check_circuit_breaker(self, model_type: ModelType, trade_result: Dict[str, Any]) -> bool:
        """[FIX #3] Check circuit breaker - rollback if new model underperforms.
        
        Args:
            model_type: Model to check
            trade_result: Latest trade result with r_multiple
        
        Returns:
            True if rollback was triggered
        """
        import json
        from pathlib import Path
        
        circuit_breaker_file = Path("/app/data") / "circuit_breakers.json"
        
        if not circuit_breaker_file.exists():
            return False
        
        try:
            with open(circuit_breaker_file, 'r') as f:
                circuit_breakers = json.load(f)
            
            breaker = circuit_breakers.get(model_type.value)
            if not breaker or breaker["status"] != "MONITORING":
                return False
            
            # Check if monitoring period expired
            monitoring_until = datetime.fromisoformat(breaker["monitoring_until"])
            if datetime.utcnow() > monitoring_until:
                breaker["status"] = "PASSED"
                with open(circuit_breaker_file, 'w') as f:
                    json.dump(circuit_breakers, f, indent=2)
                logger.info(f"[FIX #3] ✅ Circuit breaker passed for {model_type.value}")
                return False
            
            # Update metrics
            breaker["trades_count"] += 1
            r_multiple = trade_result.get("r_multiple", 0.0)
            breaker["total_r"] += r_multiple
            
            if r_multiple > 0:
                breaker["wins"] += 1
            else:
                breaker["losses"] += 1
            
            # Check rollback conditions (after min 10 trades)
            if breaker["trades_count"] >= 10:
                avg_r = breaker["total_r"] / breaker["trades_count"]
                winrate = breaker["wins"] / breaker["trades_count"]
                
                # Trigger rollback if: avg_R < -0.3 OR winrate < 35%
                should_rollback = (avg_r < -0.3) or (winrate < 0.35)
                
                if should_rollback:
                    logger.error(
                        f"[FIX #3] 🚨 CIRCUIT BREAKER TRIGGERED for {model_type.value}! "
                        f"avg_R={avg_r:.2f}, WR={winrate:.1%} - ROLLING BACK"
                    )
                    
                    # Perform rollback
                    old_version = breaker.get("old_version")
                    if old_version:
                        self.registry.demote(model_type, breaker["new_version"])
                        self.registry.promote(model_type, old_version)
                        
                        breaker["status"] = "ROLLED_BACK"
                        breaker["rollback_at"] = datetime.utcnow().isoformat()
                        
                        logger.info(
                            f"[FIX #3] ✅ Rollback complete: {breaker['new_version']} → {old_version}"
                        )
                        
                        with open(circuit_breaker_file, 'w') as f:
                            json.dump(circuit_breakers, f, indent=2)
                        
                        return True
            
            # Save updated metrics
            with open(circuit_breaker_file, 'w') as f:
                json.dump(circuit_breakers, f, indent=2)
            
        except Exception as e:
            logger.error(f"[FIX #3] Error checking circuit breaker: {e}")
        
        return False

