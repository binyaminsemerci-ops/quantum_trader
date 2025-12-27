"""
RETRAINING SYSTEM ORCHESTRATOR

Automates model retraining based on performance drift, schedule, and regime changes.
Maintains model versioning, safe deployment policies, and rollback capabilities.

Author: Quantum Trader AI Team
Date: November 23, 2025
"""

import os
import json
import logging
import pickle
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

class TriggerType(Enum):
    """Types of retraining triggers."""
    PERFORMANCE_DRIVEN = "performance_driven"  # Model performance degraded
    TIME_DRIVEN = "time_driven"                # Scheduled periodic retrain
    REGIME_DRIVEN = "regime_driven"            # Market regime changed
    MANUAL = "manual"                          # Manual override
    DRIFT_DETECTED = "drift_detected"          # Model drift detected


class RetrainingStatus(Enum):
    """Status of retraining job."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentDecision(Enum):
    """Deployment decision for new model."""
    DEPLOY_IMMEDIATELY = "deploy_immediately"  # Deploy to production
    RUN_CANARY = "run_canary"                  # Run alongside old model
    KEEP_OLD = "keep_old"                      # Keep old model, reject new
    REQUIRES_REVIEW = "requires_review"        # Manual review needed


@dataclass
class ModelVersion:
    """Represents a trained model version."""
    model_id: str  # e.g., "xgboost_v1"
    version_tag: str  # e.g., "v20251123_120000"
    training_date: str  # ISO timestamp
    data_range_start: str  # Data start date
    data_range_end: str  # Data end date
    sample_count: int
    feature_count: int
    
    # Training config
    hyperparameters: Dict[str, Any]
    features_used: List[str]
    
    # Performance metrics
    train_metrics: Dict[str, float]  # winrate, avg_R, calibration
    validation_metrics: Dict[str, float]
    
    # Deployment status
    is_deployed: bool
    deployment_date: Optional[str]
    
    # Paths
    model_path: str
    scaler_path: Optional[str]
    metadata_path: str


@dataclass
class RetrainingTrigger:
    """Represents a trigger for retraining."""
    trigger_id: str
    trigger_type: TriggerType
    model_id: str
    timestamp: str
    reason: str
    priority: str  # URGENT, HIGH, MEDIUM, LOW
    metadata: Dict[str, Any]


@dataclass
class RetrainingJob:
    """Represents a retraining job."""
    job_id: str
    model_id: str
    trigger: RetrainingTrigger
    status: RetrainingStatus
    
    # Timing
    scheduled_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    
    # Configuration
    data_window_days: int
    features_config: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    
    # Result
    new_version: Optional[ModelVersion]
    error_message: Optional[str]
    logs: List[str]


@dataclass
class DeploymentRecommendation:
    """Recommendation for deploying new model version."""
    model_id: str
    old_version: Optional[ModelVersion]
    new_version: ModelVersion
    decision: DeploymentDecision
    
    # Comparison
    performance_improvement: float  # % improvement in key metric
    is_safe_to_deploy: bool
    requires_canary: bool
    
    # Reasoning
    reasons: List[str]
    concerns: List[str]
    
    # Thresholds
    min_improvement_threshold: float  # e.g., 0.05 = 5% improvement required


@dataclass
class RetrainingPlan:
    """Plan for upcoming retraining jobs."""
    plan_id: str
    created_at: str
    jobs: List[RetrainingJob]
    total_jobs: int
    estimated_duration_minutes: float


# ============================================================
# RETRAINING ORCHESTRATOR
# ============================================================

class RetrainingOrchestrator:
    """
    Manages model retraining lifecycle:
    - Evaluates triggers (performance, time, regime)
    - Coordinates training pipeline
    - Maintains model versions
    - Makes deployment decisions
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        models_dir: str = "/app/ai_engine/models",
        scripts_dir: str = "/app/scripts",
        
        # Performance thresholds
        min_winrate: float = 0.50,
        min_avg_r: float = 0.0,
        min_calibration: float = 0.70,
        
        # Deployment thresholds
        min_improvement_pct: float = 0.05,  # 5% improvement to deploy
        canary_threshold_pct: float = 0.02,  # 2-5% improvement = canary
        
        # Time-driven schedule
        periodic_retrain_days: int = 7,  # Weekly retraining
        
        # Regime-driven
        regime_stability_days: int = 3,  # New regime sustained for 3 days
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.scripts_dir = Path(scripts_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Thresholds
        self.min_winrate = min_winrate
        self.min_avg_r = min_avg_r
        self.min_calibration = min_calibration
        self.min_improvement_pct = min_improvement_pct
        self.canary_threshold_pct = canary_threshold_pct
        self.periodic_retrain_days = periodic_retrain_days
        self.regime_stability_days = regime_stability_days
        
        # State tracking
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.deployed_versions: Dict[str, ModelVersion] = {}
        
        # Load existing state
        self._load_state()
    
    # --------------------------------------------------------
    # TRIGGER EVALUATION
    # --------------------------------------------------------
    
    def evaluate_triggers(
        self,
        supervisor_output: Dict[str, Any],
        current_regime: Optional[str] = None,
    ) -> List[RetrainingTrigger]:
        """
        Evaluate all trigger conditions and return list of triggered retrains.
        
        Args:
            supervisor_output: Output from Model Supervisor
            current_regime: Current market regime (TRENDING/RANGING)
        
        Returns:
            List of RetrainingTrigger objects
        """
        triggers = []
        now = datetime.now(timezone.utc).isoformat()
        
        # 1. PERFORMANCE-DRIVEN TRIGGERS
        for model_id, metrics in supervisor_output.get("model_metrics", {}).items():
            # Check critical health
            if metrics.get("health_status") == "CRITICAL":
                triggers.append(RetrainingTrigger(
                    trigger_id=f"perf_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    trigger_type=TriggerType.PERFORMANCE_DRIVEN,
                    model_id=model_id,
                    timestamp=now,
                    reason=f"Model health CRITICAL: WR={metrics.get('winrate', 0):.1%}, R={metrics.get('avg_R', 0):.2f}",
                    priority="URGENT",
                    metadata={"metrics": metrics}
                ))
            
            # Check degraded performance
            elif metrics.get("health_status") == "DEGRADED":
                triggers.append(RetrainingTrigger(
                    trigger_id=f"perf_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    trigger_type=TriggerType.PERFORMANCE_DRIVEN,
                    model_id=model_id,
                    timestamp=now,
                    reason=f"Model health DEGRADED: {metrics.get('performance_trend', 'UNKNOWN')}",
                    priority="HIGH",
                    metadata={"metrics": metrics}
                ))
            
            # Check performance trend
            elif metrics.get("performance_trend") == "DEGRADING":
                triggers.append(RetrainingTrigger(
                    trigger_id=f"drift_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    trigger_type=TriggerType.DRIFT_DETECTED,
                    model_id=model_id,
                    timestamp=now,
                    reason=f"Performance degrading over time",
                    priority="MEDIUM",
                    metadata={"metrics": metrics}
                ))
        
        # 2. TIME-DRIVEN TRIGGERS
        for model_id in self.deployed_versions.keys():
            deployed_version = self.deployed_versions[model_id]
            deployment_date = datetime.fromisoformat(deployed_version.deployment_date)
            days_since_deploy = (datetime.now(timezone.utc) - deployment_date).days
            
            if days_since_deploy >= self.periodic_retrain_days:
                triggers.append(RetrainingTrigger(
                    trigger_id=f"time_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    trigger_type=TriggerType.TIME_DRIVEN,
                    model_id=model_id,
                    timestamp=now,
                    reason=f"Periodic retrain: {days_since_deploy} days since last deploy",
                    priority="LOW",
                    metadata={"days_since_deploy": days_since_deploy}
                ))
        
        # 3. REGIME-DRIVEN TRIGGERS
        if current_regime:
            # Check if regime is stable and different from training regime
            for model_id, version in self.deployed_versions.items():
                training_regime = version.hyperparameters.get("regime", None)
                if training_regime and training_regime != current_regime:
                    triggers.append(RetrainingTrigger(
                        trigger_id=f"regime_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                        trigger_type=TriggerType.REGIME_DRIVEN,
                        model_id=model_id,
                        timestamp=now,
                        reason=f"Regime changed: {training_regime} → {current_regime}",
                        priority="MEDIUM",
                        metadata={"old_regime": training_regime, "new_regime": current_regime}
                    ))
        
        logger.info(f"Evaluated triggers: {len(triggers)} triggers found")
        for trigger in triggers:
            logger.info(f"  - [{trigger.priority}] {trigger.model_id}: {trigger.reason}")
        
        return triggers
    
    # --------------------------------------------------------
    # TRAINING COORDINATION
    # --------------------------------------------------------
    
    def create_retraining_plan(
        self,
        triggers: List[RetrainingTrigger],
        batch_size: int = 3,  # Train 3 models in parallel
    ) -> RetrainingPlan:
        """
        Create a plan for retraining jobs based on triggers.
        
        Args:
            triggers: List of retraining triggers
            batch_size: Max models to train in parallel
        
        Returns:
            RetrainingPlan with scheduled jobs
        """
        now = datetime.now(timezone.utc).isoformat()
        plan_id = f"plan_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Group triggers by model_id (deduplicate)
        model_triggers: Dict[str, List[RetrainingTrigger]] = {}
        for trigger in triggers:
            if trigger.model_id not in model_triggers:
                model_triggers[trigger.model_id] = []
            model_triggers[trigger.model_id].append(trigger)
        
        # Prioritize: URGENT > HIGH > MEDIUM > LOW
        priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_models = sorted(
            model_triggers.items(),
            key=lambda x: min(priority_order.get(t.priority, 99) for t in x[1])
        )
        
        # Create jobs
        jobs = []
        for model_id, model_trig_list in sorted_models:
            # Use highest priority trigger
            main_trigger = sorted(
                model_trig_list,
                key=lambda t: priority_order.get(t.priority, 99)
            )[0]
            
            job = RetrainingJob(
                job_id=f"job_{model_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                trigger=main_trigger,
                status=RetrainingStatus.SCHEDULED,
                scheduled_at=now,
                started_at=None,
                completed_at=None,
                duration_seconds=None,
                data_window_days=self._get_data_window_days(main_trigger),
                features_config=self._get_features_config(model_id),
                hyperparameters=self._get_hyperparameters(model_id),
                new_version=None,
                error_message=None,
                logs=[]
            )
            jobs.append(job)
        
        # Estimate duration (15 min per model avg)
        estimated_minutes = len(jobs) * 15 / batch_size
        
        plan = RetrainingPlan(
            plan_id=plan_id,
            created_at=now,
            jobs=jobs,
            total_jobs=len(jobs),
            estimated_duration_minutes=estimated_minutes
        )
        
        logger.info(f"Created retraining plan: {len(jobs)} jobs, ~{estimated_minutes:.0f} minutes")
        return plan
    
    def execute_retraining_job(
        self,
        job: RetrainingJob,
        async_mode: bool = False,
    ) -> RetrainingJob:
        """
        Execute a single retraining job.
        
        Args:
            job: RetrainingJob to execute
            async_mode: If True, run in background
        
        Returns:
            Updated RetrainingJob with results
        """
        logger.info(f"[RETRAIN] Starting job {job.job_id} for {job.model_id}")
        
        job.status = RetrainingStatus.IN_PROGRESS
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.logs.append(f"Started at {job.started_at}")
        
        try:
            # Determine training script
            script_path = self._get_training_script(job.model_id)
            if not script_path.exists():
                raise FileNotFoundError(f"Training script not found: {script_path}")
            
            job.logs.append(f"Using script: {script_path}")
            
            # Prepare training command
            cmd = self._build_training_command(job, script_path)
            job.logs.append(f"Command: {' '.join(cmd)}")
            
            # Execute training
            result = subprocess.run(
                cmd,
                cwd=str(self.scripts_dir.parent),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode != 0:
                job.status = RetrainingStatus.FAILED
                job.error_message = f"Training failed: {result.stderr[-500:]}"
                job.logs.append(f"ERROR: {job.error_message}")
                logger.error(f"[RETRAIN] Job {job.job_id} failed: {job.error_message}")
            else:
                # Training succeeded - load new model version
                job.logs.append("Training succeeded")
                new_version = self._load_new_model_version(job)
                job.new_version = new_version
                job.status = RetrainingStatus.COMPLETED
                logger.info(f"[RETRAIN] Job {job.job_id} completed: {new_version.version_tag}")
                
                # Register version
                self._register_model_version(new_version)
        
        except subprocess.TimeoutExpired:
            job.status = RetrainingStatus.FAILED
            job.error_message = "Training timed out after 2 hours"
            job.logs.append(f"ERROR: {job.error_message}")
            logger.error(f"[RETRAIN] Job {job.job_id} timed out")
        
        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = f"Unexpected error: {str(e)}"
            job.logs.append(f"ERROR: {job.error_message}")
            logger.exception(f"[RETRAIN] Job {job.job_id} crashed")
        
        finally:
            job.completed_at = datetime.now(timezone.utc).isoformat()
            start_dt = datetime.fromisoformat(job.started_at)
            end_dt = datetime.fromisoformat(job.completed_at)
            job.duration_seconds = (end_dt - start_dt).total_seconds()
            job.logs.append(f"Completed in {job.duration_seconds:.0f}s")
        
        return job
    
    # --------------------------------------------------------
    # MODEL VERSIONING
    # --------------------------------------------------------
    
    def _register_model_version(self, version: ModelVersion):
        """Register a new model version."""
        if version.model_id not in self.model_versions:
            self.model_versions[version.model_id] = []
        
        self.model_versions[version.model_id].append(version)
        
        # Keep last 10 versions
        if len(self.model_versions[version.model_id]) > 10:
            # Remove oldest versions (but keep deployed ones)
            non_deployed = [v for v in self.model_versions[version.model_id] if not v.is_deployed]
            if len(non_deployed) > 5:
                # Delete old model files
                oldest = non_deployed[0]
                if Path(oldest.model_path).exists():
                    Path(oldest.model_path).unlink()
                if oldest.scaler_path and Path(oldest.scaler_path).exists():
                    Path(oldest.scaler_path).unlink()
                self.model_versions[version.model_id].remove(oldest)
        
        self._save_state()
        logger.info(f"Registered version: {version.model_id} {version.version_tag}")
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model, sorted newest first."""
        versions = self.model_versions.get(model_id, [])
        return sorted(versions, key=lambda v: v.training_date, reverse=True)
    
    def get_deployed_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get currently deployed version of a model."""
        return self.deployed_versions.get(model_id)
    
    def rollback_model(self, model_id: str, target_version_tag: str) -> bool:
        """
        Rollback model to a previous version.
        
        Args:
            model_id: Model identifier
            target_version_tag: Version tag to rollback to
        
        Returns:
            True if rollback succeeded
        """
        versions = self.get_model_versions(model_id)
        target = next((v for v in versions if v.version_tag == target_version_tag), None)
        
        if not target:
            logger.error(f"Rollback failed: version {target_version_tag} not found")
            return False
        
        # Deploy target version
        current = self.deployed_versions.get(model_id)
        if current:
            current.is_deployed = False
            current.deployment_date = None
        
        target.is_deployed = True
        target.deployment_date = datetime.now(timezone.utc).isoformat()
        self.deployed_versions[model_id] = target
        
        self._save_state()
        logger.info(f"Rolled back {model_id} to {target_version_tag}")
        return True
    
    # --------------------------------------------------------
    # DEPLOYMENT DECISIONS
    # --------------------------------------------------------
    
    def evaluate_deployment(
        self,
        new_version: ModelVersion,
    ) -> DeploymentRecommendation:
        """
        Evaluate whether to deploy a new model version.
        
        Args:
            new_version: Newly trained model version
        
        Returns:
            DeploymentRecommendation with decision and reasoning
        """
        model_id = new_version.model_id
        old_version = self.deployed_versions.get(model_id)
        
        reasons = []
        concerns = []
        
        # Extract key metrics
        new_wr = new_version.validation_metrics.get("winrate", 0.0)
        new_r = new_version.validation_metrics.get("avg_R", 0.0)
        new_cal = new_version.validation_metrics.get("calibration", 0.0)
        
        if old_version:
            old_wr = old_version.validation_metrics.get("winrate", 0.0)
            old_r = old_version.validation_metrics.get("avg_R", 0.0)
            old_cal = old_version.validation_metrics.get("calibration", 0.0)
            
            # Calculate improvement
            wr_improvement = (new_wr - old_wr) / old_wr if old_wr > 0 else 0.0
            r_improvement = (new_r - old_r) / abs(old_r) if old_r != 0 else 0.0
            cal_improvement = (new_cal - old_cal) / old_cal if old_cal > 0 else 0.0
            
            # Overall improvement (weighted average)
            overall_improvement = (
                wr_improvement * 0.35 +
                r_improvement * 0.35 +
                cal_improvement * 0.30
            )
        else:
            # No old version - first deployment
            overall_improvement = 1.0
            reasons.append("First version of this model")
        
        # Safety checks
        is_safe = True
        
        if new_wr < self.min_winrate:
            is_safe = False
            concerns.append(f"Winrate {new_wr:.1%} below minimum {self.min_winrate:.1%}")
        
        if new_r < self.min_avg_r:
            is_safe = False
            concerns.append(f"Avg R {new_r:.2f} below minimum {self.min_avg_r:.2f}")
        
        if new_cal < self.min_calibration:
            concerns.append(f"Calibration {new_cal:.1%} below ideal {self.min_calibration:.1%}")
        
        # Make decision
        if not is_safe:
            decision = DeploymentDecision.KEEP_OLD
            reasons.append("New version failed safety checks")
        elif overall_improvement >= self.min_improvement_pct:
            decision = DeploymentDecision.DEPLOY_IMMEDIATELY
            reasons.append(f"Performance improved by {overall_improvement:.1%}")
        elif overall_improvement >= self.canary_threshold_pct:
            decision = DeploymentDecision.RUN_CANARY
            reasons.append(f"Moderate improvement ({overall_improvement:.1%}) - run canary test")
        elif overall_improvement < 0:
            decision = DeploymentDecision.KEEP_OLD
            reasons.append(f"Performance regressed by {abs(overall_improvement):.1%}")
        else:
            decision = DeploymentDecision.REQUIRES_REVIEW
            reasons.append(f"Minimal improvement ({overall_improvement:.1%}) - manual review needed")
        
        recommendation = DeploymentRecommendation(
            model_id=model_id,
            old_version=old_version,
            new_version=new_version,
            decision=decision,
            performance_improvement=overall_improvement,
            is_safe_to_deploy=is_safe,
            requires_canary=(decision == DeploymentDecision.RUN_CANARY),
            reasons=reasons,
            concerns=concerns,
            min_improvement_threshold=self.min_improvement_pct
        )
        
        logger.info(f"Deployment evaluation for {model_id}: {decision.value}")
        logger.info(f"  Improvement: {overall_improvement:.1%}")
        logger.info(f"  Reasons: {', '.join(reasons)}")
        if concerns:
            logger.warning(f"  Concerns: {', '.join(concerns)}")
        
        return recommendation
    
    def deploy_model(self, version: ModelVersion) -> bool:
        """
        Deploy a model version to production.
        
        Args:
            version: ModelVersion to deploy
        
        Returns:
            True if deployment succeeded
        """
        model_id = version.model_id
        
        # Undeploy old version
        old_version = self.deployed_versions.get(model_id)
        if old_version:
            old_version.is_deployed = False
            old_version.deployment_date = None
        
        # Deploy new version
        version.is_deployed = True
        version.deployment_date = datetime.now(timezone.utc).isoformat()
        self.deployed_versions[model_id] = version
        
        self._save_state()
        logger.info(f"Deployed {model_id} version {version.version_tag}")
        return True
    
    # --------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------
    
    def _get_data_window_days(self, trigger: RetrainingTrigger) -> int:
        """Determine data window based on trigger type."""
        if trigger.trigger_type == TriggerType.PERFORMANCE_DRIVEN:
            return 60  # Use last 2 months for performance-driven
        elif trigger.trigger_type == TriggerType.REGIME_DRIVEN:
            return 30  # Use last month for regime change
        else:
            return 90  # Default 3 months
    
    def _get_features_config(self, model_id: str) -> Dict[str, Any]:
        """Get feature configuration for model."""
        # Use same features as current version
        current = self.deployed_versions.get(model_id)
        if current:
            return {
                "features": current.features_used,
                "feature_count": current.feature_count
            }
        
        # Default config
        return {
            "features": ["ohlcv", "technical", "sentiment", "regime"],
            "feature_count": 50
        }
    
    def _get_hyperparameters(self, model_id: str) -> Dict[str, Any]:
        """Get hyperparameters for model."""
        # Use same hyperparameters as current version
        current = self.deployed_versions.get(model_id)
        if current:
            return current.hyperparameters
        
        # Default hyperparameters by model type
        if "xgboost" in model_id.lower():
            return {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        elif "lgbm" in model_id.lower() or "lightgbm" in model_id.lower():
            return {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        elif "nhits" in model_id.lower():
            return {"hidden_size": 128, "num_blocks": 3, "epochs": 50}
        elif "patchtst" in model_id.lower():
            return {"d_model": 128, "nhead": 8, "num_layers": 3, "epochs": 50}
        else:
            return {}
    
    def _get_training_script(self, model_id: str) -> Path:
        """Get training script path for model."""
        if "xgboost" in model_id.lower() or "xgb" in model_id.lower():
            return self.scripts_dir / "train_futures_xgboost.py"
        elif "lgbm" in model_id.lower() or "lightgbm" in model_id.lower():
            return self.scripts_dir / "train_lightgbm.py"
        elif "nhits" in model_id.lower():
            return self.scripts_dir / "train_futures_nhits.py"
        elif "patchtst" in model_id.lower():
            return self.scripts_dir / "train_futures_patchtst.py"
        else:
            return self.scripts_dir / "train_all_models_futures.py"
    
    def _build_training_command(self, job: RetrainingJob, script_path: Path) -> List[str]:
        """Build command to execute training script."""
        cmd = ["python", str(script_path)]
        
        # Add data window
        cmd.extend(["--days", str(job.data_window_days)])
        
        # Add hyperparameters
        for key, value in job.hyperparameters.items():
            cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def _load_new_model_version(self, job: RetrainingJob) -> ModelVersion:
        """
        Load newly trained model version from disk.
        
        Args:
            job: Completed retraining job
        
        Returns:
            ModelVersion object
        """
        model_id = job.model_id
        now = datetime.now(timezone.utc)
        version_tag = f"v{now.strftime('%Y%m%d_%H%M%S')}"
        
        # Find model files
        if "xgboost" in model_id.lower():
            model_filename = "xgb_model.pkl"
            scaler_filename = "scaler.pkl"
            metadata_filename = "metadata.json"
        elif "lgbm" in model_id.lower():
            model_filename = "lgbm_model.pkl"
            scaler_filename = "lgbm_scaler.pkl"
            metadata_filename = "lgbm_metadata.json"
        elif "nhits" in model_id.lower():
            model_filename = "nhits_model.pth"
            scaler_filename = None
            metadata_filename = "nhits_metadata.json"
        elif "patchtst" in model_id.lower():
            model_filename = "patchtst_model.pth"
            scaler_filename = None
            metadata_filename = "patchtst_metadata.json"
        else:
            model_filename = f"{model_id}_model.pkl"
            scaler_filename = f"{model_id}_scaler.pkl"
            metadata_filename = f"{model_id}_metadata.json"
        
        model_path = self.models_dir / model_filename
        scaler_path = self.models_dir / scaler_filename if scaler_filename else None
        metadata_path = self.models_dir / metadata_filename
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Create version
        version = ModelVersion(
            model_id=model_id,
            version_tag=version_tag,
            training_date=now.isoformat(),
            data_range_start=(now - timedelta(days=job.data_window_days)).isoformat(),
            data_range_end=now.isoformat(),
            sample_count=metadata.get("sample_count", 0),
            feature_count=metadata.get("feature_count", 0),
            hyperparameters=job.hyperparameters,
            features_used=job.features_config.get("features", []),
            train_metrics=metadata.get("train_metrics", {}),
            validation_metrics=metadata.get("validation_metrics", {}),
            is_deployed=False,
            deployment_date=None,
            model_path=str(model_path),
            scaler_path=str(scaler_path) if scaler_path else None,
            metadata_path=str(metadata_path)
        )
        
        return version
    
    def _load_state(self):
        """Load orchestrator state from disk."""
        state_path = self.data_dir / "retraining_orchestrator_state.json"
        
        if not state_path.exists():
            logger.info("No saved state found, starting fresh")
            return
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Load model versions
            for model_id, versions_data in state.get("model_versions", {}).items():
                self.model_versions[model_id] = [
                    ModelVersion(**v) for v in versions_data
                ]
            
            # Load deployed versions
            for model_id, version_data in state.get("deployed_versions", {}).items():
                self.deployed_versions[model_id] = ModelVersion(**version_data)
            
            logger.info(f"Loaded state: {len(self.deployed_versions)} deployed models")
        
        except Exception as e:
            logger.exception("Failed to load state, starting fresh")
    
    def _save_state(self):
        """Save orchestrator state to disk."""
        state_path = self.data_dir / "retraining_orchestrator_state.json"
        
        state = {
            "model_versions": {
                model_id: [asdict(v) for v in versions]
                for model_id, versions in self.model_versions.items()
            },
            "deployed_versions": {
                model_id: asdict(version)
                for model_id, version in self.deployed_versions.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.debug("Saved orchestrator state")
    
    async def run(self):
        """Continuous monitoring loop for retraining triggers."""
        import asyncio
        
        logger.info("\n" + "=" * 80)
        logger.info("🔄 RETRAINING ORCHESTRATOR - STARTING CONTINUOUS MONITORING")
        logger.info("=" * 80)
        logger.info(f"Mode: ADVISORY (evaluate triggers every hour)")
        logger.info(f"Check interval: Every 3600 seconds")
        logger.info("=" * 80 + "\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                logger.info(
                    f"🔄 [RETRAINING] Evaluation cycle #{iteration} - "
                    f"Checking for retraining triggers"
                )
                
                # In a real implementation, would fetch supervisor output and evaluate
                # For now, just log that we're checking
                logger.info("🔄 [RETRAINING] No immediate retraining needs detected")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                logger.info("🔄 [RETRAINING] Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"[RETRAINING] Monitor loop error: {e}", exc_info=True)
                await asyncio.sleep(3600)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("RETRAINING ORCHESTRATOR - Standalone Test")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = RetrainingOrchestrator(
        data_dir="./data",
        models_dir="./ai_engine/models",
        scripts_dir="./scripts",
        min_winrate=0.50,
        min_improvement_pct=0.05,
        periodic_retrain_days=7
    )
    
    print(f"\n[OK] Orchestrator initialized")
    print(f"  Models dir: {orchestrator.models_dir}")
    print(f"  Min winrate: {orchestrator.min_winrate:.1%}")
    print(f"  Min improvement: {orchestrator.min_improvement_pct:.1%}")
    
    # Simulate Model Supervisor output
    print("\n" + "=" * 60)
    print("TEST 1: Evaluate Triggers")
    print("=" * 60)
    
    supervisor_output = {
        "model_metrics": {
            "xgboost_v1": {
                "winrate": 0.45,
                "avg_R": -0.05,
                "calibration_quality": 0.60,
                "health_status": "CRITICAL",
                "performance_trend": "DEGRADING"
            },
            "ensemble_v2": {
                "winrate": 0.58,
                "avg_R": 0.15,
                "calibration_quality": 0.75,
                "health_status": "DEGRADED",
                "performance_trend": "STABLE"
            },
            "lstm_v1": {
                "winrate": 0.65,
                "avg_R": 0.25,
                "calibration_quality": 0.85,
                "health_status": "HEALTHY",
                "performance_trend": "IMPROVING"
            }
        }
    }
    
    triggers = orchestrator.evaluate_triggers(
        supervisor_output=supervisor_output,
        current_regime="TRENDING"
    )
    
    print(f"\n[OK] Found {len(triggers)} triggers:")
    for trigger in triggers:
        print(f"  [{trigger.priority}] {trigger.model_id}: {trigger.reason}")
    
    # Create retraining plan
    print("\n" + "=" * 60)
    print("TEST 2: Create Retraining Plan")
    print("=" * 60)
    
    plan = orchestrator.create_retraining_plan(triggers, batch_size=2)
    
    print(f"\n[OK] Created plan: {plan.plan_id}")
    print(f"  Total jobs: {plan.total_jobs}")
    print(f"  Estimated duration: {plan.estimated_duration_minutes:.0f} minutes")
    print(f"\n  Jobs:")
    for job in plan.jobs:
        print(f"    - {job.model_id}: {job.trigger.reason}")
    
    # Test deployment evaluation
    print("\n" + "=" * 60)
    print("TEST 3: Evaluate Deployment")
    print("=" * 60)
    
    # Create mock old version
    old_version = ModelVersion(
        model_id="xgboost_v1",
        version_tag="v20251120_120000",
        training_date="2025-11-20T12:00:00Z",
        data_range_start="2025-10-01T00:00:00Z",
        data_range_end="2025-11-20T00:00:00Z",
        sample_count=5000,
        feature_count=45,
        hyperparameters={"n_estimators": 100, "max_depth": 6},
        features_used=["ohlcv", "technical"],
        train_metrics={"winrate": 0.55, "avg_R": 0.12},
        validation_metrics={"winrate": 0.52, "avg_R": 0.10, "calibration": 0.70},
        is_deployed=True,
        deployment_date="2025-11-20T12:00:00Z",
        model_path="./ai_engine/models/xgb_model.pkl",
        scaler_path="./ai_engine/models/scaler.pkl",
        metadata_path="./ai_engine/models/metadata.json"
    )
    orchestrator.deployed_versions["xgboost_v1"] = old_version
    
    # Create mock new version (improved)
    new_version = ModelVersion(
        model_id="xgboost_v1",
        version_tag="v20251123_120000",
        training_date="2025-11-23T12:00:00Z",
        data_range_start="2025-10-23T00:00:00Z",
        data_range_end="2025-11-23T00:00:00Z",
        sample_count=6000,
        feature_count=50,
        hyperparameters={"n_estimators": 100, "max_depth": 6},
        features_used=["ohlcv", "technical", "sentiment"],
        train_metrics={"winrate": 0.60, "avg_R": 0.18},
        validation_metrics={"winrate": 0.58, "avg_R": 0.15, "calibration": 0.78},
        is_deployed=False,
        deployment_date=None,
        model_path="./ai_engine/models/xgb_model_new.pkl",
        scaler_path="./ai_engine/models/scaler_new.pkl",
        metadata_path="./ai_engine/models/metadata_new.json"
    )
    
    recommendation = orchestrator.evaluate_deployment(new_version)
    
    print(f"\n[OK] Deployment recommendation for {new_version.model_id}:")
    print(f"  Decision: {recommendation.decision.value}")
    print(f"  Performance improvement: {recommendation.performance_improvement:.1%}")
    print(f"  Safe to deploy: {recommendation.is_safe_to_deploy}")
    print(f"  Requires canary: {recommendation.requires_canary}")
    print(f"\n  Reasons:")
    for reason in recommendation.reasons:
        print(f"    - {reason}")
    if recommendation.concerns:
        print(f"\n  Concerns:")
        for concern in recommendation.concerns:
            print(f"    - {concern}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)
    
    # Save output
    output = {
        "test_timestamp": datetime.now(timezone.utc).isoformat(),
        "triggers_found": len(triggers),
        "plan_created": {
            "plan_id": plan.plan_id,
            "total_jobs": plan.total_jobs,
            "estimated_minutes": plan.estimated_duration_minutes
        },
        "deployment_recommendation": {
            "model_id": recommendation.model_id,
            "decision": recommendation.decision.value,
            "improvement": recommendation.performance_improvement,
            "reasons": recommendation.reasons
        }
    }
    
    output_path = Path("./data/retraining_orchestrator_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[OK] Test output saved to: {output_path}")
