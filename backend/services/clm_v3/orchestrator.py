"""
CLM v3 Orchestrator - Main training workflow coordinator.

Orchestrates the complete training pipeline:
1. Fetch training data
2. Train model (via adapters)
3. Evaluate model (backtest)
4. Make promotion decision
5. Publish events
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

from backend.services.clm_v3.adapters import ModelTrainingAdapter, BacktestAdapter
from backend.services.clm_v3.models import (
    EvaluationResult,
    ModelStatus,
    ModelTrainedEvent,
    ModelEvaluatedEvent,
    ModelPromotedEvent,
    ModelVersion,
    TrainingJob,
    TriggerReason,
)
from backend.services.clm_v3.storage import ModelRegistryV3

logger = logging.getLogger(__name__)


class ClmOrchestrator:
    """
    CLM v3 Orchestrator - Main training workflow coordinator.
    
    Pipeline:
    1. handle_training_job(job) → fetch data
    2. _train_model(job, data) → call training adapter
    3. _evaluate_model(model_version) → backtest
    4. _decide_promotion(evaluation) → check criteria
    5. _promote_if_approved(model_version) → promote to CANDIDATE/PRODUCTION
    """
    
    def __init__(
        self,
        registry: ModelRegistryV3,
        training_adapter: ModelTrainingAdapter,
        backtest_adapter: BacktestAdapter,
        event_bus: Optional[any] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize CLM Orchestrator.
        
        Args:
            registry: ModelRegistryV3 instance
            training_adapter: Adapter for model training
            backtest_adapter: Adapter for backtesting
            event_bus: EventBus for publishing events (optional)
            config: Orchestrator configuration
        """
        self.registry = registry
        self.training_adapter = training_adapter
        self.backtest_adapter = backtest_adapter
        self.event_bus = event_bus
        
        # Merge provided config with defaults (similar to scheduler)
        default_config = self._default_config()
        if config:
            for key, value in config.items():
                if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        self.config = default_config
        
        logger.info("[CLM v3 Orchestrator] Initialized")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default orchestrator configuration."""
        return {
            # Promotion criteria
            "promotion_criteria": {
                "min_sharpe_ratio": 1.0,
                "min_win_rate": 0.52,
                "min_profit_factor": 1.3,
                "max_drawdown": 0.15,
                "min_trades": 50,
            },
            
            # Auto-promotion settings
            "auto_promote_to_candidate": True,  # Auto-promote to CANDIDATE if passed
            "auto_promote_to_production": False,  # Require manual approval for PRODUCTION
            
            # Safety settings
            "require_shadow_testing": True,  # Require shadow testing before production
            "shadow_test_min_trades": 100,
        }
    
    # ========================================================================
    # Main Pipeline
    # ========================================================================
    
    async def handle_training_job(self, job: TrainingJob) -> Optional[ModelVersion]:
        """
        Handle complete training job pipeline.
        
        Steps:
        1. Update job status to in_progress
        2. Fetch training data
        3. Train model
        4. Save model version to registry
        5. Evaluate model (backtest)
        6. Make promotion decision
        7. Publish events
        8. Update job status to completed
        
        Args:
            job: TrainingJob to execute
        
        Returns:
            ModelVersion if successful, None if failed
        """
        logger.info(
            f"[CLM v3 Orchestrator] Starting training job {job.id} "
            f"(model={job.model_type.value}, trigger={job.trigger_reason.value})"
        )
        
        try:
            # Update job status
            job.status = "in_progress"
            job.started_at = datetime.utcnow()
            self.registry.update_training_job(job.id, {
                "status": "in_progress",
                "started_at": job.started_at,
            })
            
            # Step 0: Enrich RL training with TP feedback (before fetching data)
            await self._enrich_rl_training_with_tp_feedback(job)
            
            # Step 1: Fetch training data
            logger.info(f"[CLM v3 Orchestrator] [{job.id}] Fetching training data...")
            training_data = await self._fetch_training_data(job)
            
            # Step 2: Train model
            logger.info(f"[CLM v3 Orchestrator] [{job.id}] Training model...")
            model_version = await self._train_model(job, training_data)
            
            # Step 3: Register model
            logger.info(f"[CLM v3 Orchestrator] [{job.id}] Registering model version...")
            self.registry.register_model_version(model_version)
            
            # Publish training event
            await self._publish_model_trained_event(job, model_version, success=True)
            
            # Step 4: Evaluate model
            logger.info(f"[CLM v3 Orchestrator] [{job.id}] Evaluating model...")
            evaluation = await self._evaluate_model(model_version, job)
            
            # Step 5: Save evaluation
            self.registry.save_evaluation_result(evaluation)
            
            # Publish evaluation event
            await self._publish_model_evaluated_event(model_version, evaluation)
            
            # Step 6: Promotion decision
            if evaluation.passed:
                logger.info(
                    f"[CLM v3 Orchestrator] [{job.id}] Model passed evaluation "
                    f"(score={evaluation.promotion_score:.2f})"
                )
                await self._handle_promotion(model_version, evaluation)
            else:
                logger.warning(
                    f"[CLM v3 Orchestrator] [{job.id}] Model FAILED evaluation "
                    f"(reason: {evaluation.failure_reason})"
                )
                model_version.status = ModelStatus.FAILED
                self.registry.register_model_version(model_version)  # Update status
            
            # Step 7: Update job status
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            self.registry.update_training_job(job.id, {
                "status": "completed",
                "completed_at": job.completed_at,
            })
            
            logger.info(
                f"[CLM v3 Orchestrator] ✅ Training job {job.id} completed successfully "
                f"(model={model_version.model_id} v{model_version.version}, status={model_version.status.value})"
            )
            
            return model_version
        
        except Exception as e:
            logger.error(
                f"[CLM v3 Orchestrator] ❌ Training job {job.id} FAILED: {e}",
                exc_info=True
            )
            
            # Update job status
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            self.registry.update_training_job(job.id, {
                "status": "failed",
                "error_message": str(e),
                "completed_at": job.completed_at,
            })
            
            # Publish failure event
            await self._publish_model_trained_event(job, None, success=False)
            
            return None
    
    # ========================================================================
    # Pipeline Steps
    # ========================================================================
    
    async def _enrich_rl_training_with_tp_feedback(self, job: TrainingJob) -> None:
        """
        Enrich RL training job with TP performance feedback.
        
        Queries TPPerformanceTracker for recent TP metrics and computes
        tp_reward_weight to modulate RL reward function.
        
        Logic:
        - Low hit rate (< 0.45) + good R → increase weight (2.0) to encourage better TP prediction
        - High hit rate (> 0.70) + low R → decrease weight (0.5) to deprioritize TP accuracy
        - Optimal → default weight (1.0)
        
        Args:
            job: TrainingJob (modified in-place)
        """
        from backend.services.monitoring.tp_performance_tracker import get_tp_tracker
        
        # Only apply to RL models
        if job.model_type not in ["rl_v2", "rl_v3"]:
            return
        
        logger.info(f"[CLM v3 Orchestrator] [{job.id}] Fetching TP performance feedback...")
        
        try:
            tp_tracker = get_tp_tracker()
            
            # Get TP feedback for this strategy (aggregate across symbols if no specific symbol)
            # Strategy ID extracted from model_type or config
            strategy_id = job.training_params.get('strategy_id', 'RL_V3')
            
            tp_feedback = tp_tracker.get_strategy_tp_feedback(
                strategy_id=strategy_id,
                symbol=job.symbol,
                min_attempts=10
            )
            
            if not tp_feedback:
                logger.info(
                    f"[CLM v3 Orchestrator] [{job.id}] Insufficient TP data for {strategy_id}, "
                    f"using default tp_reward_weight=1.0"
                )
                job.training_params['tp_reward_weight'] = 1.0
                return
            
            # Extract metrics
            hit_rate = tp_feedback['tp_hit_rate']
            avg_r = tp_feedback['avg_r_multiple']
            
            # Compute tp_reward_weight based on performance
            if hit_rate < 0.45 and avg_r >= 1.2:
                # Low hit rate but good R → TPs too far, increase weight to improve prediction
                tp_reward_weight = 2.0
                reason = f"Low hit rate ({hit_rate:.1%}) but good R ({avg_r:.2f}) - increasing TP weight"
            elif hit_rate > 0.70 and avg_r < 1.2:
                # High hit rate but low R → TPs too close, decrease weight
                tp_reward_weight = 0.5
                reason = f"High hit rate ({hit_rate:.1%}) but low R ({avg_r:.2f}) - decreasing TP weight"
            elif hit_rate >= 0.45 and hit_rate <= 0.70 and avg_r < 1.2:
                # Hit rate ok but R low → moderate increase
                tp_reward_weight = 1.2
                reason = f"Hit rate ok ({hit_rate:.1%}) but low R ({avg_r:.2f}) - slight TP weight increase"
            else:
                # Optimal performance
                tp_reward_weight = 1.0
                reason = f"TP performance optimal (hit_rate={hit_rate:.1%}, R={avg_r:.2f})"
            
            # Store in training params
            job.training_params['tp_hit_rate'] = hit_rate
            job.training_params['avg_tp_r_multiple'] = avg_r
            job.training_params['tp_reward_weight'] = tp_reward_weight
            job.training_params['tp_feedback_reason'] = reason
            
            logger.info(
                f"[CLM v3 Orchestrator] [{job.id}] TP Feedback: {reason} "
                f"(weight={tp_reward_weight:.2f})"
            )
            
            # Optional: Get TPOptimizer recommendations for observability
            if self.config.get('enable_tp_optimizer_logging', False):
                await self._log_tp_optimizer_recommendations(job, strategy_id)
        
        except Exception as e:
            logger.error(
                f"[CLM v3 Orchestrator] [{job.id}] Failed to fetch TP feedback: {e}",
                exc_info=True
            )
            # Fallback to default
            job.training_params['tp_reward_weight'] = 1.0
    
    async def _log_tp_optimizer_recommendations(self, job: TrainingJob, strategy_id: str) -> None:
        """
        Log TPOptimizerV3 recommendations for observability.
        
        Does not auto-apply adjustments - just logs for manual review.
        
        Args:
            job: TrainingJob
            strategy_id: Strategy identifier
        """
        try:
            from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer
            
            optimizer = get_tp_optimizer()
            
            if job.symbol:
                # Single symbol
                rec = optimizer.evaluate_profile(strategy_id, job.symbol)
                if rec:
                    logger.info(
                        f"[CLM v3 Orchestrator] [{job.id}] TPOptimizer recommendation for {job.symbol}: "
                        f"{rec.direction.value} ({rec.suggested_scale_factor:.3f}x) - {rec.reason}"
                    )
                    # Store in job metadata
                    if 'tp_optimizer_recommendations' not in job.training_params:
                        job.training_params['tp_optimizer_recommendations'] = []
                    job.training_params['tp_optimizer_recommendations'].append({
                        'symbol': rec.symbol,
                        'direction': rec.direction.value,
                        'scale_factor': rec.suggested_scale_factor,
                        'confidence': rec.confidence,
                        'reason': rec.reason
                    })
            else:
                # Multi-symbol: get batch recommendations
                recommendations = optimizer.optimize_all_profiles_once()
                if recommendations:
                    logger.info(
                        f"[CLM v3 Orchestrator] [{job.id}] TPOptimizer generated "
                        f"{len(recommendations)} recommendations"
                    )
                    job.training_params['tp_optimizer_recommendations'] = [
                        {
                            'symbol': rec.symbol,
                            'direction': rec.direction.value,
                            'scale_factor': rec.suggested_scale_factor,
                            'confidence': rec.confidence,
                            'reason': rec.reason
                        }
                        for rec in recommendations
                    ]
        except Exception as e:
            logger.warning(
                f"[CLM v3 Orchestrator] [{job.id}] Failed to get TPOptimizer recommendations: {e}"
            )
    
    async def _fetch_training_data(self, job: TrainingJob) -> Dict:
        """
        Fetch training data for the job.
        
        Args:
            job: TrainingJob
        
        Returns:
            Dict with training data (features, labels, etc.)
        """
        # TODO: Implement data fetching
        # For now, return placeholder
        logger.warning("[CLM v3 Orchestrator] _fetch_training_data not implemented - using placeholder")
        return {
            "symbol": job.symbol or "MULTI",
            "timeframe": job.timeframe,
            "dataset_span_days": job.dataset_span_days,
            "features": [],  # Placeholder
            "labels": [],    # Placeholder
            "dates": [],     # Placeholder
        }
    
    async def _train_model(self, job: TrainingJob, training_data: Dict) -> ModelVersion:
        """
        Train model using training adapter.
        
        Args:
            job: TrainingJob
            training_data: Training data dict
        
        Returns:
            ModelVersion with trained model
        """
        # Call training adapter
        model_version = await self.training_adapter.train_model(job, training_data)
        
        # Set training job reference
        model_version.training_job_id = job.id
        
        return model_version
    
    async def _evaluate_model(
        self,
        model_version: ModelVersion,
        job: TrainingJob,
    ) -> EvaluationResult:
        """
        Evaluate model via backtest.
        
        Args:
            model_version: ModelVersion to evaluate
            job: Original TrainingJob
        
        Returns:
            EvaluationResult
        """
        # Run backtest via adapter
        evaluation = await self.backtest_adapter.evaluate_model(
            model_version,
            evaluation_period_days=job.dataset_span_days,
        )
        
        # Apply promotion criteria
        evaluation = self._apply_promotion_criteria(evaluation)
        
        return evaluation
    
    def _apply_promotion_criteria(self, evaluation: EvaluationResult) -> EvaluationResult:
        """
        Apply promotion criteria to evaluation result.
        
        Updates:
        - evaluation.passed (bool)
        - evaluation.promotion_score (0-100)
        - evaluation.failure_reason (if failed)
        """
        criteria = self.config["promotion_criteria"]
        
        failures = []
        
        # Check Sharpe ratio
        if evaluation.sharpe_ratio < criteria["min_sharpe_ratio"]:
            failures.append(
                f"Sharpe {evaluation.sharpe_ratio:.3f} < {criteria['min_sharpe_ratio']}"
            )
        
        # Check win rate
        if evaluation.win_rate < criteria["min_win_rate"]:
            failures.append(
                f"WinRate {evaluation.win_rate:.3f} < {criteria['min_win_rate']}"
            )
        
        # Check profit factor
        if evaluation.profit_factor < criteria["min_profit_factor"]:
            failures.append(
                f"ProfitFactor {evaluation.profit_factor:.3f} < {criteria['min_profit_factor']}"
            )
        
        # Check max drawdown
        if evaluation.max_drawdown > criteria["max_drawdown"]:
            failures.append(
                f"MaxDD {evaluation.max_drawdown:.3f} > {criteria['max_drawdown']}"
            )
        
        # Check min trades
        if evaluation.total_trades < criteria["min_trades"]:
            failures.append(
                f"Trades {evaluation.total_trades} < {criteria['min_trades']}"
            )
        
        # Calculate promotion score (0-100)
        # Higher is better
        score = 0.0
        
        if not failures:
            # Passed all criteria - calculate score
            sharpe_score = min(evaluation.sharpe_ratio / 2.0 * 100, 100)  # Sharpe 2.0 = 100 pts
            wr_score = (evaluation.win_rate - 0.5) / 0.5 * 100  # 50% = 0, 100% = 100
            pf_score = min((evaluation.profit_factor - 1.0) / 2.0 * 100, 100)  # PF 3.0 = 100
            
            score = (sharpe_score * 0.5 + wr_score * 0.3 + pf_score * 0.2)
            evaluation.passed = True
        else:
            evaluation.passed = False
            evaluation.failure_reason = "; ".join(failures)
        
        evaluation.promotion_score = max(0, min(100, score))
        
        return evaluation
    
    async def _handle_promotion(
        self,
        model_version: ModelVersion,
        evaluation: EvaluationResult,
    ):
        """Handle promotion logic based on config."""
        if self.config["auto_promote_to_candidate"]:
            # Auto-promote to CANDIDATE
            model_version.status = ModelStatus.CANDIDATE
            self.registry.register_model_version(model_version)  # Update status
            
            logger.info(
                f"[CLM v3 Orchestrator] Auto-promoted {model_version.model_id} "
                f"v{model_version.version} to CANDIDATE"
            )
            
            # Check if should auto-promote to PRODUCTION
            if self.config["auto_promote_to_production"]:
                if self.config["require_shadow_testing"]:
                    # Mark as SHADOW first
                    model_version.status = ModelStatus.SHADOW
                    self.registry.register_model_version(model_version)
                    logger.info(
                        f"[CLM v3 Orchestrator] Model requires shadow testing before production"
                    )
                else:
                    # Direct to production (dangerous!)
                    await self._promote_to_production(model_version)
        else:
            # Keep as SHADOW for manual review
            model_version.status = ModelStatus.SHADOW
            self.registry.register_model_version(model_version)
            logger.info(
                f"[CLM v3 Orchestrator] Model marked as SHADOW - awaiting manual promotion"
            )
    
    async def _promote_to_production(self, model_version: ModelVersion):
        """Promote model to production."""
        success = self.registry.promote_model(
            model_version.model_id,
            model_version.version,
            promoted_by="clm_v3_orchestrator",
        )
        
        if success:
            await self._publish_model_promoted_event(model_version)
    
    # ========================================================================
    # Events
    # ========================================================================
    
    async def _publish_model_trained_event(
        self,
        job: TrainingJob,
        model_version: Optional[ModelVersion],
        success: bool,
    ):
        """Publish model_trained event."""
        if not self.event_bus:
            return
        
        event = ModelTrainedEvent(
            job_id=job.id,
            model_id=model_version.model_id if model_version else f"{job.model_type.value}_unknown",
            version=model_version.version if model_version else "unknown",
            model_type=job.model_type,
            success=success,
            train_metrics=model_version.train_metrics if model_version else {},
        )
        
        # TODO: Publish to EventBus
        logger.info(f"[CLM v3 Orchestrator] Event: model_trained (success={success})")
    
    async def _publish_model_evaluated_event(
        self,
        model_version: ModelVersion,
        evaluation: EvaluationResult,
    ):
        """Publish model_evaluated event."""
        if not self.event_bus:
            return
        
        event = ModelEvaluatedEvent(
            model_id=model_version.model_id,
            version=model_version.version,
            passed=evaluation.passed,
            promotion_score=evaluation.promotion_score,
            metrics={
                "sharpe_ratio": evaluation.sharpe_ratio,
                "win_rate": evaluation.win_rate,
                "profit_factor": evaluation.profit_factor,
                "max_drawdown": evaluation.max_drawdown,
            },
        )
        
        logger.info(
            f"[CLM v3 Orchestrator] Event: model_evaluated "
            f"(passed={evaluation.passed}, score={evaluation.promotion_score:.2f})"
        )
    
    async def _publish_model_promoted_event(self, model_version: ModelVersion):
        """Publish model_promoted event."""
        if not self.event_bus:
            return
        
        event = ModelPromotedEvent(
            model_id=model_version.model_id,
            version=model_version.version,
            previous_version=model_version.parent_version,
            promoted_by="clm_v3_orchestrator",
        )
        
        logger.info(f"[CLM v3 Orchestrator] Event: model_promoted")
