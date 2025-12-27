"""
Shadow Evaluator - evaluates new model versions against live models.
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from .models import ModelVersion, ShadowEvaluationResult

logger = logging.getLogger(__name__)


class ShadowEvaluator:
    """
    Evaluates shadow models against live models.
    
    Collects predictions from both and compares performance.
    """
    
    def __init__(self):
        self._shadow_predictions: Dict[str, list] = {}
        self._live_predictions: Dict[str, list] = {}
        self._ground_truth: Dict[str, list] = {}
    
    def record_shadow_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[float] = None,
    ):
        """Record a prediction from shadow model."""
        if model_name not in self._shadow_predictions:
            self._shadow_predictions[model_name] = []
            self._ground_truth[model_name] = []
        
        self._shadow_predictions[model_name].append(prediction)
        if actual is not None:
            self._ground_truth[model_name].append(actual)
    
    def record_live_prediction(
        self,
        model_name: str,
        prediction: float,
    ):
        """Record a prediction from live model."""
        if model_name not in self._live_predictions:
            self._live_predictions[model_name] = []
        
        self._live_predictions[model_name].append(prediction)
    
    def calculate_accuracy(
        self,
        predictions: list[float],
        actuals: list[float],
        threshold: float = 0.5,
    ) -> float:
        """Calculate accuracy for binary classification."""
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0
        
        correct = sum(
            1 for pred, actual in zip(predictions, actuals)
            if (pred >= threshold) == (actual >= threshold)
        )
        
        return correct / len(predictions)
    
    def calculate_f1_score(
        self,
        predictions: list[float],
        actuals: list[float],
        threshold: float = 0.5,
    ) -> float:
        """Calculate F1 score."""
        if len(predictions) == 0 or len(actuals) == 0:
            return 0.0
        
        # Binary classification
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        actual_binary = [1 if a >= threshold else 0 for a in actuals]
        
        tp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(pred_binary, actual_binary) if p == 0 and a == 1)
        
        if tp == 0:
            return 0.0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    async def evaluate(
        self,
        shadow_model: ModelVersion,
        live_model: ModelVersion,
        evaluation_hours: float = 24.0,
        min_samples: int = 100,
    ) -> ShadowEvaluationResult:
        """
        Evaluate shadow model against live model.
        
        Returns recommendation on whether to promote.
        """
        model_name = shadow_model.model_name
        
        # Get predictions
        shadow_preds = self._shadow_predictions.get(model_name, [])
        live_preds = self._live_predictions.get(model_name, [])
        actuals = self._ground_truth.get(model_name, [])
        
        # Check if we have enough samples
        if len(shadow_preds) < min_samples or len(actuals) < min_samples:
            logger.warning(
                f"Insufficient samples for {model_name}: "
                f"{len(shadow_preds)} shadow, {len(actuals)} ground truth"
            )
            return ShadowEvaluationResult(
                model_name=model_name,
                shadow_version=shadow_model.version,
                live_version=live_model.version,
                shadow_accuracy=0.0,
                live_accuracy=0.0,
                accuracy_improvement=0.0,
                shadow_f1=0.0,
                live_f1=0.0,
                f1_improvement=0.0,
                samples_evaluated=len(shadow_preds),
                evaluation_period_hours=evaluation_hours,
                should_promote=False,
                confidence=0.0,
                reason="Insufficient samples for evaluation",
            )
        
        # Calculate metrics
        shadow_accuracy = self.calculate_accuracy(shadow_preds, actuals)
        live_accuracy = self.calculate_accuracy(live_preds, actuals)
        accuracy_improvement = shadow_accuracy - live_accuracy
        
        shadow_f1 = self.calculate_f1_score(shadow_preds, actuals)
        live_f1 = self.calculate_f1_score(live_preds, actuals)
        f1_improvement = shadow_f1 - live_f1
        
        # Determine if should promote
        should_promote = (
            accuracy_improvement > 0.01 and  # At least 1% improvement
            f1_improvement > 0.0  # F1 not worse
        )
        
        confidence = min(1.0, (accuracy_improvement + f1_improvement) / 0.1)
        
        if should_promote:
            reason = f"Shadow model shows {accuracy_improvement*100:.1f}% accuracy improvement"
        else:
            reason = "Shadow model performance not better than live"
        
        logger.info(
            f"Shadow evaluation for {model_name}: "
            f"Accuracy {shadow_accuracy:.3f} vs {live_accuracy:.3f}, "
            f"F1 {shadow_f1:.3f} vs {live_f1:.3f}, "
            f"Promote: {should_promote}"
        )
        
        return ShadowEvaluationResult(
            model_name=model_name,
            shadow_version=shadow_model.version,
            live_version=live_model.version,
            shadow_accuracy=shadow_accuracy,
            live_accuracy=live_accuracy,
            accuracy_improvement=accuracy_improvement,
            shadow_f1=shadow_f1,
            live_f1=live_f1,
            f1_improvement=f1_improvement,
            samples_evaluated=len(shadow_preds),
            evaluation_period_hours=evaluation_hours,
            should_promote=should_promote,
            confidence=confidence,
            reason=reason,
        )
    
    def clear_predictions(self, model_name: str):
        """Clear recorded predictions for a model."""
        self._shadow_predictions.pop(model_name, None)
        self._live_predictions.pop(model_name, None)
        self._ground_truth.pop(model_name, None)
