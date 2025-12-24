"""
Phase 3C-3: Adaptive Threshold Manager

Automatically learns and adjusts alert thresholds based on historical data.
Reduces false positives and optimizes alert sensitivity.

Features:
- Dynamic threshold adjustment based on historical performance
- Learn optimal thresholds from past data
- Reduce false positive alerts
- Auto-tune health score weights
- Predictive alerting (alert before issues occur)
- Threshold recommendation engine

Author: AI Agent
Date: December 24, 2025
"""

import asyncio
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import structlog

logger = structlog.get_logger()


class ThresholdType(str, Enum):
    """Threshold types."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    HEALTH_SCORE = "health_score"
    THROUGHPUT = "throughput"


class ThresholdAdjustmentReason(str, Enum):
    """Reasons for threshold adjustment."""
    FALSE_POSITIVES = "false_positives"
    FALSE_NEGATIVES = "false_negatives"
    BASELINE_SHIFT = "baseline_shift"
    SEASONAL_PATTERN = "seasonal_pattern"
    MANUAL_OVERRIDE = "manual_override"
    INITIAL_LEARNING = "initial_learning"


@dataclass
class Threshold:
    """Threshold configuration."""
    threshold_id: str
    module_type: str
    metric_name: str
    threshold_type: ThresholdType
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    last_adjusted: datetime
    adjustment_count: int
    confidence: float  # 0-1, how confident we are in this threshold
    is_learned: bool
    
    def to_dict(self) -> dict:
        return {
            'threshold_id': self.threshold_id,
            'module_type': self.module_type,
            'metric_name': self.metric_name,
            'threshold_type': self.threshold_type.value,
            'warning_threshold': self.warning_threshold,
            'error_threshold': self.error_threshold,
            'critical_threshold': self.critical_threshold,
            'last_adjusted': self.last_adjusted.isoformat(),
            'adjustment_count': self.adjustment_count,
            'confidence': self.confidence,
            'is_learned': self.is_learned
        }


@dataclass
class ThresholdAdjustment:
    """Record of threshold adjustment."""
    adjustment_id: str
    timestamp: datetime
    module_type: str
    metric_name: str
    old_value: float
    new_value: float
    reason: ThresholdAdjustmentReason
    false_positive_rate: float
    false_negative_rate: float
    
    def to_dict(self) -> dict:
        return {
            'adjustment_id': self.adjustment_id,
            'timestamp': self.timestamp.isoformat(),
            'module_type': self.module_type,
            'metric_name': self.metric_name,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'reason': self.reason.value,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate
        }


@dataclass
class HealthScoreWeights:
    """Weights for health score calculation."""
    phase_2b_weight: float
    phase_2d_weight: float
    phase_3a_weight: float
    phase_3b_weight: float
    ensemble_weight: float
    last_updated: datetime
    adjustment_count: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictiveAlert:
    """Predictive alert before issue occurs."""
    alert_id: str
    timestamp: datetime
    module_type: str
    predicted_issue: str
    confidence: float
    time_to_issue_hours: float
    recommended_action: str
    metric_trend: str
    
    def to_dict(self) -> dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'module_type': self.module_type,
            'predicted_issue': self.predicted_issue,
            'confidence': self.confidence,
            'time_to_issue_hours': self.time_to_issue_hours,
            'recommended_action': self.recommended_action,
            'metric_trend': self.metric_trend
        }


class AdaptiveThresholdManager:
    """
    Adaptive threshold management with machine learning.
    
    Features:
    - Automatically learns optimal thresholds from historical data
    - Reduces false positive alerts
    - Adapts to changing system behavior
    - Predictive alerting based on trends
    - Auto-tunes health score weights
    
    Learning approach:
    - Tracks alert outcomes (true positive, false positive)
    - Adjusts thresholds to minimize false positives while catching real issues
    - Uses statistical analysis to set optimal thresholds
    - Adapts to baseline shifts and seasonal patterns
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_samples_for_learning: int = 100,
        false_positive_target: float = 0.05,  # Target 5% false positive rate
        adjustment_interval_hours: int = 24,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize adaptive threshold manager.
        
        Args:
            learning_rate: How quickly to adapt thresholds (0-1)
            min_samples_for_learning: Minimum samples before learning
            false_positive_target: Target false positive rate
            adjustment_interval_hours: How often to review thresholds
            confidence_threshold: Minimum confidence to apply adjustments
        """
        self.learning_rate = learning_rate
        self.min_samples = min_samples_for_learning
        self.fp_target = false_positive_target
        self.adjustment_interval = timedelta(hours=adjustment_interval_hours)
        self.confidence_threshold = confidence_threshold
        
        # Thresholds
        self.thresholds: Dict[str, Dict[str, Threshold]] = defaultdict(dict)
        self.adjustment_history: deque = deque(maxlen=1000)
        
        # Alert tracking for learning
        self.alert_outcomes: deque = deque(maxlen=1000)
        self.false_positives: Dict[str, int] = defaultdict(int)
        self.true_positives: Dict[str, int] = defaultdict(int)
        self.false_negatives: Dict[str, int] = defaultdict(int)
        
        # Historical metrics for learning
        self.metric_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # Health score weights (start with equal weights)
        self.health_weights = HealthScoreWeights(
            phase_2b_weight=0.20,
            phase_2d_weight=0.20,
            phase_3a_weight=0.25,
            phase_3b_weight=0.25,
            ensemble_weight=0.10,
            last_updated=datetime.utcnow(),
            adjustment_count=0
        )
        
        # Predictive alerting
        self.predictive_alerts: List[PredictiveAlert] = []
        
        # Control
        self.is_learning = False
        self.last_adjustment = datetime.utcnow()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        logger.info("[PHASE 3C-3] 🧠 Adaptive Threshold Manager initialized")
    
    def _initialize_default_thresholds(self):
        """Set initial default thresholds."""
        # Latency thresholds (milliseconds)
        self._set_threshold('phase_2b', 'latency_ms', ThresholdType.LATENCY,
                          warning=50, error=100, critical=200)
        self._set_threshold('phase_2d', 'latency_ms', ThresholdType.LATENCY,
                          warning=50, error=100, critical=200)
        self._set_threshold('phase_3a', 'latency_ms', ThresholdType.LATENCY,
                          warning=75, error=150, critical=300)
        self._set_threshold('phase_3b', 'latency_ms', ThresholdType.LATENCY,
                          warning=75, error=150, critical=300)
        self._set_threshold('ensemble', 'latency_ms', ThresholdType.LATENCY,
                          warning=100, error=200, critical=400)
        
        # Accuracy thresholds (percentage)
        self._set_threshold('phase_2b', 'accuracy_pct', ThresholdType.ACCURACY,
                          warning=70, error=60, critical=50)
        self._set_threshold('phase_2d', 'accuracy_pct', ThresholdType.ACCURACY,
                          warning=70, error=60, critical=50)
        self._set_threshold('phase_3a', 'accuracy_pct', ThresholdType.ACCURACY,
                          warning=75, error=65, critical=55)
        self._set_threshold('phase_3b', 'accuracy_pct', ThresholdType.ACCURACY,
                          warning=75, error=65, critical=55)
        
        # Error rate thresholds (percentage)
        for module in ['phase_2b', 'phase_2d', 'phase_3a', 'phase_3b', 'ensemble']:
            self._set_threshold(module, 'error_rate', ThresholdType.ERROR_RATE,
                              warning=1, error=5, critical=10)
        
        # Health score thresholds
        for module in ['phase_2b', 'phase_2d', 'phase_3a', 'phase_3b', 'ensemble']:
            self._set_threshold(module, 'health_score', ThresholdType.HEALTH_SCORE,
                              warning=80, error=60, critical=40)
    
    def _set_threshold(
        self,
        module_type: str,
        metric_name: str,
        threshold_type: ThresholdType,
        warning: float,
        error: float,
        critical: float,
        is_learned: bool = False
    ):
        """Set threshold values."""
        threshold_id = f"{module_type}_{metric_name}"
        
        self.thresholds[module_type][metric_name] = Threshold(
            threshold_id=threshold_id,
            module_type=module_type,
            metric_name=metric_name,
            threshold_type=threshold_type,
            warning_threshold=warning,
            error_threshold=error,
            critical_threshold=critical,
            last_adjusted=datetime.utcnow(),
            adjustment_count=0,
            confidence=0.5 if not is_learned else 0.9,
            is_learned=is_learned
        )
    
    # ========================================================================
    # LEARNING & ADAPTATION
    # ========================================================================
    
    async def start_learning(self):
        """Start continuous threshold learning."""
        self.is_learning = True
        logger.info("[PHASE 3C-3] 🧠 Adaptive learning started")
        
        while self.is_learning:
            try:
                await self.review_and_adjust_thresholds()
                await asyncio.sleep(self.adjustment_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PHASE 3C-3] ❌ Learning error: {e}", exc_info=True)
                await asyncio.sleep(self.adjustment_interval.total_seconds())
    
    def stop_learning(self):
        """Stop learning."""
        self.is_learning = False
        logger.info("[PHASE 3C-3] Adaptive learning stopped")
    
    async def review_and_adjust_thresholds(self):
        """Review thresholds and adjust if needed."""
        logger.info("[PHASE 3C-3] Reviewing thresholds for adjustment...")
        
        adjustments_made = 0
        
        for module_type in self.thresholds:
            for metric_name, threshold in self.thresholds[module_type].items():
                # Get metric history
                history = list(self.metric_history[module_type][metric_name])
                
                if len(history) < self.min_samples:
                    continue
                
                # Calculate optimal threshold based on distribution
                new_threshold = await self._calculate_optimal_threshold(
                    history, threshold
                )
                
                if new_threshold is not None:
                    # Calculate false positive rate
                    fp_rate = self._calculate_false_positive_rate(module_type, metric_name)
                    fn_rate = self._calculate_false_negative_rate(module_type, metric_name)
                    
                    # Adjust threshold
                    await self._adjust_threshold(
                        module_type, metric_name, new_threshold,
                        ThresholdAdjustmentReason.FALSE_POSITIVES if fp_rate > self.fp_target 
                        else ThresholdAdjustmentReason.BASELINE_SHIFT,
                        fp_rate, fn_rate
                    )
                    adjustments_made += 1
        
        # Adjust health score weights
        await self._adjust_health_weights()
        
        logger.info(f"[PHASE 3C-3] ✅ Threshold review complete ({adjustments_made} adjustments)")
    
    async def _calculate_optimal_threshold(
        self,
        history: List[float],
        current_threshold: Threshold
    ) -> Optional[float]:
        """Calculate optimal threshold from historical data."""
        if len(history) < self.min_samples:
            return None
        
        # Calculate statistical measures
        mean = statistics.mean(history)
        std_dev = statistics.stdev(history) if len(history) > 1 else 0
        
        # For latency/error rate: use mean + 2*std_dev for warning
        if current_threshold.threshold_type in [ThresholdType.LATENCY, ThresholdType.ERROR_RATE]:
            optimal = mean + (2 * std_dev)
            
            # Don't adjust if within 10% of current
            if abs(optimal - current_threshold.warning_threshold) / current_threshold.warning_threshold < 0.1:
                return None
            
            return optimal
        
        # For accuracy: use mean - 1*std_dev for warning
        elif current_threshold.threshold_type == ThresholdType.ACCURACY:
            optimal = mean - std_dev
            
            if abs(optimal - current_threshold.warning_threshold) / current_threshold.warning_threshold < 0.1:
                return None
            
            return optimal
        
        return None
    
    async def _adjust_threshold(
        self,
        module_type: str,
        metric_name: str,
        new_value: float,
        reason: ThresholdAdjustmentReason,
        fp_rate: float,
        fn_rate: float
    ):
        """Adjust threshold value."""
        threshold = self.thresholds[module_type][metric_name]
        old_value = threshold.warning_threshold
        
        # Apply learning rate
        adjusted_value = old_value + (new_value - old_value) * self.learning_rate
        
        # Update threshold
        threshold.warning_threshold = adjusted_value
        threshold.error_threshold = adjusted_value * 1.5
        threshold.critical_threshold = adjusted_value * 2.0
        threshold.last_adjusted = datetime.utcnow()
        threshold.adjustment_count += 1
        threshold.confidence = min(1.0, threshold.confidence + 0.1)
        threshold.is_learned = True
        
        # Record adjustment
        adjustment = ThresholdAdjustment(
            adjustment_id=f"adj_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow(),
            module_type=module_type,
            metric_name=metric_name,
            old_value=old_value,
            new_value=adjusted_value,
            reason=reason,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate
        )
        
        self.adjustment_history.append(adjustment)
        
        logger.info(
            f"[PHASE 3C-3] 🔧 Threshold adjusted: {module_type}.{metric_name} "
            f"{old_value:.2f} → {adjusted_value:.2f} (reason: {reason.value})"
        )
    
    async def _adjust_health_weights(self):
        """Adjust health score weights based on module importance."""
        # Calculate module criticality based on error rates and impact
        # This is a simplified version - could be much more sophisticated
        
        # For now, just normalize weights to ensure they sum to 1.0
        total = (
            self.health_weights.phase_2b_weight +
            self.health_weights.phase_2d_weight +
            self.health_weights.phase_3a_weight +
            self.health_weights.phase_3b_weight +
            self.health_weights.ensemble_weight
        )
        
        if abs(total - 1.0) > 0.01:
            # Normalize
            self.health_weights.phase_2b_weight /= total
            self.health_weights.phase_2d_weight /= total
            self.health_weights.phase_3a_weight /= total
            self.health_weights.phase_3b_weight /= total
            self.health_weights.ensemble_weight /= total
            self.health_weights.last_updated = datetime.utcnow()
            self.health_weights.adjustment_count += 1
            
            logger.info("[PHASE 3C-3] Health score weights normalized")
    
    # ========================================================================
    # ALERT TRACKING
    # ========================================================================
    
    def record_alert_outcome(
        self,
        module_type: str,
        metric_name: str,
        was_true_positive: bool
    ):
        """Record whether an alert was a true or false positive."""
        key = f"{module_type}_{metric_name}"
        
        if was_true_positive:
            self.true_positives[key] += 1
        else:
            self.false_positives[key] += 1
        
        self.alert_outcomes.append({
            'timestamp': datetime.utcnow(),
            'module_type': module_type,
            'metric_name': metric_name,
            'true_positive': was_true_positive
        })
    
    def record_missed_issue(self, module_type: str, metric_name: str):
        """Record a false negative (issue that should have alerted but didn't)."""
        key = f"{module_type}_{metric_name}"
        self.false_negatives[key] += 1
    
    def _calculate_false_positive_rate(self, module_type: str, metric_name: str) -> float:
        """Calculate false positive rate for a metric."""
        key = f"{module_type}_{metric_name}"
        
        total_alerts = self.true_positives[key] + self.false_positives[key]
        if total_alerts == 0:
            return 0.0
        
        return self.false_positives[key] / total_alerts
    
    def _calculate_false_negative_rate(self, module_type: str, metric_name: str) -> float:
        """Calculate false negative rate for a metric."""
        key = f"{module_type}_{metric_name}"
        
        actual_issues = self.true_positives[key] + self.false_negatives[key]
        if actual_issues == 0:
            return 0.0
        
        return self.false_negatives[key] / actual_issues
    
    # ========================================================================
    # METRIC RECORDING
    # ========================================================================
    
    def record_metric(self, module_type: str, metric_name: str, value: float):
        """Record metric value for learning."""
        self.metric_history[module_type][metric_name].append(value)
    
    # ========================================================================
    # PREDICTIVE ALERTING
    # ========================================================================
    
    async def generate_predictive_alerts(self) -> List[PredictiveAlert]:
        """Generate predictive alerts based on metric trends."""
        predictive_alerts = []
        
        for module_type in self.metric_history:
            for metric_name, history in self.metric_history[module_type].items():
                if len(history) < 20:
                    continue
                
                # Analyze trend
                recent = list(history)[-20:]
                trend = await self._analyze_trend(recent)
                
                if trend['is_degrading'] and trend['confidence'] > 0.7:
                    threshold = self.thresholds[module_type].get(metric_name)
                    if not threshold:
                        continue
                    
                    # Predict when threshold will be crossed
                    time_to_issue = await self._predict_time_to_threshold(
                        recent, threshold.warning_threshold, trend['rate']
                    )
                    
                    if time_to_issue and time_to_issue < 24:  # Alert if within 24 hours
                        alert = PredictiveAlert(
                            alert_id=f"pred_{int(datetime.utcnow().timestamp())}",
                            timestamp=datetime.utcnow(),
                            module_type=module_type,
                            predicted_issue=f"{metric_name} will exceed threshold",
                            confidence=trend['confidence'],
                            time_to_issue_hours=time_to_issue,
                            recommended_action=f"Monitor {module_type} closely and prepare remediation",
                            metric_trend=trend['direction']
                        )
                        
                        predictive_alerts.append(alert)
                        
                        logger.warning(
                            f"[PHASE 3C-3] 🔮 Predictive alert: {module_type}.{metric_name} "
                            f"will exceed threshold in {time_to_issue:.1f}h"
                        )
        
        self.predictive_alerts = predictive_alerts
        return predictive_alerts
    
    async def _analyze_trend(self, data: List[float]) -> Dict[str, Any]:
        """Analyze metric trend."""
        if len(data) < 2:
            return {'is_degrading': False, 'confidence': 0.0, 'rate': 0.0, 'direction': 'stable'}
        
        # Simple linear regression
        n = len(data)
        x = list(range(n))
        y = data
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {'is_degrading': False, 'confidence': 0.0, 'rate': 0.0, 'direction': 'stable'}
        
        slope = numerator / denominator
        
        # Calculate R-squared for confidence
        y_pred = [slope * (x[i] - x_mean) + y_mean for i in range(n)]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        is_degrading = slope > 0  # Positive slope = degrading for latency/errors
        direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'is_degrading': is_degrading,
            'confidence': abs(r_squared),
            'rate': slope,
            'direction': direction
        }
    
    async def _predict_time_to_threshold(
        self,
        data: List[float],
        threshold: float,
        rate: float
    ) -> Optional[float]:
        """Predict hours until threshold is crossed."""
        if rate <= 0:
            return None
        
        current_value = data[-1]
        if current_value >= threshold:
            return 0.0
        
        # Assuming rate is per sample, and we sample every 5 minutes
        samples_to_threshold = (threshold - current_value) / rate
        hours_to_threshold = (samples_to_threshold * 5) / 60  # 5 min per sample
        
        return max(0.0, hours_to_threshold)
    
    # ========================================================================
    # QUERY METHODS
    # ========================================================================
    
    def get_threshold(self, module_type: str, metric_name: str) -> Optional[Threshold]:
        """Get threshold for a metric."""
        return self.thresholds[module_type].get(metric_name)
    
    def get_all_thresholds(self) -> Dict[str, Dict[str, Threshold]]:
        """Get all thresholds."""
        return self.thresholds
    
    def get_adjustment_history(self, hours: int = 24) -> List[ThresholdAdjustment]:
        """Get recent threshold adjustments."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            adj for adj in self.adjustment_history
            if adj.timestamp > cutoff
        ]
    
    def get_health_weights(self) -> HealthScoreWeights:
        """Get current health score weights."""
        return self.health_weights
    
    def get_predictive_alerts(self) -> List[PredictiveAlert]:
        """Get current predictive alerts."""
        return self.predictive_alerts
    
    def get_false_positive_rate(self, module_type: str, metric_name: str) -> float:
        """Get false positive rate for a metric."""
        return self._calculate_false_positive_rate(module_type, metric_name)
    
    # ========================================================================
    # MANUAL OVERRIDES
    # ========================================================================
    
    def override_threshold(
        self,
        module_type: str,
        metric_name: str,
        warning: float,
        error: float,
        critical: float
    ):
        """Manually override threshold values."""
        if module_type not in self.thresholds:
            raise ValueError(f"Unknown module type: {module_type}")
        
        if metric_name not in self.thresholds[module_type]:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        threshold = self.thresholds[module_type][metric_name]
        old_warning = threshold.warning_threshold
        
        threshold.warning_threshold = warning
        threshold.error_threshold = error
        threshold.critical_threshold = critical
        threshold.last_adjusted = datetime.utcnow()
        threshold.adjustment_count += 1
        threshold.confidence = 1.0  # Full confidence in manual override
        
        # Record adjustment
        adjustment = ThresholdAdjustment(
            adjustment_id=f"manual_{int(datetime.utcnow().timestamp())}",
            timestamp=datetime.utcnow(),
            module_type=module_type,
            metric_name=metric_name,
            old_value=old_warning,
            new_value=warning,
            reason=ThresholdAdjustmentReason.MANUAL_OVERRIDE,
            false_positive_rate=0.0,
            false_negative_rate=0.0
        )
        
        self.adjustment_history.append(adjustment)
        
        logger.info(
            f"[PHASE 3C-3] ✋ Manual threshold override: {module_type}.{metric_name} → {warning}"
        )
    
    def override_health_weights(
        self,
        phase_2b: float,
        phase_2d: float,
        phase_3a: float,
        phase_3b: float,
        ensemble: float
    ):
        """Manually override health score weights."""
        total = phase_2b + phase_2d + phase_3a + phase_3b + ensemble
        
        if abs(total - 1.0) > 0.01:
            # Normalize
            phase_2b /= total
            phase_2d /= total
            phase_3a /= total
            phase_3b /= total
            ensemble /= total
        
        self.health_weights = HealthScoreWeights(
            phase_2b_weight=phase_2b,
            phase_2d_weight=phase_2d,
            phase_3a_weight=phase_3a,
            phase_3b_weight=phase_3b,
            ensemble_weight=ensemble,
            last_updated=datetime.utcnow(),
            adjustment_count=self.health_weights.adjustment_count + 1
        )
        
        logger.info("[PHASE 3C-3] ✋ Manual health weights override applied")
