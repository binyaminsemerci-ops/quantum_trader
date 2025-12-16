"""
DRIFT DETECTION MANAGER - MODULE 3

Monitors AI model performance degradation and triggers retraining workflows.

Key Capabilities:
- Population Stability Index (PSI) for feature distribution shifts
- Kolmogorov-Smirnov test for prediction distribution changes
- Performance degradation tracking (win rate, F1, calibration)
- Automatic retraining trigger logic
- Integration with Memory States and RL Signal Manager

Author: Quantum Trader AI System
Date: November 2025
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from scipy.stats import ks_2samp, chi2_contingency
from collections import defaultdict, deque
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class DriftSeverity(Enum):
    """Drift severity levels"""
    NONE = "none"
    MINOR = "minor"          # 0.10 ≤ PSI < 0.15
    MODERATE = "moderate"    # 0.15 ≤ PSI < 0.25
    SEVERE = "severe"        # PSI ≥ 0.25
    CRITICAL = "critical"    # Performance drop + severe PSI


class DriftType(Enum):
    """Types of drift detected"""
    FEATURE_DRIFT = "feature_drift"              # Input distribution changed
    PREDICTION_DRIFT = "prediction_drift"        # Output distribution changed
    PERFORMANCE_DRIFT = "performance_drift"      # Accuracy degraded
    CONCEPT_DRIFT = "concept_drift"              # Feature-target relationship changed


class RetrainingUrgency(Enum):
    """Urgency levels for retraining"""
    NONE = "none"
    MONITOR = "monitor"              # Watch closely, no action yet
    SCHEDULED = "scheduled"          # Retrain within 72 hours
    URGENT = "urgent"                # Retrain within 24 hours
    IMMEDIATE = "immediate"          # Retrain ASAP (within 4 hours)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FeatureDistribution:
    """Feature distribution snapshot"""
    feature_name: str
    timestamp: str
    bins: List[float]           # Bin edges
    frequencies: List[float]    # Frequency per bin
    mean: float
    std: float
    min_val: float
    max_val: float
    sample_count: int


@dataclass
class PSIResult:
    """Population Stability Index result"""
    feature_name: str
    psi_score: float
    severity: str  # DriftSeverity
    bin_contributions: List[float]  # PSI contribution per bin
    timestamp: str


@dataclass
class KSTestResult:
    """Kolmogorov-Smirnov test result"""
    feature_or_prediction: str
    ks_statistic: float
    p_value: float
    is_significant: bool  # True if p < 0.01
    timestamp: str


@dataclass
class PerformanceMetrics:
    """Model performance metrics snapshot"""
    timestamp: str
    window_size: int
    trades_count: int
    
    # Core metrics
    win_rate: float
    precision: float
    recall: float
    f1_score: float
    
    # Calibration
    calibration_error: float
    brier_score: float
    
    # Financial
    avg_pnl: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    timestamp: str
    model_name: str
    drift_types: List[str]  # List of DriftType values
    severity: str  # DriftSeverity
    urgency: str  # RetrainingUrgency
    
    # Diagnostic details
    psi_scores: Dict[str, float]  # Feature name → PSI
    ks_p_values: Dict[str, float]  # Feature/prediction → p-value
    performance_delta: Dict[str, float]  # Metric → change
    
    # Actions
    recommended_action: str
    auto_retrain_scheduled: bool
    estimated_retrain_hours: float


@dataclass
class DriftContext:
    """Current drift detection context"""
    baseline_timestamp: str
    trades_since_baseline: int
    
    # Drift status
    active_alerts: List[DriftAlert]
    last_retrain_timestamp: Optional[str]
    
    # Performance trends
    recent_win_rate: float
    baseline_win_rate: float
    win_rate_delta: float
    
    consecutive_poor_windows: int  # Windows with degraded performance
    
    # Feature drift summary
    features_with_drift: List[str]
    max_psi_score: float
    max_psi_feature: str


# ============================================================
# DRIFT DETECTION MANAGER
# ============================================================

class DriftDetectionManager:
    """
    Manages drift detection across all AI models.
    
    Monitors:
    - Feature distributions (PSI)
    - Prediction distributions (KS-test)
    - Performance metrics (win rate, F1, calibration)
    - Triggers retraining workflows when drift detected
    """
    
    def __init__(
        self,
        # PSI thresholds
        psi_minor_threshold: float = 0.10,
        psi_moderate_threshold: float = 0.15,
        psi_severe_threshold: float = 0.25,
        psi_bins: int = 10,
        
        # KS test parameters
        ks_p_value_threshold: float = 0.01,
        
        # Performance thresholds
        win_rate_drop_threshold: float = 0.05,  # 5 pp drop
        f1_drop_threshold: float = 0.10,        # 10% drop
        calibration_error_threshold: float = 0.08,  # 8 pp increase
        
        # Window parameters
        performance_window_size: int = 100,  # Trades per window
        min_trades_for_detection: int = 50,
        consecutive_windows_threshold: int = 2,
        
        # Retraining parameters
        urgent_retrain_trades: int = 200,  # Trigger urgent if 200+ trades with issues
        scheduled_retrain_hours: int = 72,
        
        # Persistence
        checkpoint_path: str = "/app/data/drift_checkpoints/drift_state.json",
        checkpoint_interval_seconds: int = 300  # 5 minutes
    ):
        # Store parameters
        self.psi_thresholds = {
            'minor': psi_minor_threshold,
            'moderate': psi_moderate_threshold,
            'severe': psi_severe_threshold
        }
        self.psi_bins = psi_bins
        self.ks_p_value_threshold = ks_p_value_threshold
        
        self.performance_thresholds = {
            'win_rate_drop': win_rate_drop_threshold,
            'f1_drop': f1_drop_threshold,
            'calibration_error': calibration_error_threshold
        }
        
        self.window_size = performance_window_size
        self.min_trades = min_trades_for_detection
        self.consecutive_threshold = consecutive_windows_threshold
        
        self.urgent_retrain_trades = urgent_retrain_trades
        self.scheduled_retrain_hours = scheduled_retrain_hours
        
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval_seconds
        
        # State management
        self.baseline_distributions: Dict[str, Dict[str, FeatureDistribution]] = {}  # model → feature → dist
        self.baseline_performance: Dict[str, PerformanceMetrics] = {}  # model → metrics
        self.baseline_predictions: Dict[str, np.ndarray] = {}  # model → predictions array
        
        self.recent_distributions: Dict[str, Dict[str, FeatureDistribution]] = {}
        self.recent_performance_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))  # Last 10 windows
        self.recent_predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 predictions
        
        # Trade tracking per model
        self.trade_outcomes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=performance_window_size))
        self.trades_since_baseline: Dict[str, int] = defaultdict(int)
        
        # Drift alerts
        self.active_alerts: Dict[str, List[DriftAlert]] = defaultdict(list)  # model → alerts
        self.alert_history: List[DriftAlert] = []
        
        # Consecutive poor performance tracking
        self.consecutive_poor_windows: Dict[str, int] = defaultdict(int)
        
        # Last checkpoint time
        self.last_checkpoint_time = datetime.now()
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        logger.info(
            f"DriftDetectionManager initialized: "
            f"PSI_severe={psi_severe_threshold}, KS_p={ks_p_value_threshold}, "
            f"WR_drop={win_rate_drop_threshold}, window={performance_window_size}"
        )
    
    # ========================================
    # BASELINE ESTABLISHMENT
    # ========================================
    
    def establish_baseline(
        self,
        model_name: str,
        feature_values: Dict[str, np.ndarray],  # feature_name → values
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,  # Binary: 1=win, 0=loss
        confidences: np.ndarray
    ) -> None:
        """
        Establish baseline distributions and performance for a model.
        
        Args:
            model_name: Name of model (e.g., 'xgboost', 'lightgbm')
            feature_values: Dictionary of feature arrays
            predictions: Model predictions (probabilities or classes)
            actual_outcomes: Actual trade outcomes (1=win, 0=loss)
            confidences: Confidence scores
        """
        logger.info(f"Establishing baseline for {model_name} with {len(predictions)} samples")
        
        # Feature distributions
        self.baseline_distributions[model_name] = {}
        for feature_name, values in feature_values.items():
            dist = self._compute_feature_distribution(feature_name, values)
            self.baseline_distributions[model_name][feature_name] = dist
        
        # Prediction distribution
        self.baseline_predictions[model_name] = predictions
        
        # Performance metrics
        metrics = self._compute_performance_metrics(
            actual_outcomes=actual_outcomes,
            predictions=predictions,
            confidences=confidences,
            window_size=len(predictions)
        )
        self.baseline_performance[model_name] = metrics
        
        # Reset trade counter
        self.trades_since_baseline[model_name] = 0
        
        logger.info(
            f"Baseline established for {model_name}: "
            f"WR={metrics.win_rate:.3f}, F1={metrics.f1_score:.3f}, "
            f"Features={len(feature_values)}"
        )
    
    def _compute_feature_distribution(
        self,
        feature_name: str,
        values: np.ndarray
    ) -> FeatureDistribution:
        """Compute feature distribution with binning"""
        # Remove NaNs
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            logger.warning(f"Empty values for feature {feature_name}")
            return FeatureDistribution(
                feature_name=feature_name,
                timestamp=datetime.now().isoformat(),
                bins=[],
                frequencies=[],
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                sample_count=0
            )
        
        # Statistics
        mean = float(np.mean(values))
        std = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        # Create bins (deciles)
        try:
            bins = np.linspace(min_val, max_val, self.psi_bins + 1)
            frequencies, _ = np.histogram(values, bins=bins)
            frequencies = frequencies / len(values)  # Normalize to probabilities
        except Exception as e:
            logger.error(f"Error creating bins for {feature_name}: {e}")
            bins = []
            frequencies = []
        
        return FeatureDistribution(
            feature_name=feature_name,
            timestamp=datetime.now().isoformat(),
            bins=bins.tolist() if len(bins) > 0 else [],
            frequencies=frequencies.tolist() if len(frequencies) > 0 else [],
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            sample_count=len(values)
        )
    
    # ========================================
    # DRIFT DETECTION
    # ========================================
    
    def detect_drift(
        self,
        model_name: str,
        feature_values: Dict[str, np.ndarray],
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        confidences: np.ndarray
    ) -> Optional[DriftAlert]:
        """
        Detect drift for a model using current data.
        
        Returns DriftAlert if drift detected, None otherwise.
        """
        # Check if baseline exists
        if model_name not in self.baseline_distributions:
            logger.warning(f"No baseline for {model_name}, cannot detect drift")
            return None
        
        # Check minimum trades
        if len(predictions) < self.min_trades:
            logger.debug(f"Insufficient data for {model_name}: {len(predictions)} < {self.min_trades}")
            return None
        
        logger.info(f"Detecting drift for {model_name} with {len(predictions)} recent samples")
        
        # Update recent distributions
        self.recent_distributions[model_name] = {}
        for feature_name, values in feature_values.items():
            dist = self._compute_feature_distribution(feature_name, values)
            self.recent_distributions[model_name][feature_name] = dist
        
        # Stage 1: Feature Distribution Drift (PSI)
        psi_results = self._compute_psi_all_features(model_name)
        
        # Stage 2: Prediction Distribution Drift (KS-test)
        ks_result = self._compute_ks_test_predictions(model_name, predictions)
        
        # Stage 3: Performance Degradation
        recent_metrics = self._compute_performance_metrics(
            actual_outcomes=actual_outcomes,
            predictions=predictions,
            confidences=confidences,
            window_size=len(predictions)
        )
        performance_drift = self._detect_performance_drift(model_name, recent_metrics)
        
        # Stage 4: Determine drift types and severity
        drift_types = []
        severity = DriftSeverity.NONE
        
        # Check feature drift (PSI)
        severe_psi_features = [r for r in psi_results if r.psi_score >= self.psi_thresholds['severe']]
        if severe_psi_features:
            drift_types.append(DriftType.FEATURE_DRIFT.value)
            severity = DriftSeverity.SEVERE
        
        # Check prediction drift (KS)
        if ks_result and ks_result.is_significant:
            drift_types.append(DriftType.PREDICTION_DRIFT.value)
            if severity == DriftSeverity.NONE:
                severity = DriftSeverity.MODERATE
        
        # Check performance drift
        if performance_drift:
            drift_types.append(DriftType.PERFORMANCE_DRIFT.value)
            severity = DriftSeverity.CRITICAL  # Performance drop is critical
        
        # If no drift detected
        if not drift_types:
            self.consecutive_poor_windows[model_name] = 0
            return None
        
        # Stage 5: Determine urgency and action
        urgency, action = self._determine_retraining_urgency(
            model_name=model_name,
            drift_types=drift_types,
            severity=severity,
            trades_since_baseline=self.trades_since_baseline[model_name]
        )
        
        # Create alert
        alert = DriftAlert(
            alert_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            drift_types=drift_types,
            severity=severity.value,
            urgency=urgency.value,
            psi_scores={r.feature_name: r.psi_score for r in psi_results},
            ks_p_values={ks_result.feature_or_prediction: ks_result.p_value} if ks_result else {},
            performance_delta={
                'win_rate': recent_metrics.win_rate - self.baseline_performance[model_name].win_rate,
                'f1_score': recent_metrics.f1_score - self.baseline_performance[model_name].f1_score,
                'calibration_error': recent_metrics.calibration_error - self.baseline_performance[model_name].calibration_error
            },
            recommended_action=action,
            auto_retrain_scheduled=(urgency in [RetrainingUrgency.URGENT, RetrainingUrgency.IMMEDIATE]),
            estimated_retrain_hours=self._estimate_retrain_time(urgency)
        )
        
        # Store alert
        self.active_alerts[model_name].append(alert)
        self.alert_history.append(alert)
        
        logger.warning(
            f"DRIFT DETECTED for {model_name}: "
            f"Types={drift_types}, Severity={severity.value}, Urgency={urgency.value}"
        )
        
        return alert
    
    def _compute_psi_all_features(self, model_name: str) -> List[PSIResult]:
        """Compute PSI for all features of a model"""
        results = []
        
        baseline_dists = self.baseline_distributions.get(model_name, {})
        recent_dists = self.recent_distributions.get(model_name, {})
        
        for feature_name in baseline_dists.keys():
            if feature_name not in recent_dists:
                continue
            
            baseline = baseline_dists[feature_name]
            recent = recent_dists[feature_name]
            
            psi_score, bin_contributions = self._calculate_psi(
                expected_frequencies=np.array(baseline.frequencies),
                actual_frequencies=np.array(recent.frequencies)
            )
            
            # Determine severity
            if psi_score >= self.psi_thresholds['severe']:
                severity = DriftSeverity.SEVERE
            elif psi_score >= self.psi_thresholds['moderate']:
                severity = DriftSeverity.MODERATE
            elif psi_score >= self.psi_thresholds['minor']:
                severity = DriftSeverity.MINOR
            else:
                severity = DriftSeverity.NONE
            
            result = PSIResult(
                feature_name=feature_name,
                psi_score=psi_score,
                severity=severity.value,
                bin_contributions=bin_contributions,
                timestamp=datetime.now().isoformat()
            )
            results.append(result)
        
        return results
    
    def _calculate_psi(
        self,
        expected_frequencies: np.ndarray,
        actual_frequencies: np.ndarray
    ) -> Tuple[float, List[float]]:
        """
        Calculate Population Stability Index.
        
        PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        expected_frequencies = np.maximum(expected_frequencies, epsilon)
        actual_frequencies = np.maximum(actual_frequencies, epsilon)
        
        # PSI calculation
        psi_contributions = (actual_frequencies - expected_frequencies) * np.log(actual_frequencies / expected_frequencies)
        psi_score = float(np.sum(psi_contributions))
        
        return psi_score, psi_contributions.tolist()
    
    def _compute_ks_test_predictions(
        self,
        model_name: str,
        recent_predictions: np.ndarray
    ) -> Optional[KSTestResult]:
        """Compute KS test on prediction distributions"""
        if model_name not in self.baseline_predictions:
            return None
        
        baseline_preds = self.baseline_predictions[model_name]
        
        # KS test
        ks_stat, p_value = ks_2samp(baseline_preds, recent_predictions)
        
        is_significant = p_value < self.ks_p_value_threshold
        
        return KSTestResult(
            feature_or_prediction="predictions",
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            is_significant=is_significant,
            timestamp=datetime.now().isoformat()
        )
    
    def _compute_performance_metrics(
        self,
        actual_outcomes: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        window_size: int
    ) -> PerformanceMetrics:
        """Compute performance metrics for a window"""
        # Binary classification metrics
        # Assume predictions are probabilities, threshold at 0.5
        pred_classes = (predictions >= 0.5).astype(int)
        
        # True positives, false positives, true negatives, false negatives
        tp = np.sum((pred_classes == 1) & (actual_outcomes == 1))
        fp = np.sum((pred_classes == 1) & (actual_outcomes == 0))
        tn = np.sum((pred_classes == 0) & (actual_outcomes == 0))
        fn = np.sum((pred_classes == 0) & (actual_outcomes == 1))
        
        # Metrics
        win_rate = float(np.mean(actual_outcomes))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calibration error (Brier score simplified)
        calibration_error = float(np.mean((predictions - actual_outcomes) ** 2))
        brier_score = calibration_error
        
        # Placeholder for financial metrics (would need PnL data)
        avg_pnl = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            window_size=window_size,
            trades_count=len(predictions),
            win_rate=win_rate,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            calibration_error=calibration_error,
            brier_score=brier_score,
            avg_pnl=avg_pnl,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
    
    def _detect_performance_drift(
        self,
        model_name: str,
        recent_metrics: PerformanceMetrics
    ) -> bool:
        """Detect if performance has degraded significantly"""
        if model_name not in self.baseline_performance:
            return False
        
        baseline = self.baseline_performance[model_name]
        
        # Check win rate drop
        win_rate_delta = recent_metrics.win_rate - baseline.win_rate
        if win_rate_delta < -self.performance_thresholds['win_rate_drop']:
            self.consecutive_poor_windows[model_name] += 1
        else:
            self.consecutive_poor_windows[model_name] = 0
        
        # Check F1 drop
        f1_delta = recent_metrics.f1_score - baseline.f1_score
        f1_drop_pct = abs(f1_delta) / baseline.f1_score if baseline.f1_score > 0 else 0
        
        # Check calibration error increase
        cal_error_delta = recent_metrics.calibration_error - baseline.calibration_error
        
        # Drift detected if:
        # - Win rate dropped AND consecutive windows threshold met
        # - OR F1 score dropped significantly
        # - OR calibration error increased significantly
        performance_drift = (
            (win_rate_delta < -self.performance_thresholds['win_rate_drop'] and
             self.consecutive_poor_windows[model_name] >= self.consecutive_threshold)
            or
            (f1_drop_pct > self.performance_thresholds['f1_drop'])
            or
            (cal_error_delta > self.performance_thresholds['calibration_error'])
        )
        
        return performance_drift
    
    def _determine_retraining_urgency(
        self,
        model_name: str,
        drift_types: List[str],
        severity: DriftSeverity,
        trades_since_baseline: int
    ) -> Tuple[RetrainingUrgency, str]:
        """Determine urgency level and recommended action"""
        # IMMEDIATE: Critical performance drop with many trades
        if (DriftType.PERFORMANCE_DRIFT.value in drift_types and
            severity == DriftSeverity.CRITICAL and
            trades_since_baseline >= self.urgent_retrain_trades):
            return RetrainingUrgency.IMMEDIATE, "RETRAIN_IMMEDIATELY"
        
        # URGENT: Performance drop detected
        if (DriftType.PERFORMANCE_DRIFT.value in drift_types and
            severity == DriftSeverity.CRITICAL):
            return RetrainingUrgency.URGENT, "RETRAIN_WITHIN_24H"
        
        # SCHEDULED: Severe feature/prediction drift
        if (severity in [DriftSeverity.SEVERE, DriftSeverity.CRITICAL] and
            (DriftType.FEATURE_DRIFT.value in drift_types or
             DriftType.PREDICTION_DRIFT.value in drift_types)):
            return RetrainingUrgency.SCHEDULED, "SCHEDULE_RETRAIN_72H"
        
        # MONITOR: Moderate drift
        if severity in [DriftSeverity.MODERATE]:
            return RetrainingUrgency.MONITOR, "MONITOR_CLOSELY"
        
        # No action
        return RetrainingUrgency.NONE, "CONTINUE_MONITORING"
    
    def _estimate_retrain_time(self, urgency: RetrainingUrgency) -> float:
        """Estimate hours until retraining based on urgency"""
        if urgency == RetrainingUrgency.IMMEDIATE:
            return 4.0
        elif urgency == RetrainingUrgency.URGENT:
            return 24.0
        elif urgency == RetrainingUrgency.SCHEDULED:
            return 72.0
        else:
            return 168.0  # 1 week
    
    # ========================================
    # TRADE OUTCOME PROCESSING
    # ========================================
    
    def process_trade_outcome(
        self,
        model_name: str,
        prediction: float,
        confidence: float,
        actual_outcome: int,  # 1=win, 0=loss
        feature_values: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Process a single trade outcome for drift tracking.
        
        Called after each trade to incrementally update drift detection state.
        """
        # Update trade counter
        self.trades_since_baseline[model_name] += 1
        
        # Store outcome
        self.trade_outcomes[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'actual_outcome': actual_outcome,
            'feature_values': feature_values or {}
        })
        
        # Store prediction
        self.recent_predictions[model_name].append(prediction)
        
        # Check if we have enough data for a performance window
        if len(self.trade_outcomes[model_name]) >= self.window_size:
            self._update_performance_window(model_name)
        
        # Auto-checkpoint periodically
        if (datetime.now() - self.last_checkpoint_time).total_seconds() >= self.checkpoint_interval:
            self.checkpoint()
    
    def _update_performance_window(self, model_name: str) -> None:
        """Update rolling performance window"""
        outcomes = list(self.trade_outcomes[model_name])
        
        predictions = np.array([t['prediction'] for t in outcomes])
        confidences = np.array([t['confidence'] for t in outcomes])
        actuals = np.array([t['actual_outcome'] for t in outcomes])
        
        metrics = self._compute_performance_metrics(
            actual_outcomes=actuals,
            predictions=predictions,
            confidences=confidences,
            window_size=len(predictions)
        )
        
        self.recent_performance_windows[model_name].append(metrics)
    
    # ========================================
    # CONTEXT AND DIAGNOSTICS
    # ========================================
    
    def get_drift_context(self, model_name: str) -> DriftContext:
        """Get current drift context for a model"""
        baseline_metrics = self.baseline_performance.get(model_name)
        recent_windows = list(self.recent_performance_windows.get(model_name, []))
        
        # Recent win rate (last window if available)
        recent_win_rate = recent_windows[-1].win_rate if recent_windows else 0.0
        baseline_win_rate = baseline_metrics.win_rate if baseline_metrics else 0.0
        win_rate_delta = recent_win_rate - baseline_win_rate
        
        # Features with drift
        features_with_drift = []
        max_psi_score = 0.0
        max_psi_feature = ""
        
        if model_name in self.baseline_distributions and model_name in self.recent_distributions:
            for feature_name in self.baseline_distributions[model_name].keys():
                if feature_name in self.recent_distributions[model_name]:
                    baseline = self.baseline_distributions[model_name][feature_name]
                    recent = self.recent_distributions[model_name][feature_name]
                    
                    psi_score, _ = self._calculate_psi(
                        np.array(baseline.frequencies),
                        np.array(recent.frequencies)
                    )
                    
                    if psi_score >= self.psi_thresholds['moderate']:
                        features_with_drift.append(feature_name)
                    
                    if psi_score > max_psi_score:
                        max_psi_score = psi_score
                        max_psi_feature = feature_name
        
        return DriftContext(
            baseline_timestamp=baseline_metrics.timestamp if baseline_metrics else "",
            trades_since_baseline=self.trades_since_baseline.get(model_name, 0),
            active_alerts=self.active_alerts.get(model_name, []),
            last_retrain_timestamp=None,  # Would be set after retrain
            recent_win_rate=recent_win_rate,
            baseline_win_rate=baseline_win_rate,
            win_rate_delta=win_rate_delta,
            consecutive_poor_windows=self.consecutive_poor_windows.get(model_name, 0),
            features_with_drift=features_with_drift,
            max_psi_score=max_psi_score,
            max_psi_feature=max_psi_feature
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics"""
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'models_tracked': list(self.baseline_distributions.keys()),
            'active_alerts_count': sum(len(alerts) for alerts in self.active_alerts.values()),
            'total_alerts_history': len(self.alert_history),
            'models_status': {}
        }
        
        for model_name in self.baseline_distributions.keys():
            context = self.get_drift_context(model_name)
            
            diagnostics['models_status'][model_name] = {
                'trades_since_baseline': context.trades_since_baseline,
                'win_rate_delta': context.win_rate_delta,
                'consecutive_poor_windows': context.consecutive_poor_windows,
                'features_with_drift_count': len(context.features_with_drift),
                'max_psi_score': context.max_psi_score,
                'max_psi_feature': context.max_psi_feature,
                'active_alerts': len(context.active_alerts),
                'status': 'DEGRADED' if context.active_alerts else 'HEALTHY'
            }
        
        return diagnostics
    
    # ========================================
    # CHECKPOINTING
    # ========================================
    
    def checkpoint(self) -> None:
        """Save drift detection state to disk"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'trades_since_baseline': dict(self.trades_since_baseline),
                'consecutive_poor_windows': dict(self.consecutive_poor_windows),
                'active_alerts': {
                    model: [asdict(alert) for alert in alerts]
                    for model, alerts in self.active_alerts.items()
                },
                'baseline_performance': {
                    model: asdict(metrics)
                    for model, metrics in self.baseline_performance.items()
                }
            }
            
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.last_checkpoint_time = datetime.now()
            logger.info(f"Checkpoint saved: {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> None:
        """Load drift detection state from disk"""
        try:
            import os
            if not os.path.exists(self.checkpoint_path):
                logger.info("No checkpoint found, starting fresh")
                return
            
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.trades_since_baseline = defaultdict(int, state.get('trades_since_baseline', {}))
            self.consecutive_poor_windows = defaultdict(int, state.get('consecutive_poor_windows', {}))
            
            # Restore alerts
            for model, alerts_data in state.get('active_alerts', {}).items():
                self.active_alerts[model] = [
                    DriftAlert(**alert_dict) for alert_dict in alerts_data
                ]
            
            # Restore baseline performance
            for model, metrics_data in state.get('baseline_performance', {}).items():
                self.baseline_performance[model] = PerformanceMetrics(**metrics_data)
            
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    # ========================================
    # RETRAINING INTEGRATION
    # ========================================
    
    def trigger_retraining(self, model_name: str, alert: DriftAlert) -> Dict[str, Any]:
        """
        Trigger retraining workflow for a model.
        
        Returns retraining job details.
        """
        logger.warning(f"TRIGGERING RETRAINING for {model_name} due to alert {alert.alert_id}")
        
        # Create retraining job (would integrate with actual retraining pipeline)
        job = {
            'job_id': f"retrain_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_name': model_name,
            'trigger_alert_id': alert.alert_id,
            'urgency': alert.urgency,
            'estimated_hours': alert.estimated_retrain_hours,
            'scheduled_start': (
                datetime.now() + timedelta(hours=alert.estimated_retrain_hours)
            ).isoformat(),
            'status': 'QUEUED'
        }
        
        logger.info(f"Retraining job created: {job['job_id']}")
        
        return job
    
    def reset_baseline_after_retrain(
        self,
        model_name: str,
        new_feature_values: Dict[str, np.ndarray],
        new_predictions: np.ndarray,
        new_actual_outcomes: np.ndarray,
        new_confidences: np.ndarray
    ) -> None:
        """
        Reset baseline after model retraining.
        
        Called after successful retraining to establish new baseline.
        """
        logger.info(f"Resetting baseline for {model_name} after retraining")
        
        # Clear old alerts
        self.active_alerts[model_name] = []
        
        # Reset counters
        self.trades_since_baseline[model_name] = 0
        self.consecutive_poor_windows[model_name] = 0
        
        # Establish new baseline
        self.establish_baseline(
            model_name=model_name,
            feature_values=new_feature_values,
            predictions=new_predictions,
            actual_outcomes=new_actual_outcomes,
            confidences=new_confidences
        )
        
        logger.info(f"Baseline reset complete for {model_name}")
