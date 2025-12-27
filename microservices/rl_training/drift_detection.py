"""
Drift Detection

Monitors feature and prediction drift to trigger retraining.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

from microservices.rl_training.models import (
    DriftSeverity,
    DriftDetectedEvent,
)


logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Drift Detector.
    
    Monitors:
    - Feature distribution drift (PSI - Population Stability Index)
    - Prediction distribution drift
    - Performance degradation
    """
    
    def __init__(
        self,
        event_bus,
        config,
        logger_instance=None
    ):
        """
        Initialize drift detector.
        
        Args:
            event_bus: EventBus for publishing drift events
            config: Service configuration
            logger_instance: Logger instance (optional)
        """
        self.event_bus = event_bus
        self.config = config
        self.logger = logger_instance or logger
        
        self._reference_distributions: Dict[str, Dict[str, Any]] = {}
        self._current_distributions: Dict[str, Dict[str, Any]] = {}
        self._drift_history: List[Dict[str, Any]] = []
    
    def set_reference_distribution(
        self,
        feature_name: str,
        distribution: Dict[str, Any]
    ) -> None:
        """
        Set reference distribution for a feature.
        
        Args:
            feature_name: Feature name
            distribution: Distribution statistics (mean, std, bins, frequencies)
        """
        self._reference_distributions[feature_name] = distribution
        
        self.logger.info(
            f"[DriftDetector] Set reference distribution for {feature_name}"
        )
    
    def calculate_psi(
        self,
        reference_freq: List[float],
        current_freq: List[float]
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = sum((current% - reference%) * ln(current% / reference%))
        
        Args:
            reference_freq: Reference distribution frequencies
            current_freq: Current distribution frequencies
        
        Returns:
            PSI score
        """
        psi = 0.0
        epsilon = 1e-10  # Avoid log(0)
        
        for ref, cur in zip(reference_freq, current_freq):
            ref = max(ref, epsilon)
            cur = max(cur, epsilon)
            psi += (cur - ref) * np.log(cur / ref)
        
        return psi
    
    def classify_psi_severity(self, psi_score: float) -> DriftSeverity:
        """
        Classify PSI score into drift severity.
        
        PSI < 0.10: No drift
        0.10 ≤ PSI < 0.15: Minor drift
        0.15 ≤ PSI < 0.25: Moderate drift
        PSI ≥ 0.25: Severe drift
        
        Args:
            psi_score: PSI score
        
        Returns:
            DriftSeverity enum
        """
        if psi_score < 0.10:
            return DriftSeverity.NONE
        elif psi_score < 0.15:
            return DriftSeverity.MINOR
        elif psi_score < 0.25:
            return DriftSeverity.MODERATE
        else:
            return DriftSeverity.SEVERE
    
    async def check_feature_drift(
        self,
        feature_name: str,
        current_distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check for feature drift.
        
        Args:
            feature_name: Feature name
            current_distribution: Current distribution statistics
        
        Returns:
            Drift detection result
        """
        if feature_name not in self._reference_distributions:
            self.logger.warning(
                f"[DriftDetector] No reference distribution for {feature_name}"
            )
            return {
                "status": "no_reference",
                "feature_name": feature_name,
                "drift_detected": False
            }
        
        reference = self._reference_distributions[feature_name]
        
        # Calculate PSI
        psi_score = self.calculate_psi(
            reference.get("frequencies", []),
            current_distribution.get("frequencies", [])
        )
        
        severity = self.classify_psi_severity(psi_score)
        drift_detected = severity != DriftSeverity.NONE
        
        self.logger.info(
            f"[DriftDetector] Feature drift check: {feature_name}, "
            f"PSI={psi_score:.4f}, severity={severity.value}"
        )
        
        # Store current distribution
        self._current_distributions[feature_name] = current_distribution
        
        # Record in history
        drift_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_name": feature_name,
            "psi_score": psi_score,
            "severity": severity.value,
            "drift_detected": drift_detected
        }
        self._drift_history.append(drift_record)
        
        # Publish drift event if significant
        if psi_score > self.config.DRIFT_TRIGGER_THRESHOLD:
            recommendation = "monitor"
            if severity == DriftSeverity.MODERATE:
                recommendation = "retrain_scheduled"
            elif severity in [DriftSeverity.SEVERE, DriftSeverity.CRITICAL]:
                recommendation = "retrain_urgent"
            
            await self.event_bus.publish(
                "data.drift_detected",
                DriftDetectedEvent(
                    drift_type="feature_drift",
                    severity=severity,
                    affected_models=[],  # Would determine affected models
                    psi_score=psi_score,
                    recommendation=recommendation,
                    detected_at=datetime.now(timezone.utc).isoformat()
                ).model_dump()
            )
            
            self.logger.warning(
                f"[DriftDetector] Drift detected for {feature_name}! "
                f"PSI={psi_score:.4f}, recommendation={recommendation}"
            )
        
        return {
            "status": "checked",
            "feature_name": feature_name,
            "psi_score": psi_score,
            "severity": severity.value,
            "drift_detected": drift_detected
        }
    
    async def check_performance_degradation(
        self,
        model_name: str,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if model performance has degraded.
        
        Args:
            model_name: Model name
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics
        
        Returns:
            Performance degradation result
        """
        # Calculate performance changes
        sharpe_change = (
            current_metrics.get("sharpe_ratio", 0) -
            baseline_metrics.get("sharpe_ratio", 0)
        )
        winrate_change = (
            current_metrics.get("win_rate", 0) -
            baseline_metrics.get("win_rate", 0)
        )
        
        # Check if degraded
        threshold = self.config.PERFORMANCE_DECAY_THRESHOLD
        degraded = (
            sharpe_change < -threshold or
            winrate_change < -threshold
        )
        
        self.logger.info(
            f"[DriftDetector] Performance check for {model_name}: "
            f"sharpe_change={sharpe_change:.3f}, winrate_change={winrate_change:.3f}, "
            f"degraded={degraded}"
        )
        
        if degraded:
            await self.event_bus.publish(
                "data.drift_detected",
                DriftDetectedEvent(
                    drift_type="performance_drift",
                    severity=DriftSeverity.MODERATE,
                    affected_models=[model_name],
                    psi_score=0.0,
                    recommendation="retrain_scheduled",
                    detected_at=datetime.now(timezone.utc).isoformat()
                ).model_dump()
            )
            
            self.logger.warning(
                f"[DriftDetector] Performance degradation detected for {model_name}!"
            )
        
        return {
            "status": "checked",
            "model_name": model_name,
            "sharpe_change": sharpe_change,
            "winrate_change": winrate_change,
            "degraded": degraded
        }
    
    def get_drift_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent drift detection history"""
        return self._drift_history[-limit:]
    
    def get_current_distributions(self) -> Dict[str, Dict[str, Any]]:
        """Get current feature distributions"""
        return self._current_distributions.copy()
