"""
Drift Detector - Monitors distribution shifts in features, predictions, and performance.

Provides:
- Feature drift detection (Kolmogorov-Smirnov test)
- Prediction drift detection
- Performance drift detection
- Automatic alerts and retraining triggers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.event_bus import EventBus
from backend.domains.learning.model_registry import ModelType

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class DriftType(str, Enum):
    """Type of drift detected."""
    FEATURE = "feature"
    PREDICTION = "prediction"
    PERFORMANCE = "performance"


class DriftSeverity(str, Enum):
    """Severity of detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DriftResult:
    """Result from drift detection."""
    
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float  # 0-1 (higher = more drift)
    p_value: float
    threshold: float
    
    # Context
    model_type: Optional[ModelType] = None
    feature_name: Optional[str] = None
    detection_time: datetime = None
    reference_period: Optional[Dict] = None  # {start, end}
    current_period: Optional[Dict] = None
    
    # Statistics
    reference_stats: Optional[Dict] = None
    current_stats: Optional[Dict] = None
    
    # Recommendation
    trigger_retraining: bool = False
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.detection_time is None:
            self.detection_time = datetime.utcnow()


# ============================================================================
# Drift Detector
# ============================================================================

class DriftDetector:
    """
    Monitors distribution shifts using statistical tests.
    
    Methods:
    - Kolmogorov-Smirnov test for feature distributions
    - Population Stability Index (PSI) for predictions
    - Performance metric tracking (winrate, sharpe, accuracy)
    
    Thresholds:
    - LOW: p-value < 0.10 (10% significance)
    - MEDIUM: p-value < 0.05 (5% significance)
    - HIGH: p-value < 0.01 (1% significance)
    - CRITICAL: p-value < 0.001 (0.1% significance)
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        performance_threshold: float = 0.15,
        reference_window_days: int = 30,
        current_window_days: int = 7,
    ):
        self.db = db_session
        self.event_bus = event_bus
        
        # Thresholds
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.performance_threshold = performance_threshold
        
        # Time windows
        self.reference_window_days = reference_window_days
        self.current_window_days = current_window_days
        
        logger.info(
            f"DriftDetector initialized: ks_threshold={ks_threshold}, "
            f"psi_threshold={psi_threshold}, reference_window={reference_window_days}d"
        )
    
    async def detect_feature_drift(
        self,
        feature_name: str,
        reference_data: pd.Series,
        current_data: pd.Series,
    ) -> DriftResult:
        """
        Detect drift in feature distribution using KS test.
        
        Args:
            feature_name: Name of the feature
            reference_data: Historical baseline data
            current_data: Recent data to compare
            
        Returns:
            DriftResult
        """
        # Remove NaN/inf
        reference = reference_data.replace([np.inf, -np.inf], np.nan).dropna()
        current = current_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(reference) < 30 or len(current) < 30:
            logger.warning(f"Insufficient data for {feature_name}: ref={len(reference)}, cur={len(current)}")
            return DriftResult(
                drift_type=DriftType.FEATURE,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.ks_threshold,
                feature_name=feature_name,
                notes="Insufficient data",
            )
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        
        # Determine severity
        severity = self._classify_severity(p_value, self.ks_threshold)
        trigger_retraining = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        
        # Compute statistics
        reference_stats = {
            "mean": float(reference.mean()),
            "std": float(reference.std()),
            "median": float(reference.median()),
            "min": float(reference.min()),
            "max": float(reference.max()),
        }
        
        current_stats = {
            "mean": float(current.mean()),
            "std": float(current.std()),
            "median": float(current.median()),
            "min": float(current.min()),
            "max": float(current.max()),
        }
        
        result = DriftResult(
            drift_type=DriftType.FEATURE,
            severity=severity,
            drift_score=ks_statistic,
            p_value=p_value,
            threshold=self.ks_threshold,
            feature_name=feature_name,
            reference_stats=reference_stats,
            current_stats=current_stats,
            trigger_retraining=trigger_retraining,
            notes=f"KS statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}",
        )
        
        if trigger_retraining:
            logger.warning(
                f"⚠️ {severity.value.upper()} feature drift detected: {feature_name} "
                f"(KS={ks_statistic:.4f}, p={p_value:.4f})"
            )
        
        return result
    
    async def detect_prediction_drift(
        self,
        model_type: ModelType,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
    ) -> DriftResult:
        """
        Detect drift in model predictions using PSI.
        
        Population Stability Index (PSI):
        - PSI < 0.1: No significant drift
        - PSI 0.1-0.2: Moderate drift
        - PSI > 0.2: Significant drift
        
        Args:
            model_type: Type of model
            reference_predictions: Historical predictions
            current_predictions: Recent predictions
            
        Returns:
            DriftResult
        """
        if len(reference_predictions) < 30 or len(current_predictions) < 30:
            logger.warning(f"Insufficient predictions for {model_type.value}")
            return DriftResult(
                drift_type=DriftType.PREDICTION,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.psi_threshold,
                model_type=model_type,
                notes="Insufficient data",
            )
        
        # Compute PSI
        psi = self._calculate_psi(reference_predictions, current_predictions)
        
        # Determine severity based on PSI
        if psi < 0.1:
            severity = DriftSeverity.LOW
        elif psi < 0.2:
            severity = DriftSeverity.MEDIUM
        elif psi < 0.3:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        trigger_retraining = severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        
        result = DriftResult(
            drift_type=DriftType.PREDICTION,
            severity=severity,
            drift_score=psi,
            p_value=0.0,  # PSI doesn't have p-value
            threshold=self.psi_threshold,
            model_type=model_type,
            trigger_retraining=trigger_retraining,
            notes=f"PSI: {psi:.4f}",
        )
        
        if trigger_retraining:
            logger.warning(
                f"⚠️ {severity.value.upper()} prediction drift detected: {model_type.value} "
                f"(PSI={psi:.4f})"
            )
        
        return result
    
    async def detect_performance_drift(
        self,
        model_type: ModelType,
        reference_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> DriftResult:
        """
        Detect drift in model performance metrics.
        
        Compares key metrics:
        - Accuracy / RMSE
        - Directional accuracy
        - Sharpe ratio
        
        Args:
            model_type: Type of model
            reference_metrics: Historical performance
            current_metrics: Recent performance
            
        Returns:
            DriftResult
        """
        # Calculate relative changes
        changes = {}
        for metric in ["accuracy", "rmse", "directional_accuracy", "sharpe_ratio"]:
            if metric in reference_metrics and metric in current_metrics:
                ref_val = reference_metrics[metric]
                cur_val = current_metrics[metric]
                
                if ref_val != 0:
                    # For RMSE, lower is better, so invert the change
                    if metric == "rmse":
                        change = (ref_val - cur_val) / ref_val
                    else:
                        change = (cur_val - ref_val) / abs(ref_val)
                    
                    changes[metric] = change
        
        if not changes:
            logger.warning(f"No comparable metrics for {model_type.value}")
            return DriftResult(
                drift_type=DriftType.PERFORMANCE,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                p_value=1.0,
                threshold=self.performance_threshold,
                model_type=model_type,
                notes="No comparable metrics",
            )
        
        # Average degradation
        avg_change = np.mean(list(changes.values()))
        drift_score = abs(avg_change)
        
        # Determine severity
        if drift_score < 0.05:
            severity = DriftSeverity.LOW
        elif drift_score < 0.10:
            severity = DriftSeverity.MEDIUM
        elif drift_score < 0.20:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        # Trigger retraining if performance degraded significantly
        trigger_retraining = (avg_change < -self.performance_threshold) and \
                            severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        
        result = DriftResult(
            drift_type=DriftType.PERFORMANCE,
            severity=severity,
            drift_score=drift_score,
            p_value=0.0,
            threshold=self.performance_threshold,
            model_type=model_type,
            reference_stats=reference_metrics,
            current_stats=current_metrics,
            trigger_retraining=trigger_retraining,
            notes=f"Avg change: {avg_change:.2%}, changes: {changes}",
        )
        
        if trigger_retraining:
            logger.warning(
                f"⚠️ {severity.value.upper()} performance drift detected: {model_type.value} "
                f"(degradation: {avg_change:.2%})"
            )
        
        return result
    
    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI = Σ (% current - % reference) * ln(% current / % reference)
        """
        # Create bins based on reference distribution
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicate bin edges
        
        if len(bins) < 2:
            return 0.0
        
        # Bin both distributions
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _classify_severity(
        self,
        p_value: float,
        threshold: float,
    ) -> DriftSeverity:
        """Classify drift severity based on p-value."""
        if p_value < 0.001:
            return DriftSeverity.CRITICAL
        elif p_value < 0.01:
            return DriftSeverity.HIGH
        elif p_value < threshold:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    async def store_drift_event(self, result: DriftResult):
        """Store drift detection result to database."""
        query = text("""
            INSERT INTO drift_events (
                drift_type, severity, drift_score, p_value, threshold,
                model_type, feature_name, detection_time,
                reference_stats, current_stats, trigger_retraining, notes
            ) VALUES (
                :drift_type, :severity, :drift_score, :p_value, :threshold,
                :model_type, :feature_name, :detection_time,
                :reference_stats, :current_stats, :trigger_retraining, :notes
            )
        """)
        
        import json
        await self.db.execute(query, {
            "drift_type": result.drift_type.value,
            "severity": result.severity.value,
            "drift_score": result.drift_score,
            "p_value": result.p_value,
            "threshold": result.threshold,
            "model_type": result.model_type.value if result.model_type else None,
            "feature_name": result.feature_name,
            "detection_time": result.detection_time,
            "reference_stats": json.dumps(result.reference_stats) if result.reference_stats else None,
            "current_stats": json.dumps(result.current_stats) if result.current_stats else None,
            "trigger_retraining": result.trigger_retraining,
            "notes": result.notes,
        })
        
        await self.db.commit()
        
        # Publish event
        if result.trigger_retraining:
            await self.event_bus.publish("learning.drift.detected", {
                "drift_type": result.drift_type.value,
                "severity": result.severity.value,
                "model_type": result.model_type.value if result.model_type else None,
                "feature_name": result.feature_name,
                "drift_score": result.drift_score,
                "trigger_retraining": result.trigger_retraining,
                "detection_time": result.detection_time.isoformat(),
            })
    
    async def get_recent_drift_events(
        self,
        days: int = 7,
        drift_type: Optional[DriftType] = None,
        model_type: Optional[ModelType] = None,
    ) -> List[Dict]:
        """Get recent drift events."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        conditions = ["detection_time >= :cutoff"]
        params = {"cutoff": cutoff}
        
        if drift_type:
            conditions.append("drift_type = :drift_type")
            params["drift_type"] = drift_type.value
        
        if model_type:
            conditions.append("model_type = :model_type")
            params["model_type"] = model_type.value
        
        where_clause = " AND ".join(conditions)
        
        query = text(f"""
            SELECT *
            FROM drift_events
            WHERE {where_clause}
            ORDER BY detection_time DESC
            LIMIT 100
        """)
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        events = []
        for row in rows:
            events.append({
                "id": row.id,
                "drift_type": row.drift_type,
                "severity": row.severity,
                "drift_score": float(row.drift_score),
                "p_value": float(row.p_value),
                "model_type": row.model_type,
                "feature_name": row.feature_name,
                "detection_time": row.detection_time.isoformat(),
                "trigger_retraining": row.trigger_retraining,
                "notes": row.notes,
            })
        
        return events


# ============================================================================
# Database Schema
# ============================================================================

async def create_drift_events_table(db_session: AsyncSession) -> None:
    """Create drift_events table."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS drift_events (
            id SERIAL PRIMARY KEY,
            drift_type VARCHAR(50) NOT NULL,
            severity VARCHAR(50) NOT NULL,
            drift_score DOUBLE PRECISION NOT NULL,
            p_value DOUBLE PRECISION NOT NULL,
            threshold DOUBLE PRECISION NOT NULL,
            model_type VARCHAR(50),
            feature_name VARCHAR(255),
            detection_time TIMESTAMP NOT NULL,
            reference_stats JSONB,
            current_stats JSONB,
            trigger_retraining BOOLEAN DEFAULT FALSE,
            notes TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_drift_time 
        ON drift_events(detection_time DESC);
        
        CREATE INDEX IF NOT EXISTS idx_drift_model 
        ON drift_events(model_type, detection_time DESC);
        
        CREATE INDEX IF NOT EXISTS idx_drift_trigger 
        ON drift_events(trigger_retraining, detection_time DESC) 
        WHERE trigger_retraining = TRUE;
    """)
    
    await db_session.execute(create_table_sql)
    await db_session.commit()
    logger.info("Created drift_events table")
