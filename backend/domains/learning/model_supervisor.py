"""
Model Supervisor - Monitors model performance, calibration, and bias.

Provides:
- Winrate tracking with confidence intervals
- Calibration monitoring (predicted vs actual probabilities)
- Bias detection (directional, volatility, regime-specific)
- Performance degradation alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
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
# Data Structures
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Model performance metrics over a time period."""
    
    model_id: str
    model_type: ModelType
    period_start: datetime
    period_end: datetime
    
    # Winrate
    n_trades: int
    n_wins: int
    n_losses: int
    winrate: float
    winrate_ci_lower: float  # 95% confidence interval
    winrate_ci_upper: float
    
    # Returns
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    
    # Accuracy metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    
    # Calibration
    calibration_error: Optional[float] = None  # Mean absolute calibration error
    
    # Bias indicators
    directional_bias: Optional[float] = None  # -1 (bearish) to +1 (bullish)
    volatility_bias: Optional[float] = None  # Overpredict or underpredict volatility
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "n_trades": self.n_trades,
            "winrate": self.winrate,
            "winrate_ci": [self.winrate_ci_lower, self.winrate_ci_upper],
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "accuracy": self.accuracy,
            "calibration_error": self.calibration_error,
            "directional_bias": self.directional_bias,
            "volatility_bias": self.volatility_bias,
        }


# ============================================================================
# Model Supervisor
# ============================================================================

class ModelSupervisor:
    """
    Monitors model performance, calibration, and bias.
    
    Functions:
    - Track winrate with statistical confidence intervals
    - Calibration: Compare predicted probabilities vs actual outcomes
    - Bias detection: Identify systematic prediction errors
    - Alert on performance degradation
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        winrate_alert_threshold: float = 0.45,
        calibration_error_threshold: float = 0.10,
        bias_threshold: float = 0.15,
    ):
        self.db = db_session
        self.event_bus = event_bus
        
        # Alert thresholds
        self.winrate_alert_threshold = winrate_alert_threshold
        self.calibration_error_threshold = calibration_error_threshold
        self.bias_threshold = bias_threshold
        
        logger.info(
            f"ModelSupervisor initialized: winrate_alert={winrate_alert_threshold}, "
            f"calibration_threshold={calibration_error_threshold}"
        )
    
    async def compute_performance_metrics(
        self,
        model_type: ModelType,
        model_id: str,
        period_days: int = 7,
    ) -> Optional[PerformanceMetrics]:
        """
        Compute comprehensive performance metrics for a model.
        
        Args:
            model_type: Type of model
            model_id: Model identifier
            period_days: Number of days to analyze
            
        Returns:
            PerformanceMetrics or None
        """
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=period_days)
        
        # Query shadow test results with outcomes
        query = text("""
            SELECT 
                prediction_value,
                confidence,
                actual_outcome
            FROM shadow_test_results
            WHERE 
                model_id = :model_id
                AND timestamp BETWEEN :start_date AND :end_date
                AND actual_outcome IS NOT NULL
        """)
        
        result = await self.db.execute(query, {
            "model_id": model_id,
            "start_date": period_start,
            "end_date": period_end,
        })
        
        rows = result.fetchall()
        
        if not rows or len(rows) < 10:
            logger.warning(f"Insufficient data for {model_id}: {len(rows) if rows else 0} trades")
            return None
        
        # Extract data
        predictions = np.array([row.prediction_value for row in rows])
        confidences = np.array([row.confidence if row.confidence else 0.5 for row in rows])
        outcomes = np.array([row.actual_outcome for row in rows])
        
        # Winrate
        wins = outcomes > 0
        n_wins = int(wins.sum())
        n_losses = int((~wins).sum())
        n_trades = len(outcomes)
        winrate = n_wins / n_trades if n_trades > 0 else 0.0
        
        # Confidence interval (binomial proportion)
        winrate_ci_lower, winrate_ci_upper = self._binomial_ci(n_wins, n_trades)
        
        # Returns
        win_outcomes = outcomes[wins]
        loss_outcomes = outcomes[~wins]
        
        avg_win = float(win_outcomes.mean()) if len(win_outcomes) > 0 else 0.0
        avg_loss = float(loss_outcomes.mean()) if len(loss_outcomes) > 0 else 0.0
        
        # Profit factor
        total_wins = float(win_outcomes.sum()) if len(win_outcomes) > 0 else 0.0
        total_losses = abs(float(loss_outcomes.sum())) if len(loss_outcomes) > 0 else 0.01
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Sharpe ratio
        if np.std(outcomes) > 0:
            sharpe = np.mean(outcomes) / np.std(outcomes) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Accuracy (directional)
        directional_correct = (np.sign(predictions) == np.sign(outcomes))
        accuracy = float(directional_correct.mean())
        
        # Precision/Recall/F1 (for classification)
        predicted_positive = predictions > 0
        actual_positive = outcomes > 0
        
        tp = int((predicted_positive & actual_positive).sum())
        fp = int((predicted_positive & ~actual_positive).sum())
        fn = int((~predicted_positive & actual_positive).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calibration error
        calibration_error = self._calculate_calibration_error(confidences, wins)
        
        # Bias
        directional_bias = float(np.mean(np.sign(predictions)))
        volatility_bias = self._calculate_volatility_bias(predictions, outcomes)
        
        metrics = PerformanceMetrics(
            model_id=model_id,
            model_type=model_type,
            period_start=period_start,
            period_end=period_end,
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            winrate=winrate,
            winrate_ci_lower=winrate_ci_lower,
            winrate_ci_upper=winrate_ci_upper,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            calibration_error=calibration_error,
            directional_bias=directional_bias,
            volatility_bias=volatility_bias,
        )
        
        # Check for alerts
        await self._check_performance_alerts(metrics)
        
        # Store metrics
        await self._store_metrics(metrics)
        
        return metrics
    
    def _binomial_ci(
        self,
        n_success: int,
        n_trials: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate binomial confidence interval using Wilson score.
        
        Returns:
            (lower_bound, upper_bound)
        """
        if n_trials == 0:
            return 0.0, 0.0
        
        p = n_success / n_trials
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / n_trials
        center = (p + z**2 / (2 * n_trials)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n_trials + z**2 / (4 * n_trials**2)) / denominator
        
        return (
            max(0.0, center - margin),
            min(1.0, center + margin)
        )
    
    def _calculate_calibration_error(
        self,
        confidences: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate mean absolute calibration error.
        
        Calibration: Do predicted probabilities match actual frequencies?
        """
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        calibration_errors = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_confidence = confidences[mask].mean()
            bin_accuracy = outcomes[mask].mean()
            
            error = abs(bin_confidence - bin_accuracy)
            calibration_errors.append(error)
        
        return float(np.mean(calibration_errors)) if calibration_errors else 0.0
    
    def _calculate_volatility_bias(
        self,
        predictions: np.ndarray,
        outcomes: np.ndarray,
    ) -> float:
        """
        Calculate volatility bias.
        
        Positive: Overpredicting volatility (predictions more extreme than outcomes)
        Negative: Underpredicting volatility
        """
        pred_std = np.std(predictions)
        outcome_std = np.std(outcomes)
        
        if outcome_std > 0:
            bias = (pred_std - outcome_std) / outcome_std
        else:
            bias = 0.0
        
        return float(bias)
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if performance metrics trigger alerts."""
        alerts = []
        
        # Winrate alert
        if metrics.winrate < self.winrate_alert_threshold:
            alerts.append({
                "type": "low_winrate",
                "severity": "high",
                "message": f"Winrate {metrics.winrate:.1%} below threshold {self.winrate_alert_threshold:.1%}",
            })
        
        # Calibration alert
        if metrics.calibration_error and metrics.calibration_error > self.calibration_error_threshold:
            alerts.append({
                "type": "poor_calibration",
                "severity": "medium",
                "message": f"Calibration error {metrics.calibration_error:.2%} above threshold {self.calibration_error_threshold:.2%}",
            })
        
        # Directional bias alert
        if metrics.directional_bias and abs(metrics.directional_bias) > self.bias_threshold:
            bias_direction = "bullish" if metrics.directional_bias > 0 else "bearish"
            alerts.append({
                "type": "directional_bias",
                "severity": "medium",
                "message": f"Strong {bias_direction} bias detected: {metrics.directional_bias:.2f}",
            })
        
        # Volatility bias alert
        if metrics.volatility_bias and abs(metrics.volatility_bias) > self.bias_threshold:
            bias_type = "overpredicting" if metrics.volatility_bias > 0 else "underpredicting"
            alerts.append({
                "type": "volatility_bias",
                "severity": "low",
                "message": f"{bias_type.capitalize()} volatility: {metrics.volatility_bias:.2%}",
            })
        
        # Publish alerts
        if alerts:
            logger.warning(
                f"⚠️ Performance alerts for {metrics.model_id}: "
                + ", ".join([a["message"] for a in alerts])
            )
            
            await self.event_bus.publish("learning.performance.alert", {
                "model_id": metrics.model_id,
                "model_type": metrics.model_type.value,
                "alerts": alerts,
                "metrics": metrics.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            })
    
    async def _store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics to database."""
        query = text("""
            INSERT INTO model_performance_logs (
                model_id, model_type, period_start, period_end,
                n_trades, winrate, winrate_ci_lower, winrate_ci_upper,
                avg_win, avg_loss, profit_factor, sharpe_ratio,
                accuracy, precision, recall, f1,
                calibration_error, directional_bias, volatility_bias
            ) VALUES (
                :model_id, :model_type, :period_start, :period_end,
                :n_trades, :winrate, :winrate_ci_lower, :winrate_ci_upper,
                :avg_win, :avg_loss, :profit_factor, :sharpe_ratio,
                :accuracy, :precision, :recall, :f1,
                :calibration_error, :directional_bias, :volatility_bias
            )
        """)
        
        await self.db.execute(query, {
            "model_id": metrics.model_id,
            "model_type": metrics.model_type.value,
            "period_start": metrics.period_start,
            "period_end": metrics.period_end,
            "n_trades": metrics.n_trades,
            "winrate": metrics.winrate,
            "winrate_ci_lower": metrics.winrate_ci_lower,
            "winrate_ci_upper": metrics.winrate_ci_upper,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "calibration_error": metrics.calibration_error,
            "directional_bias": metrics.directional_bias,
            "volatility_bias": metrics.volatility_bias,
        })
        
        await self.db.commit()
    
    async def get_performance_history(
        self,
        model_id: str,
        days: int = 30,
    ) -> List[Dict]:
        """Get historical performance metrics."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT *
            FROM model_performance_logs
            WHERE 
                model_id = :model_id
                AND period_end >= :cutoff
            ORDER BY period_end DESC
        """)
        
        result = await self.db.execute(query, {
            "model_id": model_id,
            "cutoff": cutoff,
        })
        
        rows = result.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "period_start": row.period_start.isoformat(),
                "period_end": row.period_end.isoformat(),
                "n_trades": row.n_trades,
                "winrate": float(row.winrate),
                "profit_factor": float(row.profit_factor),
                "sharpe_ratio": float(row.sharpe_ratio),
                "calibration_error": float(row.calibration_error) if row.calibration_error else None,
            })
        
        return history


# ============================================================================
# Database Schema
# ============================================================================

async def create_model_performance_logs_table(db_session: AsyncSession) -> None:
    """Create model_performance_logs table."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS model_performance_logs (
            id SERIAL PRIMARY KEY,
            model_id VARCHAR(255) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            period_start TIMESTAMP NOT NULL,
            period_end TIMESTAMP NOT NULL,
            n_trades INT NOT NULL,
            winrate DOUBLE PRECISION NOT NULL,
            winrate_ci_lower DOUBLE PRECISION NOT NULL,
            winrate_ci_upper DOUBLE PRECISION NOT NULL,
            avg_win DOUBLE PRECISION NOT NULL,
            avg_loss DOUBLE PRECISION NOT NULL,
            profit_factor DOUBLE PRECISION NOT NULL,
            sharpe_ratio DOUBLE PRECISION NOT NULL,
            accuracy DOUBLE PRECISION,
            precision DOUBLE PRECISION,
            recall DOUBLE PRECISION,
            f1 DOUBLE PRECISION,
            calibration_error DOUBLE PRECISION,
            directional_bias DOUBLE PRECISION,
            volatility_bias DOUBLE PRECISION,
            FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_perf_model_period 
        ON model_performance_logs(model_id, period_end DESC);
        
        CREATE INDEX IF NOT EXISTS idx_perf_type_period 
        ON model_performance_logs(model_type, period_end DESC);
    """)
    
    await db_session.execute(create_table_sql)
    await db_session.commit()
    logger.info("Created model_performance_logs table")
