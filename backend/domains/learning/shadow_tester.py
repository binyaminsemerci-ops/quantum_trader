"""
Shadow Testing - Parallel model evaluation and auto-promotion system.

Provides:
- Shadow model predictions alongside active models
- Performance comparison tracking
- Automatic promotion based on metrics
- EventBus integration for live signals
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.event_bus import EventBus
from backend.domains.learning.model_registry import ModelRegistry, ModelStatus, ModelType

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ShadowPrediction:
    """A single shadow model prediction."""
    
    prediction_id: str
    model_id: str
    model_type: ModelType
    timestamp: datetime
    symbol: str
    prediction_value: float
    confidence: Optional[float] = None
    features: Optional[Dict] = None
    
    # Outcome (filled after trade closes)
    actual_outcome: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "prediction_value": self.prediction_value,
            "confidence": self.confidence,
            "features": self.features,
            "actual_outcome": self.actual_outcome,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
        }


@dataclass
class ShadowTestResult:
    """Aggregated results from shadow testing."""
    
    model_id: str
    model_type: ModelType
    test_start: datetime
    test_end: datetime
    n_predictions: int
    
    # Metrics
    rmse: float
    mae: float
    directional_accuracy: float
    sharpe_ratio: float
    
    # Comparison with active model
    active_model_id: str
    rmse_improvement: float  # Percentage improvement
    mae_improvement: float
    directional_accuracy_improvement: float
    sharpe_improvement: float
    
    # Status
    ready_for_promotion: bool = False
    promoted: bool = False
    promoted_at: Optional[datetime] = None
    notes: Optional[str] = None


# ============================================================================
# Shadow Tester
# ============================================================================

class ShadowTester:
    """
    Evaluates shadow models in parallel with active models.
    
    Workflow:
    1. Subscribe to ai.signal.generated events
    2. For each signal, run both ACTIVE and SHADOW models
    3. Record predictions
    4. When trades close, record actual outcomes
    5. Aggregate metrics and compare
    6. Auto-promote if shadow model outperforms
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        event_bus: EventBus,
        model_registry: ModelRegistry,
        min_predictions: int = 100,
        promotion_threshold: float = 0.05,  # 5% improvement required
        test_duration_days: int = 7,
    ):
        self.db = db_session
        self.event_bus = event_bus
        self.registry = model_registry
        self.min_predictions = min_predictions
        self.promotion_threshold = promotion_threshold
        self.test_duration_days = test_duration_days
        
        self.running = False
        
        logger.info(
            f"ShadowTester initialized: min_predictions={min_predictions}, "
            f"promotion_threshold={promotion_threshold:.1%}, "
            f"test_duration={test_duration_days}d"
        )
    
    async def start(self):
        """Start shadow testing (subscribe to events)."""
        if self.running:
            logger.warning("ShadowTester already running")
            return
        
        self.running = True
        
        # Subscribe to trading signals (synchronous subscription)
        self.event_bus.subscribe("ai.signal.generated", self._on_signal_generated)
        
        # Subscribe to trade closures (synchronous subscription)
        self.event_bus.subscribe("execution.trade.closed", self._on_trade_closed)
        
        logger.info("ShadowTester started")
    
    async def stop(self):
        """Stop shadow testing."""
        self.running = False
        logger.info("ShadowTester stopped")
    
    async def _on_signal_generated(self, event: Dict):
        """
        Handle ai.signal.generated event.
        
        Event payload:
        {
            "symbol": "BTCUSDT",
            "signal": "LONG" / "SHORT",
            "confidence": 0.75,
            "model_predictions": {...},
            "features": {...}
        }
        """
        try:
            symbol = event.get("symbol")
            if not symbol:
                return
            
            # Get active and shadow models for each type
            for model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.NHITS, ModelType.PATCHTST]:
                active = await self.registry.get_active_model(model_type)
                shadow = await self.registry.get_shadow_model(model_type)
                
                if not active or not shadow:
                    continue
                
                # Record shadow prediction (active already recorded by ai_trading_engine)
                prediction = ShadowPrediction(
                    prediction_id=f"{shadow.model_id}_{symbol}_{int(datetime.utcnow().timestamp())}",
                    model_id=shadow.model_id,
                    model_type=model_type,
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    prediction_value=event["model_predictions"].get(model_type.value, 0.0),
                    confidence=event.get("confidence"),
                    features=event.get("features"),
                )
                
                await self._store_prediction(prediction)
        
        except Exception as e:
            logger.error(f"Error in _on_signal_generated: {e}", exc_info=True)
    
    async def _on_trade_closed(self, event: Dict):
        """
        Handle execution.trade.closed event.
        
        Event payload:
        {
            "symbol": "BTCUSDT",
            "pnl_percent": 1.25,
            "closed_at": "2024-01-15T10:30:00Z",
            ...
        }
        """
        try:
            symbol = event.get("symbol")
            pnl_percent = event.get("pnl_percent")
            
            if not symbol or pnl_percent is None:
                return
            
            # Update predictions with actual outcome
            await self._record_outcome(
                symbol=symbol,
                actual_outcome=pnl_percent,
                outcome_timestamp=datetime.fromisoformat(event.get("closed_at", datetime.utcnow().isoformat())),
            )
        
        except Exception as e:
            logger.error(f"Error in _on_trade_closed: {e}", exc_info=True)
    
    async def _store_prediction(self, prediction: ShadowPrediction):
        """Store shadow prediction to database."""
        query = text("""
            INSERT INTO shadow_test_results (
                prediction_id, model_id, model_type, timestamp, symbol,
                prediction_value, confidence, features
            ) VALUES (
                :prediction_id, :model_id, :model_type, :timestamp, :symbol,
                :prediction_value, :confidence, :features
            )
            ON CONFLICT (prediction_id) DO NOTHING
        """)
        
        import json
        await self.db.execute(query, {
            "prediction_id": prediction.prediction_id,
            "model_id": prediction.model_id,
            "model_type": prediction.model_type.value,
            "timestamp": prediction.timestamp,
            "symbol": prediction.symbol,
            "prediction_value": prediction.prediction_value,
            "confidence": prediction.confidence,
            "features": json.dumps(prediction.features) if prediction.features else None,
        })
        
        await self.db.commit()
    
    async def _record_outcome(
        self,
        symbol: str,
        actual_outcome: float,
        outcome_timestamp: datetime,
    ):
        """Update predictions with actual outcomes."""
        # Find recent predictions for this symbol (last 1 hour)
        cutoff = outcome_timestamp - timedelta(hours=1)
        
        query = text("""
            UPDATE shadow_test_results
            SET 
                actual_outcome = :actual_outcome,
                outcome_timestamp = :outcome_timestamp
            WHERE 
                symbol = :symbol
                AND timestamp >= :cutoff
                AND actual_outcome IS NULL
        """)
        
        result = await self.db.execute(query, {
            "actual_outcome": actual_outcome,
            "outcome_timestamp": outcome_timestamp,
            "symbol": symbol,
            "cutoff": cutoff,
        })
        
        await self.db.commit()
        
        logger.debug(f"Recorded outcome for {symbol}: {actual_outcome:.2f}% ({result.rowcount} predictions updated)")
    
    async def evaluate_shadow_model(
        self,
        model_type: ModelType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[ShadowTestResult]:
        """
        Evaluate shadow model performance vs active model.
        
        Args:
            model_type: Type of model to evaluate
            start_date: Start of evaluation period (default: test_duration_days ago)
            end_date: End of evaluation period (default: now)
            
        Returns:
            ShadowTestResult or None
        """
        if not end_date:
            end_date = datetime.utcnow()
        
        if not start_date:
            start_date = end_date - timedelta(days=self.test_duration_days)
        
        # Get active and shadow models
        active = await self.registry.get_active_model(model_type)
        shadow = await self.registry.get_shadow_model(model_type)
        
        if not active or not shadow:
            logger.warning(f"Cannot evaluate {model_type.value}: missing active or shadow model")
            return None
        
        # Get predictions with outcomes
        query = text("""
            SELECT 
                model_id,
                prediction_value,
                actual_outcome
            FROM shadow_test_results
            WHERE 
                model_type = :model_type
                AND model_id IN (:active_id, :shadow_id)
                AND timestamp BETWEEN :start_date AND :end_date
                AND actual_outcome IS NOT NULL
        """)
        
        result = await self.db.execute(query, {
            "model_type": model_type.value,
            "active_id": active.model_id,
            "shadow_id": shadow.model_id,
            "start_date": start_date,
            "end_date": end_date,
        })
        
        rows = result.fetchall()
        
        if not rows:
            logger.warning(f"No predictions with outcomes found for {model_type.value}")
            return None
        
        # Separate active and shadow predictions
        active_preds = []
        active_outcomes = []
        shadow_preds = []
        shadow_outcomes = []
        
        for row in rows:
            if row.model_id == active.model_id:
                active_preds.append(row.prediction_value)
                active_outcomes.append(row.actual_outcome)
            else:
                shadow_preds.append(row.prediction_value)
                shadow_outcomes.append(row.actual_outcome)
        
        if len(shadow_preds) < self.min_predictions:
            logger.info(
                f"Shadow model {shadow.model_id} has only {len(shadow_preds)} predictions "
                f"(minimum: {self.min_predictions})"
            )
            return None
        
        # Calculate metrics
        shadow_metrics = self._calculate_metrics(shadow_preds, shadow_outcomes)
        active_metrics = self._calculate_metrics(active_preds, active_outcomes)
        
        # Calculate improvements
        rmse_improvement = (active_metrics["rmse"] - shadow_metrics["rmse"]) / active_metrics["rmse"]
        mae_improvement = (active_metrics["mae"] - shadow_metrics["mae"]) / active_metrics["mae"]
        directional_improvement = (shadow_metrics["directional_accuracy"] - active_metrics["directional_accuracy"]) / active_metrics["directional_accuracy"]
        sharpe_improvement = (shadow_metrics["sharpe_ratio"] - active_metrics["sharpe_ratio"]) / abs(active_metrics["sharpe_ratio"]) if active_metrics["sharpe_ratio"] != 0 else 0
        
        # Check if ready for promotion
        ready = (
            rmse_improvement >= self.promotion_threshold and
            mae_improvement >= self.promotion_threshold and
            directional_improvement >= self.promotion_threshold and
            sharpe_improvement >= self.promotion_threshold
        )
        
        result = ShadowTestResult(
            model_id=shadow.model_id,
            model_type=model_type,
            test_start=start_date,
            test_end=end_date,
            n_predictions=len(shadow_preds),
            rmse=shadow_metrics["rmse"],
            mae=shadow_metrics["mae"],
            directional_accuracy=shadow_metrics["directional_accuracy"],
            sharpe_ratio=shadow_metrics["sharpe_ratio"],
            active_model_id=active.model_id,
            rmse_improvement=rmse_improvement,
            mae_improvement=mae_improvement,
            directional_accuracy_improvement=directional_improvement,
            sharpe_improvement=sharpe_improvement,
            ready_for_promotion=ready,
        )
        
        logger.info(
            f"Shadow evaluation for {model_type.value}:\n"
            f"  Predictions: {len(shadow_preds)}\n"
            f"  RMSE: {shadow_metrics['rmse']:.4f} (active: {active_metrics['rmse']:.4f}, improvement: {rmse_improvement:.1%})\n"
            f"  MAE: {shadow_metrics['mae']:.4f} (active: {active_metrics['mae']:.4f}, improvement: {mae_improvement:.1%})\n"
            f"  Directional: {shadow_metrics['directional_accuracy']:.1%} (active: {active_metrics['directional_accuracy']:.1%}, improvement: {directional_improvement:.1%})\n"
            f"  Sharpe: {shadow_metrics['sharpe_ratio']:.2f} (active: {active_metrics['sharpe_ratio']:.2f}, improvement: {sharpe_improvement:.1%})\n"
            f"  Ready for promotion: {ready}"
        )
        
        return result
    
    def _calculate_metrics(
        self,
        predictions: List[float],
        outcomes: List[float],
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        preds = np.array(predictions)
        actual = np.array(outcomes)
        
        # RMSE
        rmse = np.sqrt(np.mean((preds - actual) ** 2))
        
        # MAE
        mae = np.mean(np.abs(preds - actual))
        
        # Directional accuracy
        directional = np.mean(np.sign(preds) == np.sign(actual))
        
        # Sharpe ratio
        if np.std(preds) > 0:
            sharpe = np.mean(preds) / np.std(preds) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": directional,
            "sharpe_ratio": sharpe,
        }
    
    async def promote_if_ready(
        self,
        model_type: ModelType,
    ) -> bool:
        """
        Evaluate and promote shadow model if ready.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if promotion occurred
        """
        result = await self.evaluate_shadow_model(model_type)
        
        if not result:
            return False
        
        if not result.ready_for_promotion:
            logger.info(f"Shadow model {result.model_id} not ready for promotion")
            return False
        
        # Promote
        success = await self.registry.promote_shadow_to_active(model_type)
        
        if success:
            # Publish event
            await self.event_bus.publish("learning.model.promoted", {
                "model_id": result.model_id,
                "model_type": model_type.value,
                "active_model_id": result.active_model_id,
                "metrics": {
                    "rmse": result.rmse,
                    "mae": result.mae,
                    "directional_accuracy": result.directional_accuracy,
                    "sharpe_ratio": result.sharpe_ratio,
                },
                "improvements": {
                    "rmse": result.rmse_improvement,
                    "mae": result.mae_improvement,
                    "directional_accuracy": result.directional_accuracy_improvement,
                    "sharpe": result.sharpe_improvement,
                },
                "promoted_at": datetime.utcnow().isoformat(),
            })
            
            logger.info(f"✅ Promoted shadow model to active: {result.model_id}")
        
        return success
    
    async def get_shadow_test_summary(
        self,
        model_type: Optional[ModelType] = None,
        days: int = 30,
    ) -> List[Dict]:
        """
        Get summary of recent shadow testing activity.
        
        Args:
            model_type: Filter by model type
            days: Number of days to include
            
        Returns:
            List of summary dictionaries
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        conditions = ["timestamp >= :cutoff"]
        params = {"cutoff": cutoff}
        
        if model_type:
            conditions.append("model_type = :model_type")
            params["model_type"] = model_type.value
        
        where_clause = " AND ".join(conditions)
        
        query = text(f"""
            SELECT 
                model_id,
                model_type,
                COUNT(*) as total_predictions,
                COUNT(actual_outcome) as predictions_with_outcomes,
                AVG(CASE WHEN actual_outcome IS NOT NULL THEN ABS(prediction_value - actual_outcome) END) as avg_error,
                MIN(timestamp) as first_prediction,
                MAX(timestamp) as last_prediction
            FROM shadow_test_results
            WHERE {where_clause}
            GROUP BY model_id, model_type
            ORDER BY last_prediction DESC
        """)
        
        result = await self.db.execute(query, params)
        rows = result.fetchall()
        
        summaries = []
        for row in rows:
            summaries.append({
                "model_id": row.model_id,
                "model_type": row.model_type,
                "total_predictions": row.total_predictions,
                "predictions_with_outcomes": row.predictions_with_outcomes,
                "avg_error": float(row.avg_error) if row.avg_error else None,
                "first_prediction": row.first_prediction.isoformat() if row.first_prediction else None,
                "last_prediction": row.last_prediction.isoformat() if row.last_prediction else None,
            })
        
        return summaries


# ============================================================================
# Database Schema
# ============================================================================

async def create_shadow_test_results_table(db_session: AsyncSession) -> None:
    """Create shadow_test_results table."""
    
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS shadow_test_results (
            prediction_id VARCHAR(255) PRIMARY KEY,
            model_id VARCHAR(255) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(50) NOT NULL,
            prediction_value DOUBLE PRECISION NOT NULL,
            confidence DOUBLE PRECISION,
            features JSONB,
            actual_outcome DOUBLE PRECISION,
            outcome_timestamp TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
        );
        
        CREATE INDEX IF NOT EXISTS idx_shadow_model_timestamp 
        ON shadow_test_results(model_id, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_shadow_symbol_timestamp 
        ON shadow_test_results(symbol, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_shadow_outcome 
        ON shadow_test_results(outcome_timestamp DESC) 
        WHERE actual_outcome IS NOT NULL;
    """)
    
    await db_session.execute(create_table_sql)
    await db_session.commit()
    logger.info("Created shadow_test_results table")
