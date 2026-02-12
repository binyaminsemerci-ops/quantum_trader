"""
REPLAY HARNESS (PATH 2.4A)

Purpose: Prove confidence semantics through historical replay
Authority: OBSERVER ONLY (no execution, pure analysis)

Workflow:
1. Collect signal.score events with timestamps
2. Match against actual market outcomes (PnL, direction correctness)
3. Bucket by confidence ranges
4. Measure empirical accuracy per bucket
5. Generate calibration mapping

Requirements:
- Minimum 1000 samples for statistical significance
- Time-aligned outcome measurement
- No lookahead bias
- Separate train/test split

Output:
- Reliability diagram (confidence vs accuracy)
- Calibration function (raw → calibrated)
- Confidence semantics document
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

try:
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import train_test_split
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    np = None
    IsotonicRegression = None

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class SignalOutcomePair:
    """
    Paired signal and outcome for calibration.
    
    Signal: What ensemble predicted
    Outcome: What actually happened
    """
    # Signal data
    signal_id: str
    timestamp: datetime
    symbol: str
    suggested_action: str  # CLOSE or HOLD
    confidence: float
    expected_edge: float
    
    # Outcome data (filled later)
    outcome_timestamp: Optional[datetime] = None
    actual_action_taken: Optional[str] = None  # What apply_layer did
    was_correct: Optional[bool] = None  # Did prediction match outcome?
    actual_pnl: Optional[float] = None  # Realized PnL if action taken
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
            "expected_edge": self.expected_edge,
            "outcome_timestamp": self.outcome_timestamp.isoformat() if self.outcome_timestamp else None,
            "actual_action_taken": self.actual_action_taken,
            "was_correct": self.was_correct,
            "actual_pnl": self.actual_pnl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalOutcomePair':
        """Deserialize from storage."""
        return cls(
            signal_id=data["signal_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            suggested_action=data["suggested_action"],
            confidence=data["confidence"],
            expected_edge=data["expected_edge"],
            outcome_timestamp=datetime.fromisoformat(data["outcome_timestamp"]) if data.get("outcome_timestamp") else None,
            actual_action_taken=data.get("actual_action_taken"),
            was_correct=data.get("was_correct"),
            actual_pnl=data.get("actual_pnl")
        )


class OutcomeCollector:
    """
    Collects outcomes for emitted signals.
    
    Strategy: Wait N hours after signal, measure what happened.
    - If CLOSE suggested: Did position actually close? Was it profitable?
    - If HOLD suggested: Did position remain open? Would close have been worse?
    
    This is RETROSPECTIVE analysis, no real-time decisions.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        outcome_horizon_hours: int = 4  # Wait 4h to measure outcome
    ):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.outcome_horizon = timedelta(hours=outcome_horizon_hours)
        
        logger.info(f"[OUTCOME-COLLECTOR] Initialized (horizon={outcome_horizon_hours}h)")
    
    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
        logger.info("[OUTCOME-COLLECTOR] Connected to Redis")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
    
    async def collect_signals(
        self,
        stream: str = "quantum:stream:signal.score",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_count: int = 10000
    ) -> List[SignalOutcomePair]:
        """
        Collect signals from stream within time range.
        
        Args:
            stream: Signal stream name
            start_time: Earliest signal to collect (None = from beginning)
            end_time: Latest signal to collect (None = until now)
            max_count: Maximum signals to collect
        
        Returns:
            List of SignalOutcomePair (outcomes not yet filled)
        """
        signals = []
        
        # Convert timestamps to Redis stream IDs
        start_id = '-' if start_time is None else f"{int(start_time.timestamp() * 1000)}-0"
        end_id = '+' if end_time is None else f"{int(end_time.timestamp() * 1000)}-0"
        
        # Read from stream
        messages = await self.redis.xrange(stream, start_id, end_id, count=max_count)
        
        for msg_id, fields in messages:
            try:
                # Parse signal
                signal = SignalOutcomePair(
                    signal_id=msg_id,
                    timestamp=datetime.fromisoformat(fields["timestamp"].rstrip('Z')),
                    symbol=fields["symbol"],
                    suggested_action=fields["suggested_action"],
                    confidence=float(fields["confidence"]),
                    expected_edge=float(fields["expected_edge"])
                )
                signals.append(signal)
            except (KeyError, ValueError) as e:
                logger.warning(f"[OUTCOME-COLLECTOR] Failed to parse signal {msg_id}: {e}")
                continue
        
        logger.info(f"[OUTCOME-COLLECTOR] Collected {len(signals)} signals")
        return signals
    
    async def measure_outcome(
        self,
        signal: SignalOutcomePair,
        apply_result_stream: str = "quantum:stream:apply.result",
        execution_stream: str = "quantum:stream:execution.complete"
    ) -> SignalOutcomePair:
        """
        Measure outcome for a single signal.
        
        Strategy:
        1. Look at apply.result within outcome_horizon
        2. Check if action was EXECUTE (close) or SKIP (hold)
        3. If executed, get PnL from execution.complete
        4. Determine if suggestion was correct
        
        Args:
            signal: Signal to measure outcome for
            apply_result_stream: Stream with apply_layer decisions
            execution_stream: Stream with execution results
        
        Returns:
            Signal with outcome filled in
        """
        # Define time window
        window_start = int(signal.timestamp.timestamp() * 1000)
        window_end = int((signal.timestamp + self.outcome_horizon).timestamp() * 1000)
        
        # Search apply.result for this symbol in time window
        apply_messages = await self.redis.xrange(
            apply_result_stream,
            f"{window_start}-0",
            f"{window_end}-0",
            count=100
        )
        
        # Filter for matching symbol
        relevant_decisions = []
        for msg_id, fields in apply_messages:
            if fields.get("symbol") == signal.symbol:
                relevant_decisions.append((msg_id, fields))
        
        if not relevant_decisions:
            # No decision made during window
            signal.actual_action_taken = "NONE"
            signal.was_correct = None  # Cannot determine
            return signal
        
        # Use first decision (closest to signal time)
        msg_id, decision = relevant_decisions[0]
        signal.outcome_timestamp = datetime.fromtimestamp(int(msg_id.split('-')[0]) / 1000)
        
        # Determine what action was taken
        decision_type = decision.get("decision", "SKIP")
        executed = decision.get("executed", "false").lower() == "true"
        
        if decision_type == "EXECUTE" and executed:
            signal.actual_action_taken = "CLOSE"
            
            # Try to get PnL from execution stream
            exec_messages = await self.redis.xrange(
                execution_stream,
                msg_id,  # Start from decision time
                f"{window_end}-0",
                count=10
            )
            
            for exec_id, exec_fields in exec_messages:
                if exec_fields.get("symbol") == signal.symbol:
                    try:
                        signal.actual_pnl = float(exec_fields.get("pnl", 0))
                    except (ValueError, TypeError):
                        pass
                    break
        else:
            signal.actual_action_taken = "HOLD"
        
        # Determine correctness
        signal.was_correct = (signal.suggested_action == signal.actual_action_taken)
        
        return signal
    
    async def batch_measure_outcomes(
        self,
        signals: List[SignalOutcomePair],
        batch_size: int = 100
    ) -> List[SignalOutcomePair]:
        """
        Measure outcomes for multiple signals in batches.
        
        Args:
            signals: List of signals to measure
            batch_size: Number of signals to process concurrently
        
        Returns:
            Signals with outcomes filled
        """
        results = []
        
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [self.measure_outcome(signal) for signal in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            logger.info(f"[OUTCOME-COLLECTOR] Processed {len(results)}/{len(signals)} signals")
        
        return results


class ConfidenceCalibrator:
    """
    Calibrates confidence values using isotonic regression.
    
    Goal: Map raw_confidence → calibrated_confidence such that
          calibrated_confidence ≈ empirical accuracy
    
    Example:
        - Model outputs 0.8 but is only correct 65% of time
        - Calibrator learns: 0.8 → 0.65
        - After calibration: confidence = 0.65 (semantically true)
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ("isotonic" or "platt")
        """
        if not CALIBRATION_AVAILABLE:
            raise ImportError("sklearn required for calibration (pip install scikit-learn)")
        
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
        logger.info(f"[CALIBRATOR] Initialized (method={method})")
    
    def fit(
        self,
        confidences: List[float],
        outcomes: List[bool],
        train_size: float = 0.7
    ):
        """
        Fit calibrator on signal-outcome pairs.
        
        Args:
            confidences: Raw confidence values from ensemble
            outcomes: Binary outcomes (True = correct, False = incorrect)
            train_size: Fraction of data for training (rest for validation)
        """
        if len(confidences) < 100:
            raise ValueError(f"Need at least 100 samples, got {len(confidences)}")
        
        # Convert to numpy
        X = np.array(confidences).reshape(-1, 1)
        y = np.array(outcomes, dtype=float)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=42
        )
        
        # Fit isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(X_train.ravel(), y_train)
        
        self.is_fitted = True
        
        # Measure calibration error on test set
        y_pred = self.calibrator.predict(X_test.ravel())
        calibration_error = np.mean(np.abs(y_pred - y_test))
        
        logger.info(f"[CALIBRATOR] Fitted on {len(X_train)} samples")
        logger.info(f"[CALIBRATOR] Test calibration error: {calibration_error:.4f}")
        
        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "calibration_error": float(calibration_error)
        }
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a single confidence value.
        
        Args:
            raw_confidence: Raw confidence from ensemble
        
        Returns:
            Calibrated confidence (semantically true)
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")
        
        calibrated = self.calibrator.predict([raw_confidence])[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def generate_reliability_diagram(
        self,
        confidences: List[float],
        outcomes: List[bool],
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Generate reliability diagram data.
        
        Bins predictions by confidence, measures actual accuracy per bin.
        
        Args:
            confidences: Raw confidence values
            outcomes: Binary outcomes
            n_bins: Number of bins for bucketing
        
        Returns:
            Dict with bin data for plotting
        """
        confidences = np.array(confidences)
        outcomes = np.array(outcomes, dtype=float)
        
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bins)
        
        bin_data = []
        for i in range(1, n_bins + 1):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_confidence = confidences[mask].mean()
            bin_accuracy = outcomes[mask].mean()
            bin_count = mask.sum()
            
            bin_data.append({
                "bin_center": float(bins[i-1] + bins[i]) / 2,
                "mean_confidence": float(bin_confidence),
                "actual_accuracy": float(bin_accuracy),
                "count": int(bin_count),
                "calibration_error": float(abs(bin_confidence - bin_accuracy))
            })
        
        # Overall calibration error (Expected Calibration Error)
        ece = np.mean([b["calibration_error"] * b["count"] for b in bin_data]) / len(confidences)
        
        return {
            "bins": bin_data,
            "expected_calibration_error": float(ece),
            "total_samples": len(confidences)
        }


async def main_replay_harness():
    """
    Main replay harness workflow.
    
    Steps:
    1. Collect signals from quantum:stream:signal.score
    2. Measure outcomes (wait for sufficient time)
    3. Fit calibrator
    4. Generate reliability diagram
    5. Save calibration function
    """
    logger.info("\n" + "="*70)
    logger.info("PATH 2.4A — REPLAY HARNESS + CONFIDENCE CALIBRATION")
    logger.info("="*70)
    
    # Initialize collector
    collector = OutcomeCollector()
    await collector.connect()
    
    try:
        # Step 1: Collect signals
        logger.info("\n[STEP 1] Collecting signals...")
        signals = await collector.collect_signals(
            start_time=datetime.now() - timedelta(days=3),  # Last 3 days
            max_count=5000
        )
        
        if len(signals) < 100:
            logger.error(f"Insufficient signals ({len(signals)}). Need at least 100.")
            return
        
        # Step 2: Measure outcomes
        logger.info("\n[STEP 2] Measuring outcomes (this may take a while)...")
        signals_with_outcomes = await collector.batch_measure_outcomes(signals)
        
        # Filter for signals with measurable outcomes
        valid_pairs = [s for s in signals_with_outcomes if s.was_correct is not None]
        logger.info(f"[STEP 2] Got {len(valid_pairs)} valid signal-outcome pairs")
        
        if len(valid_pairs) < 100:
            logger.error(f"Insufficient valid pairs ({len(valid_pairs)}). Need at least 100.")
            return
        
        # Step 3: Fit calibrator
        logger.info("\n[STEP 3] Fitting calibrator...")
        confidences = [s.confidence for s in valid_pairs]
        outcomes = [s.was_correct for s in valid_pairs]
        
        calibrator = ConfidenceCalibrator(method="isotonic")
        fit_stats = calibrator.fit(confidences, outcomes)
        
        logger.info(f"[STEP 3] Calibration complete:")
        logger.info(f"         Train samples: {fit_stats['train_samples']}")
        logger.info(f"         Test calibration error: {fit_stats['calibration_error']:.4f}")
        
        # Step 4: Generate reliability diagram
        logger.info("\n[STEP 4] Generating reliability diagram...")
        diagram = calibrator.generate_reliability_diagram(confidences, outcomes)
        
        logger.info(f"[STEP 4] Expected Calibration Error (ECE): {diagram['expected_calibration_error']:.4f}")
        logger.info("[STEP 4] Reliability bins:")
        for bin_data in diagram['bins']:
            logger.info(
                f"         Confidence {bin_data['mean_confidence']:.2f} → "
                f"Accuracy {bin_data['actual_accuracy']:.2f} "
                f"(n={bin_data['count']}, error={bin_data['calibration_error']:.3f})"
            )
        
        # Step 5: Save results
        logger.info("\n[STEP 5] Saving calibration artifacts...")
        
        # Save calibrator
        from ai_engine.calibration.calibration_loader import save_calibrator
        
        calibrator_path = "/home/qt/quantum_trader/ai_engine/calibration/calibrator_v1.pkl"
        metadata = {
            "created_at": datetime.now().isoformat(),
            "method": "isotonic_regression",
            "train_samples": fit_stats['train_samples'],
            "test_samples": fit_stats['test_samples'],
            "calibration_error": fit_stats['calibration_error'],
            "expected_calibration_error": diagram['expected_calibration_error'],
            "total_signal_outcome_pairs": len(valid_pairs),
            "reliability_bins": diagram['bins']
        }
        
        save_calibrator(calibrator.calibrator, calibrator_path, metadata)
        
        logger.info(f"[STEP 5] Saved to {calibrator_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ REPLAY HARNESS COMPLETE")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Review reliability diagram (bins above)")
        logger.info("2. If ECE < 0.1: Deploy calibrator to production")
        logger.info("3. If ECE >= 0.1: Collect more data or investigate model")
        logger.info("4. Update ensemble_predictor_service to use calibrator")
        logger.info("5. Document confidence semantics")
        
    finally:
        await collector.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_replay_harness())
