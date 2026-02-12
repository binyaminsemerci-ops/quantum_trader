#!/usr/bin/env python3
"""
PATH 2.4A — Calibration Using Existing AI Signals

Uses existing quantum:stream:ai.signal_generated data for calibration.
No need to wait for new signals - 8574 signals already available!

Key insight: Existing AI engine already produces confidence values
(stored in ai.signal_generated stream). We can calibrate these immediately.
"""
import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import redis.asyncio as aioredis
import numpy as np
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExistingSignalOutcomePair:
    """Signal-outcome pair from existing AI system."""
    signal_id: str
    timestamp: datetime
    symbol: str
    action: str  # "buy" or "sell"
    confidence: float
    
    # Outcome (measured retrospectively)
    outcome_timestamp: Optional[datetime] = None
    actual_result: Optional[str] = None  # "profit", "loss", "neutral"
    was_correct: Optional[bool] = None
    actual_pnl: Optional[float] = None


class ExistingSignalCollector:
    """Collects signals from ai.signal_generated stream."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=True
        )
        logger.info("✅ Connected to Redis")
        
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            
    async def collect_signals(
        self,
        start_time: Optional[datetime] = None,
        max_count: int = 5000
    ) -> List[ExistingSignalOutcomePair]:
        """
        Collect signals from ai.signal_generated stream.
        
        Args:
            start_time: Collect signals after this time (default: 3 days ago)
            max_count: Maximum number of signals to collect
            
        Returns:
            List of signal-outcome pairs (outcomes not yet measured)
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=3)
            
        logger.info(f"Collecting signals from ai.signal_generated (after {start_time}, max {max_count})...")
        
        # Convert datetime to Redis stream ID
        start_ms = int(start_time.timestamp() * 1000)
        start_id = f"{start_ms}-0"
        
        # Read from stream
        signals = []
        
        try:
            results = await self.redis.xrevrange(
                "quantum:stream:ai.signal_generated",
                "+",  # Latest
                start_id,  # Start from this ID
                count=max_count
            )
            
            logger.info(f"Retrieved {len(results)} raw entries")
            
            for entry_id, fields in results:
                try:
                    # Parse payload
                    payload_json = fields.get("payload", "{}")
                    payload = json.loads(payload_json)
                    
                    # Extract signal data
                    symbol = payload.get("symbol", "UNKNOWN")
                    action = payload.get("action", "hold")
                    confidence = payload.get("confidence", 0.5)
                    
                    # Parse timestamp
                    ts_str = payload.get("timestamp")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        # Fallback: use stream entry ID
                        ts_ms = int(entry_id.split("-")[0])
                        ts = datetime.fromtimestamp(ts_ms / 1000)
                    
                    signal = ExistingSignalOutcomePair(
                        signal_id=entry_id,
                        timestamp=ts,
                        symbol=symbol,
                        action=action,
                        confidence=float(confidence)
                    )
                    
                    signals.append(signal)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse signal {entry_id}: {e}")
                    continue
            
            logger.info(f"✅ Collected {len(signals)} signals")
            return signals
            
        except Exception as e:
            logger.error(f"Failed to collect signals: {e}")
            return []
    
    async def measure_outcome(
        self,
        signal: ExistingSignalOutcomePair,
        horizon_hours: int = 4
    ) -> Optional[ExistingSignalOutcomePair]:
        """
        Measure outcome for a single signal by looking at:
        1. apply.result stream (was signal used?)
        2. execution.result stream (what was the trade result?)
        3. trade.closed stream (what was the final PnL?)
        
        Args:
            signal: Signal to measure
            horizon_hours: Time horizon for outcome measurement
            
        Returns:
            Signal with outcome measured (or None if insufficient data)
        """
        try:
            # Calculate outcome window
            end_time = signal.timestamp + timedelta(hours=horizon_hours)
            
            # Check if enough time has passed
            if datetime.now() < end_time:
                return None  # Too early to measure
            
            # Convert to stream IDs
            start_ms = int(signal.timestamp.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            start_id = f"{start_ms}-0"
            end_id = f"{end_ms}-0"
            
            # Look for related trades in execution.result stream
            exec_results = await self.redis.xrange(
                "quantum:stream:execution.result",
                start_id,
                end_id,
                count=100
            )
            
            # Look for closed trades
            closed_trades = await self.redis.xrange(
                "quantum:stream:trade.closed",
                start_id,
                end_id,
                count=100
            )
            
            # Simple heuristic: if signal said "sell" and we see profit > 0, signal was correct
            # This is simplified - real logic would correlate by symbol and exact timing
            
            total_pnl = 0.0
            trade_count = 0
            
            for entry_id, fields in closed_trades:
                try:
                    pnl_str = fields.get("realized_pnl", "0")
                    pnl = float(pnl_str)
                    total_pnl += pnl
                    trade_count += 1
                except:
                    continue
            
            if trade_count == 0:
                return None  # No trades found
            
            avg_pnl = total_pnl / trade_count
            
            # Determine if signal was "correct"
            # Simplified logic: if confidence > 0.5 and PnL > 0, correct
            # This is a placeholder - real logic needs proper correlation
            was_correct = (signal.confidence > 0.5 and avg_pnl > 0) or \
                          (signal.confidence <= 0.5 and avg_pnl <= 0)
            
            signal.outcome_timestamp = end_time
            signal.actual_pnl = avg_pnl
            signal.was_correct = was_correct
            signal.actual_result = "profit" if avg_pnl > 0 else "loss" if avg_pnl < 0 else "neutral"
            
            return signal
            
        except Exception as e:
            logger.warning(f"Failed to measure outcome for {signal.signal_id}: {e}")
            return None
    
    async def batch_measure_outcomes(
        self,
        signals: List[ExistingSignalOutcomePair],
        horizon_hours: int = 4
    ) -> List[ExistingSignalOutcomePair]:
        """
        Measure outcomes for batch of signals (with progress bar).
        
        Args:
            signals: List of signals to measure
            horizon_hours: Outcome horizon
            
        Returns:
            List of signals with outcomes measured (filtered to only valid)
        """
        logger.info(f"Measuring outcomes for {len(signals)} signals (horizon={horizon_hours}h)...")
        
        # Measure outcomes (sequential for now - could parallelize)
        measured = []
        for i, signal in enumerate(signals):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(signals)} ({i*100//len(signals)}%)")
            
            outcome = await self.measure_outcome(signal, horizon_hours)
            if outcome and outcome.was_correct is not None:
                measured.append(outcome)
        
        logger.info(f"✅ Measured outcomes for {len(measured)} signals ({len(measured)*100//len(signals)}% success rate)")
        return measured


async def main_calibrate_existing_signals():
    """Main calibration workflow using existing AI signals."""
    
    logger.info("\n" + "="*70)
    logger.info("PATH 2.4A — Calibrating Existing AI Signals")
    logger.info("="*70)
    logger.info("")
    
    collector = ExistingSignalCollector()
    
    try:
        await collector.connect()
        
        # Step 1: Collect signals
        logger.info("\n[STEP 1] Collecting signals from ai.signal_generated...")
        signals = await collector.collect_signals(
            start_time=datetime.now() - timedelta(days=7),  # Last 7 days
            max_count=5000
        )
        
        if len(signals) < 100:
            logger.error(f"⚠️ Insufficient signals collected: {len(signals)}")
            logger.error("   Need at least 100 signals for calibration")
            return
        
        # Step 2: Measure outcomes
        logger.info("\n[STEP 2] Measuring outcomes...")
        signals_with_outcomes = await collector.batch_measure_outcomes(signals)
        
        if len(signals_with_outcomes) < 100:
            logger.error(f"⚠️ Insufficient valid outcomes: {len(signals_with_outcomes)}")
            logger.error("   Need at least 100 signal-outcome pairs")
            logger.error("   Try collecting older data or wait longer for outcomes")
            return
        
        # Step 3: Fit calibrator
        logger.info("\n[STEP 3] Fitting calibrator...")
        
        confidences = np.array([s.confidence for s in signals_with_outcomes])
        outcomes = np.array([1.0 if s.was_correct else 0.0 for s in signals_with_outcomes])
        
        # Train/test split
        train_size = int(0.7 * len(confidences))
        train_conf, test_conf = confidences[:train_size], confidences[train_size:]
        train_out, test_out = outcomes[:train_size], outcomes[train_size:]
        
        logger.info(f"Training samples: {len(train_conf)}")
        logger.info(f"Test samples: {len(test_conf)}")
        
        # Fit isotonic regression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_conf, train_out)
        
        # Evaluate
        calibrated_test = calibrator.predict(test_conf)
        test_errors = np.abs(calibrated_test - test_out)
        test_calibration_error = np.mean(test_errors)
        
        logger.info(f"Test Calibration Error: {test_calibration_error:.4f}")
        
        # Step 4: Generate reliability diagram
        logger.info("\n[STEP 4] Generating reliability diagram...")
        
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        bin_data = []
        total_error = 0.0
        total_count = 0
        
        for bin_start, bin_end in bins:
            mask = (test_conf >= bin_start) & (test_conf < bin_end)
            if mask.sum() == 0:
                continue
            
            bin_confidences = test_conf[mask]
            bin_outcomes = test_out[mask]
            
            mean_conf = bin_confidences.mean()
            mean_acc = bin_outcomes.mean()
            count = len(bin_confidences)
            error = abs(mean_conf - mean_acc)
            
            bin_data.append({
                "range": f"[{bin_start:.1f}-{bin_end:.1f}]",
                "mean_confidence": float(mean_conf),
                "actual_accuracy": float(mean_acc),
                "count": int(count),
                "calibration_error": float(error)
            })
            
            total_error += error * count
            total_count += count
            
            logger.info(f"Bin {bin_start:.1f}-{bin_end:.1f}: conf={mean_conf:.3f}, acc={mean_acc:.3f}, n={count}, err={error:.3f}")
        
        ece = total_error / total_count if total_count > 0 else 0.0
        logger.info(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        
        # Step 5: Save calibrator
        logger.info("\n[STEP 5] Saving calibrator...")
        
        import pickle
        
        calibrator_path = "/home/qt/quantum_trader/ai_engine/calibration/calibrator_existing_v1.pkl"
        metadata_path = "/home/qt/quantum_trader/ai_engine/calibration/calibrator_existing_v1.pkl.json"
        
        # Save calibrator
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "method": "isotonic_regression",
            "source_stream": "quantum:stream:ai.signal_generated",
            "train_samples": int(len(train_conf)),
            "test_samples": int(len(test_conf)),
            "test_calibration_error": float(test_calibration_error),
            "expected_calibration_error": float(ece),
            "total_signal_outcome_pairs": len(signals_with_outcomes),
            "reliability_bins": bin_data
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Saved calibrator: {calibrator_path}")
        logger.info(f"✅ Saved metadata: {metadata_path}")
        
        logger.info("\n" + "="*70)
        logger.info("✅ CALIBRATION COMPLETE")
        logger.info("="*70)
        logger.info(f"\nResults:")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  Train samples: {len(train_conf)}")
        logger.info(f"  Test samples: {len(test_conf)}")
        logger.info(f"\nCALIBRATION {'EXCELLENT' if ece < 0.05 else 'ACCEPTABLE' if ece < 0.10 else 'NEEDS IMPROVEMENT'}")
        
    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main_calibrate_existing_signals())
