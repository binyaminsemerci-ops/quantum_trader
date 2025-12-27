"""
Confidence Calibrator - Phase 3C Integration

Calibrates AI signal confidence scores using historical accuracy data from Phase 3C-2,
ensuring confidence scores reflect actual prediction accuracy.

Problem: AI models may output confidence scores that don't match real-world accuracy.
Example: A model that claims 80% confidence but only achieves 60% accuracy should
have its confidence adjusted to reflect this gap.

Solution: Track actual vs predicted accuracy and calibrate confidence scores using
historical performance data.

Architecture:
- Uses Phase 3C-2 (Performance Benchmarker) for accuracy tracking
- Applies calibration before publishing AI signals
- Maintains calibration factors per module
- Provides calibration transparency and logging

Key Features:
1. Historical Accuracy Calibration: Adjust confidence based on actual performance
2. Per-Module Calibration: Each AI module has its own calibration factor
3. Smoothing: Gradual calibration to avoid overcorrection
4. Transparency: Log all calibrations for debugging

Author: AI Assistant
Date: 2025-12-24
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of confidence calibration"""
    raw_confidence: float  # Original confidence from AI model
    calibrated_confidence: float  # Calibrated confidence
    calibration_factor: float  # Factor applied (calibrated / raw)
    actual_accuracy: float  # Historical accuracy used for calibration
    sample_count: int  # Number of predictions used for calibration
    reason: str  # Explanation of calibration


@dataclass
class ModuleCalibrationStats:
    """Calibration statistics for a single module"""
    module_type: str
    total_predictions: int = 0
    correct_predictions: int = 0
    total_confidence_claimed: float = 0.0  # Sum of all raw confidences
    calibration_factor: float = 1.0  # Current calibration factor
    last_updated: Optional[datetime] = None


class ConfidenceCalibrator:
    """
    Calibrates AI signal confidence using historical accuracy from Phase 3C-2.
    
    Calibration Algorithm:
    1. Track raw confidence vs actual accuracy per module
    2. Calculate calibration factor = actual_accuracy / claimed_confidence
    3. Apply smoothing to avoid overcorrection
    4. Clamp result to [0.1, 0.95] to avoid extremes
    
    Example:
        Module claims 80% confidence on average, achieves 60% accuracy
        → Calibration factor = 60/80 = 0.75
        → New signal with 85% confidence → 85% * 0.75 = 63.75% calibrated
    
    Integration Points:
    - Phase 3C-2: Performance Benchmarker (for accuracy data)
    - AI Engine Service: Apply calibration before publishing signals
    """
    
    def __init__(
        self,
        performance_benchmarker=None,
        smoothing_factor: float = 0.7,
        min_confidence: float = 0.1,
        max_confidence: float = 0.95,
        min_sample_size: int = 20
    ):
        """
        Initialize Confidence Calibrator.
        
        Args:
            performance_benchmarker: Phase 3C-2 instance for accuracy data
            smoothing_factor: Weight for calibration factor (0-1)
                0 = no calibration, 1 = full calibration
            min_confidence: Minimum allowed confidence after calibration
            max_confidence: Maximum allowed confidence after calibration
            min_sample_size: Minimum predictions needed for calibration
        """
        self.benchmarker = performance_benchmarker
        self.smoothing_factor = smoothing_factor
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_samples = min_sample_size
        
        # Per-module calibration stats
        self.module_stats: Dict[str, ModuleCalibrationStats] = {}
        
        # Calibration history for analysis
        self.calibration_history: list = []
        
        logger.info(
            f"[CONF_CALIBRATOR] 🎯 Initialized with smoothing={smoothing_factor}, "
            f"range=[{min_confidence}, {max_confidence}], min_samples={min_sample_size}"
        )
    
    async def calibrate_confidence(
        self,
        signal_source: str,
        raw_confidence: float,
        symbol: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Calibrate confidence score based on historical accuracy.
        
        Args:
            signal_source: Module that generated the signal (e.g., 'ensemble', 'phase_3a')
            raw_confidence: Original confidence from AI model (0-1)
            symbol: Trading symbol (optional, for logging)
            metadata: Additional context (optional)
        
        Returns:
            CalibrationResult with calibrated confidence and explanation
        """
        
        # Step 1: Get module accuracy from Phase 3C-2
        actual_accuracy, sample_count = await self._get_module_accuracy(signal_source)
        
        # Step 2: Check if we have enough data for calibration
        if sample_count < self.min_samples:
            logger.debug(
                f"[CONF_CALIBRATOR] {signal_source}: Insufficient data ({sample_count} < {self.min_samples}), "
                f"using raw confidence={raw_confidence:.3f}"
            )
            return CalibrationResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_factor=1.0,
                actual_accuracy=0.0,
                sample_count=sample_count,
                reason=f"Insufficient data ({sample_count} < {self.min_samples})"
            )
        
        # Step 3: Calculate calibration factor
        calibration_factor = self._calculate_calibration_factor(
            actual_accuracy=actual_accuracy,
            raw_confidence=raw_confidence
        )
        
        # Step 4: Apply calibration with smoothing
        calibrated = raw_confidence * (
            self.smoothing_factor * calibration_factor +
            (1 - self.smoothing_factor) * 1.0
        )
        
        # Step 5: Clamp to allowed range
        calibrated = max(self.min_confidence, min(self.max_confidence, calibrated))
        
        # Step 6: Build result
        result = CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_factor=calibration_factor,
            actual_accuracy=actual_accuracy,
            sample_count=sample_count,
            reason=(
                f"Calibrated from {raw_confidence:.3f} to {calibrated:.3f} "
                f"(factor={calibration_factor:.3f}, actual_acc={actual_accuracy:.3f}, "
                f"samples={sample_count})"
            )
        )
        
        # Step 7: Log and track
        self._record_calibration(signal_source, result, symbol)
        
        return result
    
    def calibrate_confidence_sync(
        self,
        signal_source: str,
        raw_confidence: float,
        symbol: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> CalibrationResult:
        """
        Synchronous wrapper for calibrate_confidence.
        
        Uses internal tracking data instead of async benchmarker queries.
        This is used by synchronous code like Risk Manager.
        
        Args:
            signal_source: Module that generated the signal
            raw_confidence: Original confidence from AI model (0-1)
            symbol: Trading symbol (optional, for logging)
            metadata: Additional context (optional)
        
        Returns:
            CalibrationResult with calibrated confidence and explanation
        """
        # Get accuracy from internal stats (synchronous)
        if signal_source not in self.module_stats:
            # No data yet, return raw confidence
            return CalibrationResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_factor=1.0,
                actual_accuracy=0.0,
                sample_count=0,
                reason="No calibration data yet"
            )
        
        stats = self.module_stats[signal_source]
        
        if stats.total_predictions < self.min_samples:
            # Insufficient data
            return CalibrationResult(
                raw_confidence=raw_confidence,
                calibrated_confidence=raw_confidence,
                calibration_factor=1.0,
                actual_accuracy=0.0,
                sample_count=stats.total_predictions,
                reason=f"Insufficient data ({stats.total_predictions} < {self.min_samples})"
            )
        
        # Calculate actual accuracy
        actual_accuracy = stats.correct_predictions / stats.total_predictions
        
        # Calculate calibration factor
        calibration_factor = self._calculate_calibration_factor(
            actual_accuracy=actual_accuracy,
            raw_confidence=raw_confidence
        )
        
        # Apply calibration with smoothing
        calibrated = raw_confidence * (
            self.smoothing_factor * calibration_factor +
            (1 - self.smoothing_factor) * 1.0
        )
        
        # Clamp to allowed range
        calibrated = max(self.min_confidence, min(self.max_confidence, calibrated))
        
        # Build result
        result = CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_factor=calibration_factor,
            actual_accuracy=actual_accuracy,
            sample_count=stats.total_predictions,
            reason=(
                f"Calibrated from {raw_confidence:.3f} to {calibrated:.3f} "
                f"(factor={calibration_factor:.3f}, actual_acc={actual_accuracy:.3f}, "
                f"samples={stats.total_predictions})"
            )
        )
        
        # Record calibration
        self._record_calibration(signal_source, result, symbol)
        
        return result
    
    def record_prediction_outcome(
        self,
        signal_source: str,
        claimed_confidence: float,
        was_correct: bool
    ):
        """
        Record prediction outcome for calibration tracking.
        
        Args:
            signal_source: Module that made the prediction
            claimed_confidence: Confidence claimed by the module
            was_correct: Whether the prediction was correct
        """
        if signal_source not in self.module_stats:
            self.module_stats[signal_source] = ModuleCalibrationStats(
                module_type=signal_source
            )
        
        stats = self.module_stats[signal_source]
        stats.total_predictions += 1
        if was_correct:
            stats.correct_predictions += 1
        stats.total_confidence_claimed += claimed_confidence
        stats.last_updated = datetime.utcnow()
        
        # Recalculate calibration factor
        if stats.total_predictions >= self.min_samples:
            avg_claimed = stats.total_confidence_claimed / stats.total_predictions
            actual_acc = stats.correct_predictions / stats.total_predictions
            
            if avg_claimed > 0.5:
                stats.calibration_factor = actual_acc / avg_claimed
            else:
                stats.calibration_factor = 1.0
        
        logger.debug(
            f"[CONF_CALIBRATOR] {signal_source}: Recorded {'correct' if was_correct else 'incorrect'} "
            f"prediction (claimed={claimed_confidence:.3f}), "
            f"total={stats.total_predictions}, acc={stats.correct_predictions/stats.total_predictions:.3f}"
        )
    
    def get_calibration_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get calibration summary for all modules."""
        summary = {}
        
        for module, stats in self.module_stats.items():
            if stats.total_predictions == 0:
                continue
            
            actual_accuracy = stats.correct_predictions / stats.total_predictions
            avg_claimed = stats.total_confidence_claimed / stats.total_predictions
            
            summary[module] = {
                "total_predictions": stats.total_predictions,
                "accuracy": actual_accuracy,
                "avg_claimed_confidence": avg_claimed,
                "calibration_factor": stats.calibration_factor,
                "last_updated": stats.last_updated.isoformat() if stats.last_updated else None
            }
        
        return summary
    
    def get_calibration_history(
        self,
        limit: int = 100,
        signal_source: Optional[str] = None
    ) -> list:
        """
        Get recent calibration history.
        
        Args:
            limit: Maximum number of records to return
            signal_source: Filter by signal source (optional)
        
        Returns:
            List of calibration records
        """
        history = self.calibration_history
        
        if signal_source:
            history = [h for h in history if h.get("signal_source") == signal_source]
        
        return history[-limit:]
    
    # ========== PRIVATE HELPER METHODS ==========
    
    async def _get_module_accuracy(
        self,
        signal_source: str
    ) -> Tuple[float, int]:
        """
        Get actual accuracy and sample count from Phase 3C-2.
        
        Returns:
            (actual_accuracy, sample_count)
        """
        if not self.benchmarker:
            # Fall back to internal tracking
            if signal_source in self.module_stats:
                stats = self.module_stats[signal_source]
                if stats.total_predictions > 0:
                    accuracy = stats.correct_predictions / stats.total_predictions
                    return (accuracy, stats.total_predictions)
            
            return (0.0, 0)
        
        try:
            # Get benchmarks from Phase 3C-2
            benchmarks = await self.benchmarker.get_current_benchmarks()
            
            if signal_source in benchmarks:
                module_perf = benchmarks[signal_source]
                
                if module_perf.accuracy_stats:
                    accuracy = module_perf.accuracy_stats.accuracy_pct / 100.0
                    sample_count = module_perf.accuracy_stats.total_predictions
                    
                    return (accuracy, sample_count)
            
            return (0.0, 0)
        
        except Exception as e:
            logger.error(f"[CONF_CALIBRATOR] Error getting accuracy for {signal_source}: {e}")
            return (0.0, 0)
    
    def _calculate_calibration_factor(
        self,
        actual_accuracy: float,
        raw_confidence: float
    ) -> float:
        """
        Calculate calibration factor.
        
        Factor = actual_accuracy / raw_confidence
        
        If model claims 80% confidence but achieves 60% accuracy:
        → factor = 60/80 = 0.75
        
        Special cases:
        - If raw_confidence < 0.5, don't calibrate (too low to trust)
        - Clamp factor to [0.5, 1.5] to avoid extreme corrections
        """
        if raw_confidence < 0.5:
            # Don't calibrate very low confidence scores
            return 1.0
        
        # Calculate basic factor
        if actual_accuracy > 0 and raw_confidence > 0:
            factor = actual_accuracy / raw_confidence
        else:
            factor = 1.0
        
        # Clamp to reasonable range
        factor = max(0.5, min(1.5, factor))
        
        return factor
    
    def _record_calibration(
        self,
        signal_source: str,
        result: CalibrationResult,
        symbol: str
    ):
        """Record calibration for history and logging."""
        
        # Log calibration
        if abs(result.raw_confidence - result.calibrated_confidence) > 0.05:
            logger.info(
                f"[CONF_CALIBRATOR] {signal_source} ({symbol}): "
                f"{result.raw_confidence:.3f} → {result.calibrated_confidence:.3f} "
                f"(factor={result.calibration_factor:.3f}, samples={result.sample_count})"
            )
        
        # Add to history
        self.calibration_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "signal_source": signal_source,
            "symbol": symbol,
            "raw_confidence": result.raw_confidence,
            "calibrated_confidence": result.calibrated_confidence,
            "calibration_factor": result.calibration_factor,
            "actual_accuracy": result.actual_accuracy,
            "sample_count": result.sample_count,
            "reason": result.reason
        })
        
        # Trim history if too long
        if len(self.calibration_history) > 1000:
            self.calibration_history = self.calibration_history[-500:]
