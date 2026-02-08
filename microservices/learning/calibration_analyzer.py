"""
Calibration Analyzer - Core Analysis Logic

This module performs calibration analysis on SimpleCLM trade data:
1. Confidence calibration (isotonic regression)
2. Ensemble weight micro-adjustments
3. HOLD bias calibration (optional)

NO TRAINING. NO MODEL CHANGES. Only numerical adjustments based on historical truth.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, precision_score, recall_score

from microservices.learning.cadence_policy import Trade
from microservices.learning.calibration_types import (
    ConfidenceCalibration,
    EnsembleWeightCalibration,
    HoldBiasCalibration,
    WeightChange,
    ValidationCheck,
    ValidationReport,
    ValidationSeverity,
    CalibrationValidationError
)

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    Analyzes SimpleCLM trade data and produces calibration parameters.
    
    Philosophy: "Vi lærer sakte av sannhet, ikke raskt av støy"
    - Only adjusts based on clear, statistically significant patterns
    - Conservative bounds on all adjustments
    - Multiple safety checks before recommendations
    """
    
    # Safety constraints
    MIN_TRADES_FOR_CALIBRATION = 50
    MIN_CONFIDENCE_IMPROVEMENT = 0.05  # 5% MSE reduction minimum
    MAX_WEIGHT_CHANGE = 0.10  # ±10% max adjustment
    MIN_WEIGHT = 0.15  # No model below 15%
    MAX_WEIGHT = 0.40  # No model above 40%
    MIN_OUTCOME_DIVERSITY = 0.15  # Need at least 15% wins AND losses
    
    def __init__(self):
        self.trades: List[Trade] = []
    
    def load_clm_data(self, file_path: str) -> List[Trade]:
        """
        Load and validate SimpleCLM trade data.
        
        Args:
            file_path: Path to clm_trades.jsonl
        
        Returns:
            List of Trade objects
        
        Raises:
            CalibrationValidationError: If data is insufficient or invalid
        """
        path = Path(file_path)
        
        if not path.exists():
            raise CalibrationValidationError(f"CLM data file not found: {file_path}")
        
        trades = []
        
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    # Parse timestamp (handle  both ISO format and string)
                    timestamp_str = data.get('timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except:
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                    
                    # Parse trade - handle both SimpleCLM format and test format
                    trade = Trade(
                        timestamp=timestamp,
                        symbol=data.get('symbol', 'UNKNOWN'),
                        entry_price=float(data.get('entry_price', 0.0)),
                        exit_price=float(data.get('exit_price', 0.0)),
                        pnl_percent=float(data.get('pnl_percent', 0.0)),
                        confidence=float(data.get('confidence', 0.5)),
                        model_id=data.get('model_id', 'ensemble'),  # Default to ensemble
                        outcome_label=data.get('outcome_label', 'UNKNOWN'),
                        duration_seconds=data.get('duration_seconds') or data.get('duration_minutes', 60) * 60,
                        strategy_id=data.get('strategy_id', 'default'),
                        position_size=float(data.get('position_size', 0.0)),
                        exit_reason=data.get('exit_reason'),
                        side=data.get('side', data.get('action', 'BUY'))
                    )
                    
                    trades.append(trade)
                    
                except KeyError as e:
                    logger.debug(f"Skipping line {line_num}: missing required field {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Skipping line {line_num}: {e}")
                    continue
        
        if len(trades) < self.MIN_TRADES_FOR_CALIBRATION:
            raise CalibrationValidationError(
                f"Insufficient data: {len(trades)} trades "
                f"(minimum: {self.MIN_TRADES_FOR_CALIBRATION})"
            )
        
        logger.info(f"✅ Loaded {len(trades)} trades from CLM")
        self.trades = trades
        return trades
    
    def calibrate_confidence(
        self, 
        trades: Optional[List[Trade]] = None,
        method: str = "isotonic"
    ) -> ConfidenceCalibration:
        """
        Calibrate model confidence using isotonic regression.
        
        Maps predicted confidence → actual win rate to improve calibration.
        
        Args:
            trades: Trade data (uses self.trades if None)
            method: Calibration method ("isotonic" or "none")
        
        Returns:
            ConfidenceCalibration object with mapping and metrics
        """
        if trades is None:
            trades = self.trades
        
        if len(trades) < self.MIN_TRADES_FOR_CALIBRATION:
            logger.warning("Insufficient trades for confidence calibration")
            return self._get_default_confidence_calibration()
        
        logger.info(f"🔄 Calibrating confidence with {len(trades)} trades...")
        
        # Extract confidence and outcomes
        y_true = []
        y_pred_conf = []
        
        for trade in trades:
            # Convert outcome to binary
            is_win = 1 if trade.outcome_label == "WIN" else 0
            y_true.append(is_win)
            y_pred_conf.append(trade.confidence)
        
        y_true = np.array(y_true)
        y_pred_conf = np.array(y_pred_conf)
        
        # Compute MSE before calibration
        mse_before = float(mean_squared_error(y_true, y_pred_conf))
        
        if method == "isotonic":
            # Fit isotonic regression (monotonic calibration)
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred_conf, y_true)
            
            # Create lookup table (binned for efficiency)
            bins = np.arange(0.5, 1.01, 0.05)  # 0.50, 0.55, ..., 1.00
            mapping = {}
            
            for bin_val in bins:
                calibrated = calibrator.predict([bin_val])[0]
                mapping[float(bin_val)] = float(calibrated)
            
            # Compute MSE after calibration
            y_calibrated = calibrator.predict(y_pred_conf)
            mse_after = float(mean_squared_error(y_true, y_calibrated))
            
        else:
            # No calibration
            mapping = {float(b): float(b) for b in np.arange(0.5, 1.01, 0.05)}
            mse_after = mse_before
        
        improvement_pct = ((mse_before - mse_after) / mse_before) * 100 if mse_before > 0 else 0
        
        logger.info(f"  MSE before: {mse_before:.4f}")
        logger.info(f"  MSE after:  {mse_after:.4f}")
        logger.info(f"  Improvement: {improvement_pct:+.1f}%")
        
        # Check if improvement is significant
        enabled = improvement_pct >= self.MIN_CONFIDENCE_IMPROVEMENT * 100
        
        if not enabled:
            logger.warning(f"⚠️  Confidence calibration improvement insufficient ({improvement_pct:.1f}% < 5%)")
            logger.warning("   Using default (no calibration)")
            return self._get_default_confidence_calibration()
        
        return ConfidenceCalibration(
            enabled=True,
            method=method,
            mapping=mapping,
            mse_before=mse_before,
            mse_after=mse_after,
            improvement_pct=improvement_pct,
            sample_size=len(trades)
        )
    
    def calibrate_ensemble_weights(
        self,
        trades: Optional[List[Trade]] = None
    ) -> EnsembleWeightCalibration:
        """
        Micro-adjust ensemble weights based on per-model performance.
        
        Analyzes:
        - Precision (accuracy when model predicts action)
        - False positive rate
        - Contribution to winning trades
        
        Constraints:
        - Max ±10% change per model
        - Weights must sum to 1.0
        - No weight < 0.15 or > 0.40
        
        Args:
            trades: Trade data (uses self.trades if None)
        
        Returns:
            EnsembleWeightCalibration with adjusted weights
        """
        if trades is None:
            trades = self.trades
        
        if len(trades) < self.MIN_TRADES_FOR_CALIBRATION:
            logger.warning("Insufficient trades for weight calibration")
            return self._get_default_weight_calibration()
        
        logger.info(f"🔄 Calibrating ensemble weights with {len(trades)} trades...")
        
        # Current weights (baseline)
        current_weights = {
            'xgb': 0.30,
            'lgbm': 0.30,
            'nhits': 0.20,
            'patchtst': 0.20
        }
        
        # Compute per-model metrics
        model_scores = {}
        
        for model in ['xgb', 'lgbm', 'nhits', 'patchtst']:
            # Note: In real implementation, we'd need model_breakdown in Trade
            # For now, use confidence as proxy (placeholder)
            
            # Compute basic performance metrics
            precision = self._compute_model_precision(trades, model)
            recall = self._compute_model_recall(trades, model)
            win_contribution = self._compute_win_contribution(trades, model)
            
            # Combined score (0.0-1.0)
            score = (precision + recall + win_contribution) / 3
            model_scores[model] = score
            
            logger.info(f"  {model:8s}: precision={precision:.3f}, recall={recall:.3f}, "
                       f"win_contrib={win_contribution:.3f} → score={score:.3f}")
        
        # Adjust weights based on scores
        new_weights = {}
        changes = []
        
        for model, score in model_scores.items():
            current = current_weights[model]
            
            # Score relative to baseline (0.5)
            # score > 0.5 → increase weight
            # score < 0.5 → decrease weight
            adjustment = (score - 0.5) * (self.MAX_WEIGHT_CHANGE * 2)  # Max ±10%
            
            new_weight = current * (1 + adjustment)
            
            # Apply constraints
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            
            new_weights[model] = new_weight
            
            delta = new_weight - current
            delta_pct = (delta / current) * 100
            
            reason = self._get_weight_change_reason(score, precision, recall, win_contribution)
            
            changes.append(WeightChange(
                model=model,
                before=current,
                after=new_weight,
                delta=delta,
                delta_pct=delta_pct,
                reason=reason
            ))
        
        # Normalize to sum=1.0
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Update changes with normalized weights
        for change in changes:
            change.after = new_weights[change.model]
            change.delta = change.after - change.before
            change.delta_pct = (change.delta / change.before) * 100
        
        total_delta = sum(abs(c.delta) for c in changes)
        
        logger.info(f"  Total weight change: {total_delta:.4f}")
        
        # Check if changes are significant
        enabled = total_delta >= 0.02  # At least 2% total change
        
        if not enabled:
            logger.warning(f"⚠️  Weight adjustments too small ({total_delta:.4f} < 0.02)")
            logger.warning("   Using default weights")
            return self._get_default_weight_calibration()
        
        return EnsembleWeightCalibration(
            enabled=True,
            weights=new_weights,
            changes=changes,
            total_delta=total_delta,
            sample_size=len(trades)
        )
    
    def calibrate_hold_bias(
        self,
        trades: Optional[List[Trade]] = None
    ) -> HoldBiasCalibration:
        """
        Calibrate HOLD bias (OPTIONAL, conservative).
        
        For now, returns disabled by default.
        Can be implemented later if HOLD signals show systematic bias.
        """
        if trades is None:
            trades = self.trades
        
        # For initial implementation, keep HOLD bias disabled
        logger.info("🔄 HOLD bias calibration: DISABLED (not yet implemented)")
        
        return HoldBiasCalibration(
            enabled=False,
            adjustment_factor=1.0,
            hold_accuracy=0.0,
            missed_opportunities=0,
            sample_size=len(trades) if trades else 0
        )
    
    def validate_calibration(
        self,
        confidence: ConfidenceCalibration,
        weights: EnsembleWeightCalibration,
        hold_bias: HoldBiasCalibration
    ) -> ValidationReport:
        """
        Validate calibration before deployment.
        
        Safety checks:
        1. Confidence calibration improves MSE
        2. Weights sum to 1.0
        3. No weight out of bounds
        4. No extreme adjustments
        5. Outcome diversity sufficient
        
        Returns:
            ValidationReport with pass/fail and risk score
        """
        logger.info("🔍 Validating calibration...")
        
        checks: List[ValidationCheck] = []
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check 1: Confidence improvement
        if confidence.enabled:
            improvement = confidence.improvement_pct
            passed = improvement > 0
            checks.append(ValidationCheck(
                name="confidence_improvement",
                passed=passed,
                severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
                description="Confidence calibration must improve MSE",
                result=f"{improvement:+.1f}%",
                details=f"MSE: {confidence.mse_before:.4f} → {confidence.mse_after:.4f}"
            ))
            
            if not passed:
                errors.append(f"Confidence calibration worsens MSE by {-improvement:.1f}%")
        
        # Check 2: Weights sum to 1.0
        if weights.enabled:
            weight_sum = sum(weights.weights.values())
            passed = abs(weight_sum - 1.0) < 0.001
            checks.append(ValidationCheck(
                name="weight_sum",
                passed=passed,
                severity=ValidationSeverity.CRITICAL if not passed else ValidationSeverity.INFO,
                description="Ensemble weights must sum to 1.0",
                result=f"{weight_sum:.6f}",
                details=f"Delta: {abs(weight_sum - 1.0):.6f}"
            ))
            
            if not passed:
                errors.append(f"Weights sum to {weight_sum:.4f}, not 1.0")
        
        # Check 3: Weight bounds
        if weights.enabled:
            for model, weight in weights.weights.items():
                passed = self.MIN_WEIGHT <= weight <= self.MAX_WEIGHT
                checks.append(ValidationCheck(
                    name=f"weight_bounds_{model}",
                    passed=passed,
                    severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
                    description=f"{model} weight must be between {self.MIN_WEIGHT} and {self.MAX_WEIGHT}",
                    result=f"{weight:.4f}",
                    details=f"Bounds: [{self.MIN_WEIGHT}, {self.MAX_WEIGHT}]"
                ))
                
                if not passed:
                    errors.append(f"{model} weight {weight:.4f} out of bounds")
        
        # Check 4: Maximum weight change check
        if weights.enabled:
            for change in weights.changes:
                passed = abs(change.delta) <= self.MAX_WEIGHT_CHANGE * 1.5  # Allow some margin
                severity = ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO
                
                checks.append(ValidationCheck(
                    name=f"max_change_{change.model}",
                    passed=passed,
                    severity=severity,
                    description=f"{change.model} weight change should be reasonable",
                    result=f"{change.delta:+.4f} ({change.delta_pct:+.1f}%)",
                    details=change.reason
                ))
                
                if not passed:
                    warnings.append(
                        f"{change.model} weight change is large: {change.delta_pct:+.1f}%"
                    )
        
        # Check 5: Outcome diversity
        if self.trades:
            wins = sum(1 for t in self.trades if t.outcome_label == "WIN")
            losses = sum(1 for t in self.trades if t.outcome_label == "LOSS")
            total = len(self.trades)
            
            win_pct = wins / total if total > 0 else 0
            loss_pct = losses / total if total > 0 else 0
            
            passed = win_pct >= self.MIN_OUTCOME_DIVERSITY and loss_pct >= self.MIN_OUTCOME_DIVERSITY
            
            checks.append(ValidationCheck(
                name="outcome_diversity",
                passed=passed,
                severity=ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO,
                description="Trade outcomes should be diverse",
                result=f"WIN={win_pct:.1%}, LOSS={loss_pct:.1%}",
                details=f"Need at least {self.MIN_OUTCOME_DIVERSITY:.0%} of each"
            ))
            
            if not passed:
                warnings.append(
                    f"Low outcome diversity: WIN={win_pct:.1%}, LOSS={loss_pct:.1%}"
                )
        
        # Compute overall pass/fail
        critical_failures = [c for c in checks if not c.passed and c.severity == ValidationSeverity.CRITICAL]
        error_failures = [c for c in checks if not c.passed and c.severity == ValidationSeverity.ERROR]
        
        passed = len(critical_failures) == 0 and len(error_failures) == 0
        
        # Compute risk score (0.0-1.0, lower is safer)
        risk_score = (
            len(critical_failures) * 0.5 +
            len(error_failures) * 0.3 +
            len(warnings) * 0.1
        ) / max(len(checks), 1)
        
        logger.info(f"  Validation: {'✅ PASSED' if passed else '❌ FAILED'}")
        logger.info(f"  Risk score: {risk_score:.2%}")
        logger.info(f"  Errors: {len(errors)}, Warnings: {len(warnings)}")
        
        return ValidationReport(
            passed=passed,
            checks=checks,
            errors=errors,
            warnings=warnings,
            risk_score=risk_score
        )
    
    # Helper methods
    
    def _get_default_confidence_calibration(self) -> ConfidenceCalibration:
        """Return default (no-op) confidence calibration"""
        bins = np.arange(0.5, 1.01, 0.05)
        mapping = {float(b): float(b) for b in bins}  # Identity mapping
        
        return ConfidenceCalibration(
            enabled=False,
            method="none",
            mapping=mapping,
            mse_before=0.0,
            mse_after=0.0,
            improvement_pct=0.0,
            sample_size=0
        )
    
    def _get_default_weight_calibration(self) -> EnsembleWeightCalibration:
        """Return default (no-op) weight calibration"""
        default_weights = {
            'xgb': 0.30,
            'lgbm': 0.30,
            'nhits': 0.20,
            'patchtst': 0.20
        }
        
        changes = [
            WeightChange(
                model=model,
                before=weight,
                after=weight,
                delta=0.0,
                delta_pct=0.0,
                reason="No adjustment (insufficient data or insignificant change)"
            )
            for model, weight in default_weights.items()
        ]
        
        return EnsembleWeightCalibration(
            enabled=False,
            weights=default_weights,
            changes=changes,
            total_delta=0.0,
            sample_size=0
        )
    
    def _compute_model_precision(self, trades: List[Trade], model: str) -> float:
        """Compute precision for a specific model (placeholder)"""
        # In real implementation, would use model_breakdown from trades
        # For now, return baseline
        return 0.5
    
    def _compute_model_recall(self, trades: List[Trade], model: str) -> float:
        """Compute recall for a specific model (placeholder)"""
        return 0.5
    
    def _compute_win_contribution(self, trades: List[Trade], model: str) -> float:
        """Compute model's contribution to winning trades (placeholder)"""
        return 0.5
    
    def _get_weight_change_reason(
        self,
        score: float,
        precision: float,
        recall: float,
        win_contrib: float
    ) -> str:
        """Generate human-readable reason for weight change"""
        if score > 0.55:
            return f"Strong performance (precision={precision:.2f}, recall={recall:.2f})"
        elif score < 0.45:
            return f"Underperformance (precision={precision:.2f}, recall={recall:.2f})"
        else:
            return "Neutral performance, minor adjustment"
