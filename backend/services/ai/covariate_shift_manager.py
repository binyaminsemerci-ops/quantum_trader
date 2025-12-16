"""
COVARIATE SHIFT MANAGER - Distribution Shift Detection and Adaptation

This module detects and adapts to covariate shift in feature distributions,
ensuring model predictions remain accurate as market conditions change.

Key Components:
- Importance weighting using likelihood ratios
- Domain adaptation through feature reweighting
- KL divergence monitoring for distribution changes
- Adaptive prediction adjustment

Author: Quantum Trader AI Team
Date: 2025-11-26
Version: 1.0
"""

import json
import logging
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class ShiftType(Enum):
    """Types of covariate shift detected"""
    NO_SHIFT = "no_shift"
    MILD_SHIFT = "mild_shift"
    MODERATE_SHIFT = "moderate_shift"
    SEVERE_SHIFT = "severe_shift"


class AdaptationMethod(Enum):
    """Methods for adapting to covariate shift"""
    IMPORTANCE_WEIGHTING = "importance_weighting"
    KERNEL_MEAN_MATCHING = "kernel_mean_matching"
    ADVERSARIAL_ADAPTATION = "adversarial_adaptation"


@dataclass
class FeatureStatistics:
    """Statistics for feature distributions"""
    mean: float
    std: float
    min: float
    max: float
    q25: float
    q50: float
    q75: float
    samples: int
    last_updated: str


@dataclass
class ShiftDetection:
    """Covariate shift detection result"""
    shift_type: ShiftType
    kl_divergence: float
    js_divergence: float
    max_mean_discrepancy: float
    confidence: float
    affected_features: List[str]
    detected_at: str
    severity_score: float  # 0-100


@dataclass
class ImportanceWeights:
    """Importance weights for domain adaptation"""
    weights: Dict[str, float]
    method: AdaptationMethod
    effectiveness: float  # 0-1
    applied_at: str
    reference_distribution: str


@dataclass
class AdaptationResult:
    """Result of adaptation to covariate shift"""
    original_prediction: float
    adapted_prediction: float
    adjustment_factor: float
    confidence_boost: float
    weights_applied: Dict[str, float]
    method_used: AdaptationMethod


class CovariateShiftManager:
    """
    Manages detection and adaptation to covariate shift.
    
    Covariate shift occurs when P(X) changes but P(Y|X) remains constant.
    This manager detects such shifts and adapts predictions accordingly.
    """
    
    def __init__(
        self,
        shift_threshold: float = 0.15,
        adaptation_rate: float = 0.10,
        history_window: int = 1000,
        min_samples: int = 100,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize Covariate Shift Manager.
        
        Args:
            shift_threshold: KL divergence threshold for shift detection
            adaptation_rate: Rate of adaptation (0-1)
            history_window: Number of samples to keep in history
            min_samples: Minimum samples needed for detection
            checkpoint_path: Path to save/load checkpoints
        """
        self.shift_threshold = shift_threshold
        self.adaptation_rate = adaptation_rate
        self.history_window = history_window
        self.min_samples = min_samples
        self.checkpoint_path = checkpoint_path
        
        # Reference distributions (training distribution)
        self.reference_stats: Dict[str, Dict[str, FeatureStatistics]] = {}
        
        # Current distributions per symbol
        self.current_stats: Dict[str, Dict[str, FeatureStatistics]] = {}
        
        # Feature history for distribution estimation
        self.feature_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_window))
        )
        
        # Detected shifts
        self.shift_detections: Dict[str, List[ShiftDetection]] = defaultdict(list)
        
        # Importance weights
        self.importance_weights: Dict[str, ImportanceWeights] = {}
        
        # Adaptation history
        self.adaptation_history: Dict[str, List[AdaptationResult]] = defaultdict(list)
        
        # Performance tracking
        self.performance_before_adaptation: Dict[str, List[float]] = defaultdict(list)
        self.performance_after_adaptation: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(
            f"[CovariateShift] Initialized | "
            f"Threshold: {shift_threshold:.3f}, "
            f"Adaptation: {adaptation_rate:.2f}, "
            f"Window: {history_window}"
        )
        
        # Load checkpoint if available
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint()
    
    def set_reference_distribution(
        self,
        symbol: str,
        features: Dict[str, np.ndarray]
    ):
        """
        Set reference distribution from training data.
        
        Args:
            symbol: Trading symbol
            features: Dictionary of feature arrays from training
        """
        self.reference_stats[symbol] = {}
        
        for feature_name, values in features.items():
            if len(values) == 0:
                continue
            
            stats = FeatureStatistics(
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                q25=float(np.percentile(values, 25)),
                q50=float(np.percentile(values, 50)),
                q75=float(np.percentile(values, 75)),
                samples=len(values),
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            self.reference_stats[symbol][feature_name] = stats
        
        logger.info(
            f"[CovariateShift] Reference distribution set for {symbol} | "
            f"Features: {len(features)}, Samples: {len(next(iter(features.values())))}"
        )
    
    def update(
        self,
        symbol: str,
        features: Dict[str, float]
    ) -> bool:
        """
        Update feature history and detect shift.
        
        Args:
            symbol: Trading symbol
            features: Current feature values
        
        Returns:
            True if shift detected
        """
        # Update feature history
        for feature_name, value in features.items():
            self.feature_history[symbol][feature_name].append(value)
        
        # Check if we have enough samples
        sample_count = len(self.feature_history[symbol][next(iter(features.keys()))])
        
        if sample_count < self.min_samples:
            return False
        
        # Update current statistics
        self._update_current_statistics(symbol)
        
        # Detect shift
        shift_detected = self._detect_shift(symbol)
        
        # Update importance weights if shift detected
        if shift_detected:
            self._update_importance_weights(symbol)
        
        return shift_detected
    
    def _update_current_statistics(self, symbol: str):
        """Update current distribution statistics"""
        self.current_stats[symbol] = {}
        
        for feature_name, values in self.feature_history[symbol].items():
            if len(values) == 0:
                continue
            
            values_array = np.array(list(values))
            
            stats = FeatureStatistics(
                mean=float(np.mean(values_array)),
                std=float(np.std(values_array)),
                min=float(np.min(values_array)),
                max=float(np.max(values_array)),
                q25=float(np.percentile(values_array, 25)),
                q50=float(np.percentile(values_array, 50)),
                q75=float(np.percentile(values_array, 75)),
                samples=len(values_array),
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            self.current_stats[symbol][feature_name] = stats
    
    def _detect_shift(self, symbol: str) -> bool:
        """
        Detect covariate shift using multiple metrics.
        
        Returns:
            True if shift detected
        """
        if symbol not in self.reference_stats:
            return False
        
        if symbol not in self.current_stats:
            return False
        
        # Calculate divergences
        kl_divs = []
        js_divs = []
        mmd_scores = []
        affected_features = []
        
        for feature_name in self.reference_stats[symbol].keys():
            if feature_name not in self.current_stats[symbol]:
                continue
            
            ref_stats = self.reference_stats[symbol][feature_name]
            cur_stats = self.current_stats[symbol][feature_name]
            
            # KL divergence (approximated using normal distributions)
            kl_div = self._calculate_kl_divergence(
                ref_stats.mean, ref_stats.std,
                cur_stats.mean, cur_stats.std
            )
            kl_divs.append(kl_div)
            
            # JS divergence
            ref_values = np.array(list(self.feature_history[symbol][feature_name]))[-self.min_samples:]
            ref_hist, _ = np.histogram(ref_values, bins=20, density=True)
            ref_hist = ref_hist + 1e-10  # Avoid zeros
            ref_hist = ref_hist / ref_hist.sum()
            
            cur_values = np.array(list(self.feature_history[symbol][feature_name]))[-self.min_samples:]
            cur_hist, _ = np.histogram(cur_values, bins=20, density=True)
            cur_hist = cur_hist + 1e-10
            cur_hist = cur_hist / cur_hist.sum()
            
            js_div = jensenshannon(ref_hist, cur_hist)
            js_divs.append(js_div)
            
            # Maximum Mean Discrepancy (simplified)
            mmd = abs(ref_stats.mean - cur_stats.mean) / (ref_stats.std + 1e-10)
            mmd_scores.append(mmd)
            
            # Track affected features
            if kl_div > self.shift_threshold:
                affected_features.append(feature_name)
        
        if not kl_divs:
            return False
        
        # Aggregate metrics
        avg_kl = np.mean(kl_divs)
        avg_js = np.mean(js_divs)
        avg_mmd = np.mean(mmd_scores)
        
        # Determine shift type
        if avg_kl > self.shift_threshold * 2:
            shift_type = ShiftType.SEVERE_SHIFT
            severity = 90 + min(10, (avg_kl - self.shift_threshold * 2) * 10)
        elif avg_kl > self.shift_threshold * 1.5:
            shift_type = ShiftType.MODERATE_SHIFT
            severity = 70 + (avg_kl - self.shift_threshold * 1.5) * 40
        elif avg_kl > self.shift_threshold:
            shift_type = ShiftType.MILD_SHIFT
            severity = 50 + (avg_kl - self.shift_threshold) * 40
        else:
            shift_type = ShiftType.NO_SHIFT
            severity = avg_kl * 50 / self.shift_threshold
        
        # Confidence based on consistency across features
        confidence = 1.0 - np.std(kl_divs) / (np.mean(kl_divs) + 1e-10)
        confidence = max(0.0, min(1.0, confidence))
        
        # Create detection record
        detection = ShiftDetection(
            shift_type=shift_type,
            kl_divergence=float(avg_kl),
            js_divergence=float(avg_js),
            max_mean_discrepancy=float(avg_mmd),
            confidence=float(confidence),
            affected_features=affected_features,
            detected_at=datetime.now(timezone.utc).isoformat(),
            severity_score=float(severity)
        )
        
        # Store detection
        self.shift_detections[symbol].append(detection)
        
        # Keep only recent detections
        if len(self.shift_detections[symbol]) > 100:
            self.shift_detections[symbol] = self.shift_detections[symbol][-100:]
        
        if shift_type != ShiftType.NO_SHIFT:
            logger.warning(
                f"[CovariateShift] {shift_type.value.upper()} detected for {symbol} | "
                f"KL: {avg_kl:.4f}, JS: {avg_js:.4f}, MMD: {avg_mmd:.4f}, "
                f"Affected: {len(affected_features)} features"
            )
            return True
        
        return False
    
    def _calculate_kl_divergence(
        self,
        mean1: float,
        std1: float,
        mean2: float,
        std2: float
    ) -> float:
        """
        Calculate KL divergence between two normal distributions.
        
        KL(P||Q) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        """
        if std1 < 1e-10 or std2 < 1e-10:
            return 0.0
        
        kl = np.log(std2 / std1) + (std1**2 + (mean1 - mean2)**2) / (2 * std2**2) - 0.5
        
        return max(0.0, float(kl))
    
    def _update_importance_weights(self, symbol: str):
        """Update importance weights for domain adaptation"""
        if symbol not in self.reference_stats or symbol not in self.current_stats:
            return
        
        weights = {}
        
        for feature_name in self.reference_stats[symbol].keys():
            if feature_name not in self.current_stats[symbol]:
                weights[feature_name] = 1.0
                continue
            
            ref_stats = self.reference_stats[symbol][feature_name]
            cur_stats = self.current_stats[symbol][feature_name]
            
            # Importance weight = P_ref(x) / P_cur(x)
            # Approximated using likelihood ratios of normal distributions
            ref_density = self._normal_density(cur_stats.mean, ref_stats.mean, ref_stats.std)
            cur_density = self._normal_density(cur_stats.mean, cur_stats.mean, cur_stats.std)
            
            weight = ref_density / (cur_density + 1e-10)
            
            # Clip weights to prevent extreme values
            weight = max(0.1, min(10.0, weight))
            
            weights[feature_name] = float(weight)
        
        # Calculate effectiveness (inverse of weight variance)
        weight_values = list(weights.values())
        effectiveness = 1.0 / (1.0 + np.std(weight_values))
        
        # Store importance weights
        self.importance_weights[symbol] = ImportanceWeights(
            weights=weights,
            method=AdaptationMethod.IMPORTANCE_WEIGHTING,
            effectiveness=float(effectiveness),
            applied_at=datetime.now(timezone.utc).isoformat(),
            reference_distribution="training"
        )
        
        logger.info(
            f"[CovariateShift] Importance weights updated for {symbol} | "
            f"Effectiveness: {effectiveness:.3f}, "
            f"Avg weight: {np.mean(weight_values):.3f}"
        )
    
    def _normal_density(self, x: float, mean: float, std: float) -> float:
        """Calculate normal distribution density"""
        if std < 1e-10:
            return 1.0
        
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def adapt_prediction(
        self,
        symbol: str,
        prediction: float,
        features: Dict[str, float],
        confidence: float
    ) -> AdaptationResult:
        """
        Adapt prediction using importance weights.
        
        Args:
            symbol: Trading symbol
            prediction: Original model prediction
            features: Feature values used for prediction
            confidence: Original confidence
        
        Returns:
            Adaptation result with adjusted prediction
        """
        if symbol not in self.importance_weights:
            # No adaptation needed
            return AdaptationResult(
                original_prediction=prediction,
                adapted_prediction=prediction,
                adjustment_factor=1.0,
                confidence_boost=0.0,
                weights_applied={},
                method_used=AdaptationMethod.IMPORTANCE_WEIGHTING
            )
        
        weights = self.importance_weights[symbol].weights
        
        # Calculate weighted adjustment
        adjustments = []
        applied_weights = {}
        
        for feature_name, feature_value in features.items():
            if feature_name in weights:
                weight = weights[feature_name]
                applied_weights[feature_name] = weight
                
                # Adjustment proportional to weight deviation from 1.0
                adjustment = (weight - 1.0) * self.adaptation_rate
                adjustments.append(adjustment)
        
        if not adjustments:
            return AdaptationResult(
                original_prediction=prediction,
                adapted_prediction=prediction,
                adjustment_factor=1.0,
                confidence_boost=0.0,
                weights_applied={},
                method_used=AdaptationMethod.IMPORTANCE_WEIGHTING
            )
        
        # Average adjustment
        avg_adjustment = np.mean(adjustments)
        
        # Apply adjustment to prediction
        adjustment_factor = 1.0 + avg_adjustment
        adapted_prediction = prediction * adjustment_factor
        
        # Clip to valid range [0, 1]
        adapted_prediction = max(0.0, min(1.0, adapted_prediction))
        
        # Confidence boost based on weight effectiveness
        effectiveness = self.importance_weights[symbol].effectiveness
        confidence_boost = effectiveness * 0.05  # Up to 5% boost
        
        result = AdaptationResult(
            original_prediction=prediction,
            adapted_prediction=adapted_prediction,
            adjustment_factor=adjustment_factor,
            confidence_boost=confidence_boost,
            weights_applied=applied_weights,
            method_used=AdaptationMethod.IMPORTANCE_WEIGHTING
        )
        
        # Store adaptation
        self.adaptation_history[symbol].append(result)
        
        if len(self.adaptation_history[symbol]) > 1000:
            self.adaptation_history[symbol] = self.adaptation_history[symbol][-1000:]
        
        logger.debug(
            f"[CovariateShift] Adapted prediction for {symbol} | "
            f"Original: {prediction:.4f}, Adapted: {adapted_prediction:.4f}, "
            f"Factor: {adjustment_factor:.4f}"
        )
        
        return result
    
    def get_detected_shifts(self, symbol: str, n: int = 10) -> List[ShiftDetection]:
        """Get recent shift detections"""
        return self.shift_detections.get(symbol, [])[-n:]
    
    def get_importance_weights(self, symbol: str) -> Optional[ImportanceWeights]:
        """Get current importance weights for a symbol"""
        return self.importance_weights.get(symbol)
    
    def get_adaptation_history(self, symbol: str, n: int = 100) -> List[AdaptationResult]:
        """Get recent adaptations"""
        return self.adaptation_history.get(symbol, [])[-n:]
    
    def get_shift(self, symbol: str) -> Dict[str, Any]:
        """Get current shift status"""
        recent_detections = self.get_detected_shifts(symbol, n=10)
        
        if not recent_detections:
            return {
                'shift_detected': False,
                'shift_type': ShiftType.NO_SHIFT.value
            }
        
        latest = recent_detections[-1]
        
        return {
            'shift_detected': latest.shift_type != ShiftType.NO_SHIFT,
            'shift_type': latest.shift_type.value,
            'kl_divergence': latest.kl_divergence,
            'js_divergence': latest.js_divergence,
            'severity_score': latest.severity_score,
            'confidence': latest.confidence,
            'affected_features': latest.affected_features,
            'detected_at': latest.detected_at
        }
    
    def get_history(self, symbol: str, n: int = 100) -> List[Dict[str, Any]]:
        """Get shift detection history"""
        detections = self.shift_detections.get(symbol, [])[-n:]
        return [
            {
                'shift_type': d.shift_type.value,
                'kl_divergence': d.kl_divergence,
                'js_divergence': d.js_divergence,
                'severity_score': d.severity_score,
                'confidence': d.confidence,
                'affected_features': d.affected_features,
                'detected_at': d.detected_at
            }
            for d in detections
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall manager status"""
        total_shifts = sum(len(shifts) for shifts in self.shift_detections.values())
        symbols_with_shifts = sum(
            1 for shifts in self.shift_detections.values()
            if any(s.shift_type != ShiftType.NO_SHIFT for s in shifts)
        )
        
        return {
            'total_shifts_detected': total_shifts,
            'symbols_monitored': len(self.feature_history),
            'symbols_with_active_shifts': symbols_with_shifts,
            'symbols_with_adaptation': len(self.importance_weights),
            'shift_threshold': self.shift_threshold,
            'adaptation_rate': self.adaptation_rate
        }
    
    def save_checkpoint(self):
        """Save manager state to checkpoint"""
        if not self.checkpoint_path:
            return
        
        checkpoint = {
            'reference_stats': {
                symbol: {
                    name: asdict(stats)
                    for name, stats in features.items()
                }
                for symbol, features in self.reference_stats.items()
            },
            'shift_detections': {
                symbol: [asdict(d) for d in detections]
                for symbol, detections in self.shift_detections.items()
            },
            'importance_weights': {
                symbol: asdict(weights)
                for symbol, weights in self.importance_weights.items()
            },
            'config': {
                'shift_threshold': self.shift_threshold,
                'adaptation_rate': self.adaptation_rate,
                'history_window': self.history_window,
                'min_samples': self.min_samples
            },
            'saved_at': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"[CovariateShift] Checkpoint saved to {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"[CovariateShift] Failed to save checkpoint: {e}")
    
    def load_checkpoint(self):
        """Load manager state from checkpoint"""
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            # Restore reference stats
            for symbol, features in checkpoint.get('reference_stats', {}).items():
                self.reference_stats[symbol] = {
                    name: FeatureStatistics(**stats)
                    for name, stats in features.items()
                }
            
            # Restore shift detections
            for symbol, detections in checkpoint.get('shift_detections', {}).items():
                self.shift_detections[symbol] = [
                    ShiftDetection(
                        shift_type=ShiftType(d['shift_type']),
                        kl_divergence=d['kl_divergence'],
                        js_divergence=d['js_divergence'],
                        max_mean_discrepancy=d['max_mean_discrepancy'],
                        confidence=d['confidence'],
                        affected_features=d['affected_features'],
                        detected_at=d['detected_at'],
                        severity_score=d['severity_score']
                    )
                    for d in detections
                ]
            
            # Restore importance weights
            for symbol, weights_data in checkpoint.get('importance_weights', {}).items():
                self.importance_weights[symbol] = ImportanceWeights(
                    weights=weights_data['weights'],
                    method=AdaptationMethod(weights_data['method']),
                    effectiveness=weights_data['effectiveness'],
                    applied_at=weights_data['applied_at'],
                    reference_distribution=weights_data['reference_distribution']
                )
            
            logger.info(
                f"[CovariateShift] Checkpoint loaded | "
                f"Symbols: {len(self.reference_stats)}, "
                f"Shifts: {sum(len(s) for s in self.shift_detections.values())}"
            )
        
        except Exception as e:
            logger.error(f"[CovariateShift] Failed to load checkpoint: {e}")
