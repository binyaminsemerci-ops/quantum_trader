"""
META-AGENT V2: Regime-Aware Stacked Ensemble with Safety Guarantees

Design Principles:
1. Only override when confident (meta_confidence >= threshold)
2. Fallback to weighted ensemble when uncertain
3. Runtime safety checks (constant output detection, shape validation)
4. Full explainability (reason for every decision)
5. Regime-aware decision making

Input Features (per timestep):
- Base agent signals: action + confidence for XGBoost, LightGBM, N-HiTS, PatchTST
- Derived metrics: mean_conf, max_conf, std_conf, disagreement, vote distribution
- Optional: volatility regime, trend strength

Output Contract:
{
    "use_meta": bool,          # True if meta overrides, False if fallback to ensemble
    "action": str,              # SELL | HOLD | BUY
    "confidence": float,        # 0.0-1.0
    "reason": str,              # Explanation of decision
    "meta_confidence": float    # Internal meta model confidence
}

Model: Logistic Regression with strong L2 regularization
Training: Time-series CV on post-fix data only
Calibration: Platt scaling for probability calibration
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class MetaAgentV2:
    """
    Meta-Agent V2: Regime-aware stacked ensemble with safety guarantees.
    
    Learns when to trust base agents vs when to defer to weighted ensemble.
    """
    
    # Class constants
    VERSION = "2.0.0"
    ACTIONS = ["SELL", "HOLD", "BUY"]
    ACTION_TO_IDX = {"SELL": 0, "HOLD": 1, "BUY": 2}
    IDX_TO_ACTION = {0: "SELL", 1: "HOLD", 2: "BUY"}
    
    # Default thresholds
    DEFAULT_META_THRESHOLD = 0.65  # Minimum confidence to override ensemble
    DEFAULT_CONSENSUS_THRESHOLD = 0.75  # Strong consensus = defer to ensemble
    DEFAULT_MAX_OVERRIDE_RATE = 0.40  # Warn if meta overrides >40% of time
    
    # Safety checks
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    CONSTANT_OUTPUT_TOLERANCE = 1e-4  # For detecting broken models
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        meta_threshold: float = DEFAULT_META_THRESHOLD,
        consensus_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
        max_override_rate: float = DEFAULT_MAX_OVERRIDE_RATE,
        enable_regime_features: bool = True
    ):
        """
        Initialize Meta-Agent V2.
        
        Args:
            model_dir: Directory containing trained model files
            meta_threshold: Minimum confidence for meta override (0.0-1.0)
            consensus_threshold: Strong consensus threshold (0.0-1.0)
            max_override_rate: Maximum allowed override rate before warning
            enable_regime_features: Include regime features if available
        """
        self.meta_threshold = meta_threshold
        self.consensus_threshold = consensus_threshold
        self.max_override_rate = max_override_rate
        self.enable_regime_features = enable_regime_features
        
        # Model artifacts
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.expected_feature_dim: int = 0
        
        # Runtime statistics
        self.total_predictions = 0
        self.meta_overrides = 0
        self.fallback_reasons: Counter = Counter()
        
        # Model directory
        if model_dir is None:
            model_dir = os.getenv(
                "META_V2_MODEL_DIR",
                "/home/qt/quantum_trader/ai_engine/models/meta_v2"
            )
        self.model_dir = Path(model_dir)
        
        # Load model if available
        self._load_model()
        
        logger.info(f"[MetaV2] Initialized (version={self.VERSION})")
        logger.info(f"[MetaV2] Meta threshold: {self.meta_threshold:.2f}")
        logger.info(f"[MetaV2] Consensus threshold: {self.consensus_threshold:.2f}")
        logger.info(f"[MetaV2] Max override rate: {self.max_override_rate:.2%}")
        logger.info(f"[MetaV2] Model ready: {self.is_ready()}")
    
    def is_ready(self) -> bool:
        """Check if meta-agent is ready for predictions."""
        return (
            self.model is not None
            and self.scaler is not None
            and self.expected_feature_dim > 0
        )
    
    def _load_model(self) -> None:
        """Load trained meta-agent model from disk."""
        if not self.model_dir.exists():
            logger.warning(f"[MetaV2] Model directory not found: {self.model_dir}")
            return
        
        model_path = self.model_dir / "meta_model.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "metadata.json"
        
        if not model_path.exists():
            logger.warning(f"[MetaV2] Model file not found: {model_path}")
            return
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                logger.warning(f"[MetaV2] Scaler not found, creating identity scaler")
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.zeros(20)
                self.scaler.scale_ = np.ones(20)
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
                self.expected_feature_dim = len(self.feature_names)
            else:
                logger.warning(f"[MetaV2] Metadata not found, using defaults")
                self.expected_feature_dim = 20  # Default
            
            logger.info(f"[MetaV2] ✅ Model loaded successfully")
            logger.info(f"[MetaV2]    Model type: {type(self.model).__name__}")
            logger.info(f"[MetaV2]    Features: {self.expected_feature_dim}")
            logger.info(f"[MetaV2]    Training date: {self.metadata.get('training_date', 'unknown')}")
            logger.info(f"[MetaV2]    Training samples: {self.metadata.get('n_samples', 'unknown')}")
            logger.info(f"[MetaV2]    CV accuracy: {self.metadata.get('cv_accuracy', 'unknown')}")
            
            # Validate model
            self._validate_model()
            
        except Exception as e:
            logger.error(f"[MetaV2] ❌ Failed to load model: {e}")
            self.model = None
            self.scaler = None
    
    def _validate_model(self) -> None:
        """Validate that loaded model produces non-constant output."""
        if not self.is_ready():
            return
        
        try:
            # Test with two random feature vectors
            np.random.seed(42)
            X1 = np.random.randn(1, self.expected_feature_dim)
            X2 = np.random.randn(1, self.expected_feature_dim)
            
            X1_scaled = self.scaler.transform(X1)
            X2_scaled = self.scaler.transform(X2)
            
            y1 = self.model.predict_proba(X1_scaled)[0]
            y2 = self.model.predict_proba(X2_scaled)[0]
            
            # Check if outputs are too similar (constant model)
            diff = np.abs(y1 - y2).max()
            if diff < self.CONSTANT_OUTPUT_TOLERANCE:
                logger.error(
                    f"[MetaV2] ❌ VALIDATION FAILED: Model produces constant output "
                    f"(max_diff={diff:.6f})"
                )
                self.model = None
                self.scaler = None
                return
            
            logger.info(f"[MetaV2] ✅ Validation passed (output variation={diff:.6f})")
            
        except Exception as e:
            logger.error(f"[MetaV2] ❌ Validation error: {e}")
            self.model = None
            self.scaler = None
    
    def _extract_features(
        self,
        base_predictions: Dict[str, Dict[str, Any]],
        regime_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract feature vector from base agent predictions.
        
        Args:
            base_predictions: Dict of {model_name: {action, confidence}}
            regime_info: Optional market regime information
        
        Returns:
            (feature_vector, feature_dict) where feature_dict is for explainability
        """
        features = []
        feature_dict = {}
        
        # Expected base models
        base_models = ['xgb', 'lgbm', 'nhits', 'patchtst']
        
        # 1. Base agent signals (action + confidence for each model)
        for model in base_models:
            if model in base_predictions:
                pred = base_predictions[model]
                action = pred.get('action', 'HOLD')
                conf = pred.get('confidence', 0.5)
                
                # One-hot encode action (3 features per model)
                action_onehot = [0, 0, 0]
                action_idx = self.ACTION_TO_IDX.get(action, 1)  # Default HOLD
                action_onehot[action_idx] = 1
                
                features.extend(action_onehot)
                features.append(conf)
                
                feature_dict[f'{model}_action'] = action
                feature_dict[f'{model}_conf'] = conf
            else:
                # Missing model: neutral HOLD with 0.5 confidence
                features.extend([0, 1, 0, 0.5])  # HOLD action + 0.5 conf
                feature_dict[f'{model}_action'] = 'HOLD'
                feature_dict[f'{model}_conf'] = 0.5
        
        # 2. Aggregate statistics
        confidences = [
            base_predictions[m].get('confidence', 0.5)
            for m in base_models
            if m in base_predictions
        ]
        
        if confidences:
            mean_conf = np.mean(confidences)
            max_conf = np.max(confidences)
            min_conf = np.min(confidences)
            std_conf = np.std(confidences) if len(confidences) > 1 else 0.0
        else:
            mean_conf = max_conf = min_conf = 0.5
            std_conf = 0.0
        
        features.extend([mean_conf, max_conf, min_conf, std_conf])
        feature_dict['mean_confidence'] = mean_conf
        feature_dict['max_confidence'] = max_conf
        feature_dict['min_confidence'] = min_conf
        feature_dict['confidence_std'] = std_conf
        
        # 3. Voting statistics
        actions = [
            base_predictions[m].get('action', 'HOLD')
            for m in base_models
            if m in base_predictions
        ]
        
        action_counts = Counter(actions)
        num_buy = action_counts.get('BUY', 0)
        num_sell = action_counts.get('SELL', 0)
        num_hold = action_counts.get('HOLD', 0)
        total = len(actions) if actions else 4
        
        # Disagreement ratio: 1 - (most_common_count / total)
        if actions:
            most_common_count = action_counts.most_common(1)[0][1]
            disagreement = 1.0 - (most_common_count / total)
        else:
            disagreement = 0.0
        
        features.extend([
            num_buy / total,
            num_sell / total,
            num_hold / total,
            disagreement
        ])
        
        feature_dict['vote_buy_pct'] = num_buy / total
        feature_dict['vote_sell_pct'] = num_sell / total
        feature_dict['vote_hold_pct'] = num_hold / total
        feature_dict['disagreement'] = disagreement
        
        # 4. Optional regime features
        if self.enable_regime_features and regime_info:
            volatility = regime_info.get('volatility', 0.5)
            trend_strength = regime_info.get('trend_strength', 0.0)
            features.extend([volatility, trend_strength])
            feature_dict['volatility'] = volatility
            feature_dict['trend_strength'] = trend_strength
        else:
            # Padding for missing regime features
            features.extend([0.5, 0.0])
            feature_dict['volatility'] = 0.5  # Neutral
            feature_dict['trend_strength'] = 0.0  # No trend
        
        return np.array(features, dtype=np.float32).reshape(1, -1), feature_dict
    
    def _compute_weighted_ensemble(
        self,
        base_predictions: Dict[str, Dict[str, Any]],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Compute weighted ensemble decision as fallback.
        
        Args:
            base_predictions: Base agent predictions
            weights: Optional custom weights (default: equal weighting)
        
        Returns:
            (action, confidence)
        """
        if weights is None:
            # Default weights from metadata or equal
            weights = self.metadata.get('base_weights', {
                'xgb': 0.25,
                'lgbm': 0.25,
                'nhits': 0.30,
                'patchtst': 0.20
            })
        
        # Accumulate weighted votes
        vote_scores = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
        total_weight = 0.0
        
        for model, weight in weights.items():
            if model in base_predictions:
                pred = base_predictions[model]
                action = pred.get('action', 'HOLD')
                conf = pred.get('confidence', 0.5)
                
                # Weighted vote: weight * confidence
                vote_scores[action] += weight * conf
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for action in vote_scores:
                vote_scores[action] /= total_weight
        
        # Select action with highest score
        best_action = max(vote_scores, key=vote_scores.get)
        best_confidence = vote_scores[best_action]
        
        return best_action, best_confidence
    
    def _check_strong_consensus(
        self,
        base_predictions: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if base agents have strong consensus.
        
        Args:
            base_predictions: Base agent predictions
        
        Returns:
            (has_consensus, consensus_action, consensus_strength)
        """
        actions = [
            pred.get('action', 'HOLD')
            for pred in base_predictions.values()
        ]
        
        if not actions:
            return False, None, 0.0
        
        action_counts = Counter(actions)
        total = len(actions)
        most_common_action, most_common_count = action_counts.most_common(1)[0]
        
        consensus_strength = most_common_count / total
        
        has_consensus = consensus_strength >= self.consensus_threshold
        
        return has_consensus, most_common_action if has_consensus else None, consensus_strength
    
    def _analyze_disagreement(
        self,
        base_predictions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze disagreement level between base models.
        
        Returns:
            {
                'num_buy': int,
                'num_sell': int,
                'num_hold': int,
                'is_split_vote': bool (2 BUY, 2 SELL),
                'disagreement_ratio': float (0.0 = all agree, 1.0 = max disagreement),
                'action_distribution': dict
            }
        """
        actions = [
            pred.get('action', 'HOLD')
            for pred in base_predictions.values()
        ]
        
        if not actions:
            return {
                'num_buy': 0,
                'num_sell': 0,
                'num_hold': 0,
                'is_split_vote': False,
                'disagreement_ratio': 0.0,
                'action_distribution': {}
            }
        
        action_counts = Counter(actions)
        num_buy = action_counts.get('BUY', 0)
        num_sell = action_counts.get('SELL', 0)
        num_hold = action_counts.get('HOLD', 0)
        
        # Check for perfect split vote (2 BUY, 2 SELL)
        is_split_vote = (num_buy == 2 and num_sell == 2)
        
        # Disagreement ratio: 1 - (max_vote_share)
        # If all agree: max_vote_share = 1.0 → disagreement = 0.0
        # If perfect split: max_vote_share = 0.5 → disagreement = 0.5
        total = len(actions)
        max_vote_share = max(action_counts.values()) / total if total > 0 else 0
        disagreement_ratio = 1.0 - max_vote_share
        
        return {
            'num_buy': num_buy,
            'num_sell': num_sell,
            'num_hold': num_hold,
            'is_split_vote': is_split_vote,
            'disagreement_ratio': disagreement_ratio,
            'action_distribution': dict(action_counts)
        }
    
    def predict(
        self,
        base_predictions: Dict[str, Dict[str, Any]],
        regime_info: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        POLICY LAYER: Decide whether to use base ensemble or escalate to Arbiter.
        
        This method does NOT make trading decisions directly.
        It decides WHETHER the base ensemble is sufficient or if we need
        to escalate to Arbiter (Agent #5) for market understanding.
        
        Args:
            base_predictions: Dict of {model_name: {action, confidence}}
            regime_info: Optional market regime information
            symbol: Trading symbol (for logging)
        
        Returns:
            DEFER (use base ensemble):
            {
                "use_meta": false,
                "action": ensemble_action,
                "confidence": ensemble_confidence,
                "reason": "strong_consensus_hold" | "clear_ensemble" | ...,
                "base_ensemble_action": ensemble_action,
                "base_ensemble_confidence": ensemble_confidence
            }
            
            ESCALATE (call Arbiter):
            {
                "use_meta": true,
                "reason": "undecided_market" | "high_disagreement" | "split_vote",
                "base_ensemble_action": ensemble_action,
                "base_ensemble_confidence": ensemble_confidence,
                "disagreement_metrics": {...}
            }
        """
        symbol_str = f" [{symbol}]" if symbol else ""
        self.total_predictions += 1
        
        # Compute base ensemble fallback
        fallback_action, fallback_conf = self._compute_weighted_ensemble(base_predictions)
        
        result = {
            "use_meta": False,
            "action": fallback_action,
            "confidence": fallback_conf,
            "reason": "fallback_default",
            "base_ensemble_action": fallback_action,
            "base_ensemble_confidence": fallback_conf
        }
        
        # Safety check: Model not ready → DEFER
        if not self.is_ready():
            result['reason'] = "model_not_loaded"
            self.fallback_reasons['model_not_loaded'] += 1
            logger.debug(f"[MetaV2-Policy]{symbol_str} DEFER: model not loaded")
            return result
        
        # Safety check: Strong consensus → DEFER (respect base agreement)
        has_consensus, consensus_action, consensus_strength = self._check_strong_consensus(base_predictions)
        if has_consensus:
            result['action'] = consensus_action
            result['confidence'] = consensus_strength
            result['reason'] = f"strong_consensus_{consensus_action.lower()}"
            result['consensus_strength'] = consensus_strength
            self.fallback_reasons['strong_consensus'] += 1
            logger.debug(
                f"[MetaV2-Policy]{symbol_str} DEFER: strong consensus "
                f"({consensus_strength:.2%} → {consensus_action})"
            )
            return result
        
        # Analyze disagreement
        disagreement_metrics = self._analyze_disagreement(base_predictions)
        
        # Check for split vote (2 BUY, 2 SELL) → ESCALATE
        if disagreement_metrics['is_split_vote']:
            result['use_meta'] = True
            result['reason'] = "split_vote"
            result['disagreement_metrics'] = disagreement_metrics
            self.meta_overrides += 1
            logger.info(
                f"[MetaV2-Policy]{symbol_str} ⬆️ ESCALATE: Split vote detected "
                f"(BUY={disagreement_metrics['num_buy']}, SELL={disagreement_metrics['num_sell']}) "
                f"→ Calling Arbiter"
            )
            return result
        
        # Check for high disagreement (no clear majority) → ESCALATE
        if disagreement_metrics['disagreement_ratio'] >= 0.50:
            result['use_meta'] = True
            result['reason'] = "high_disagreement"
            result['disagreement_metrics'] = disagreement_metrics
            self.meta_overrides += 1
            logger.info(
                f"[MetaV2-Policy]{symbol_str} ⬆️ ESCALATE: High disagreement "
                f"({disagreement_metrics['disagreement_ratio']:.2%}) → Calling Arbiter"
            )
            return result
        
        # Check for low ensemble confidence → ESCALATE
        if fallback_conf < 0.55:
            result['use_meta'] = True
            result['reason'] = "low_ensemble_confidence"
            result['disagreement_metrics'] = disagreement_metrics
            self.meta_overrides += 1
            logger.info(
                f"[MetaV2-Policy]{symbol_str} ⬆️ ESCALATE: Low ensemble confidence "
                f"({fallback_conf:.3f}) → Calling Arbiter"
            )
            return result
        
        # Extract features for additional uncertainty check
        try:
            X, feature_dict = self._extract_features(base_predictions, regime_info)
        except Exception as e:
            result['reason'] = f"feature_extraction_error: {e}"
            self.fallback_reasons['feature_error'] += 1
            logger.error(f"[MetaV2-Policy]{symbol_str} Feature extraction failed: {e}")
            return result
        
        # Safety check: Feature dimension mismatch → DEFER
        if X.shape[1] != self.expected_feature_dim:
            result['reason'] = f"feature_dim_mismatch"
            self.fallback_reasons['dim_mismatch'] += 1
            logger.debug(f"[MetaV2-Policy]{symbol_str} DEFER: feature dimension mismatch")
            return result
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            result['reason'] = f"scaling_error"
            self.fallback_reasons['scaling_error'] += 1
            logger.debug(f"[MetaV2-Policy]{symbol_str} DEFER: scaling error")
            return result
        
        # Check meta uncertainty via prediction entropy
        try:
            proba = self.model.predict_proba(X_scaled)[0]  # Shape: (3,)
            
            # Compute entropy (higher = more uncertain)
            entropy = -np.sum(proba * np.log(proba + 1e-10))
            max_entropy = -np.log(1/3)  # log(3) for 3 classes
            normalized_entropy = entropy / max_entropy
            
            # High entropy → Market undecided → ESCALATE
            if normalized_entropy > 0.80:  # Very uncertain
                result['use_meta'] = True
                result['reason'] = "undecided_market"
                result['disagreement_metrics'] = disagreement_metrics
                result['entropy'] = float(normalized_entropy)
                self.meta_overrides += 1
                logger.info(
                    f"[MetaV2-Policy]{symbol_str} ⬆️ ESCALATE: Undecided market "
                    f"(entropy={normalized_entropy:.3f}) → Calling Arbiter"
                )
                return result
            
            # Clear decision → DEFER to ensemble
            result['reason'] = "clear_ensemble"
            result['entropy'] = float(normalized_entropy)
            self.fallback_reasons['clear_ensemble'] += 1
            logger.debug(
                f"[MetaV2-Policy]{symbol_str} DEFER: Clear ensemble decision "
                f"(entropy={normalized_entropy:.3f}, action={fallback_action})"
            )
            
            return result
            
        except Exception as e:
            result['reason'] = f"prediction_error"
            self.fallback_reasons['prediction_error'] += 1
            logger.error(f"[MetaV2-Policy]{symbol_str} Prediction failed: {e}")
            return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        override_rate = (
            self.meta_overrides / self.total_predictions
            if self.total_predictions > 0
            else 0.0
        )
        
        return {
            "total_predictions": self.total_predictions,
            "meta_overrides": self.meta_overrides,
            "override_rate": override_rate,
            "fallback_reasons": dict(self.fallback_reasons),
            "model_ready": self.is_ready(),
            "meta_threshold": self.meta_threshold,
            "consensus_threshold": self.consensus_threshold
        }
    
    def reset_statistics(self) -> None:
        """Reset runtime statistics."""
        self.total_predictions = 0
        self.meta_overrides = 0
        self.fallback_reasons.clear()
        logger.info("[MetaV2] Statistics reset")


# Backward compatibility alias
MetaPredictorAgentV2 = MetaAgentV2
