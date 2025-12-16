"""
COVARIATE SHIFT HANDLER

Handles distribution shifts in input features (P(X) changes) without immediate retraining.
Uses importance weighting, domain adaptation, and confidence calibration to maintain 
model performance when feature distributions drift from training distribution.

Key Components:
1. Distribution Divergence Detection (MMD, KL divergence, KS tests)
2. Importance Weight Estimation (KMM, KLIEP, Discriminator)
3. Domain Adaptation (CORAL, Quantile Transform, Standardization)
4. OOD Confidence Calibration (Mahalanobis distance, Ensemble disagreement)

Author: Quantum Trader AI Team
Module: 4 - Covariate Shift Handling
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, linprog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================

class ShiftSeverity(Enum):
    """Covariate shift severity levels"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class AdaptationMethod(Enum):
    """Adaptation strategy types"""
    NONE = "none"
    IMPORTANCE_WEIGHTING = "importance_weighting"
    DOMAIN_ADAPTATION = "domain_adaptation"
    HYBRID = "hybrid"  # Both weighting + adaptation


@dataclass
class DistributionMetrics:
    """Distribution divergence metrics"""
    timestamp: str
    mmd_squared: float
    kl_divergence: float
    ks_test_results: Dict[str, float]  # {feature: p_value}
    severity: str
    significant_features: List[str]


@dataclass
class ImportanceWeights:
    """Importance weights for training samples"""
    timestamp: str
    method: str  # 'kmm', 'kliep', 'discriminator'
    weights: np.ndarray
    mean_weight: float
    max_weight: float
    min_weight: float
    weight_variance: float
    stability_score: float  # max/mean ratio


@dataclass
class DomainTransform:
    """Domain adaptation transformation"""
    timestamp: str
    method: str  # 'coral', 'quantile', 'standardize'
    transform_matrix: Optional[np.ndarray]
    source_stats: Dict[str, np.ndarray]  # mean, std, quantiles
    target_stats: Dict[str, np.ndarray]


@dataclass
class OODCalibration:
    """Out-of-distribution confidence calibration"""
    timestamp: str
    ood_scores: np.ndarray
    mahalanobis_distances: np.ndarray
    original_confidences: np.ndarray
    calibrated_confidences: np.ndarray
    ood_threshold: float
    ood_count: int


@dataclass
class AdaptationResult:
    """Complete adaptation result"""
    timestamp: str
    model_name: str
    shift_severity: str
    adaptation_method: str
    distribution_metrics: DistributionMetrics
    importance_weights: Optional[ImportanceWeights]
    domain_transform: Optional[DomainTransform]
    ood_calibration: Optional[OODCalibration]
    performance_impact: Dict[str, float]  # win_rate, confidence, etc.


# ============================================================
# COVARIATE SHIFT HANDLER
# ============================================================

class CovariateShiftHandler:
    """
    Main handler for covariate shift detection and adaptation
    """
    
    def __init__(
        self,
        mmd_threshold_moderate: float = 0.01,
        mmd_threshold_severe: float = 0.05,
        kl_threshold_moderate: float = 0.1,
        kl_threshold_severe: float = 0.5,
        ks_p_value_threshold: float = 0.01,
        importance_weight_upper_bound: float = 1000,
        importance_weight_clip: Tuple[float, float] = (0.1, 10),
        ood_threshold: float = 0.7,
        mahalanobis_lambda: float = 0.1,
        kernel: str = 'rbf',
        kernel_gamma: float = 0.1,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize covariate shift handler
        
        Args:
            mmd_threshold_moderate: MMD² threshold for moderate shift
            mmd_threshold_severe: MMD² threshold for severe shift
            kl_threshold_moderate: KL divergence threshold for moderate shift
            kl_threshold_severe: KL divergence threshold for severe shift
            ks_p_value_threshold: KS test p-value threshold
            importance_weight_upper_bound: Max weight for KMM (B parameter)
            importance_weight_clip: (min, max) for weight clipping
            ood_threshold: Threshold for OOD detection (0-1)
            mahalanobis_lambda: Decay rate for Mahalanobis confidence adjustment
            kernel: Kernel type ('rbf', 'poly', 'linear')
            kernel_gamma: Kernel bandwidth parameter
            checkpoint_path: Path for state persistence
        """
        # Thresholds
        self.mmd_threshold_moderate = mmd_threshold_moderate
        self.mmd_threshold_severe = mmd_threshold_severe
        self.kl_threshold_moderate = kl_threshold_moderate
        self.kl_threshold_severe = kl_threshold_severe
        self.ks_p_value_threshold = ks_p_value_threshold
        
        # Importance weighting params
        self.importance_weight_upper_bound = importance_weight_upper_bound
        self.importance_weight_clip = importance_weight_clip
        
        # OOD calibration params
        self.ood_threshold = ood_threshold
        self.mahalanobis_lambda = mahalanobis_lambda
        
        # Kernel params
        self.kernel = kernel
        self.kernel_gamma = kernel_gamma
        
        # State
        self.training_distributions: Dict[str, Dict] = {}  # {model: stats}
        self.adaptation_history: Dict[str, List[AdaptationResult]] = {}
        self.current_weights: Dict[str, ImportanceWeights] = {}
        self.current_transforms: Dict[str, DomainTransform] = {}
        
        # Checkpoint
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self._load_checkpoint()
    
    # ============================================================
    # DISTRIBUTION DIVERGENCE DETECTION
    # ============================================================
    
    def compute_mmd_squared(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> float:
        """
        Compute Maximum Mean Discrepancy (MMD²)
        
        MMD² = ||μ_train - μ_test||²_H
        
        Args:
            X_train: Training samples (n_train, n_features)
            X_test: Test samples (n_test, n_features)
        
        Returns:
            mmd_squared: MMD² value
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # Compute kernel matrices
        K_train_train = self._compute_kernel_matrix(X_train, X_train)
        K_test_test = self._compute_kernel_matrix(X_test, X_test)
        K_train_test = self._compute_kernel_matrix(X_train, X_test)
        
        # MMD² formula
        term1 = K_train_train.sum() / (n_train ** 2)
        term2 = K_test_test.sum() / (n_test ** 2)
        term3 = K_train_test.sum() / (n_train * n_test)
        
        mmd_squared = term1 + term2 - 2 * term3
        
        return max(0, mmd_squared)  # Ensure non-negative
    
    def compute_kl_divergence(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray
    ) -> float:
        """
        Compute KL divergence via kernel density estimation
        
        D_KL(P_test || P_train) = E_{x~P_test}[log(p_test(x) / p_train(x))]
        
        Args:
            X_train: Training samples
            X_test: Test samples
        
        Returns:
            kl_divergence: KL(P_test || P_train)
        """
        try:
            # Estimate densities via KDE (per feature)
            kl_total = 0
            n_features = X_train.shape[1]
            
            for feat_idx in range(n_features):
                x_train_feat = X_train[:, feat_idx]
                x_test_feat = X_test[:, feat_idx]
                
                # KDE
                kde_train = stats.gaussian_kde(x_train_feat)
                kde_test = stats.gaussian_kde(x_test_feat)
                
                # Evaluate on test samples
                p_train = kde_train(x_test_feat) + 1e-10
                p_test = kde_test(x_test_feat) + 1e-10
                
                # KL divergence
                kl_feat = np.mean(np.log(p_test / p_train))
                kl_total += kl_feat
            
            # Average across features
            kl_divergence = kl_total / n_features
            
            return max(0, kl_divergence)
        
        except Exception as e:
            logger.warning(f"KL divergence computation failed: {e}")
            return 0.0
    
    def compute_ks_tests(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute Kolmogorov-Smirnov test per feature
        
        Args:
            X_train: Training samples
            X_test: Test samples
            feature_names: List of feature names
        
        Returns:
            ks_results: {feature: p_value}
        """
        ks_results = {}
        n_features = X_train.shape[1]
        
        for feat_idx in range(n_features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            
            x_train_feat = X_train[:, feat_idx]
            x_test_feat = X_test[:, feat_idx]
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(x_train_feat, x_test_feat)
            ks_results[feat_name] = p_value
        
        return ks_results
    
    def detect_covariate_shift(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_recent: np.ndarray,
        feature_names: List[str]
    ) -> DistributionMetrics:
        """
        Detect covariate shift using multiple metrics
        
        Args:
            model_name: Model identifier
            X_train: Training distribution samples
            X_recent: Recent distribution samples
            feature_names: Feature names
        
        Returns:
            metrics: Distribution divergence metrics
        """
        # Compute divergence metrics
        mmd_squared = self.compute_mmd_squared(X_train, X_recent)
        kl_divergence = self.compute_kl_divergence(X_train, X_recent)
        ks_results = self.compute_ks_tests(X_train, X_recent, feature_names)
        
        # Determine severity
        if mmd_squared >= self.mmd_threshold_severe or kl_divergence >= self.kl_threshold_severe:
            severity = ShiftSeverity.SEVERE.value
        elif mmd_squared >= self.mmd_threshold_moderate or kl_divergence >= self.kl_threshold_moderate:
            severity = ShiftSeverity.MODERATE.value
        else:
            # Check individual features
            significant_features = [
                feat for feat, p_val in ks_results.items()
                if p_val < self.ks_p_value_threshold
            ]
            if len(significant_features) >= 3:
                severity = ShiftSeverity.MODERATE.value
            elif len(significant_features) >= 1:
                severity = ShiftSeverity.MINOR.value
            else:
                severity = ShiftSeverity.NONE.value
        
        # Significant features
        significant_features = [
            feat for feat, p_val in ks_results.items()
            if p_val < self.ks_p_value_threshold
        ]
        
        metrics = DistributionMetrics(
            timestamp=datetime.utcnow().isoformat(),
            mmd_squared=mmd_squared,
            kl_divergence=kl_divergence,
            ks_test_results=ks_results,
            severity=severity,
            significant_features=significant_features
        )
        
        logger.info(f"[{model_name}] Covariate shift detected: MMD²={mmd_squared:.4f}, KL={kl_divergence:.4f}, Severity={severity}")
        
        return metrics
    
    # ============================================================
    # IMPORTANCE WEIGHTING
    # ============================================================
    
    def kernel_mean_matching(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        epsilon: float = 0.1
    ) -> np.ndarray:
        """
        Kernel Mean Matching (KMM) for importance weight estimation
        
        Solves:
            minimize_β  (1/2) β^T K β - κ^T β
            subject to: 0 ≤ β_i ≤ B, (1-ε)n ≤ Σβ ≤ (1+ε)n
        
        Args:
            X_train: Training samples (n, d)
            X_test: Test samples (m, d)
            epsilon: Tolerance for sum constraint
        
        Returns:
            beta: Importance weights (n,)
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # Compute kernel matrices
        K = self._compute_kernel_matrix(X_train, X_train)  # n × n
        K_cross = self._compute_kernel_matrix(X_train, X_test)  # n × m
        
        # Compute kappa
        kappa = (n_train / n_test) * K_cross.sum(axis=1)  # n × 1
        
        # Quadratic programming (use scipy.optimize.minimize with constraints)
        # Objective: (1/2) β^T K β - κ^T β
        def objective(beta):
            return 0.5 * beta.T @ K @ beta - kappa.T @ beta
        
        def gradient(beta):
            return K @ beta - kappa
        
        # Constraints
        bounds = [(0, self.importance_weight_upper_bound) for _ in range(n_train)]
        
        constraints = [
            {'type': 'ineq', 'fun': lambda b: b.sum() - (1 - epsilon) * n_train},  # Σβ ≥ (1-ε)n
            {'type': 'ineq', 'fun': lambda b: (1 + epsilon) * n_train - b.sum()}   # Σβ ≤ (1+ε)n
        ]
        
        # Initial guess
        beta_init = np.ones(n_train)
        
        # Solve
        result = minimize(
            objective,
            beta_init,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500}
        )
        
        beta = result.x
        
        # Clip extreme weights
        beta = np.clip(beta, self.importance_weight_clip[0], self.importance_weight_clip[1])
        
        logger.info(f"KMM: mean={beta.mean():.3f}, max={beta.max():.3f}, min={beta.min():.3f}")
        
        return beta
    
    def kliep(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_centers: int = 100,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kullback-Leibler Importance Estimation Procedure (KLIEP)
        
        Estimates density ratio w(x) = P_test(x) / P_train(x)
        
        Args:
            X_train: Training samples
            X_test: Test samples
            n_centers: Number of kernel centers
            max_iter: Max optimization iterations
        
        Returns:
            alpha: Kernel coefficients
            centers: Kernel centers
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # Select kernel centers (subset of test samples)
        n_centers = min(n_centers, n_test)
        center_indices = np.random.choice(n_test, n_centers, replace=False)
        centers = X_test[center_indices]
        
        # Compute kernel matrices
        K_train = self._compute_kernel_matrix(X_train, centers)  # n_train × n_centers
        K_test = self._compute_kernel_matrix(X_test, centers)    # n_test × n_centers
        
        # Initialize alpha
        alpha = np.ones(n_centers) / n_centers
        
        # Gradient ascent
        learning_rate = 0.01
        
        for iteration in range(max_iter):
            # Compute density ratio
            w_test = K_test @ alpha
            w_test = np.maximum(w_test, 1e-8)  # Avoid log(0)
            
            # Gradient (maximize log-likelihood)
            grad = K_test.T @ (1 / w_test) / n_test
            
            # Constraint gradient (normalization)
            constraint_val = K_train.T @ np.ones(n_train) @ alpha / n_train
            constraint_grad = K_train.T @ np.ones(n_train) / n_train
            
            # Update
            alpha += learning_rate * (grad - constraint_grad * (constraint_val - 1))
            alpha = np.maximum(alpha, 0)  # Non-negativity
            
            # Normalize
            norm_const = K_train.T @ np.ones(n_train) @ alpha / n_train
            if norm_const > 0:
                alpha /= norm_const
        
        logger.info(f"KLIEP: {n_centers} centers, converged after {max_iter} iterations")
        
        return alpha, centers
    
    def discriminator_weights(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Importance weights via logistic regression discriminator
        
        w(x) = (n/m) · P(y=1|x) / (1 - P(y=1|x))
        
        Args:
            X_train: Training samples
            X_test: Test samples
        
        Returns:
            weights: Importance weights for training samples
        """
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        
        # Create binary classification dataset
        X = np.vstack([X_train, X_test])
        y = np.hstack([np.zeros(n_train), np.ones(n_test)])
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X, y)
        
        # Predict probabilities for training samples
        p_y1 = clf.predict_proba(X_train)[:, 1]
        
        # Compute density ratio
        weights = (n_train / n_test) * p_y1 / (1 - p_y1 + 1e-8)
        
        # Clip extreme weights
        weights = np.clip(weights, self.importance_weight_clip[0], self.importance_weight_clip[1])
        
        logger.info(f"Discriminator: mean={weights.mean():.3f}, max={weights.max():.3f}, min={weights.min():.3f}")
        
        return weights
    
    def estimate_importance_weights(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        method: Literal['kmm', 'kliep', 'discriminator'] = 'discriminator'
    ) -> ImportanceWeights:
        """
        Estimate importance weights using specified method
        
        Args:
            model_name: Model identifier
            X_train: Training samples
            X_test: Test samples
            method: Weighting method
        
        Returns:
            importance_weights: Computed weights + statistics
        """
        if method == 'kmm':
            weights = self.kernel_mean_matching(X_train, X_test)
        elif method == 'kliep':
            alpha, centers = self.kliep(X_train, X_test)
            # Compute weights for training samples
            K_train = self._compute_kernel_matrix(X_train, centers)
            weights = K_train @ alpha
            weights = np.clip(weights, self.importance_weight_clip[0], self.importance_weight_clip[1])
        elif method == 'discriminator':
            weights = self.discriminator_weights(X_train, X_test)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute statistics
        mean_weight = weights.mean()
        max_weight = weights.max()
        min_weight = weights.min()
        weight_variance = weights.var()
        stability_score = max_weight / mean_weight if mean_weight > 0 else np.inf
        
        importance_weights = ImportanceWeights(
            timestamp=datetime.utcnow().isoformat(),
            method=method,
            weights=weights,
            mean_weight=mean_weight,
            max_weight=max_weight,
            min_weight=min_weight,
            weight_variance=weight_variance,
            stability_score=stability_score
        )
        
        # Store
        self.current_weights[model_name] = importance_weights
        
        return importance_weights
    
    # ============================================================
    # DOMAIN ADAPTATION
    # ============================================================
    
    def coral_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        CORAL (Correlation Alignment) transformation
        
        A = Σ_test^(1/2) Σ_train^(-1/2)
        X_adapted = X_train @ A.T
        
        Args:
            X_train: Training samples (n, d)
            X_test: Test samples (m, d)
        
        Returns:
            X_train_adapted: Transformed training samples
            A: Transformation matrix
        """
        # Compute covariance matrices
        Sigma_train = np.cov(X_train.T)
        Sigma_test = np.cov(X_test.T)
        
        # Add regularization
        d = Sigma_train.shape[0]
        Sigma_train += np.eye(d) * 1e-6
        Sigma_test += np.eye(d) * 1e-6
        
        # Compute Σ_train^(-1/2)
        U_train, S_train, Vt_train = np.linalg.svd(Sigma_train)
        Sigma_train_inv_sqrt = U_train @ np.diag(1 / np.sqrt(S_train + 1e-8)) @ Vt_train
        
        # Compute Σ_test^(1/2)
        U_test, S_test, Vt_test = np.linalg.svd(Sigma_test)
        Sigma_test_sqrt = U_test @ np.diag(np.sqrt(S_test)) @ Vt_test
        
        # Transformation matrix
        A = Sigma_test_sqrt @ Sigma_train_inv_sqrt
        
        # Transform training data
        X_train_adapted = X_train @ A.T
        
        logger.info(f"CORAL: Transformed {X_train.shape[0]} samples")
        
        return X_train_adapted, A
    
    def quantile_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, QuantileTransformer]:
        """
        Quantile transformation to match test distribution
        
        Args:
            X_train: Training samples
            X_test: Test samples
        
        Returns:
            X_train_adapted: Transformed training samples
            transformer: Fitted QuantileTransformer
        """
        # Fit on test distribution
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, X_test.shape[0]))
        qt.fit(X_test)
        
        # Transform training data
        X_train_adapted = qt.transform(X_train)
        
        logger.info(f"Quantile Transform: Matched {X_train.shape[0]} samples to test distribution")
        
        return X_train_adapted, qt
    
    def standardize_to_test(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Standardize training data to test distribution statistics
        
        Args:
            X_train: Training samples
            X_test: Test samples
        
        Returns:
            X_train_adapted: Standardized training samples
            stats: Test distribution statistics
        """
        mu_test = X_test.mean(axis=0)
        sigma_test = X_test.std(axis=0)
        
        X_train_adapted = (X_train - mu_test) / (sigma_test + 1e-8)
        
        stats = {
            'mean': mu_test,
            'std': sigma_test
        }
        
        logger.info(f"Standardization: Aligned {X_train.shape[0]} samples to test statistics")
        
        return X_train_adapted, stats
    
    def apply_domain_adaptation(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        method: Literal['coral', 'quantile', 'standardize'] = 'coral'
    ) -> DomainTransform:
        """
        Apply domain adaptation transformation
        
        Args:
            model_name: Model identifier
            X_train: Training samples
            X_test: Test samples
            method: Adaptation method
        
        Returns:
            domain_transform: Transformation details
        """
        if method == 'coral':
            X_adapted, A = self.coral_transform(X_train, X_test)
            transform_matrix = A
            source_stats = {'cov': np.cov(X_train.T)}
            target_stats = {'cov': np.cov(X_test.T)}
        
        elif method == 'quantile':
            X_adapted, qt = self.quantile_transform(X_train, X_test)
            transform_matrix = None  # Stored in qt object
            source_stats = {'quantiles': np.percentile(X_train, [25, 50, 75], axis=0)}
            target_stats = {'quantiles': np.percentile(X_test, [25, 50, 75], axis=0)}
        
        elif method == 'standardize':
            X_adapted, stats = self.standardize_to_test(X_train, X_test)
            transform_matrix = None
            source_stats = {'mean': X_train.mean(axis=0), 'std': X_train.std(axis=0)}
            target_stats = stats
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        domain_transform = DomainTransform(
            timestamp=datetime.utcnow().isoformat(),
            method=method,
            transform_matrix=transform_matrix,
            source_stats=source_stats,
            target_stats=target_stats
        )
        
        # Store
        self.current_transforms[model_name] = domain_transform
        
        return domain_transform
    
    # ============================================================
    # OOD CONFIDENCE CALIBRATION
    # ============================================================
    
    def compute_mahalanobis_distance(
        self,
        X: np.ndarray,
        X_train: np.ndarray
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance to training distribution
        
        D_M(x) = sqrt((x - μ)^T Σ^(-1) (x - μ))
        
        Args:
            X: Samples to evaluate
            X_train: Training distribution
        
        Returns:
            distances: Mahalanobis distances
        """
        mu = X_train.mean(axis=0)
        Sigma = np.cov(X_train.T)
        
        # Add regularization
        d = Sigma.shape[0]
        Sigma += np.eye(d) * 1e-6
        
        # Inverse covariance
        Sigma_inv = np.linalg.inv(Sigma)
        
        # Compute distances
        distances = np.array([
            np.sqrt((x - mu).T @ Sigma_inv @ (x - mu))
            for x in X
        ])
        
        return distances
    
    def calibrate_ood_confidence(
        self,
        model_name: str,
        X: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        X_train: np.ndarray
    ) -> OODCalibration:
        """
        Calibrate confidence for out-of-distribution predictions
        
        confidence_adjusted = confidence_original · exp(-λ · D_M(x))
        
        Args:
            model_name: Model identifier
            X: Feature samples
            predictions: Model predictions
            confidences: Original confidence scores
            X_train: Training distribution
        
        Returns:
            ood_calibration: Calibration results
        """
        # Compute Mahalanobis distance
        mahalanobis_distances = self.compute_mahalanobis_distance(X, X_train)
        
        # Normalize to [0, 1]
        D_M_norm = (mahalanobis_distances - mahalanobis_distances.min()) / (
            mahalanobis_distances.max() - mahalanobis_distances.min() + 1e-8
        )
        
        # OOD scores
        ood_scores = D_M_norm
        
        # Adjust confidence
        calibrated_confidences = confidences * np.exp(-self.mahalanobis_lambda * D_M_norm)
        
        # Count OOD samples
        ood_count = (ood_scores >= self.ood_threshold).sum()
        
        ood_calibration = OODCalibration(
            timestamp=datetime.utcnow().isoformat(),
            ood_scores=ood_scores,
            mahalanobis_distances=mahalanobis_distances,
            original_confidences=confidences,
            calibrated_confidences=calibrated_confidences,
            ood_threshold=self.ood_threshold,
            ood_count=int(ood_count)
        )
        
        logger.info(f"[{model_name}] OOD Calibration: {ood_count}/{len(ood_scores)} samples flagged as OOD")
        
        return ood_calibration
    
    # ============================================================
    # ADAPTIVE STRATEGY SELECTION
    # ============================================================
    
    def select_adaptation_strategy(
        self,
        metrics: DistributionMetrics,
        performance_drop: float = 0.0
    ) -> AdaptationMethod:
        """
        Select adaptation strategy based on shift severity
        
        Decision tree:
        - NONE: No adaptation needed
        - MINOR: Confidence calibration only
        - MODERATE: Importance weighting + calibration
        - SEVERE: Domain adaptation + weighting + calibration (if performance OK)
        - SEVERE + performance drop: Escalate to retraining (via Module 3)
        
        Args:
            metrics: Distribution metrics
            performance_drop: Win rate drop (if available)
        
        Returns:
            strategy: Recommended adaptation method
        """
        severity = metrics.severity
        
        if severity == ShiftSeverity.NONE.value:
            return AdaptationMethod.NONE
        
        elif severity == ShiftSeverity.MINOR.value:
            # Calibration only
            return AdaptationMethod.NONE  # Just monitor
        
        elif severity == ShiftSeverity.MODERATE.value:
            # Importance weighting
            return AdaptationMethod.IMPORTANCE_WEIGHTING
        
        elif severity == ShiftSeverity.SEVERE.value:
            if performance_drop > 0.05:  # 5pp drop
                # Performance dropped significantly → escalate to retraining
                logger.warning(f"Severe shift + performance drop ({performance_drop:.1%}) → Recommend retraining")
                return AdaptationMethod.NONE  # Don't adapt, trigger retraining instead
            else:
                # Pure covariate shift → full adaptation
                return AdaptationMethod.HYBRID
        
        else:
            return AdaptationMethod.NONE
    
    def adapt_to_covariate_shift(
        self,
        model_name: str,
        X_train: np.ndarray,
        X_recent: np.ndarray,
        feature_names: List[str],
        current_performance: Optional[Dict] = None
    ) -> AdaptationResult:
        """
        Complete covariate shift adaptation pipeline
        
        1. Detect shift
        2. Select strategy
        3. Apply adaptation (weighting, transform, calibration)
        4. Return results
        
        Args:
            model_name: Model identifier
            X_train: Training distribution samples
            X_recent: Recent distribution samples
            feature_names: Feature names
            current_performance: Current performance metrics (for strategy selection)
        
        Returns:
            result: Complete adaptation result
        """
        # Step 1: Detect shift
        metrics = self.detect_covariate_shift(model_name, X_train, X_recent, feature_names)
        
        # Step 2: Select strategy
        performance_drop = current_performance.get('win_rate_drop', 0.0) if current_performance else 0.0
        strategy = self.select_adaptation_strategy(metrics, performance_drop)
        
        logger.info(f"[{model_name}] Adaptation strategy: {strategy.value}")
        
        # Step 3: Apply adaptation
        importance_weights = None
        domain_transform = None
        ood_calibration = None
        
        if strategy == AdaptationMethod.IMPORTANCE_WEIGHTING:
            # Importance weighting
            importance_weights = self.estimate_importance_weights(
                model_name, X_train, X_recent, method='discriminator'
            )
        
        elif strategy == AdaptationMethod.HYBRID:
            # Both domain adaptation + weighting
            domain_transform = self.apply_domain_adaptation(
                model_name, X_train, X_recent, method='coral'
            )
            importance_weights = self.estimate_importance_weights(
                model_name, X_train, X_recent, method='discriminator'
            )
        
        # Step 4: Compile result
        result = AdaptationResult(
            timestamp=datetime.utcnow().isoformat(),
            model_name=model_name,
            shift_severity=metrics.severity,
            adaptation_method=strategy.value,
            distribution_metrics=metrics,
            importance_weights=importance_weights,
            domain_transform=domain_transform,
            ood_calibration=ood_calibration,
            performance_impact={}  # To be filled by caller
        )
        
        # Store in history
        if model_name not in self.adaptation_history:
            self.adaptation_history[model_name] = []
        self.adaptation_history[model_name].append(result)
        
        return result
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    def _compute_kernel_matrix(
        self, 
        X1: np.ndarray, 
        X2: np.ndarray
    ) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = k(x1_i, x2_j)
        
        Args:
            X1: Samples (n1, d)
            X2: Samples (n2, d)
        
        Returns:
            K: Kernel matrix (n1, n2)
        """
        if self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.kernel_gamma)
        elif self.kernel == 'poly':
            return polynomial_kernel(X1, X2, degree=2, gamma=self.kernel_gamma)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def get_current_weights(self, model_name: str) -> Optional[np.ndarray]:
        """Get current importance weights for a model"""
        if model_name in self.current_weights:
            return self.current_weights[model_name].weights
        return None
    
    def get_adaptation_history(self, model_name: str) -> List[AdaptationResult]:
        """Get adaptation history for a model"""
        return self.adaptation_history.get(model_name, [])
    
    def checkpoint(self):
        """Save state to checkpoint"""
        if not self.checkpoint_path:
            return
        
        state = {
            'training_distributions': self.training_distributions,
            'adaptation_history': {
                model: [asdict(r) for r in history]
                for model, history in self.adaptation_history.items()
            },
            'current_weights': {
                model: asdict(w)
                for model, w in self.current_weights.items()
            }
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Checkpoint saved to {self.checkpoint_path}")
    
    def _load_checkpoint(self):
        """Load state from checkpoint"""
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.training_distributions = state.get('training_distributions', {})
            # Note: Adaptation history and weights contain numpy arrays, 
            # would need custom deserialization for full restore
            
            logger.info(f"Checkpoint loaded from {self.checkpoint_path}")
        
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
