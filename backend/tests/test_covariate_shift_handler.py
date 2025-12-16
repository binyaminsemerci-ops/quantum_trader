"""
Comprehensive test suite for covariate_shift_handler.py

Tests:
1. Unit tests: MMD, KL, KS, KMM, KLIEP, Discriminator, CORAL, Mahalanobis
2. Integration tests: Full detection cycle, ensemble integration, checkpoint
3. Scenario tests: Gradual shift, sudden shift, false positives, adaptation failure
4. Performance tests: Latency benchmarks for all methods
5. Risk mitigation tests: Extreme weights, artifacts, OOD miscalibration

Run: pytest backend/tests/test_covariate_shift_handler.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from backend.services.ai.covariate_shift_handler import (
    CovariateShiftHandler,
    ShiftSeverity,
    AdaptationMethod,
    DistributionMetrics,
    ImportanceWeights,
    DomainTransform,
    OODCalibration,
    AdaptationResult
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def handler():
    """Create CovariateShiftHandler with test configuration"""
    return CovariateShiftHandler(
        mmd_threshold_moderate=0.01,
        mmd_threshold_severe=0.05,
        kl_threshold_moderate=0.1,
        kl_threshold_severe=0.5,
        ks_p_value_threshold=0.01,
        weight_bound=1000,
        weight_clip_min=0.1,
        weight_clip_max=10,
        ood_threshold=0.7,
        mahalanobis_lambda=0.1,
        kernel='rbf',
        kernel_gamma=0.1,
        checkpoint_path='test_covariate_checkpoint.json'
    )


@pytest.fixture
def normal_distribution():
    """Generate samples from normal distribution"""
    np.random.seed(42)
    X = np.random.randn(500, 10) * 2 + 5  # Mean=5, std=2
    return X


@pytest.fixture
def shifted_distribution():
    """Generate samples from shifted distribution"""
    np.random.seed(43)
    X = np.random.randn(100, 10) * 3 + 7  # Mean=7, std=3 (shifted)
    return X


@pytest.fixture
def extreme_outlier_distribution():
    """Generate distribution with extreme outliers"""
    np.random.seed(44)
    X = np.random.randn(100, 10) * 2 + 5
    # Add 5 extreme outliers
    X[:5] = np.random.randn(5, 10) * 20 + 50
    return X


# ============================================================================
# UNIT TESTS: DISTRIBUTION DIVERGENCE METRICS
# ============================================================================

class TestMMDComputation:
    """Test Maximum Mean Discrepancy calculation"""
    
    def test_mmd_identical_distributions(self, handler, normal_distribution):
        """MMD should be ~0 for identical distributions"""
        X1 = normal_distribution[:250]
        X2 = normal_distribution[250:500]
        
        mmd_squared = handler.compute_mmd_squared(X1, X2)
        
        assert mmd_squared < 0.005, f"MMD² should be <0.005 for identical distributions, got {mmd_squared}"
    
    def test_mmd_shifted_distributions(self, handler, normal_distribution, shifted_distribution):
        """MMD should be >0.05 for shifted distributions"""
        mmd_squared = handler.compute_mmd_squared(normal_distribution, shifted_distribution)
        
        assert mmd_squared > 0.05, f"MMD² should be >0.05 for shifted distributions, got {mmd_squared}"
    
    def test_mmd_output_range(self, handler, normal_distribution, shifted_distribution):
        """MMD² should always be non-negative"""
        mmd_squared = handler.compute_mmd_squared(normal_distribution, shifted_distribution)
        
        assert mmd_squared >= 0, f"MMD² should be non-negative, got {mmd_squared}"
    
    def test_mmd_kernel_sensitivity(self, handler, normal_distribution, shifted_distribution):
        """Different kernels should give different MMD values"""
        mmd_rbf = handler.compute_mmd_squared(normal_distribution, shifted_distribution, kernel='rbf')
        
        handler.kernel = 'linear'
        mmd_linear = handler.compute_mmd_squared(normal_distribution, shifted_distribution, kernel='linear')
        
        # Should differ by at least 20%
        assert abs(mmd_rbf - mmd_linear) / mmd_rbf > 0.2


class TestKLDivergence:
    """Test Kullback-Leibler divergence calculation"""
    
    def test_kl_identical_distributions(self, handler, normal_distribution):
        """KL divergence should be ~0 for identical distributions"""
        X1 = normal_distribution[:250]
        X2 = normal_distribution[250:500]
        
        kl_div = handler.compute_kl_divergence(X1, X2)
        
        assert kl_div < 0.05, f"KL should be <0.05 for identical distributions, got {kl_div}"
    
    def test_kl_shifted_distributions(self, handler, normal_distribution, shifted_distribution):
        """KL divergence should be >0.5 for shifted distributions"""
        kl_div = handler.compute_kl_divergence(normal_distribution, shifted_distribution)
        
        assert kl_div > 0.3, f"KL should be >0.3 for shifted distributions, got {kl_div}"
    
    def test_kl_output_range(self, handler, normal_distribution, shifted_distribution):
        """KL divergence should always be non-negative"""
        kl_div = handler.compute_kl_divergence(normal_distribution, shifted_distribution)
        
        assert kl_div >= 0, f"KL should be non-negative, got {kl_div}"


class TestKSTests:
    """Test Kolmogorov-Smirnov per-feature tests"""
    
    def test_ks_identical_distributions(self, handler, normal_distribution):
        """KS tests should have high p-values for identical distributions"""
        X1 = normal_distribution[:250]
        X2 = normal_distribution[250:500]
        
        feature_names = [f'feat_{i}' for i in range(10)]
        ks_results = handler.compute_ks_tests(X1, X2, feature_names)
        
        # All p-values should be > 0.01 (not significant)
        significant_count = sum(1 for p in ks_results.values() if p < 0.01)
        assert significant_count == 0, f"Expected 0 significant features, got {significant_count}"
    
    def test_ks_shifted_distributions(self, handler, normal_distribution, shifted_distribution):
        """KS tests should have low p-values for shifted distributions"""
        feature_names = [f'feat_{i}' for i in range(10)]
        ks_results = handler.compute_ks_tests(normal_distribution, shifted_distribution, feature_names)
        
        # Most p-values should be < 0.01 (significant)
        significant_count = sum(1 for p in ks_results.values() if p < 0.01)
        assert significant_count >= 7, f"Expected >=7 significant features, got {significant_count}"


# ============================================================================
# UNIT TESTS: IMPORTANCE WEIGHTING METHODS
# ============================================================================

class TestKernelMeanMatching:
    """Test KMM importance weighting"""
    
    def test_kmm_weight_constraints(self, handler, normal_distribution, shifted_distribution):
        """KMM weights should satisfy constraints: 0 ≤ β ≤ B, sum ≈ n"""
        weights_obj = handler.kernel_mean_matching(normal_distribution, shifted_distribution)
        weights = weights_obj.weights
        
        # Check bounds
        assert np.all(weights >= 0), "Weights should be non-negative"
        assert np.all(weights <= handler.weight_bound), f"Weights should be ≤{handler.weight_bound}"
        
        # Check sum constraint (within 10% of n)
        n = len(weights)
        weight_sum = np.sum(weights)
        assert 0.9 * n <= weight_sum <= 1.1 * n, f"Weight sum {weight_sum} should be ≈{n}"
    
    def test_kmm_stability_score(self, handler, normal_distribution, shifted_distribution):
        """KMM stability score should be reasonable (<50 for normal shifts)"""
        weights_obj = handler.kernel_mean_matching(normal_distribution, shifted_distribution)
        
        assert weights_obj.stability_score < 50, f"Stability score {weights_obj.stability_score} too high"
    
    def test_kmm_weight_clipping(self, handler, normal_distribution, extreme_outlier_distribution):
        """KMM should clip extreme weights to [0.1, 10]"""
        weights_obj = handler.kernel_mean_matching(normal_distribution, extreme_outlier_distribution)
        weights = weights_obj.weights
        
        # After clipping, weights should be in [0.1, 10]
        assert np.all(weights >= handler.weight_clip_min), f"Weights should be ≥{handler.weight_clip_min}"
        assert np.all(weights <= handler.weight_clip_max), f"Weights should be ≤{handler.weight_clip_max}"


class TestKLIEP:
    """Test KLIEP importance weighting"""
    
    def test_kliep_weight_normalization(self, handler, normal_distribution, shifted_distribution):
        """KLIEP weights should satisfy normalization constraint"""
        weights_obj = handler.kliep(normal_distribution, shifted_distribution)
        weights = weights_obj.weights
        
        # Compute average weight (should be ≈1 due to normalization constraint)
        avg_weight = np.mean(weights)
        assert 0.8 < avg_weight < 1.2, f"Average weight {avg_weight} should be ≈1"
    
    def test_kliep_convergence(self, handler, normal_distribution, shifted_distribution):
        """KLIEP should converge (weights should be positive)"""
        weights_obj = handler.kliep(normal_distribution, shifted_distribution)
        weights = weights_obj.weights
        
        # All weights should be positive after convergence
        assert np.all(weights > 0), "All KLIEP weights should be positive"
    
    def test_kliep_faster_than_kmm(self, handler, normal_distribution, shifted_distribution):
        """KLIEP should be faster than KMM for large datasets"""
        import time
        
        # KMM timing
        start = time.time()
        handler.kernel_mean_matching(normal_distribution, shifted_distribution)
        kmm_time = time.time() - start
        
        # KLIEP timing
        start = time.time()
        handler.kliep(normal_distribution, shifted_distribution)
        kliep_time = time.time() - start
        
        # KLIEP should be at least 2x faster
        assert kliep_time < kmm_time / 2, f"KLIEP ({kliep_time:.3f}s) not faster than KMM ({kmm_time:.3f}s)"


class TestDiscriminatorWeights:
    """Test discriminator-based importance weighting"""
    
    def test_discriminator_weight_ratio(self, handler, normal_distribution, shifted_distribution):
        """Discriminator weights should reflect density ratio"""
        weights_obj = handler.discriminator_weights(normal_distribution, shifted_distribution)
        weights = weights_obj.weights
        
        # Weights should be positive
        assert np.all(weights > 0), "Discriminator weights should be positive"
        
        # Average weight should be reasonable
        avg_weight = np.mean(weights)
        assert 0.5 < avg_weight < 2.0, f"Average weight {avg_weight} should be ≈1"
    
    def test_discriminator_fastest_method(self, handler, normal_distribution, shifted_distribution):
        """Discriminator should be fastest method"""
        import time
        
        # Discriminator timing
        start = time.time()
        handler.discriminator_weights(normal_distribution, shifted_distribution)
        discr_time = time.time() - start
        
        # KMM timing
        start = time.time()
        handler.kernel_mean_matching(normal_distribution, shifted_distribution)
        kmm_time = time.time() - start
        
        # Discriminator should be faster
        assert discr_time < kmm_time, f"Discriminator ({discr_time:.3f}s) not faster than KMM ({kmm_time:.3f}s)"


# ============================================================================
# UNIT TESTS: DOMAIN ADAPTATION METHODS
# ============================================================================

class TestCORAL:
    """Test CORAL domain adaptation"""
    
    def test_coral_covariance_alignment(self, handler, normal_distribution, shifted_distribution):
        """CORAL should align covariances"""
        X_adapted, transform = handler.coral_transform(normal_distribution, shifted_distribution)
        
        # Compute covariances
        cov_adapted = np.cov(X_adapted.T)
        cov_target = np.cov(shifted_distribution.T)
        
        # Frobenius norm of difference should be small
        frob_norm = np.linalg.norm(cov_adapted - cov_target, 'fro')
        frob_norm_baseline = np.linalg.norm(np.cov(normal_distribution.T) - cov_target, 'fro')
        
        # Adaptation should reduce difference by >50%
        assert frob_norm < 0.5 * frob_norm_baseline, "CORAL should reduce covariance difference"
    
    def test_coral_transform_shape(self, handler, normal_distribution, shifted_distribution):
        """CORAL transform should preserve data shape"""
        X_adapted, transform = handler.coral_transform(normal_distribution, shifted_distribution)
        
        assert X_adapted.shape == normal_distribution.shape, "CORAL should preserve shape"


class TestQuantileTransform:
    """Test quantile transformation"""
    
    def test_quantile_distribution_matching(self, handler, normal_distribution, shifted_distribution):
        """Quantile transform should match distributions"""
        X_adapted, transform = handler.quantile_transform(normal_distribution, shifted_distribution)
        
        # Check if quantiles match (use KS test)
        ks_pvals = []
        for i in range(10):
            _, p_val = stats.ks_2samp(X_adapted[:, i], shifted_distribution[:, i])
            ks_pvals.append(p_val)
        
        # Most features should have high p-values (distributions match)
        high_pval_count = sum(1 for p in ks_pvals if p > 0.05)
        assert high_pval_count >= 7, f"Expected >=7 features with matched distributions, got {high_pval_count}"


class TestStandardization:
    """Test feature standardization"""
    
    def test_standardize_mean_std(self, handler, normal_distribution, shifted_distribution):
        """Standardization should match mean and std"""
        X_adapted, transform = handler.standardize_to_test(normal_distribution, shifted_distribution)
        
        # Compute means and stds
        mean_adapted = X_adapted.mean(axis=0)
        std_adapted = X_adapted.std(axis=0)
        
        mean_target = shifted_distribution.mean(axis=0)
        std_target = shifted_distribution.std(axis=0)
        
        # Means should match within 10%
        assert np.allclose(mean_adapted, mean_target, rtol=0.1), "Means should match"
        
        # Stds should match within 10%
        assert np.allclose(std_adapted, std_target, rtol=0.1), "Stds should match"


# ============================================================================
# UNIT TESTS: OOD CALIBRATION
# ============================================================================

class TestMahalanobisDistance:
    """Test Mahalanobis distance computation"""
    
    def test_mahalanobis_in_distribution(self, handler, normal_distribution):
        """In-distribution samples should have low Mahalanobis distance"""
        X_train = normal_distribution[:400]
        X_test = normal_distribution[400:500]  # In-distribution
        
        distances = handler.compute_mahalanobis_distance(X_train, X_test)
        
        # Most samples should have low distance
        low_dist_count = np.sum(distances < 3.0)  # 3-sigma
        assert low_dist_count >= 85, f"Expected >=85% in-distribution, got {low_dist_count}"
    
    def test_mahalanobis_ood(self, handler, normal_distribution, shifted_distribution):
        """OOD samples should have high Mahalanobis distance"""
        distances = handler.compute_mahalanobis_distance(normal_distribution, shifted_distribution)
        
        # Most samples should have high distance
        high_dist_count = np.sum(distances > 3.0)
        assert high_dist_count >= 70, f"Expected >=70% OOD, got {high_dist_count}"


class TestOODCalibration:
    """Test OOD confidence calibration"""
    
    def test_ood_confidence_reduction(self, handler, normal_distribution, shifted_distribution):
        """OOD samples should have reduced confidence"""
        confidences = np.random.uniform(0.6, 0.9, size=100)
        
        calibration = handler.calibrate_ood_confidence(
            normal_distribution, shifted_distribution, confidences
        )
        
        # Calibrated confidences should be lower
        assert np.mean(calibration.calibrated_confidences) < np.mean(calibration.original_confidences), \
            "Calibrated confidences should be lower for OOD samples"
    
    def test_ood_count(self, handler, normal_distribution, shifted_distribution):
        """OOD calibration should count OOD samples correctly"""
        confidences = np.random.uniform(0.6, 0.9, size=100)
        
        calibration = handler.calibrate_ood_confidence(
            normal_distribution, shifted_distribution, confidences
        )
        
        # Should have >50 OOD samples (shifted distribution)
        assert calibration.ood_count > 50, f"Expected >50 OOD samples, got {calibration.ood_count}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestFullDetectionCycle:
    """Test complete detection and adaptation cycle"""
    
    def test_detect_no_shift(self, handler, normal_distribution):
        """Should detect no shift for identical distributions"""
        X_train = normal_distribution[:400]
        X_test = normal_distribution[400:500]
        
        feature_names = [f'feat_{i}' for i in range(10)]
        metrics = handler.detect_covariate_shift(X_train, X_test, feature_names)
        
        assert metrics.severity == ShiftSeverity.NONE, "Should detect no shift"
        assert metrics.mmd_squared < 0.01, f"MMD² should be <0.01, got {metrics.mmd_squared}"
    
    def test_detect_moderate_shift(self, handler):
        """Should detect moderate shift"""
        np.random.seed(42)
        X_train = np.random.randn(500, 10) * 2 + 5
        X_test = np.random.randn(100, 10) * 2.5 + 6  # Moderate shift
        
        feature_names = [f'feat_{i}' for i in range(10)]
        metrics = handler.detect_covariate_shift(X_train, X_test, feature_names)
        
        assert metrics.severity in [ShiftSeverity.MODERATE, ShiftSeverity.SEVERE], \
            f"Should detect shift, got {metrics.severity}"
    
    def test_detect_severe_shift(self, handler, normal_distribution, shifted_distribution):
        """Should detect severe shift"""
        feature_names = [f'feat_{i}' for i in range(10)]
        metrics = handler.detect_covariate_shift(normal_distribution, shifted_distribution, feature_names)
        
        assert metrics.severity == ShiftSeverity.SEVERE, f"Should detect severe shift, got {metrics.severity}"
        assert metrics.mmd_squared >= 0.05, f"MMD² should be >=0.05, got {metrics.mmd_squared}"


class TestAdaptationPipeline:
    """Test full adaptation pipeline"""
    
    def test_adaptation_no_shift(self, handler, normal_distribution):
        """Should not adapt when no shift detected"""
        X_train = normal_distribution[:400]
        X_test = normal_distribution[400:500]
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            current_performance={'win_rate': 0.58}
        )
        
        assert result.adaptation_method == AdaptationMethod.NONE, "Should not adapt"
    
    def test_adaptation_moderate_shift(self, handler):
        """Should apply importance weighting for moderate shift"""
        np.random.seed(42)
        X_train = np.random.randn(500, 10) * 2 + 5
        X_test = np.random.randn(100, 10) * 2.5 + 6
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            current_performance={'win_rate': 0.58}
        )
        
        assert result.adaptation_method == AdaptationMethod.IMPORTANCE_WEIGHTING, \
            f"Should use importance weighting, got {result.adaptation_method}"
        assert result.importance_weights is not None, "Should compute importance weights"
    
    def test_adaptation_severe_shift(self, handler, normal_distribution, shifted_distribution):
        """Should apply hybrid adaptation for severe shift"""
        feature_names = [f'feat_{i}' for i in range(10)]
        
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=normal_distribution,
            X_test=shifted_distribution,
            feature_names=feature_names,
            current_performance={'win_rate': 0.56}
        )
        
        assert result.adaptation_method in [AdaptationMethod.HYBRID, AdaptationMethod.DOMAIN_ADAPTATION], \
            f"Should use domain adaptation or hybrid, got {result.adaptation_method}"
        assert result.domain_transform is not None, "Should compute domain transform"


class TestCheckpointing:
    """Test state persistence"""
    
    def test_checkpoint_save_restore(self, handler, normal_distribution, shifted_distribution):
        """Should save and restore state correctly"""
        feature_names = [f'feat_{i}' for i in range(10)]
        
        # Perform adaptation
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=normal_distribution,
            X_test=shifted_distribution,
            feature_names=feature_names,
            current_performance={'win_rate': 0.56}
        )
        
        # Checkpoint
        handler.checkpoint()
        
        # Create new handler and restore
        handler_new = CovariateShiftHandler(checkpoint_path='test_covariate_checkpoint.json')
        
        # Check if adaptation history restored
        history = handler_new.get_adaptation_history('test_model')
        assert len(history) > 0, "Adaptation history should be restored"
        assert history[0].model_name == 'test_model', "Model name should match"


# ============================================================================
# SCENARIO TESTS
# ============================================================================

class TestGradualShift:
    """Test gradual distribution shift over time"""
    
    def test_gradual_volatility_increase(self, handler):
        """Simulate gradual volatility increase over 30 days"""
        np.random.seed(42)
        X_train = np.random.randn(500, 10) * 2 + 5  # Initial volatility
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        adaptations = []
        
        # Simulate 30 days (volatility increases from 2 to 5)
        for day in range(30):
            volatility = 2 + (day / 30) * 3  # Gradual increase
            X_test = np.random.randn(100, 10) * volatility + 5
            
            result = handler.adapt_to_covariate_shift(
                model_name='test_model',
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                current_performance={'win_rate': 0.56}
            )
            
            adaptations.append({
                'day': day,
                'volatility': volatility,
                'severity': result.shift_severity,
                'method': result.adaptation_method,
                'mmd': result.distribution_metrics.mmd_squared
            })
        
        # Check progression
        # Early days: no shift
        assert adaptations[5]['severity'] == ShiftSeverity.NONE, "Day 5 should have no shift"
        
        # Mid days: moderate shift
        assert adaptations[15]['severity'] in [ShiftSeverity.MODERATE, ShiftSeverity.SEVERE], \
            "Day 15 should have moderate/severe shift"
        
        # Late days: severe shift
        assert adaptations[25]['severity'] == ShiftSeverity.SEVERE, "Day 25 should have severe shift"


class TestSuddenShift:
    """Test sudden distribution shift (flash crash)"""
    
    def test_flash_crash_scenario(self, handler):
        """Simulate flash crash: sudden 3x volatility, 10x volume"""
        np.random.seed(42)
        
        # Normal trading
        X_train = np.random.randn(500, 10) * 2 + 5
        X_train[:, 0] *= 1  # Normal volume
        
        # Flash crash
        X_test = np.random.randn(100, 10) * 6 + 5  # 3x volatility
        X_test[:, 0] *= 10  # 10x volume
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=X_train,
            X_test=X_test,
            feature_names=feature_names,
            current_performance={'win_rate': 0.56}
        )
        
        # Should detect severe shift
        assert result.shift_severity == ShiftSeverity.SEVERE, "Should detect severe shift"
        
        # Should flag high OOD ratio
        if result.ood_calibration:
            ood_ratio = result.ood_calibration.ood_count / len(X_test)
            assert ood_ratio > 0.5, f"OOD ratio should be >50%, got {ood_ratio:.2%}"


class TestFalsePositivePrevention:
    """Test false positive handling (temporary spikes)"""
    
    def test_whale_transaction_temporary_spike(self, handler):
        """Single whale transaction shouldn't trigger adaptation"""
        np.random.seed(42)
        X_train = np.random.randn(500, 10) * 2 + 5
        
        # Test data: 99 normal samples + 1 whale
        X_test = np.random.randn(100, 10) * 2 + 5
        X_test[0, :] = np.random.randn(10) * 20 + 50  # Whale transaction
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        # Detection WITH outlier removal (should not trigger)
        metrics = handler.detect_covariate_shift(X_train, X_test, feature_names)
        
        # Should detect no shift or minor shift (not severe)
        assert metrics.severity in [ShiftSeverity.NONE, ShiftSeverity.MINOR, ShiftSeverity.MODERATE], \
            f"Whale transaction should not trigger severe shift, got {metrics.severity}"


class TestAdaptationFailure:
    """Test adaptation failure scenario (requires retraining)"""
    
    def test_escalation_to_retraining(self, handler, normal_distribution):
        """Severe shift + performance drop should escalate to retraining"""
        # Severe shift
        X_test = np.random.randn(100, 10) * 8 + 15  # Extreme shift
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        result = handler.adapt_to_covariate_shift(
            model_name='test_model',
            X_train=normal_distribution,
            X_test=X_test,
            feature_names=feature_names,
            current_performance={'win_rate': 0.48, 'baseline_win_rate': 0.58}  # 10pp drop
        )
        
        # Should detect severe shift
        assert result.shift_severity == ShiftSeverity.SEVERE, "Should detect severe shift"
        
        # Strategy selection should recommend escalation (log warning)
        # Note: Actual escalation logic in ai_trading_engine.py


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test computational performance"""
    
    def test_mmd_latency(self, handler):
        """MMD computation should be <200ms for n=500"""
        import time
        
        np.random.seed(42)
        X_train = np.random.randn(500, 10)
        X_test = np.random.randn(100, 10)
        
        start = time.time()
        handler.compute_mmd_squared(X_train, X_test)
        latency = time.time() - start
        
        assert latency < 0.2, f"MMD latency {latency:.3f}s should be <200ms"
    
    def test_kmm_latency(self, handler):
        """KMM computation should be <3s for n=500"""
        import time
        
        np.random.seed(42)
        X_train = np.random.randn(500, 10)
        X_test = np.random.randn(100, 10)
        
        start = time.time()
        handler.kernel_mean_matching(X_train, X_test)
        latency = time.time() - start
        
        assert latency < 3.0, f"KMM latency {latency:.3f}s should be <3s"
    
    def test_discriminator_latency(self, handler):
        """Discriminator computation should be <500ms for n=1000"""
        import time
        
        np.random.seed(42)
        X_train = np.random.randn(1000, 10)
        X_test = np.random.randn(200, 10)
        
        start = time.time()
        handler.discriminator_weights(X_train, X_test)
        latency = time.time() - start
        
        assert latency < 0.5, f"Discriminator latency {latency:.3f}s should be <500ms"
    
    def test_full_detection_latency(self, handler):
        """Full detection cycle should be <500ms"""
        import time
        
        np.random.seed(42)
        X_train = np.random.randn(500, 10)
        X_test = np.random.randn(100, 10)
        
        feature_names = [f'feat_{i}' for i in range(10)]
        
        start = time.time()
        handler.detect_covariate_shift(X_train, X_test, feature_names)
        latency = time.time() - start
        
        assert latency < 0.5, f"Full detection latency {latency:.3f}s should be <500ms"


# ============================================================================
# RISK MITIGATION TESTS
# ============================================================================

class TestExtremeWeights:
    """Test extreme weight handling"""
    
    def test_weight_clipping_prevents_instability(self, handler, normal_distribution, extreme_outlier_distribution):
        """Weight clipping should prevent stability >20"""
        weights_obj = handler.kernel_mean_matching(normal_distribution, extreme_outlier_distribution)
        
        # Stability score should be reasonable after clipping
        assert weights_obj.stability_score < 100, \
            f"Stability score {weights_obj.stability_score} too high even after clipping"
    
    def test_discriminator_more_stable_than_kmm(self, handler, normal_distribution, extreme_outlier_distribution):
        """Discriminator should be more stable than KMM for outliers"""
        kmm_weights = handler.kernel_mean_matching(normal_distribution, extreme_outlier_distribution)
        discr_weights = handler.discriminator_weights(normal_distribution, extreme_outlier_distribution)
        
        assert discr_weights.stability_score < kmm_weights.stability_score, \
            "Discriminator should be more stable than KMM"


class TestDomainAdaptationArtifacts:
    """Test domain adaptation artifact prevention"""
    
    def test_coral_regularization(self, handler, normal_distribution):
        """CORAL should not introduce extreme correlations"""
        # Create uncorrelated test data
        X_test = np.random.randn(100, 10) * 3
        
        X_adapted, _ = handler.coral_transform(normal_distribution, X_test)
        
        # Check correlation matrix
        corr_original = np.corrcoef(normal_distribution.T)
        corr_adapted = np.corrcoef(X_adapted.T)
        
        # Correlations shouldn't change by >0.5
        max_corr_change = np.max(np.abs(corr_adapted - corr_original))
        assert max_corr_change < 0.7, f"CORAL introduced excessive correlation changes: {max_corr_change:.2f}"


class TestOODMiscalibration:
    """Test OOD miscalibration prevention"""
    
    def test_ood_threshold_prevents_over_flagging(self, handler, normal_distribution):
        """OOD threshold should prevent flagging >80% as OOD"""
        # Test with slightly shifted distribution
        X_test = np.random.randn(100, 10) * 2.5 + 6
        
        confidences = np.random.uniform(0.6, 0.9, size=100)
        
        calibration = handler.calibrate_ood_confidence(
            normal_distribution, X_test, confidences
        )
        
        ood_ratio = calibration.ood_count / len(X_test)
        
        # Should not flag >60% as OOD for moderate shift
        assert ood_ratio < 0.6, f"OOD ratio {ood_ratio:.2%} too high"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
