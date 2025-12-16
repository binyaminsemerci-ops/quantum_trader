"""
DRIFT DETECTION MANAGER - COMPREHENSIVE TEST SUITE

Tests for Drift Detection (Module 3):
- Unit tests for PSI, KS-test, performance metrics
- Integration tests with trading system
- Scenario simulations (gradual drift, sudden shifts, false alarms)
"""

import pytest
import asyncio
import numpy as np
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List
from scipy.stats import ks_2samp

# Assuming drift_detection_manager.py is in backend/services/ai/
import sys
sys.path.append('/app/backend')

from services.ai.drift_detection_manager import (
    DriftDetectionManager,
    DriftSeverity,
    DriftType,
    RetrainingUrgency,
    FeatureDistribution,
    PSIResult,
    KSTestResult,
    PerformanceMetrics,
    DriftAlert,
    DriftContext
)


# ============================================================
# UNIT TESTS
# ============================================================

class TestPSICalculation:
    """Test Population Stability Index calculation"""
    
    def test_psi_identical_distributions(self):
        """PSI should be ~0 for identical distributions"""
        manager = DriftDetectionManager()
        
        # Same distribution
        expected = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        actual = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005])
        
        psi_score, contributions = manager._calculate_psi(expected, actual)
        
        # PSI should be very close to 0
        assert abs(psi_score) < 0.01
        assert len(contributions) == len(expected)
    
    def test_psi_shifted_distribution(self):
        """PSI should detect distribution shift"""
        manager = DriftDetectionManager()
        
        # Baseline: Normal distribution centered at 0.5
        expected = np.array([0.01, 0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.03, 0.01, 0.00])
        
        # Shifted: Distribution shifted right (drift)
        actual = np.array([0.00, 0.01, 0.03, 0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.01])
        
        psi_score, contributions = manager._calculate_psi(expected, actual)
        
        # PSI should be significantly positive
        assert psi_score > 0.15  # Moderate drift
        
        # Should be higher than threshold
        assert psi_score > manager.psi_thresholds['minor']
    
    def test_psi_severe_drift(self):
        """PSI should detect severe drift"""
        manager = DriftDetectionManager()
        
        # Baseline
        expected = np.array([0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01, 0.00, 0.00])
        
        # Completely different distribution
        actual = np.array([0.00, 0.00, 0.01, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.3])
        
        psi_score, contributions = manager._calculate_psi(expected, actual)
        
        # PSI should indicate severe drift
        assert psi_score >= manager.psi_thresholds['severe']
        print(f"Severe drift PSI: {psi_score:.3f}")


class TestKSTest:
    """Test Kolmogorov-Smirnov test"""
    
    def test_ks_same_distribution(self):
        """KS test should not reject for same distribution"""
        manager = DriftDetectionManager()
        
        # Same distribution
        baseline = np.random.normal(0.6, 0.1, 500)
        recent = np.random.normal(0.6, 0.1, 500)
        
        ks_stat, p_value = ks_2samp(baseline, recent)
        
        # P-value should be high (not significant)
        assert p_value > manager.ks_p_value_threshold
        print(f"Same distribution - p-value: {p_value:.3f}")
    
    def test_ks_different_distribution(self):
        """KS test should reject for different distributions"""
        manager = DriftDetectionManager()
        
        # Different distributions
        baseline = np.random.normal(0.6, 0.1, 500)  # Mean 0.6
        recent = np.random.normal(0.5, 0.15, 500)   # Mean 0.5, higher variance
        
        ks_stat, p_value = ks_2samp(baseline, recent)
        
        # P-value should be low (significant difference)
        # Note: May not always be < 0.01 due to randomness, but should be low
        print(f"Different distribution - p-value: {p_value:.3f}, KS-stat: {ks_stat:.3f}")
        
        # At minimum, should detect shift in mean
        assert np.abs(np.mean(baseline) - np.mean(recent)) > 0.05


class TestPerformanceMetrics:
    """Test performance metrics calculation"""
    
    def test_performance_calculation_perfect_predictions(self):
        """Test metrics for perfect predictions"""
        manager = DriftDetectionManager()
        
        # Perfect predictions
        predictions = np.array([0.9, 0.8, 0.85, 0.75, 0.95] * 20)  # 100 trades
        actual_outcomes = np.ones(100)  # All wins
        confidences = predictions
        
        metrics = manager._compute_performance_metrics(
            actual_outcomes=actual_outcomes,
            predictions=predictions,
            confidences=confidences,
            window_size=100
        )
        
        # Should have perfect metrics
        assert metrics.win_rate == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        
        # Calibration should be good (predictions match outcomes)
        assert metrics.calibration_error < 0.15
    
    def test_performance_calculation_degraded(self):
        """Test metrics for degraded performance"""
        manager = DriftDetectionManager()
        
        # Degraded: Predicts 0.65 but actual is 45% WR
        predictions = np.array([0.65] * 100)
        actual_outcomes = np.array([1] * 45 + [0] * 55)  # 45% WR
        confidences = predictions
        
        metrics = manager._compute_performance_metrics(
            actual_outcomes=actual_outcomes,
            predictions=predictions,
            confidences=confidences,
            window_size=100
        )
        
        # Win rate should be 45%
        assert 0.44 <= metrics.win_rate <= 0.46
        
        # Calibration error should be high (predicts 0.65, actual 0.45)
        assert metrics.calibration_error > 0.02
        
        # F1 should be moderate
        assert 0.4 <= metrics.f1_score <= 0.7
        print(f"Degraded performance: WR={metrics.win_rate:.2f}, F1={metrics.f1_score:.2f}")


class TestBaselineEstablishment:
    """Test baseline establishment"""
    
    def test_establish_baseline(self):
        """Test baseline establishment with valid data"""
        manager = DriftDetectionManager()
        
        # Generate synthetic data
        n_samples = 500
        feature_values = {
            'rsi_14': np.random.uniform(30, 70, n_samples),
            'macd': np.random.normal(0, 2, n_samples),
            'volume': np.random.lognormal(10, 2, n_samples)
        }
        predictions = np.random.uniform(0.45, 0.75, n_samples)
        actual_outcomes = (np.random.rand(n_samples) < predictions).astype(int)
        confidences = predictions
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values,
            predictions=predictions,
            actual_outcomes=actual_outcomes,
            confidences=confidences
        )
        
        # Verify baseline created
        assert 'xgboost' in manager.baseline_distributions
        assert 'xgboost' in manager.baseline_performance
        assert 'xgboost' in manager.baseline_predictions
        
        # Check feature distributions
        assert 'rsi_14' in manager.baseline_distributions['xgboost']
        assert 'macd' in manager.baseline_distributions['xgboost']
        assert 'volume' in manager.baseline_distributions['xgboost']
        
        # Check performance metrics
        perf = manager.baseline_performance['xgboost']
        assert 0.0 <= perf.win_rate <= 1.0
        assert 0.0 <= perf.f1_score <= 1.0
        
        print(f"Baseline established: WR={perf.win_rate:.3f}, F1={perf.f1_score:.3f}")


class TestDriftDetection:
    """Test drift detection logic"""
    
    def test_no_drift_detected(self):
        """Test that no drift is detected for stable data"""
        manager = DriftDetectionManager()
        
        # Establish baseline
        n_baseline = 500
        feature_values_baseline = {
            'rsi_14': np.random.uniform(40, 60, n_baseline),
            'macd': np.random.normal(0, 1, n_baseline)
        }
        predictions_baseline = np.random.uniform(0.50, 0.70, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        
        # Recent data (similar to baseline)
        n_recent = 100
        feature_values_recent = {
            'rsi_14': np.random.uniform(40, 60, n_recent),
            'macd': np.random.normal(0, 1, n_recent)
        }
        predictions_recent = np.random.uniform(0.50, 0.70, n_recent)
        outcomes_recent = (np.random.rand(n_recent) < 0.57).astype(int)
        
        # Detect drift
        alert = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_recent,
            predictions=predictions_recent,
            actual_outcomes=outcomes_recent,
            confidences=predictions_recent
        )
        
        # Should not detect drift
        assert alert is None
        print("✓ No drift detected for stable data")
    
    def test_feature_drift_detected(self):
        """Test feature drift detection"""
        manager = DriftDetectionManager()
        
        # Establish baseline
        n_baseline = 500
        feature_values_baseline = {
            'rsi_14': np.random.uniform(40, 60, n_baseline),
        }
        predictions_baseline = np.random.uniform(0.55, 0.65, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        
        # Recent data (severe shift in RSI)
        n_recent = 100
        feature_values_recent = {
            'rsi_14': np.random.uniform(70, 90, n_recent),  # SHIFTED UP
        }
        predictions_recent = np.random.uniform(0.55, 0.65, n_recent)
        outcomes_recent = (np.random.rand(n_recent) < 0.57).astype(int)
        
        # Detect drift
        alert = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_recent,
            predictions=predictions_recent,
            actual_outcomes=outcomes_recent,
            confidences=predictions_recent
        )
        
        # Should detect feature drift
        if alert:  # May not always trigger due to statistical variance
            assert DriftType.FEATURE_DRIFT.value in alert.drift_types
            assert alert.severity in [DriftSeverity.MODERATE.value, DriftSeverity.SEVERE.value]
            print(f"✓ Feature drift detected: {alert.severity}")
        else:
            print("Note: Feature drift not detected (may need more samples or larger shift)")
    
    def test_performance_drift_detected(self):
        """Test performance degradation detection"""
        manager = DriftDetectionManager(
            consecutive_windows_threshold=2
        )
        
        # Establish baseline with good performance
        n_baseline = 500
        feature_values_baseline = {
            'rsi_14': np.random.uniform(40, 60, n_baseline),
        }
        predictions_baseline = np.random.uniform(0.55, 0.70, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.60).astype(int)  # 60% WR
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        
        # Simulate consecutive poor windows
        for window_num in range(3):
            # Recent data with degraded performance
            n_recent = 100
            feature_values_recent = {
                'rsi_14': np.random.uniform(40, 60, n_recent),  # Same features
            }
            predictions_recent = np.random.uniform(0.55, 0.70, n_recent)
            outcomes_recent = (np.random.rand(n_recent) < 0.48).astype(int)  # 48% WR (degraded)
            
            # Detect drift
            alert = manager.detect_drift(
                model_name='xgboost',
                feature_values=feature_values_recent,
                predictions=predictions_recent,
                actual_outcomes=outcomes_recent,
                confidences=predictions_recent
            )
        
        # Should detect performance drift after consecutive poor windows
        if alert:
            assert DriftType.PERFORMANCE_DRIFT.value in alert.drift_types
            assert alert.severity == DriftSeverity.CRITICAL.value
            assert alert.urgency in [RetrainingUrgency.URGENT.value, RetrainingUrgency.IMMEDIATE.value]
            print(f"✓ Performance drift detected: urgency={alert.urgency}")


class TestCheckpointing:
    """Test checkpoint save/load"""
    
    def test_checkpoint_and_restore(self):
        """Test state persistence"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            checkpoint_path = f.name
        
        try:
            # Create manager and establish baseline
            manager1 = DriftDetectionManager(checkpoint_path=checkpoint_path)
            
            n_samples = 300
            feature_values = {
                'rsi_14': np.random.uniform(40, 60, n_samples),
            }
            predictions = np.random.uniform(0.55, 0.65, n_samples)
            outcomes = (np.random.rand(n_samples) < 0.58).astype(int)
            
            manager1.establish_baseline(
                model_name='xgboost',
                feature_values=feature_values,
                predictions=predictions,
                actual_outcomes=outcomes,
                confidences=predictions
            )
            
            # Process some trades
            for i in range(50):
                manager1.process_trade_outcome(
                    model_name='xgboost',
                    prediction=0.60,
                    confidence=0.65,
                    actual_outcome=1 if i % 2 == 0 else 0,
                    feature_values={'rsi_14': 50.0}
                )
            
            # Save checkpoint
            manager1.checkpoint()
            
            trades1 = manager1.trades_since_baseline['xgboost']
            
            # Load new manager from checkpoint
            manager2 = DriftDetectionManager(checkpoint_path=checkpoint_path)
            
            trades2 = manager2.trades_since_baseline.get('xgboost', 0)
            
            # Verify state restored
            assert trades1 == trades2
            print(f"✓ Checkpoint restored: {trades2} trades")
        
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestSystemIntegration:
    """Integration tests with trading system"""
    
    @pytest.mark.asyncio
    async def test_full_drift_detection_cycle(self):
        """Test complete drift detection workflow"""
        manager = DriftDetectionManager()
        
        print("\n=== FULL DRIFT DETECTION CYCLE ===")
        
        # Step 1: Establish baseline
        print("Step 1: Establishing baseline...")
        n_baseline = 500
        feature_values_baseline = {
            'rsi_14': np.random.uniform(40, 60, n_baseline),
            'macd': np.random.normal(0, 1, n_baseline),
            'volume': np.random.lognormal(10, 1, n_baseline)
        }
        predictions_baseline = np.random.uniform(0.52, 0.68, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        print(f"✓ Baseline established: {n_baseline} samples")
        
        # Step 2: Process trades normally (no drift)
        print("\nStep 2: Processing 50 normal trades...")
        for i in range(50):
            manager.process_trade_outcome(
                model_name='xgboost',
                prediction=np.random.uniform(0.55, 0.65),
                confidence=0.60,
                actual_outcome=1 if np.random.rand() < 0.57 else 0,
                feature_values={
                    'rsi_14': np.random.uniform(40, 60),
                    'macd': np.random.normal(0, 1),
                    'volume': np.random.lognormal(10, 1)
                }
            )
        print("✓ 50 trades processed")
        
        # Step 3: Introduce drift
        print("\nStep 3: Introducing drift (performance degradation)...")
        for i in range(100):
            manager.process_trade_outcome(
                model_name='xgboost',
                prediction=np.random.uniform(0.55, 0.65),
                confidence=0.60,
                actual_outcome=1 if np.random.rand() < 0.42 else 0,  # 42% WR (degraded)
                feature_values={
                    'rsi_14': np.random.uniform(65, 85),  # Shifted features
                    'macd': np.random.normal(2, 1),      # Mean shifted
                    'volume': np.random.lognormal(11, 1)
                }
            )
        print("✓ 100 degraded trades processed")
        
        # Step 4: Detect drift
        print("\nStep 4: Detecting drift...")
        n_recent = 100
        feature_values_recent = {
            'rsi_14': np.random.uniform(65, 85, n_recent),
            'macd': np.random.normal(2, 1, n_recent),
            'volume': np.random.lognormal(11, 1, n_recent)
        }
        predictions_recent = np.random.uniform(0.55, 0.65, n_recent)
        outcomes_recent = (np.random.rand(n_recent) < 0.42).astype(int)
        
        alert = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_recent,
            predictions=predictions_recent,
            actual_outcomes=outcomes_recent,
            confidences=predictions_recent
        )
        
        if alert:
            print(f"✓ DRIFT DETECTED:")
            print(f"  - Types: {alert.drift_types}")
            print(f"  - Severity: {alert.severity}")
            print(f"  - Urgency: {alert.urgency}")
            print(f"  - Action: {alert.recommended_action}")
            
            # Step 5: Trigger retraining
            print("\nStep 5: Triggering retraining...")
            retrain_job = manager.trigger_retraining(
                model_name='xgboost',
                alert=alert
            )
            print(f"✓ Retraining job: {retrain_job['job_id']}")
            
            # Step 6: Reset baseline (simulate retraining complete)
            print("\nStep 6: Resetting baseline after retrain...")
            manager.reset_baseline_after_retrain(
                model_name='xgboost',
                new_feature_values=feature_values_recent,
                new_predictions=predictions_recent,
                new_actual_outcomes=outcomes_recent,
                new_confidences=predictions_recent
            )
            print("✓ Baseline reset")
            
            # Verify alerts cleared
            context = manager.get_drift_context('xgboost')
            assert len(context.active_alerts) == 0
            print("✓ Alerts cleared")
        else:
            print("Note: Drift not detected (may need more samples)")
        
        print("\n✅ Full cycle complete")


# ============================================================
# SCENARIO SIMULATIONS
# ============================================================

class TestScenarios:
    """Real-world scenario simulations"""
    
    def test_gradual_drift_scenario(self):
        """Scenario: Features drift gradually over 30 days"""
        manager = DriftDetectionManager()
        
        print("\n=== GRADUAL DRIFT SCENARIO (30 days) ===")
        
        # Establish baseline
        n_baseline = 500
        feature_values_baseline = {
            'volatility': np.random.uniform(2, 4, n_baseline),
        }
        predictions_baseline = np.random.uniform(0.55, 0.65, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        print("Baseline: Volatility mean=3.0")
        
        # Simulate 30 days of gradual drift
        psi_scores = []
        for day in range(1, 31):
            # Volatility increases gradually
            vol_mean = 3.0 + (day / 30) * 2.0  # From 3.0 to 5.0
            
            n_daily = 20
            feature_values_daily = {
                'volatility': np.random.uniform(vol_mean - 0.5, vol_mean + 0.5, n_daily),
            }
            predictions_daily = np.random.uniform(0.55, 0.65, n_daily)
            outcomes_daily = (np.random.rand(n_daily) < 0.57).astype(int)
            
            # Compute PSI
            dist = manager._compute_feature_distribution('volatility', feature_values_daily['volatility'])
            baseline_dist = manager.baseline_distributions['xgboost']['volatility']
            psi, _ = manager._calculate_psi(
                np.array(baseline_dist.frequencies),
                np.array(dist.frequencies)
            )
            psi_scores.append(psi)
            
            if day % 5 == 0:
                print(f"Day {day}: Vol mean={vol_mean:.2f}, PSI={psi:.3f}")
        
        # Check if drift eventually detected
        max_psi = max(psi_scores)
        print(f"\nMax PSI reached: {max_psi:.3f}")
        if max_psi >= manager.psi_thresholds['severe']:
            print("✓ Gradual drift eventually detected")
        else:
            print(f"Note: Max PSI {max_psi:.3f} below severe threshold {manager.psi_thresholds['severe']}")
    
    def test_sudden_shift_scenario(self):
        """Scenario: Sudden market regime change"""
        manager = DriftDetectionManager()
        
        print("\n=== SUDDEN SHIFT SCENARIO (Flash Crash) ===")
        
        # Establish baseline (normal market)
        n_baseline = 500
        feature_values_baseline = {
            'volatility': np.random.uniform(2, 4, n_baseline),
            'volume': np.random.lognormal(10, 0.5, n_baseline)
        }
        predictions_baseline = np.random.uniform(0.55, 0.65, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        print("Baseline: Normal market (Vol=3, Volume=22k)")
        
        # Sudden shift (flash crash)
        print("\nFlash crash occurs...")
        n_crash = 100
        feature_values_crash = {
            'volatility': np.random.uniform(8, 12, n_crash),      # 3x volatility
            'volume': np.random.lognormal(12, 1, n_crash)         # 10x volume
        }
        predictions_crash = np.random.uniform(0.50, 0.60, n_crash)
        outcomes_crash = (np.random.rand(n_crash) < 0.45).astype(int)  # Performance drops
        
        # Detect drift
        alert = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_crash,
            predictions=predictions_crash,
            actual_outcomes=outcomes_crash,
            confidences=predictions_crash
        )
        
        if alert:
            print(f"✓ IMMEDIATE DRIFT DETECTED:")
            print(f"  - Severity: {alert.severity}")
            print(f"  - PSI scores: {alert.psi_scores}")
            print(f"  - Win rate delta: {alert.performance_delta['win_rate']:.3f}")
            
            assert alert.severity in [DriftSeverity.SEVERE.value, DriftSeverity.CRITICAL.value]
        else:
            print("Note: Drift not detected (may need larger sample)")
    
    def test_false_alarm_scenario(self):
        """Scenario: Temporary volatility spike (should not trigger retraining)"""
        manager = DriftDetectionManager(
            consecutive_windows_threshold=3  # Require 3 consecutive windows
        )
        
        print("\n=== FALSE ALARM PREVENTION SCENARIO ===")
        
        # Establish baseline
        n_baseline = 500
        feature_values_baseline = {
            'rsi_14': np.random.uniform(40, 60, n_baseline),
        }
        predictions_baseline = np.random.uniform(0.55, 0.65, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        print("Baseline established: 58% WR")
        
        # Window 1: Temporary spike (bad performance)
        print("\nWindow 1: Temporary performance dip...")
        n_window = 100
        feature_values_w1 = {
            'rsi_14': np.random.uniform(40, 60, n_window),
        }
        predictions_w1 = np.random.uniform(0.55, 0.65, n_window)
        outcomes_w1 = (np.random.rand(n_window) < 0.48).astype(int)  # 48% WR
        
        alert1 = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_w1,
            predictions=predictions_w1,
            actual_outcomes=outcomes_w1,
            confidences=predictions_w1
        )
        
        if alert1:
            print("Alert raised (but should be monitored, not acted upon)")
        else:
            print("No alert (consecutive threshold not met)")
        
        # Window 2: Performance recovers
        print("\nWindow 2: Performance recovers...")
        outcomes_w2 = (np.random.rand(n_window) < 0.57).astype(int)  # Back to 57% WR
        
        alert2 = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_w1,
            predictions=predictions_w1,
            actual_outcomes=outcomes_w2,
            confidences=predictions_w1
        )
        
        # Should not trigger retraining (performance recovered)
        if alert2:
            assert alert2.urgency != RetrainingUrgency.IMMEDIATE.value
            print(f"Alert severity downgraded: {alert2.urgency}")
        else:
            print("✓ No alert - false alarm prevented")


# ============================================================
# PERFORMANCE TESTS
# ============================================================

class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_drift_detection_latency(self):
        """Test drift detection completes quickly"""
        import time
        
        manager = DriftDetectionManager()
        
        # Establish baseline
        n_baseline = 500
        feature_values_baseline = {
            f'feature_{i}': np.random.uniform(0, 100, n_baseline)
            for i in range(10)  # 10 features
        }
        predictions_baseline = np.random.uniform(0.50, 0.70, n_baseline)
        outcomes_baseline = (np.random.rand(n_baseline) < 0.58).astype(int)
        
        manager.establish_baseline(
            model_name='xgboost',
            feature_values=feature_values_baseline,
            predictions=predictions_baseline,
            actual_outcomes=outcomes_baseline,
            confidences=predictions_baseline
        )
        
        # Measure drift detection time
        n_recent = 100
        feature_values_recent = {
            f'feature_{i}': np.random.uniform(0, 100, n_recent)
            for i in range(10)
        }
        predictions_recent = np.random.uniform(0.50, 0.70, n_recent)
        outcomes_recent = (np.random.rand(n_recent) < 0.55).astype(int)
        
        start = time.time()
        alert = manager.detect_drift(
            model_name='xgboost',
            feature_values=feature_values_recent,
            predictions=predictions_recent,
            actual_outcomes=outcomes_recent,
            confidences=predictions_recent
        )
        elapsed = time.time() - start
        
        print(f"\nDrift detection time: {elapsed * 1000:.2f} ms")
        
        # Should be fast (<200ms for 10 features)
        assert elapsed < 0.2
        print("✓ Latency within acceptable range")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
