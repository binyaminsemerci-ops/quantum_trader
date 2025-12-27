"""
TEST SUITE: Continuous Learning Manager

Tests for Module 6: Continuous Learning
- Performance monitoring (EWMA, CUSUM, SPC)
- Feature tracking (SHAP, drift)
- Online learning (SGD/Adam)
- Model versioning
- Retraining triggers
"""

import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import shutil

from backend.services.ai.continuous_learning_manager import (
    ContinuousLearningManager,
    PerformanceMonitor,
    FeatureTracker,
    OnlineLearner,
    ModelVersioner,
    RetrainingTrigger,
    PerformanceMetrics,
    RetrainingEvent
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_storage():
    """Temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def performance_monitor():
    """Create performance monitor"""
    return PerformanceMonitor(
        ewma_alpha=0.1,
        ewma_threshold=3.0,
        cusum_k=0.5,
        cusum_h=5.0
    )


@pytest.fixture
def feature_tracker():
    """Create feature tracker"""
    return FeatureTracker(
        drift_threshold=0.3,
        update_alpha=0.05
    )


@pytest.fixture
def online_learner():
    """Create online learner"""
    return OnlineLearner(
        learning_rate=0.01,
        update_frequency=10
    )


@pytest.fixture
def model_versioner(temp_storage):
    """Create model versioner"""
    return ModelVersioner(storage_path=temp_storage)


@pytest.fixture
def cl_manager(temp_storage):
    """Create continuous learning manager"""
    checkpoint_path = temp_storage / "checkpoint.json"
    return ContinuousLearningManager(
        storage_path=str(temp_storage / "models"),
        checkpoint_path=str(checkpoint_path),
        ewma_alpha=0.1,
        ewma_threshold=3.0,
        enable_online_learning=True
    )


# ============================================================================
# UNIT TESTS: PERFORMANCE MONITOR
# ============================================================================

def test_performance_monitor_initialization(performance_monitor):
    """Test performance monitor initializes correctly"""
    assert performance_monitor.ewma_alpha == 0.1
    assert performance_monitor.ewma_threshold == 3.0
    assert performance_monitor.baseline_wr is None
    assert performance_monitor.ewma_wr is None


def test_set_baseline(performance_monitor):
    """Test setting baseline"""
    performance_monitor.set_baseline(0.58)
    assert performance_monitor.baseline_wr == 0.58
    assert performance_monitor.ewma_wr == 0.58


def test_ewma_decay_detection(performance_monitor):
    """Test EWMA decay detection"""
    performance_monitor.set_baseline(0.58)
    
    # Simulate gradual decay
    for _ in range(100):
        status = performance_monitor.update(0.50)  # 50% WR (8pp below baseline)
    
    # Should detect decay
    assert status['ewma_decay'] > 3.0
    assert RetrainingTrigger.EWMA_DECAY in status['triggers']
    assert status['should_retrain'] == True


def test_cusum_shift_detection(performance_monitor):
    """Test CUSUM sudden shift detection"""
    performance_monitor.set_baseline(0.58)
    
    # Simulate sudden shift
    for _ in range(20):
        status = performance_monitor.update(0.45)  # Sudden drop
    
    # Should trigger CUSUM
    assert status['cusum_negative'] > 5.0
    assert RetrainingTrigger.CUSUM_SHIFT in status['triggers']


def test_spc_violations(performance_monitor):
    """Test SPC control chart violations"""
    performance_monitor.set_baseline(0.58)
    
    # 8 consecutive points below baseline
    for _ in range(8):
        status = performance_monitor.update(0.55)
    
    # Should trigger SPC
    assert status['spc_violations'] > 0
    assert RetrainingTrigger.SPC_OUT_OF_CONTROL in status['triggers']


def test_urgency_score_computation(performance_monitor):
    """Test urgency score computation"""
    performance_monitor.set_baseline(0.58)
    
    # Severe decay
    for _ in range(50):
        status = performance_monitor.update(0.45)
    
    decay_metrics = status
    urgency = performance_monitor.compute_urgency_score(decay_metrics)
    
    # Should be high urgency
    assert urgency > 50.0


# ============================================================================
# UNIT TESTS: FEATURE TRACKER
# ============================================================================

def test_feature_tracker_initialization(feature_tracker):
    """Test feature tracker initializes"""
    assert feature_tracker.drift_threshold == 0.3
    assert feature_tracker.update_alpha == 0.05


def test_set_baseline_features(feature_tracker):
    """Test setting baseline feature importance"""
    baseline = {
        'rsi': 0.30,
        'macd': 0.25,
        'volume': 0.20,
        'atr': 0.15,
        'ob': 0.10
    }
    feature_tracker.set_baseline(baseline)
    
    assert feature_tracker.baseline_importance == baseline
    assert feature_tracker.current_importance == baseline


def test_feature_importance_update(feature_tracker):
    """Test feature importance updates with SHAP"""
    baseline = {'rsi': 0.30, 'macd': 0.25, 'volume': 0.20}
    feature_tracker.set_baseline(baseline)
    
    # Update with new SHAP values
    shap_values = {'rsi': 0.05, 'macd': 0.03, 'volume': 0.08}
    status = feature_tracker.update(shap_values)
    
    assert 'current_importance' in status
    assert 'js_divergence' in status


def test_feature_drift_detection(feature_tracker):
    """Test feature drift detection"""
    baseline = {
        'rsi': 0.30,
        'macd': 0.25,
        'volume': 0.20,
        'atr': 0.15,
        'ob': 0.10
    }
    feature_tracker.set_baseline(baseline)
    
    # Simulate major shift (OB becomes dominant)
    for _ in range(100):
        shap = {
            'rsi': 0.02,
            'macd': 0.01,
            'volume': 0.02,
            'atr': 0.01,
            'ob': 0.50  # 5x increase
        }
        status = feature_tracker.update(shap)
    
    # Should detect drift
    assert status['js_divergence'] > 0.3
    assert status['drift_detected'] == True


def test_top_features(feature_tracker):
    """Test getting top features"""
    baseline = {'rsi': 0.30, 'macd': 0.25, 'volume': 0.20}
    feature_tracker.set_baseline(baseline)
    
    top = feature_tracker._get_top_features(2)
    
    assert len(top) == 2
    assert top[0][0] == 'rsi'
    assert top[0][1] == 0.30


# ============================================================================
# UNIT TESTS: ONLINE LEARNER
# ============================================================================

def test_online_learner_initialization(online_learner):
    """Test online learner initializes"""
    assert online_learner.lr == 0.01
    assert online_learner.update_freq == 10
    assert online_learner.weights is None


def test_initialize_weights(online_learner):
    """Test weight initialization"""
    initial_weights = np.array([0.5, 0.3, 0.2, 0.1])
    online_learner.initialize(initial_weights)
    
    assert online_learner.weights is not None
    assert np.array_equal(online_learner.weights, initial_weights)
    assert online_learner.velocity is not None
    assert online_learner.m is not None
    assert online_learner.v is not None


def test_record_trade_and_batch_update(online_learner):
    """Test recording trades and batch update"""
    initial_weights = np.array([0.5, 0.3, 0.2, 0.1])
    online_learner.initialize(initial_weights)
    
    # Record 10 trades (triggers batch update)
    for i in range(10):
        features = np.array([65.0, -0.02, 1.2, 0.15])
        outcome = 1.0 if i < 6 else 0.0  # 60% WR
        online_learner.record_trade(features, outcome)
    
    # Should have triggered update
    assert online_learner.t == 1
    assert len(online_learner.pending_updates) == 0


def test_weight_delta_clipping(online_learner):
    """Test weight change is clipped"""
    initial_weights = np.array([0.5, 0.3, 0.2, 0.1])
    online_learner.initialize(initial_weights)
    
    # Record extreme trades
    for i in range(10):
        features = np.array([100.0, 50.0, 10.0, 5.0])  # Extreme values
        outcome = 1.0
        online_learner.record_trade(features, outcome)
    
    # Weight changes should be clipped
    new_weights = online_learner.get_weights()
    delta = np.abs(new_weights - initial_weights)
    assert np.all(delta <= 0.1)  # Max delta


# ============================================================================
# UNIT TESTS: MODEL VERSIONER
# ============================================================================

def test_versioner_initialization(model_versioner):
    """Test versioner initializes"""
    assert model_versioner.current_version == "1.0.0"
    assert len(model_versioner.version_history) == 0


def test_save_version(model_versioner):
    """Test saving model version"""
    model_data = {'weights': [0.5, 0.3, 0.2]}
    metrics = PerformanceMetrics(
        win_rate=0.58,
        sharpe_ratio=1.85,
        sortino_ratio=2.10,
        max_drawdown=0.12,
        calmar_ratio=15.4,
        n_trades=1000
    )
    metadata = {'update_type': 'minor'}
    
    new_version = model_versioner.save_version(model_data, metrics, metadata)
    
    assert new_version == "1.1.0"
    assert model_versioner.current_version == "1.1.0"
    assert len(model_versioner.version_history) == 1


def test_load_version(model_versioner):
    """Test loading model version"""
    model_data = {'weights': [0.5, 0.3, 0.2]}
    metrics = PerformanceMetrics(
        win_rate=0.58,
        sharpe_ratio=1.85,
        sortino_ratio=2.10,
        max_drawdown=0.12,
        calmar_ratio=15.4,
        n_trades=1000
    )
    
    version = model_versioner.save_version(model_data, metrics, {})
    
    loaded_data, loaded_metadata = model_versioner.load_version(version)
    
    assert loaded_data == model_data
    assert loaded_metadata['version'] == version


def test_version_increment(model_versioner):
    """Test version incrementing"""
    # Minor update
    assert model_versioner._increment_version('minor') == "1.1.0"
    
    # Patch update
    model_versioner.current_version = "1.1.0"
    assert model_versioner._increment_version('patch') == "1.1.1"
    
    # Major update
    model_versioner.current_version = "1.1.1"
    assert model_versioner._increment_version('major') == "2.0.0"


def test_rollback(model_versioner):
    """Test version rollback"""
    model_data = {'weights': [0.5, 0.3, 0.2]}
    metrics = PerformanceMetrics(
        win_rate=0.58, sharpe_ratio=1.85, sortino_ratio=2.10,
        max_drawdown=0.12, calmar_ratio=15.4, n_trades=1000
    )
    
    v1 = model_versioner.save_version(model_data, metrics, {})
    v2 = model_versioner.save_version(model_data, metrics, {})
    
    assert model_versioner.current_version == v2
    
    model_versioner.rollback(v1)
    
    assert model_versioner.current_version == v1


# ============================================================================
# INTEGRATION TESTS: CONTINUOUS LEARNING MANAGER
# ============================================================================

def test_cl_manager_initialization(cl_manager):
    """Test CL manager initializes"""
    assert cl_manager.performance_monitor is not None
    assert cl_manager.feature_tracker is not None
    assert cl_manager.online_learner is not None
    assert cl_manager.versioner is not None


def test_initialize_baseline(cl_manager):
    """Test baseline initialization"""
    cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.30, 'macd': 0.25},
        model_weights=np.array([0.5, 0.3, 0.2, 0.1])
    )
    
    assert cl_manager.performance_monitor.baseline_wr == 0.58
    assert cl_manager.feature_tracker.baseline_importance['rsi'] == 0.30
    assert cl_manager.online_learner.weights is not None


def test_record_trade(cl_manager):
    """Test recording trade"""
    cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.30, 'macd': 0.25}
    )
    
    status = cl_manager.record_trade(
        outcome=1.0,
        features={'rsi': 65.0, 'macd': -0.02},
        shap_values={'rsi': 0.05, 'macd': 0.03}
    )
    
    assert status is not None
    assert 'trade_count' in status
    assert 'should_retrain' in status


def test_retraining_trigger(cl_manager):
    """Test retraining trigger"""
    cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.30}
    )
    
    metrics = PerformanceMetrics(
        win_rate=0.55, sharpe_ratio=1.5, sortino_ratio=1.8,
        max_drawdown=0.15, calmar_ratio=10.0, n_trades=500
    )
    
    event_id = cl_manager.trigger_retraining(
        trigger=RetrainingTrigger.EWMA_DECAY,
        urgency_score=75.0,
        metrics_before=metrics,
        reason="Performance decay 3.5pp"
    )
    
    assert event_id == "0"
    assert len(cl_manager.retraining_events) == 1


def test_complete_retraining(cl_manager):
    """Test completing retraining event"""
    cl_manager.initialize_baseline(win_rate=0.58, feature_importance={'rsi': 0.30})
    
    metrics_before = PerformanceMetrics(
        win_rate=0.55, sharpe_ratio=1.5, sortino_ratio=1.8,
        max_drawdown=0.15, calmar_ratio=10.0, n_trades=500
    )
    
    event_id = cl_manager.trigger_retraining(
        trigger=RetrainingTrigger.EWMA_DECAY,
        urgency_score=75.0,
        metrics_before=metrics_before
    )
    
    metrics_after = PerformanceMetrics(
        win_rate=0.61, sharpe_ratio=2.0, sortino_ratio=2.3,
        max_drawdown=0.10, calmar_ratio=20.0, n_trades=1000
    )
    
    cl_manager.complete_retraining(event_id, "1.1.0", metrics_after)
    
    assert cl_manager.retraining_events[0].new_version == "1.1.0"
    assert cl_manager.retraining_events[0].metrics_after.win_rate == 0.61


def test_get_status(cl_manager):
    """Test getting CL status"""
    cl_manager.initialize_baseline(win_rate=0.58, feature_importance={'rsi': 0.30})
    
    status = cl_manager.get_status()
    
    assert 'trade_count' in status
    assert 'current_version' in status
    assert 'last_retrain' in status
    assert 'online_learning' in status


def test_checkpoint_save_load(cl_manager, temp_storage):
    """Test checkpoint saving and loading"""
    cl_manager.initialize_baseline(win_rate=0.58, feature_importance={'rsi': 0.30})
    
    # Record some trades
    for _ in range(10):
        cl_manager.record_trade(
            outcome=1.0,
            features={'rsi': 65.0}
        )
    
    # Checkpoint should exist
    checkpoint_path = temp_storage / "checkpoint.json"
    assert checkpoint_path.exists()
    
    # Create new manager and load checkpoint
    cl_manager2 = ContinuousLearningManager(
        storage_path=str(temp_storage / "models"),
        checkpoint_path=str(checkpoint_path)
    )
    
    # Should have loaded trade count
    assert cl_manager2.trade_count == 10


# ============================================================================
# SCENARIO TESTS
# ============================================================================

def test_scenario_gradual_decay(cl_manager):
    """Test scenario: gradual performance decay"""
    cl_manager.initialize_baseline(win_rate=0.58, feature_importance={'rsi': 0.30})
    
    # Simulate gradual decay over 100 trades
    for i in range(100):
        wr = 0.58 - (i * 0.001)  # Linear decay
        status = cl_manager.record_trade(
            outcome=1.0 if np.random.rand() < wr else 0.0,
            features={'rsi': 65.0}
        )
    
    # Should trigger retraining
    assert status['should_retrain'] == True
    assert len(status['triggers']) > 0


def test_scenario_sudden_shift(cl_manager):
    """Test scenario: sudden market shift"""
    cl_manager.initialize_baseline(win_rate=0.58, feature_importance={'rsi': 0.30})
    
    # Normal performance
    for _ in range(50):
        cl_manager.record_trade(outcome=1.0 if np.random.rand() < 0.58 else 0.0, features={'rsi': 65.0})
    
    # Sudden drop
    for _ in range(20):
        status = cl_manager.record_trade(outcome=0.0, features={'rsi': 65.0})
    
    # Should trigger CUSUM
    assert RetrainingTrigger.CUSUM_SHIFT.value in status['triggers']


def test_scenario_feature_drift(cl_manager):
    """Test scenario: feature importance shift"""
    cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.30, 'macd': 0.25, 'volume': 0.20, 'ob': 0.10}
    )
    
    # Simulate OB becoming dominant
    for _ in range(100):
        shap = {'rsi': 0.02, 'macd': 0.01, 'volume': 0.02, 'ob': 0.50}
        status = cl_manager.record_trade(
            outcome=1.0,
            features={'rsi': 65.0, 'macd': -0.02, 'volume': 1.2, 'ob': 0.15},
            shap_values=shap
        )
    
    # Should trigger feature drift
    assert RetrainingTrigger.FEATURE_DRIFT.value in status['triggers']


def test_scenario_online_learning(cl_manager):
    """Test scenario: online learning updates"""
    weights = np.array([0.5, 0.3, 0.2, 0.1])
    cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.30},
        model_weights=weights
    )
    
    # Record 100 trades (10 batch updates)
    for i in range(100):
        features_vec = np.array([65.0, -0.02, 1.2, 0.15])
        outcome = 1.0 if i < 60 else 0.0
        cl_manager.record_trade(
            outcome=outcome,
            features={'rsi': 65.0},
            feature_vector=features_vec
        )
    
    # Weights should have changed
    new_weights = cl_manager.online_learner.get_weights()
    assert not np.array_equal(new_weights, weights)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_performance_monitoring_latency(performance_monitor):
    """Test performance monitoring is fast (<10ms)"""
    import time
    
    performance_monitor.set_baseline(0.58)
    
    start = time.time()
    for _ in range(1000):
        performance_monitor.update(0.55)
    elapsed = time.time() - start
    
    avg_latency = (elapsed / 1000) * 1000  # Convert to ms
    assert avg_latency < 10  # <10ms per update


def test_feature_tracking_latency(feature_tracker):
    """Test feature tracking is fast (<50ms)"""
    import time
    
    baseline = {'rsi': 0.30, 'macd': 0.25, 'volume': 0.20}
    feature_tracker.set_baseline(baseline)
    
    start = time.time()
    for _ in range(100):
        shap = {'rsi': 0.05, 'macd': 0.03, 'volume': 0.08}
        feature_tracker.update(shap)
    elapsed = time.time() - start
    
    avg_latency = (elapsed / 100) * 1000  # Convert to ms
    assert avg_latency < 50  # <50ms per update


def test_online_update_latency(online_learner):
    """Test online learning update is fast (<100ms)"""
    import time
    
    weights = np.array([0.5, 0.3, 0.2, 0.1])
    online_learner.initialize(weights)
    
    start = time.time()
    for i in range(10):
        features = np.array([65.0, -0.02, 1.2, 0.15])
        online_learner.record_trade(features, 1.0)
    elapsed = time.time() - start
    
    latency_ms = elapsed * 1000
    assert latency_ms < 100  # <100ms for batch update


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
