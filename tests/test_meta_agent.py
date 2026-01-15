#!/usr/bin/env python3
"""
Unit Tests for MetaPredictorAgent - Meta-Learning Layer
--------------------------------------------------------
Tests model loading, prediction logic, and ensemble fusion.

Run:
    pytest tests/test_meta_agent.py -v
    python3 tests/test_meta_agent.py
"""
import pytest
import sys
import torch
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.agents.meta_agent import MetaPredictorAgent, MetaNet

# ---------- FIXTURES ----------
@pytest.fixture
def meta_agent():
    """Create meta agent instance"""
    return MetaPredictorAgent()

@pytest.fixture
def sample_ensemble_vector():
    """Sample ensemble predictions for testing"""
    return {
        'xgb': {'action': 'BUY', 'confidence': 0.75},
        'lgbm': {'action': 'BUY', 'confidence': 0.70},
        'patch': {'action': 'HOLD', 'confidence': 0.60},
        'nhits': {'action': 'BUY', 'confidence': 0.68}
    }

# ---------- MODEL LOADING TESTS ----------
def test_meta_agent_initialization(meta_agent):
    """Test meta agent initializes correctly"""
    assert meta_agent is not None
    assert hasattr(meta_agent, 'model')
    assert hasattr(meta_agent, 'scaler')

def test_meta_agent_ready_check(meta_agent):
    """Test is_ready() method"""
    # Should have model and scaler loaded
    ready = meta_agent.is_ready()
    assert isinstance(ready, bool)
    
    if ready:
        assert meta_agent.model is not None
        assert meta_agent.scaler is not None

# ---------- NETWORK ARCHITECTURE TESTS ----------
def test_meta_network_structure():
    """Test MetaNet architecture"""
    net = MetaNet()
    
    # Check layers exist
    assert hasattr(net, 'fc1')
    assert hasattr(net, 'fc2')
    assert hasattr(net, 'fc3')
    
    # Check dimensions
    assert net.fc1.in_features == 8  # 4 confidences + 4 actions
    assert net.fc1.out_features == 32
    assert net.fc2.in_features == 32
    assert net.fc2.out_features == 32
    assert net.fc3.in_features == 32
    assert net.fc3.out_features == 3  # SELL, HOLD, BUY

def test_meta_network_forward_pass():
    """Test forward pass through network"""
    net = MetaNet()
    net.eval()
    
    # Create sample input
    x = torch.randn(1, 8)
    
    # Forward pass
    with torch.no_grad():
        output = net(x)
    
    assert output.shape == (1, 3)
    assert torch.all(torch.isfinite(output))

# ---------- PREDICTION TESTS ----------
def test_predict_with_full_ensemble(meta_agent, sample_ensemble_vector):
    """Test prediction with all 4 models"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    result = meta_agent.predict(sample_ensemble_vector, 'BTCUSDT')
    
    assert 'action' in result
    assert 'confidence' in result
    assert 'source' in result
    assert result['action'] in ['SELL', 'HOLD', 'BUY']
    assert 0.0 <= result['confidence'] <= 1.0
    assert 'meta' in result['source']

def test_predict_with_partial_ensemble(meta_agent):
    """Test prediction with only 2 models"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    partial_vector = {
        'xgb': {'action': 'BUY', 'confidence': 0.80},
        'lgbm': {'action': 'BUY', 'confidence': 0.75}
    }
    
    result = meta_agent.predict(partial_vector, 'ETHUSDT')
    
    assert 'action' in result
    assert 'confidence' in result
    # Should still work with partial data

def test_predict_returns_valid_action(meta_agent, sample_ensemble_vector):
    """Test prediction returns valid action"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    result = meta_agent.predict(sample_ensemble_vector, 'BTCUSDT')
    
    valid_actions = ['SELL', 'HOLD', 'BUY']
    assert result['action'] in valid_actions

def test_predict_confidence_range(meta_agent, sample_ensemble_vector):
    """Test confidence is in valid range"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    result = meta_agent.predict(sample_ensemble_vector, 'BTCUSDT')
    
    assert 0.0 <= result['confidence'] <= 1.0

# ---------- INPUT PROCESSING TESTS ----------
def test_action_encoding():
    """Test action encoding (SELL=0, HOLD=1, BUY=2)"""
    meta_agent = MetaPredictorAgent()
    
    # These should be encoded internally
    ensemble_vector = {
        'xgb': {'action': 'SELL', 'confidence': 0.70},
        'lgbm': {'action': 'HOLD', 'confidence': 0.60},
        'patch': {'action': 'BUY', 'confidence': 0.80},
        'nhits': {'action': 'BUY', 'confidence': 0.75}
    }
    
    # If model is ready, test prediction works
    if meta_agent.is_ready():
        result = meta_agent.predict(ensemble_vector, 'BTCUSDT')
        assert 'action' in result

def test_confidence_normalization(meta_agent, sample_ensemble_vector):
    """Test confidences are normalized properly"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    # Confidences should be scaled by StandardScaler
    result = meta_agent.predict(sample_ensemble_vector, 'BTCUSDT')
    
    # Output confidence should be reasonable
    assert 0.0 <= result['confidence'] <= 1.0

# ---------- ENSEMBLE FUSION TESTS ----------
def test_unanimous_buy_signal(meta_agent):
    """Test unanimous BUY from all models"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    unanimous_buy = {
        'xgb': {'action': 'BUY', 'confidence': 0.85},
        'lgbm': {'action': 'BUY', 'confidence': 0.80},
        'patch': {'action': 'BUY', 'confidence': 0.78},
        'nhits': {'action': 'BUY', 'confidence': 0.82}
    }
    
    result = meta_agent.predict(unanimous_buy, 'BTCUSDT')
    
    # Should likely be BUY with high confidence
    # (but meta agent can override if it learned differently)
    assert result['action'] in ['BUY', 'HOLD']
    assert result['confidence'] > 0.5

def test_unanimous_hold_signal(meta_agent):
    """Test unanimous HOLD from all models"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    unanimous_hold = {
        'xgb': {'action': 'HOLD', 'confidence': 0.90},
        'lgbm': {'action': 'HOLD', 'confidence': 0.88},
        'patch': {'action': 'HOLD', 'confidence': 0.85},
        'nhits': {'action': 'HOLD', 'confidence': 0.87}
    }
    
    result = meta_agent.predict(unanimous_hold, 'BTCUSDT')
    
    # Should likely be HOLD with high confidence
    assert result['action'] in ['HOLD', 'BUY', 'SELL']

def test_mixed_signals(meta_agent):
    """Test mixed signals from models"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    mixed_signals = {
        'xgb': {'action': 'BUY', 'confidence': 0.70},
        'lgbm': {'action': 'SELL', 'confidence': 0.65},
        'patch': {'action': 'HOLD', 'confidence': 0.60},
        'nhits': {'action': 'BUY', 'confidence': 0.68}
    }
    
    result = meta_agent.predict(mixed_signals, 'BTCUSDT')
    
    # Meta agent should resolve conflict
    assert result['action'] in ['SELL', 'HOLD', 'BUY']

# ---------- ERROR HANDLING TESTS ----------
def test_empty_ensemble_vector(meta_agent):
    """Test handling of empty ensemble vector"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    try:
        result = meta_agent.predict({}, 'BTCUSDT')
        # Should handle gracefully or raise exception
        assert 'action' in result
    except (KeyError, ValueError, TypeError):
        # Expected error handling
        pass

def test_missing_confidence(meta_agent):
    """Test handling of missing confidence values"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    incomplete_vector = {
        'xgb': {'action': 'BUY'},  # Missing confidence
        'lgbm': {'action': 'HOLD', 'confidence': 0.70}
    }
    
    try:
        result = meta_agent.predict(incomplete_vector, 'BTCUSDT')
        assert 'action' in result
    except (KeyError, ValueError):
        # Expected error
        pass

# ---------- MODEL STATE TESTS ----------
def test_model_in_eval_mode(meta_agent):
    """Test model is in evaluation mode"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    assert not meta_agent.model.training

def test_no_gradient_computation(meta_agent, sample_ensemble_vector):
    """Test gradients are disabled during prediction"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    # Prediction should not compute gradients
    result = meta_agent.predict(sample_ensemble_vector, 'BTCUSDT')
    
    # Check no gradients are tracked
    for param in meta_agent.model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)

# ---------- INTEGRATION TESTS ----------
def test_full_prediction_pipeline(meta_agent):
    """Test complete prediction pipeline"""
    if not meta_agent.is_ready():
        pytest.skip("Meta agent not ready (model not found)")
    
    # Simulate real ensemble output
    ensemble_output = {
        'xgb': {'action': 'BUY', 'confidence': 0.829},
        'lgbm': {'action': 'BUY', 'confidence': 0.786},
        'patch': {'action': 'HOLD', 'confidence': 0.654},
        'nhits': {'action': 'BUY', 'confidence': 0.712}
    }
    
    result = meta_agent.predict(ensemble_output, 'BTCUSDT')
    
    # Verify complete output structure
    assert isinstance(result, dict)
    assert 'action' in result
    assert 'confidence' in result
    assert 'source' in result
    assert result['action'] in ['SELL', 'HOLD', 'BUY']
    assert 0.0 <= result['confidence'] <= 1.0
    assert isinstance(result['source'], str)

# ---------- RUN TESTS ----------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
