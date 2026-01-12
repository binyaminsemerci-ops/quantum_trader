#!/usr/bin/env python3
"""
Unit Tests for GovernerAgent - Risk Management Layer
----------------------------------------------------
Tests Kelly Criterion calculations, circuit breakers, cooldowns,
and trade approval logic.

Run:
    pytest tests/test_governer_agent.py -v
    python3 tests/test_governer_agent.py
"""
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.agents.governer_agent import GovernerAgent, RiskConfig, PositionAllocation

# ---------- FIXTURES ----------
@pytest.fixture
def default_config():
    """Default risk configuration for testing"""
    return RiskConfig(
        max_position_size_pct=0.10,
        max_total_exposure_pct=0.50,
        max_drawdown_pct=0.15,
        min_confidence_threshold=0.65,
        kelly_fraction=0.25,
        cooldown_after_loss_minutes=60,
        max_daily_trades=20,
        emergency_stop=False
    )

@pytest.fixture
def governer(default_config, tmp_path):
    """Create governer agent with temporary state file"""
    state_file = tmp_path / "governer_test_state.json"
    return GovernerAgent(config=default_config, state_file=str(state_file))

# ---------- KELLY CRITERION TESTS ----------
def test_kelly_calculation_high_confidence(governer):
    """Test Kelly sizing with high confidence"""
    kelly = governer._calculate_kelly_position(confidence=0.80)
    assert 0.05 <= kelly <= 0.10, "High confidence should give 5-10% position"

def test_kelly_calculation_low_confidence(governer):
    """Test Kelly sizing with low confidence"""
    kelly = governer._calculate_kelly_position(confidence=0.50)
    assert 0 <= kelly <= 0.03, "Low confidence should give small position"

def test_kelly_respects_max_position(governer):
    """Test that Kelly never exceeds max position size"""
    kelly = governer._calculate_kelly_position(confidence=0.99)
    assert kelly <= governer.config.max_position_size_pct

# ---------- CIRCUIT BREAKER TESTS ----------
def test_emergency_stop(governer):
    """Test emergency stop circuit breaker"""
    governer.config.emergency_stop = True
    is_safe, reason = governer._check_circuit_breakers()
    assert not is_safe
    assert "EMERGENCY_STOP" in reason

def test_max_drawdown_breaker(governer):
    """Test max drawdown circuit breaker"""
    governer.peak_balance = 10000
    governer.current_balance = 8000  # 20% drawdown
    is_safe, reason = governer._check_circuit_breakers()
    assert not is_safe
    assert "MAX_DRAWDOWN" in reason

def test_daily_trade_limit(governer):
    """Test daily trade limit"""
    governer.daily_trade_count = 20
    governer.last_trade_date = datetime.utcnow().strftime("%Y-%m-%d")
    is_safe, reason = governer._check_circuit_breakers()
    assert not is_safe
    assert "DAILY_TRADE_LIMIT" in reason

def test_max_exposure_limit(governer):
    """Test max exposure limit"""
    governer.current_balance = 10000
    governer.active_positions = {
        'BTCUSDT': 3000,
        'ETHUSDT': 2500  # Total: 5500 > 50% of 10000
    }
    is_safe, reason = governer._check_circuit_breakers()
    assert not is_safe
    assert "MAX_EXPOSURE" in reason

# ---------- COOLDOWN TESTS ----------
def test_cooldown_after_recent_loss(governer):
    """Test cooldown prevents trading after recent loss"""
    # Add recent loss
    governer.trade_history.append({
        'symbol': 'BTCUSDT',
        'pnl': -100,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    can_trade, reason = governer._check_cooldown('BTCUSDT')
    assert not can_trade
    assert "COOLDOWN" in reason

def test_cooldown_expired(governer):
    """Test trading allowed after cooldown expires"""
    # Add old loss (>60 min ago)
    old_time = datetime.utcnow() - timedelta(minutes=70)
    governer.trade_history.append({
        'symbol': 'BTCUSDT',
        'pnl': -100,
        'timestamp': old_time.isoformat()
    })
    
    can_trade, reason = governer._check_cooldown('BTCUSDT')
    assert can_trade
    assert reason == "OK"

def test_cooldown_different_symbol(governer):
    """Test cooldown only affects specific symbol"""
    # Add loss for BTCUSDT
    governer.trade_history.append({
        'symbol': 'BTCUSDT',
        'pnl': -100,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Should still allow ETHUSDT
    can_trade, reason = governer._check_cooldown('ETHUSDT')
    assert can_trade

# ---------- POSITION ALLOCATION TESTS ----------
def test_hold_signal_rejected(governer):
    """Test HOLD signals are not approved"""
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='HOLD',
        confidence=0.90
    )
    assert not allocation.approved
    assert allocation.reason == "HOLD_SIGNAL"

def test_low_confidence_rejected(governer):
    """Test low confidence signals are rejected"""
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.50  # Below 0.65 threshold
    )
    assert not allocation.approved
    assert "LOW_CONFIDENCE" in allocation.reason

def test_high_confidence_approved(governer):
    """Test high confidence signals are approved"""
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.80,
        balance=10000
    )
    assert allocation.approved
    assert allocation.position_size_usd > 0
    assert allocation.risk_amount_usd > 0

def test_position_size_calculation(governer):
    """Test position size is calculated correctly"""
    balance = 10000
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.80,
        balance=balance
    )
    
    # Position should be percentage of balance
    expected_min = balance * 0.01  # At least 1%
    expected_max = balance * 0.10  # Max 10%
    assert expected_min <= allocation.position_size_usd <= expected_max

def test_risk_amount_calculation(governer):
    """Test risk amount is 2% of position"""
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.80,
        balance=10000
    )
    
    if allocation.approved:
        expected_risk = allocation.position_size_usd * 0.02
        assert abs(allocation.risk_amount_usd - expected_risk) < 0.01

# ---------- TRADE RECORDING TESTS ----------
def test_record_winning_trade(governer):
    """Test recording winning trade"""
    initial_balance = governer.current_balance
    governer.record_trade_result(
        symbol='BTCUSDT',
        action='BUY',
        entry_price=50000,
        exit_price=51000,
        position_size=1000,
        pnl=100
    )
    
    assert governer.current_balance == initial_balance + 100
    assert len(governer.trade_history) == 1
    assert governer.trade_history[0]['pnl'] == 100

def test_record_losing_trade(governer):
    """Test recording losing trade"""
    initial_balance = governer.current_balance
    governer.record_trade_result(
        symbol='BTCUSDT',
        action='BUY',
        entry_price=50000,
        exit_price=49000,
        position_size=1000,
        pnl=-50
    )
    
    assert governer.current_balance == initial_balance - 50
    assert governer.trade_history[0]['pnl'] == -50

def test_peak_balance_tracking(governer):
    """Test peak balance is tracked correctly"""
    initial_peak = governer.peak_balance
    
    # Win increases balance
    governer.record_trade_result(
        symbol='BTCUSDT',
        action='BUY',
        entry_price=50000,
        exit_price=51000,
        position_size=1000,
        pnl=500
    )
    
    assert governer.peak_balance > initial_peak
    assert governer.peak_balance == governer.current_balance

# ---------- STATS TESTS ----------
def test_get_stats(governer):
    """Test stats calculation"""
    # Add some trades
    governer.record_trade_result('BTCUSDT', 'BUY', 50000, 51000, 1000, 100)
    governer.record_trade_result('ETHUSDT', 'SELL', 3000, 2900, 500, 50)
    governer.record_trade_result('BNBUSDT', 'BUY', 400, 390, 200, -20)
    
    stats = governer.get_stats()
    
    assert 'balance' in stats
    assert 'drawdown_pct' in stats
    assert 'recent_win_rate' in stats
    assert stats['total_trades'] == 3
    assert 0 <= stats['recent_win_rate'] <= 1.0

# ---------- INTEGRATION TESTS ----------
def test_full_approval_flow(governer):
    """Test complete approval workflow"""
    # High confidence BUY
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.85,
        balance=10000
    )
    
    assert allocation.approved
    assert allocation.action == 'BUY'
    assert allocation.symbol == 'BTCUSDT'
    assert allocation.confidence == 0.85
    assert allocation.position_size_usd > 0
    assert allocation.kelly_optimal > 0

def test_rejection_flow(governer):
    """Test rejection workflow"""
    # Set emergency stop
    governer.config.emergency_stop = True
    
    allocation = governer.allocate_position(
        symbol='BTCUSDT',
        action='BUY',
        confidence=0.85,
        balance=10000
    )
    
    assert not allocation.approved
    assert "EMERGENCY_STOP" in allocation.reason

# ---------- RUN TESTS ----------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
