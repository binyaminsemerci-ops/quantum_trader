"""
Tests for StrategyService
EPIC: DASHBOARD-V3-TRADING-PANELS
"""

import pytest
from unittest.mock import Mock
from backend.domains.strategies import StrategyService, StrategyInfo


class MockPolicyStore:
    """Mock PolicyStore for testing"""
    def __init__(self, policy_data=None, should_fail=False):
        self.policy_data = policy_data or {}
        self.should_fail = should_fail
    
    def get(self):
        if self.should_fail:
            raise Exception("PolicyStore error")
        return self.policy_data


class TestStrategyService:
    """Test suite for StrategyService"""
    
    def test_get_active_strategies_without_policy_store(self):
        """Test strategy retrieval when PolicyStore is None"""
        service = StrategyService(policy_store=None)
        strategies = service.get_active_strategies()
        
        # Should return default strategy
        assert len(strategies) == 1
        assert strategies[0].name == "default"
        assert strategies[0].enabled is True
        assert strategies[0].profile == "normal"
    
    def test_get_active_strategies_with_policy_store(self):
        """Test strategy retrieval with working PolicyStore"""
        # Mock policy data
        policy_data = {
            "risk_mode": "AGGRESSIVE",
            "max_positions": 10,
            "global_min_confidence": 0.70
        }
        mock_store = MockPolicyStore(policy_data)
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Should return main strategy + ai_ensemble
        assert len(strategies) == 2
        assert strategies[0].name == "quantum_trader_aggressive"
        assert strategies[0].profile == "agg"
        assert strategies[0].min_confidence == 0.70
        assert "max 10 positions" in strategies[0].description
        assert strategies[1].name == "ai_ensemble"
    
    def test_get_active_strategies_risk_mode_mapping(self):
        """Test risk mode to profile mapping"""
        test_cases = [
            ("AGGRESSIVE", "agg"),
            ("NORMAL", "normal"),
            ("DEFENSIVE", "low"),
            ("CONSERVATIVE", "micro"),
        ]
        
        for risk_mode, expected_profile in test_cases:
            policy_data = {"risk_mode": risk_mode}
            mock_store = MockPolicyStore(policy_data)
            
            service = StrategyService(policy_store=mock_store)
            strategies = service.get_active_strategies()
            
            assert strategies[0].profile == expected_profile
    
    def test_get_active_strategies_policy_store_error(self):
        """Test fallback when PolicyStore.get() raises exception"""
        mock_store = MockPolicyStore(should_fail=True)
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Should still return strategies with default values
        assert len(strategies) == 2
        assert strategies[0].name == "quantum_trader_normal"
        assert strategies[0].profile == "normal"
    
    def test_get_active_strategies_default_values(self):
        """Test default values when policy fields are missing"""
        # Empty policy data
        mock_store = MockPolicyStore(policy_data={})
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Should use default values
        assert len(strategies) == 2
        assert strategies[0].name == "quantum_trader_normal"
        assert strategies[0].profile == "normal"
        assert strategies[0].min_confidence == 0.65
        assert "max 5 positions" in strategies[0].description
    
    def test_get_active_strategies_general_error(self):
        """Test error handling for general exceptions"""
        # Create a mock that raises on attribute access
        mock_store = Mock()
        mock_store.get.side_effect = AttributeError("Unexpected error")
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Should return fallback strategies (not default, but normal+ai_ensemble)
        assert len(strategies) == 2
        assert strategies[0].name == "quantum_trader_normal"
        assert strategies[1].name == "ai_ensemble"
    
    def test_get_strategy_by_name_found(self):
        """Test retrieving specific strategy by name"""
        policy_data = {"risk_mode": "NORMAL"}
        mock_store = MockPolicyStore(policy_data)
        
        service = StrategyService(policy_store=mock_store)
        strategy = service.get_strategy_by_name("quantum_trader_normal")
        
        assert strategy is not None
        assert strategy.name == "quantum_trader_normal"
    
    def test_get_strategy_by_name_not_found(self):
        """Test retrieving non-existent strategy"""
        policy_data = {"risk_mode": "NORMAL"}
        mock_store = MockPolicyStore(policy_data)
        
        service = StrategyService(policy_store=mock_store)
        strategy = service.get_strategy_by_name("nonexistent_strategy")
        
        assert strategy is None
    
    def test_strategy_info_fields(self):
        """Test that StrategyInfo has all required fields"""
        policy_data = {
            "risk_mode": "AGGRESSIVE",
            "max_positions": 10,
            "global_min_confidence": 0.75
        }
        mock_store = MockPolicyStore(policy_data)
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Verify all required fields are present
        strategy = strategies[0]
        assert hasattr(strategy, 'name')
        assert hasattr(strategy, 'enabled')
        assert hasattr(strategy, 'profile')
        assert hasattr(strategy, 'exchanges')
        assert hasattr(strategy, 'symbols')
        assert hasattr(strategy, 'description')
        assert hasattr(strategy, 'min_confidence')
        
        assert isinstance(strategy.name, str)
        assert isinstance(strategy.enabled, bool)
        assert isinstance(strategy.profile, str)
        assert isinstance(strategy.exchanges, list)
        assert isinstance(strategy.symbols, list)
        assert isinstance(strategy.description, str)
        assert isinstance(strategy.min_confidence, (int, float))
    
    def test_ai_ensemble_strategy(self):
        """Test that AI ensemble strategy is included"""
        policy_data = {"risk_mode": "NORMAL"}
        mock_store = MockPolicyStore(policy_data)
        
        service = StrategyService(policy_store=mock_store)
        strategies = service.get_active_strategies()
        
        # Find AI ensemble strategy
        ai_ensemble = next((s for s in strategies if s.name == "ai_ensemble"), None)
        assert ai_ensemble is not None
        assert ai_ensemble.enabled is True
        assert "ensemble" in ai_ensemble.description.lower()
