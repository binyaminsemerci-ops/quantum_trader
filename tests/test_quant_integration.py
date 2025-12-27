"""Integration tests for quant modules in event_driven_executor."""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from backend.services.execution.event_driven_executor import EventDrivenExecutor
from backend.services.ai_trading_engine import AITradingEngine


class TestQuantModulesIntegration:
    """Test that all quant modules are properly integrated."""
    
    def test_executor_initializes_quant_modules(self):
        """Test that EventDrivenExecutor initializes all quant modules."""
        # Mock AI engine
        mock_ai_engine = Mock(spec=AITradingEngine)
        
        # Create executor
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["BTCUSDT", "ETHUSDT"],
            confidence_threshold=0.45,
            check_interval_seconds=30,
            cooldown_seconds=300
        )
        
        # Verify all quant modules are initialized
        assert hasattr(executor, 'regime_detector'), "RegimeDetector not initialized"
        assert hasattr(executor, 'cost_model'), "CostModel not initialized"
        assert hasattr(executor, 'symbol_perf'), "SymbolPerformanceManager not initialized"
        
        assert executor.regime_detector is not None
        assert executor.cost_model is not None
        assert executor.symbol_perf is not None
    
    def test_regime_detector_works(self):
        """Test that RegimeDetector can detect regimes."""
        mock_ai_engine = Mock(spec=AITradingEngine)
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["BTCUSDT"],
            confidence_threshold=0.45
        )
        
        # Mock market data with RegimeIndicators
        from backend.services.regime_detector import RegimeIndicators
        indicators = RegimeIndicators(
            price=50000.0,
            atr=1000.0,
            ema_200=48000.0,
            adx=25.0,
            range_high=51000.0,
            range_low=49000.0
        )
        
        # Detect regime
        regime = executor.regime_detector.detect_regime(indicators)
        
        assert regime is not None
        assert hasattr(regime, 'regime')
        assert regime.regime in [
            'LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL', 'EXTREME_VOL',
            'TRENDING', 'RANGING'
        ]
    
    def test_cost_model_estimates_costs(self):
        """Test that CostModel can estimate costs."""
        mock_ai_engine = Mock(spec=AITradingEngine)
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["BTCUSDT"],
            confidence_threshold=0.45
        )
        
        # Estimate costs using convenience function
        from backend.services.cost_model import estimate_trade_cost
        cost = estimate_trade_cost(
            entry_price=50000.0,
            exit_price=52000.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=1000.0
        )
        
        assert cost is not None
        assert cost.total_cost > 0
        assert cost.cost_in_R >= 0
        assert 0 < cost.total_cost_pct < 1.0  # Should be less than 100%
    
    def test_symbol_performance_tracks_trades(self):
        """Test that SymbolPerformanceManager tracks trades."""
        mock_ai_engine = Mock(spec=AITradingEngine)
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["BTCUSDT"],
            confidence_threshold=0.45
        )
        
        # Should be able to trade any symbol
        assert executor.symbol_perf.should_trade_symbol("BTCUSDT") is True
        
        # Get initial trade count (may have historical data)
        initial_stats = executor.symbol_perf.get_stats("BTCUSDT")
        initial_count = initial_stats.trade_count
        
        # Update with a winning trade using TradeResult
        from backend.services.symbol_performance import TradeResult
        trade_result = TradeResult(
            symbol="BTCUSDT",
            pnl=100.0,
            R_multiple=2.0,
            was_winner=True
        )
        executor.symbol_perf.update_stats(trade_result)
        
        # Check stats increased by 1
        updated_stats = executor.symbol_perf.get_stats("BTCUSDT")
        assert updated_stats.trade_count == initial_count + 1
        assert updated_stats.wins >= 1
    
    def test_symbol_performance_risk_adjustment(self):
        """Test that poor performance reduces risk."""
        mock_ai_engine = Mock(spec=AITradingEngine)
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["TESTUSDT"],  # Use non-existent symbol to avoid historical data
            confidence_threshold=0.45
        )
        
        from backend.services.symbol_performance import TradeResult
        
        # Simulate 5 losing trades (poor R)
        for _ in range(5):
            trade_result = TradeResult(
                symbol="TESTUSDT",
                pnl=-50.0,
                R_multiple=-1.0,
                was_winner=False
            )
            executor.symbol_perf.update_stats(trade_result)
        
        # Check that we have enough trades to trigger adjustment
        stats = executor.symbol_perf.get_stats("TESTUSDT")
        min_trades = executor.symbol_perf.config.min_trades_for_adjustment
        
        if stats.trade_count >= min_trades:
            # Check risk modifier (should be reduced due to poor performance)
            risk_modifier = executor.symbol_perf.get_risk_modifier("TESTUSDT")
            assert risk_modifier < 1.0, f"Risk should be reduced after {stats.trade_count} poor trades"
        else:
            # Not enough trades yet, should be 1.0
            risk_modifier = executor.symbol_perf.get_risk_modifier("TESTUSDT")
            assert risk_modifier == 1.0
    
    def test_symbol_gets_disabled_after_losses(self):
        """Test that symbols get disabled after consecutive losses."""
        mock_ai_engine = Mock(spec=AITradingEngine)
        executor = EventDrivenExecutor(
            ai_engine=mock_ai_engine,
            symbols=["APTUSDT"],
            confidence_threshold=0.45
        )
        
        from backend.services.symbol_performance import TradeResult
        
        # Simulate 10 consecutive losses
        for _ in range(10):
            trade_result = TradeResult(
                symbol="APTUSDT",
                pnl=-50.0,
                R_multiple=-1.0,
                was_winner=False
            )
            executor.symbol_perf.update_stats(trade_result)
        
        # Symbol should be disabled
        assert executor.symbol_perf.should_trade_symbol("APTUSDT") is False
        
        # Stats should show disabled
        stats = executor.symbol_perf.get_stats("APTUSDT")
        assert stats.is_enabled is False
        assert stats.consecutive_losses >= 10  # May have historical data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
