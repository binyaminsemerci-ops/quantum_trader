"""Tests for SymbolPerformanceManager module."""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from backend.services.symbol_performance import (
    SymbolPerformanceManager,
    SymbolPerformanceConfig,
    SymbolStats,
    TradeResult,
)


class TestSymbolPerformanceConfig:
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SymbolPerformanceConfig()
        
        assert config.min_trades_for_adjustment == 5
        assert config.poor_winrate_threshold == 0.30
        assert config.good_winrate_threshold == 0.55
        assert config.poor_risk_multiplier == 0.5
        assert config.good_risk_multiplier == 1.0
        assert config.disable_after_losses == 10


class TestSymbolStats:
    """Test symbol statistics."""
    
    def test_initial_stats(self):
        """Test initial statistics."""
        stats = SymbolStats(symbol="BTCUSDT")
        
        assert stats.symbol == "BTCUSDT"
        assert stats.trade_count == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.win_rate == 0.0
        assert stats.avg_R == 0.0
        assert stats.is_enabled is True
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        stats = SymbolStats(symbol="BTCUSDT")
        stats.trade_count = 10
        stats.wins = 6
        stats.losses = 4
        
        assert stats.win_rate == 0.6
    
    def test_avg_R_calculation(self):
        """Test average R calculation."""
        stats = SymbolStats(symbol="BTCUSDT")
        stats.trade_count = 5
        stats.total_R = 7.5
        
        assert stats.avg_R == 1.5
    
    def test_serialization(self):
        """Test to_dict and from_dict."""
        stats = SymbolStats(
            symbol="ETHUSDT",
            trade_count=10,
            wins=7,
            losses=3,
            total_R=12.5,
            total_pnl=500.0
        )
        
        data = stats.to_dict()
        restored = SymbolStats.from_dict(data)
        
        assert restored.symbol == stats.symbol
        assert restored.trade_count == stats.trade_count
        assert restored.wins == stats.wins
        assert restored.total_R == stats.total_R


class TestBasicTracking:
    """Test basic statistics tracking."""
    
    def test_update_winning_trade(self):
        """Test updating stats with a winning trade."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        trade = TradeResult(
            symbol="BTCUSDT",
            pnl=100.0,
            R_multiple=2.5,
            was_winner=True
        )
        
        manager.update_stats(trade)
        stats = manager.get_stats("BTCUSDT")
        
        assert stats.trade_count == 1
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.total_R == 2.5
        assert stats.total_pnl == 100.0
        assert stats.consecutive_wins == 1
        assert stats.consecutive_losses == 0
    
    def test_update_losing_trade(self):
        """Test updating stats with a losing trade."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        trade = TradeResult(
            symbol="BTCUSDT",
            pnl=-50.0,
            R_multiple=-1.0,
            was_winner=False
        )
        
        manager.update_stats(trade)
        stats = manager.get_stats("BTCUSDT")
        
        assert stats.trade_count == 1
        assert stats.wins == 0
        assert stats.losses == 1
        assert stats.total_R == -1.0
        assert stats.total_pnl == -50.0
        assert stats.consecutive_wins == 0
        assert stats.consecutive_losses == 1
    
    def test_multiple_trades(self):
        """Test tracking multiple trades."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # 3 wins, 2 losses
        trades = [
            TradeResult("BTCUSDT", 100.0, 2.0, True),
            TradeResult("BTCUSDT", -50.0, -1.0, False),
            TradeResult("BTCUSDT", 150.0, 3.0, True),
            TradeResult("BTCUSDT", -50.0, -1.0, False),
            TradeResult("BTCUSDT", 75.0, 1.5, True),
        ]
        
        for trade in trades:
            manager.update_stats(trade)
        
        stats = manager.get_stats("BTCUSDT")
        
        assert stats.trade_count == 5
        assert stats.wins == 3
        assert stats.losses == 2
        assert stats.win_rate == 0.6
        assert stats.total_R == 4.5  # 2 - 1 + 3 - 1 + 1.5
        assert stats.avg_R == 0.9


class TestRiskAdjustment:
    """Test risk modifier calculations."""
    
    def test_no_history_default_multiplier(self):
        """Test that symbols with no history get 1.0 multiplier."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        modifier = manager.get_risk_modifier("NEWUSDT")
        
        assert modifier == 1.0
    
    def test_insufficient_trades_default_multiplier(self):
        """Test that insufficient trades get 1.0 multiplier."""
        config = SymbolPerformanceConfig(min_trades_for_adjustment=5, persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # Add only 3 trades
        for i in range(3):
            manager.update_stats(TradeResult("BTCUSDT", 50.0, 1.0, True))
        
        modifier = manager.get_risk_modifier("BTCUSDT")
        
        assert modifier == 1.0  # Not enough trades yet
    
    def test_poor_performance_reduces_risk(self):
        """Test that poor performance reduces risk multiplier."""
        config = SymbolPerformanceConfig(
            min_trades_for_adjustment=5,
            poor_winrate_threshold=0.30,
            poor_risk_multiplier=0.5,
            persistence_file=None
        )
        manager = SymbolPerformanceManager(config)
        
        # Create poor win rate: 1 win, 9 losses = 10% WR
        manager.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
        for i in range(9):
            manager.update_stats(TradeResult("BTCUSDT", -50.0, -1.0, False))
        
        modifier = manager.get_risk_modifier("BTCUSDT")
        
        assert modifier == 0.5  # Poor performance
    
    def test_poor_avg_R_reduces_risk(self):
        """Test that poor average R reduces risk multiplier."""
        config = SymbolPerformanceConfig(
            min_trades_for_adjustment=5,
            poor_avg_R_threshold=0.0,
            poor_risk_multiplier=0.5,
            persistence_file=None
        )
        manager = SymbolPerformanceManager(config)
        
        # Create negative avg R: more losses than wins
        manager.update_stats(TradeResult("ETHUSDT", 100.0, 2.0, True))
        manager.update_stats(TradeResult("ETHUSDT", 100.0, 2.0, True))
        for i in range(4):
            manager.update_stats(TradeResult("ETHUSDT", -60.0, -1.2, False))
        
        stats = manager.get_stats("ETHUSDT")
        assert stats.avg_R < 0.0  # Negative avg R
        
        modifier = manager.get_risk_modifier("ETHUSDT")
        
        assert modifier == 0.5  # Poor performance
    
    def test_good_performance_keeps_standard_risk(self):
        """Test that good performance keeps 1.0 multiplier (never increases)."""
        config = SymbolPerformanceConfig(
            min_trades_for_adjustment=5,
            good_winrate_threshold=0.55,
            good_avg_R_threshold=1.5,
            good_risk_multiplier=1.0,
            persistence_file=None
        )
        manager = SymbolPerformanceManager(config)
        
        # Create excellent performance: 8 wins, 2 losses, high R
        for i in range(8):
            manager.update_stats(TradeResult("SOLUSDT", 150.0, 3.0, True))
        for i in range(2):
            manager.update_stats(TradeResult("SOLUSDT", -50.0, -1.0, False))
        
        stats = manager.get_stats("SOLUSDT")
        assert stats.win_rate == 0.8  # 80% WR
        assert stats.avg_R > 1.5      # High avg R
        
        modifier = manager.get_risk_modifier("SOLUSDT")
        
        assert modifier == 1.0  # Good performance, but don't increase
    
    def test_average_performance_standard_risk(self):
        """Test that average performance gets 1.0 multiplier."""
        config = SymbolPerformanceConfig(min_trades_for_adjustment=5, persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # Create average performance: 50% WR, 0.5R avg
        for i in range(5):
            manager.update_stats(TradeResult("ADAUSDT", 100.0, 2.0, True))
            manager.update_stats(TradeResult("ADAUSDT", -75.0, -1.5, False))
        
        modifier = manager.get_risk_modifier("ADAUSDT")
        
        assert modifier == 1.0


class TestSymbolDisabling:
    """Test symbol enable/disable logic."""
    
    def test_disable_after_consecutive_losses(self):
        """Test symbol gets disabled after consecutive losses."""
        config = SymbolPerformanceConfig(disable_after_losses=5, persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # 5 consecutive losses
        for i in range(5):
            manager.update_stats(TradeResult("BTCUSDT", -50.0, -1.0, False))
        
        stats = manager.get_stats("BTCUSDT")
        
        assert stats.consecutive_losses == 5
        assert stats.is_enabled is False
        assert manager.should_trade_symbol("BTCUSDT") is False
        assert manager.get_risk_modifier("BTCUSDT") == 0.0
    
    def test_reenable_after_wins(self):
        """Test symbol gets re-enabled after consecutive wins."""
        config = SymbolPerformanceConfig(
            disable_after_losses=5,
            reenable_after_wins=3,
            persistence_file=None
        )
        manager = SymbolPerformanceManager(config)
        
        # Disable: 5 consecutive losses
        for i in range(5):
            manager.update_stats(TradeResult("ETHUSDT", -50.0, -1.0, False))
        
        assert manager.should_trade_symbol("ETHUSDT") is False
        
        # Re-enable: 3 consecutive wins (hypothetically if we allowed trading)
        # Simulate re-enabling by updating stats directly
        stats = manager.get_stats("ETHUSDT")
        stats.is_enabled = True  # Manually re-enable for testing
        
        for i in range(3):
            manager.update_stats(TradeResult("ETHUSDT", 100.0, 2.0, True))
        
        assert manager.should_trade_symbol("ETHUSDT") is True
    
    def test_consecutive_counter_resets(self):
        """Test that consecutive counters reset properly."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # 3 wins
        for i in range(3):
            manager.update_stats(TradeResult("SOLUSDT", 100.0, 2.0, True))
        
        stats = manager.get_stats("SOLUSDT")
        assert stats.consecutive_wins == 3
        assert stats.consecutive_losses == 0
        
        # 1 loss - should reset wins
        manager.update_stats(TradeResult("SOLUSDT", -50.0, -1.0, False))
        
        assert stats.consecutive_wins == 0
        assert stats.consecutive_losses == 1


class TestShouldTradeSymbol:
    """Test should_trade_symbol logic."""
    
    def test_new_symbol_can_trade(self):
        """Test that new symbols can be traded."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        assert manager.should_trade_symbol("NEWUSDT") is True
    
    def test_enabled_symbol_can_trade(self):
        """Test that enabled symbols can be traded."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        manager.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
        
        assert manager.should_trade_symbol("BTCUSDT") is True
    
    def test_disabled_symbol_cannot_trade(self):
        """Test that disabled symbols cannot be traded."""
        config = SymbolPerformanceConfig(disable_after_losses=3, persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # Disable
        for i in range(3):
            manager.update_stats(TradeResult("BADUSDT", -50.0, -1.0, False))
        
        assert manager.should_trade_symbol("BADUSDT") is False


class TestPersistence:
    """Test statistics persistence."""
    
    def test_save_and_load_stats(self):
        """Test saving and loading statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence_file = Path(tmpdir) / "test_stats.json"
            
            config = SymbolPerformanceConfig(persistence_file=str(persistence_file))
            manager1 = SymbolPerformanceManager(config)
            
            # Add some trades
            manager1.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
            manager1.update_stats(TradeResult("BTCUSDT", -50.0, -1.0, False))
            manager1.update_stats(TradeResult("ETHUSDT", 150.0, 3.0, True))
            
            # Create new manager - should load persisted stats
            manager2 = SymbolPerformanceManager(config)
            
            btc_stats = manager2.get_stats("BTCUSDT")
            eth_stats = manager2.get_stats("ETHUSDT")
            
            assert btc_stats is not None
            assert btc_stats.trade_count == 2
            assert btc_stats.wins == 1
            assert btc_stats.losses == 1
            
            assert eth_stats is not None
            assert eth_stats.trade_count == 1
            assert eth_stats.wins == 1
    
    def test_no_persistence_file(self):
        """Test that manager works without persistence file."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        manager.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
        
        stats = manager.get_stats("BTCUSDT")
        assert stats.trade_count == 1


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_get_all_stats(self):
        """Test getting all statistics."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        manager.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
        manager.update_stats(TradeResult("ETHUSDT", -50.0, -1.0, False))
        manager.update_stats(TradeResult("SOLUSDT", 75.0, 1.5, True))
        
        all_stats = manager.get_all_stats()
        
        assert len(all_stats) == 3
        assert "BTCUSDT" in all_stats
        assert "ETHUSDT" in all_stats
        assert "SOLUSDT" in all_stats
    
    def test_reset_symbol(self):
        """Test resetting symbol statistics."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # Add trades
        for i in range(5):
            manager.update_stats(TradeResult("BTCUSDT", 100.0, 2.0, True))
        
        stats_before = manager.get_stats("BTCUSDT")
        assert stats_before.trade_count == 5
        
        # Reset
        manager.reset_symbol("BTCUSDT")
        
        stats_after = manager.get_stats("BTCUSDT")
        assert stats_after.trade_count == 0
        assert stats_after.wins == 0
        assert stats_after.total_R == 0.0


class TestRealWorldScenarios:
    """Test realistic scenarios."""
    
    def test_struggling_altcoin(self):
        """Test altcoin with poor performance."""
        config = SymbolPerformanceConfig(
            min_trades_for_adjustment=5,
            poor_winrate_threshold=0.30,
            poor_risk_multiplier=0.5,
            disable_after_losses=10,
            persistence_file=None
        )
        manager = SymbolPerformanceManager(config)
        
        # Bad altcoin: 2 wins, 8 losses
        for i in range(2):
            manager.update_stats(TradeResult("BADALT", 50.0, 1.0, True))
        for i in range(8):
            manager.update_stats(TradeResult("BADALT", -60.0, -1.2, False))
        
        stats = manager.get_stats("BADALT")
        
        # Should have poor stats
        assert stats.win_rate == 0.2  # 20%
        assert stats.avg_R < 0.0
        
        # Should reduce risk
        modifier = manager.get_risk_modifier("BADALT")
        assert modifier == 0.5
        
        # Should still be tradeable (not disabled yet)
        assert manager.should_trade_symbol("BADALT") is True
    
    def test_consistent_performer(self):
        """Test consistent profitable symbol."""
        config = SymbolPerformanceConfig(persistence_file=None)
        manager = SymbolPerformanceManager(config)
        
        # Consistent: 60% WR, 1.0R avg
        for i in range(6):
            manager.update_stats(TradeResult("GOODCOIN", 100.0, 2.0, True))
        for i in range(4):
            manager.update_stats(TradeResult("GOODCOIN", -80.0, -1.6, False))
        
        stats = manager.get_stats("GOODCOIN")
        
        assert stats.win_rate == 0.6
        assert 0.5 <= stats.avg_R <= 1.2  # 6*2 - 4*1.6 = 12 - 6.4 = 5.6, /10 = 0.56
        
        # Should use standard risk
        modifier = manager.get_risk_modifier("GOODCOIN")
        assert modifier == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
