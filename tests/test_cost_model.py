"""Tests for CostModel module."""

import pytest
from backend.services.cost_model import (
    CostModel,
    CostConfig,
    TradeCost,
    estimate_trade_cost,
    calculate_net_R,
)


class TestCostConfig:
    """Test cost configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CostConfig()
        
        assert config.maker_fee_rate == 0.0002  # 0.02%
        assert config.taker_fee_rate == 0.0004  # 0.04%
        assert config.base_slippage_bps == 2.0
        assert config.volatility_slippage_factor == 50.0
        assert config.funding_rate_per_8h == 0.0001
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostConfig(
            maker_fee_rate=0.0001,
            taker_fee_rate=0.0005,
            base_slippage_bps=3.0
        )
        
        assert config.maker_fee_rate == 0.0001
        assert config.taker_fee_rate == 0.0005
        assert config.base_slippage_bps == 3.0


class TestBasicCostEstimation:
    """Test basic cost estimation."""
    
    def test_simple_long_trade(self):
        """Test cost estimation for a simple long trade."""
        model = CostModel()
        
        # BTC long: entry $50k, exit $51k, size 0.1 BTC
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0,
            holding_hours=24.0,
            entry_is_maker=True,
            exit_is_maker=False
        )
        
        # Check components exist and are positive
        assert cost.entry_fee > 0
        assert cost.exit_fee > 0
        assert cost.entry_slippage >= 0
        assert cost.exit_slippage >= 0
        assert cost.funding_cost >= 0
        assert cost.total_cost > 0
        
        # Entry fee: $5000 * 0.0002 = $1.00
        assert cost.entry_fee == pytest.approx(1.0, abs=0.01)
        
        # Exit fee: $5100 * 0.0004 = $2.04
        assert cost.exit_fee == pytest.approx(2.04, abs=0.01)
    
    def test_simple_short_trade(self):
        """Test cost estimation for a short trade."""
        model = CostModel()
        
        # BTC short: entry $50k, exit $49k, size 0.1 BTC
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=49000.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0,
            holding_hours=12.0
        )
        
        assert cost.total_cost > 0
        assert cost.entry_fee > 0
        assert cost.exit_fee > 0
        
        # Funding cost should be lower (12h vs 24h)
        assert cost.funding_cost > 0


class TestSlippageEstimation:
    """Test slippage estimation."""
    
    def test_maker_vs_taker_slippage(self):
        """Test that maker orders have lower slippage than taker."""
        model = CostModel()
        
        maker_slippage = model.estimate_slippage(
            price=50000.0,
            size=0.1,
            atr_ratio=0.01,
            is_maker=True
        )
        
        taker_slippage = model.estimate_slippage(
            price=50000.0,
            size=0.1,
            atr_ratio=0.01,
            is_maker=False
        )
        
        # Taker should have higher slippage
        assert taker_slippage > maker_slippage
        assert maker_slippage > 0
    
    def test_volatility_increases_slippage(self):
        """Test that higher volatility increases slippage."""
        model = CostModel()
        
        low_vol_slippage = model.estimate_slippage(
            price=50000.0,
            size=0.1,
            atr_ratio=0.005,  # 0.5% ATR
            is_maker=False
        )
        
        high_vol_slippage = model.estimate_slippage(
            price=50000.0,
            size=0.1,
            atr_ratio=0.03,   # 3% ATR
            is_maker=False
        )
        
        # Higher volatility should increase slippage
        assert high_vol_slippage > low_vol_slippage
    
    def test_larger_size_increases_slippage(self):
        """Test that larger position size increases slippage."""
        model = CostModel()
        
        small_slippage = model.estimate_slippage(
            price=50000.0,
            size=0.1,
            atr_ratio=0.01,
            is_maker=False
        )
        
        large_slippage = model.estimate_slippage(
            price=50000.0,
            size=1.0,
            atr_ratio=0.01,
            is_maker=False
        )
        
        # Larger size should have proportionally higher slippage
        assert large_slippage > small_slippage
        assert large_slippage == pytest.approx(small_slippage * 10, rel=0.01)


class TestNetRCalculation:
    """Test net R-multiple calculations."""
    
    def test_winning_trade_net_R(self):
        """Test net R calculation for a winning trade."""
        model = CostModel()
        
        # BTC long: entry $50k, SL $49k, TP $52.5k (2.5R raw)
        net_R = model.net_R_after_costs(
            raw_R=2.5,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52500.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0
        )
        
        # Net R should be less than raw R due to costs
        assert net_R < 2.5
        assert net_R > 2.0  # But still profitable
        
        # Typical costs should reduce by ~0.05-0.3R
        assert 2.2 <= net_R <= 2.5
    
    def test_losing_trade_net_R(self):
        """Test net R calculation for a losing trade."""
        model = CostModel()
        
        # BTC long: entry $50k, SL $49k (hit stop = -1R raw)
        net_R = model.net_R_after_costs(
            raw_R=-1.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52500.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0
        )
        
        # Net R should be more negative due to costs
        assert net_R < -1.0
        assert net_R > -1.5  # But not drastically worse
        
        # Typical costs add ~0.1-0.2R to loss
        assert -1.3 <= net_R <= -1.05
    
    def test_breakeven_becomes_small_loss(self):
        """Test that breakeven trade becomes small loss after costs."""
        model = CostModel()
        
        net_R = model.net_R_after_costs(
            raw_R=0.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52500.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0
        )
        
        # Breakeven should become small loss due to costs
        assert net_R < 0.0
        assert net_R > -0.5


class TestBreakevenPrice:
    """Test breakeven price calculation."""
    
    def test_long_breakeven(self):
        """Test breakeven price for long position."""
        model = CostModel()
        
        breakeven = model.breakeven_price(
            entry_price=50000.0,
            size=0.1,
            direction="LONG",
            symbol="BTCUSDT",
            atr=500.0,
            holding_hours=24.0
        )
        
        # Breakeven should be above entry
        assert breakeven > 50000.0
        
        # Typical breakeven ~0.2-0.5% above entry
        breakeven_pct = (breakeven - 50000.0) / 50000.0
        assert 0.001 <= breakeven_pct <= 0.01  # 0.1% to 1%
    
    def test_short_breakeven(self):
        """Test breakeven price for short position."""
        model = CostModel()
        
        breakeven = model.breakeven_price(
            entry_price=50000.0,
            size=0.1,
            direction="SHORT",
            symbol="BTCUSDT",
            atr=500.0,
            holding_hours=24.0
        )
        
        # Breakeven should be below entry
        assert breakeven < 50000.0
        
        # Similar distance as long
        breakeven_pct = (50000.0 - breakeven) / 50000.0
        assert 0.001 <= breakeven_pct <= 0.01


class TestMinimumProfitTarget:
    """Test minimum profit target calculation."""
    
    def test_long_profit_target(self):
        """Test minimum profit target for long position."""
        model = CostModel()
        
        # Want 2.0R net, need to calculate gross R including costs
        min_tp = model.minimum_profit_target(
            entry_price=50000.0,
            stop_loss=49000.0,  # 1R = $1000
            size=0.1,
            direction="LONG",
            symbol="BTCUSDT",
            atr=500.0,
            target_net_R=2.0
        )
        
        # TP should be above entry
        assert min_tp > 50000.0
        
        # For 2.0R net, need ~2.2R gross
        # 1R = $1000, so 2.2R = $52,200
        assert 52000.0 <= min_tp <= 52500.0
    
    def test_short_profit_target(self):
        """Test minimum profit target for short position."""
        model = CostModel()
        
        min_tp = model.minimum_profit_target(
            entry_price=50000.0,
            stop_loss=51000.0,  # 1R = $1000
            size=0.1,
            direction="SHORT",
            symbol="BTCUSDT",
            atr=500.0,
            target_net_R=2.0
        )
        
        # TP should be below entry
        assert min_tp < 50000.0
        
        # For 2.0R net, need ~2.2R gross = $47,800
        assert 47500.0 <= min_tp <= 48000.0
    
    def test_higher_target_increases_tp(self):
        """Test that higher target net R increases TP distance."""
        model = CostModel()
        
        tp_2R = model.minimum_profit_target(
            entry_price=50000.0,
            stop_loss=49000.0,
            size=0.1,
            direction="LONG",
            symbol="BTCUSDT",
            atr=500.0,
            target_net_R=2.0
        )
        
        tp_3R = model.minimum_profit_target(
            entry_price=50000.0,
            stop_loss=49000.0,
            size=0.1,
            direction="LONG",
            symbol="BTCUSDT",
            atr=500.0,
            target_net_R=3.0
        )
        
        # 3R target should be further than 2R target
        assert tp_3R > tp_2R
        
        # Difference should be roughly 1R = $1000
        assert (tp_3R - tp_2R) > 900


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_estimate_trade_cost_function(self):
        """Test estimate_trade_cost convenience function."""
        cost = estimate_trade_cost(
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0
        )
        
        assert isinstance(cost, TradeCost)
        assert cost.total_cost > 0
        assert cost.entry_fee > 0
    
    def test_calculate_net_R_function(self):
        """Test calculate_net_R convenience function."""
        net_R = calculate_net_R(
            raw_R=2.5,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52500.0,
            size=0.1,
            symbol="BTCUSDT",
            atr=500.0
        )
        
        assert isinstance(net_R, float)
        assert net_R < 2.5
        assert net_R > 2.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_size(self):
        """Test handling of zero size."""
        model = CostModel()
        
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.0,
            symbol="BTCUSDT"
        )
        
        # All costs should be zero
        assert cost.total_cost == 0.0
        assert cost.entry_fee == 0.0
        assert cost.exit_fee == 0.0
    
    def test_zero_holding_period(self):
        """Test very short holding period."""
        model = CostModel()
        
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.1,
            symbol="BTCUSDT",
            holding_hours=0.1  # 6 minutes
        )
        
        # Funding cost should be minimal
        assert cost.funding_cost < 0.1
        # But fees and slippage still exist
        assert cost.entry_fee > 0
        assert cost.exit_fee > 0
    
    def test_very_small_position(self):
        """Test very small position size."""
        model = CostModel()
        
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=51000.0,
            size=0.001,  # 0.001 BTC = $50
            symbol="BTCUSDT",
            atr=500.0
        )
        
        # Costs should be proportionally small
        assert cost.total_cost < 1.0
        assert cost.total_cost > 0.0


class TestRealWorldScenarios:
    """Test realistic trading scenarios."""
    
    def test_btc_scalp_trade(self):
        """Test BTC scalp trade (quick in/out)."""
        model = CostModel()
        
        # Quick scalp: 0.5% profit target, 2-hour hold
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=50250.0,  # 0.5% profit
            size=0.2,
            symbol="BTCUSDT",
            atr=300.0,
            holding_hours=2.0,
            entry_is_maker=True,
            exit_is_maker=True  # Try to exit with limit
        )
        
        gross_profit = (50250.0 - 50000.0) * 0.2  # $50
        net_profit = gross_profit - cost.total_cost
        
        # Net profit should be positive but reduced
        assert net_profit > 0
        assert net_profit < gross_profit
        
        # Costs eat significant % of small profit
        cost_pct_of_profit = cost.total_cost / gross_profit
        assert cost_pct_of_profit > 0.1  # At least 10%
    
    def test_btc_swing_trade(self):
        """Test BTC swing trade (multi-day hold)."""
        model = CostModel()
        
        # Swing trade: 5% profit target, 72-hour hold
        cost = model.estimate_cost(
            entry_price=50000.0,
            exit_price=52500.0,  # 5% profit
            size=0.2,
            symbol="BTCUSDT",
            atr=800.0,
            holding_hours=72.0
        )
        
        gross_profit = (52500.0 - 50000.0) * 0.2  # $500
        net_profit = gross_profit - cost.total_cost
        
        # Costs should be smaller % of larger profit
        cost_pct_of_profit = cost.total_cost / gross_profit
        assert cost_pct_of_profit < 0.1  # Less than 10%
        
        # But funding costs accumulate over time
        assert cost.funding_cost > 0
    
    def test_altcoin_high_fee(self):
        """Test altcoin with higher fees and slippage."""
        config = CostConfig(
            maker_fee_rate=0.0004,  # Higher fees
            taker_fee_rate=0.0008,
            base_slippage_bps=5.0   # More slippage
        )
        model = CostModel(config)
        
        cost = model.estimate_cost(
            entry_price=1.50,
            exit_price=1.65,
            size=1000.0,
            symbol="ALTUSDT",
            atr=0.08,  # 5.3% ATR - volatile
            holding_hours=24.0
        )
        
        # Higher fees and slippage
        assert cost.total_cost > 2.0  # Should be several dollars
        assert cost.total_cost_pct > 0.001  # >0.1%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
