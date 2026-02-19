"""Cost Model - Trading cost estimation and net R-multiple calculations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_float(value: str | None, *, default: float) -> float:
    """Parse float from environment variable."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CostConfig:
    """Configuration for cost estimation."""
    
    # Fee rates (as decimals, e.g., 0.0004 = 0.04% = 4 basis points)
    maker_fee_rate: float = 0.0002      # 0.02% maker fee (Binance Futures)
    taker_fee_rate: float = 0.0004      # 0.04% taker fee (Binance Futures)
    
    # Slippage parameters
    base_slippage_bps: float = 2.0      # Base slippage in basis points
    volatility_slippage_factor: float = 50.0  # Multiply ATR ratio by this
    
    # Additional costs
    funding_rate_per_8h: float = 0.0001  # 0.01% per 8 hours (typical)
    
    @classmethod
    def from_env(cls) -> CostConfig:
        """Load configuration from environment variables."""
        return cls(
            maker_fee_rate=_parse_float(
                os.getenv("COST_MAKER_FEE_RATE"),
                default=0.0002
            ),
            taker_fee_rate=_parse_float(
                os.getenv("COST_TAKER_FEE_RATE"),
                default=0.0004
            ),
            base_slippage_bps=_parse_float(
                os.getenv("COST_BASE_SLIPPAGE_BPS"),
                default=2.0
            ),
            volatility_slippage_factor=_parse_float(
                os.getenv("COST_VOLATILITY_SLIPPAGE_FACTOR"),
                default=50.0
            ),
            funding_rate_per_8h=_parse_float(
                os.getenv("COST_FUNDING_RATE_PER_8H"),
                default=0.0001
            ),
        )


@dataclass
class TradeCost:
    """Complete cost breakdown for a trade."""
    entry_fee: float           # $ cost of entry fee
    exit_fee: float            # $ cost of exit fee
    entry_slippage: float      # $ cost of entry slippage
    exit_slippage: float       # $ cost of exit slippage
    funding_cost: float        # $ estimated funding cost
    total_cost: float          # $ total cost
    total_cost_pct: float      # % of position notional
    cost_in_R: float           # Cost as fraction of 1R (risk amount)
    
    def __str__(self) -> str:
        """String representation of costs."""
        return (
            f"TradeCost(total=${self.total_cost:.2f} / "
            f"{self.total_cost_pct:.3%} / {self.cost_in_R:.3f}R)"
        )


class CostModel:
    """
    Estimate trading costs including fees, slippage, and funding rates.
    
    Provides methods to:
    - Estimate total trade costs (fees + slippage)
    - Calculate net R-multiples after costs
    - Determine minimum profit targets to overcome costs
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost model.
        
        Args:
            config: Cost configuration. If None, loads from environment.
        """
        self.config = config or CostConfig.from_env()
        logger.info(
            f"[OK] CostModel initialized: "
            f"Maker={self.config.maker_fee_rate:.4%}, "
            f"Taker={self.config.taker_fee_rate:.4%}, "
            f"Base slippage={self.config.base_slippage_bps} bps"
        )
    
    def estimate_cost(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        symbol: str,
        atr: Optional[float] = None,
        holding_hours: float = 24.0,
        entry_is_maker: bool = True,
        exit_is_maker: bool = False
    ) -> TradeCost:
        """
        Estimate total cost for a trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price (can be SL or TP)
            size: Position size in base currency
            symbol: Trading symbol
            atr: Average True Range (for slippage estimation)
            holding_hours: Expected holding period in hours
            entry_is_maker: Whether entry is limit order (maker)
            exit_is_maker: Whether exit is limit order (maker)
        
        Returns:
            TradeCost with complete breakdown
        """
        # Calculate position notional values
        entry_notional = entry_price * size
        exit_notional = exit_price * size
        
        # Fee estimation
        entry_fee_rate = self.config.maker_fee_rate if entry_is_maker else self.config.taker_fee_rate
        exit_fee_rate = self.config.maker_fee_rate if exit_is_maker else self.config.taker_fee_rate
        
        entry_fee = entry_notional * entry_fee_rate
        exit_fee = exit_notional * exit_fee_rate
        
        # Slippage estimation
        if atr:
            atr_ratio = atr / entry_price if entry_price > 0 else 0.0
        else:
            atr_ratio = 0.01  # Default 1% if ATR not provided
        
        entry_slippage = self.estimate_slippage(entry_price, size, atr_ratio, entry_is_maker)
        exit_slippage = self.estimate_slippage(exit_price, size, atr_ratio, exit_is_maker)
        
        # Funding cost estimation
        funding_periods = holding_hours / 8.0  # Funding every 8 hours
        funding_cost = abs(entry_notional * self.config.funding_rate_per_8h * funding_periods)
        
        # Total costs
        total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage + funding_cost
        
        # Calculate percentages and R metrics
        avg_notional = (entry_notional + exit_notional) / 2
        total_cost_pct = total_cost / avg_notional if avg_notional > 0 else 0.0
        
        # Cost in R: For proper calculation, we need the actual 1R distance (entry to SL)
        # Since we don't have SL here, we'll set cost_in_R to 0 and calculate it properly in net_R_after_costs
        cost_in_R = 0.0
        
        return TradeCost(
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            funding_cost=funding_cost,
            total_cost=total_cost,
            total_cost_pct=total_cost_pct,
            cost_in_R=cost_in_R
        )
    
    def estimate_slippage(
        self,
        price: float,
        size: float,
        atr_ratio: float,
        is_maker: bool = False
    ) -> float:
        """
        Estimate slippage cost.
        
        Args:
            price: Execution price
            size: Position size in base currency
            atr_ratio: ATR / Price ratio (volatility measure)
            is_maker: Whether order is limit (maker) or market (taker)
        
        Returns:
            Estimated slippage cost in dollars
        """
        if is_maker:
            # Limit orders have minimal slippage (just price improvement uncertainty)
            slippage_bps = self.config.base_slippage_bps * 0.2  # 20% of base for makers
        else:
            # Market orders pay the spread plus volatility-based slippage
            slippage_bps = self.config.base_slippage_bps + (atr_ratio * self.config.volatility_slippage_factor)
        
        notional = price * size
        slippage_cost = notional * (slippage_bps / 10000)
        
        return slippage_cost
    
    def net_R_after_costs(
        self,
        raw_R: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        symbol: str,
        atr: Optional[float] = None,
        holding_hours: float = 24.0
    ) -> float:
        """
        Calculate net R-multiple after accounting for costs.
        
        Args:
            raw_R: Raw R-multiple before costs (e.g., 2.5 for a 2.5R win)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            size: Position size in base currency
            symbol: Trading symbol
            atr: Average True Range
            holding_hours: Expected holding period
        
        Returns:
            Net R-multiple after costs
        """
        # Determine which exit price to use based on raw_R
        if raw_R > 0:
            # Winner - use take profit
            exit_price = take_profit
        elif raw_R < 0:
            # Loser - use stop loss
            exit_price = stop_loss
        else:
            # Breakeven
            exit_price = entry_price
        
        # Calculate costs
        costs = self.estimate_cost(
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            symbol=symbol,
            atr=atr,
            holding_hours=holding_hours,
            entry_is_maker=True,   # Assume limit entry
            exit_is_maker=False    # Assume market exit
        )
        
        # Calculate cost in R (cost as fraction of 1R risk amount)
        risk_distance = abs(entry_price - stop_loss)
        one_R_dollar_amount = risk_distance * size
        cost_in_R = costs.total_cost / one_R_dollar_amount if one_R_dollar_amount > 0 else 0.0
        
        # Adjust R-multiple by subtracting cost_in_R
        net_R = raw_R - cost_in_R
        
        logger.debug(
            f"{symbol} Net R: {raw_R:.2f}R → {net_R:.2f}R "
            f"(cost: {cost_in_R:.3f}R / ${costs.total_cost:.2f})"
        )
        
        return net_R
    
    def breakeven_price(
        self,
        entry_price: float,
        size: float,
        direction: str,
        symbol: str,
        atr: Optional[float] = None,
        holding_hours: float = 24.0
    ) -> float:
        """
        Calculate breakeven price that covers all costs.
        
        Args:
            entry_price: Entry price
            size: Position size in base currency
            direction: "LONG" or "SHORT"
            symbol: Trading symbol
            atr: Average True Range
            holding_hours: Expected holding period
        
        Returns:
            Breakeven price (price needed to cover all costs)
        """
        # Estimate costs for a small move (1 ATR)
        if atr is None:
            atr = entry_price * 0.01  # Default 1%
        
        # Calculate notional
        notional = entry_price * size
        
        # Total fee rate (entry + exit, assume taker for conservative estimate)
        total_fee_rate = self.config.taker_fee_rate * 2
        
        # Slippage estimate
        atr_ratio = atr / entry_price if entry_price > 0 else 0.01
        entry_slippage = self.estimate_slippage(entry_price, size, atr_ratio, is_maker=False)
        exit_slippage = entry_slippage  # Assume similar
        total_slippage = entry_slippage + exit_slippage
        
        # Funding cost
        funding_periods = holding_hours / 8.0
        funding_cost = notional * self.config.funding_rate_per_8h * funding_periods
        
        # Total cost
        total_cost = (notional * total_fee_rate) + total_slippage + funding_cost
        
        # Breakeven distance (how much price must move to cover costs)
        breakeven_distance = total_cost / size
        
        # Calculate breakeven price
        if direction.upper() in ["LONG", "BUY"]:
            breakeven = entry_price + breakeven_distance
        else:  # SHORT/SELL
            breakeven = entry_price - breakeven_distance
        
        logger.debug(
            f"{symbol} {direction} Breakeven: ${entry_price:.2f} → ${breakeven:.2f} "
            f"(distance: ${breakeven_distance:.2f}, cost: ${total_cost:.2f})"
        )
        
        return breakeven
    
    def minimum_profit_target(
        self,
        entry_price: float,
        stop_loss: float,
        size: float,
        direction: str,
        symbol: str,
        atr: Optional[float] = None,
        target_net_R: float = 1.0
    ) -> float:
        """
        Calculate minimum profit target to achieve desired net R-multiple.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            size: Position size in base currency
            direction: "LONG" or "SHORT"
            symbol: Trading symbol
            atr: Average True Range
            target_net_R: Desired net R-multiple after costs (e.g., 2.0)
        
        Returns:
            Take profit price needed to achieve target net R
        """
        # Calculate 1R (risk amount per unit)
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Initial guess: target gross R = target net R + typical cost (0.2R)
        gross_R_estimate = target_net_R + 0.2
        
        # Calculate take profit for this R
        if direction.upper() in ["LONG", "BUY"]:
            take_profit = entry_price + (gross_R_estimate * risk_per_unit)
        else:  # SHORT/SELL
            take_profit = entry_price - (gross_R_estimate * risk_per_unit)
        
        # Calculate actual costs
        costs = self.estimate_cost(
            entry_price=entry_price,
            exit_price=take_profit,
            size=size,
            symbol=symbol,
            atr=atr,
            holding_hours=24.0,
            entry_is_maker=True,
            exit_is_maker=False
        )
        
        # Calculate cost in R
        one_R_dollar_amount = risk_per_unit * size
        cost_in_R = costs.total_cost / one_R_dollar_amount if one_R_dollar_amount > 0 else 0.0
        
        # Adjust take profit to account for costs
        adjusted_gross_R = target_net_R + cost_in_R
        
        if direction.upper() in ["LONG", "BUY"]:
            final_take_profit = entry_price + (adjusted_gross_R * risk_per_unit)
        else:
            final_take_profit = entry_price - (adjusted_gross_R * risk_per_unit)
        
        logger.debug(
            f"{symbol} {direction} Min TP for {target_net_R:.1f}R net: "
            f"${final_take_profit:.2f} (gross R: {adjusted_gross_R:.2f})"
        )
        
        return final_take_profit


# Convenience functions for quick calculations
def estimate_trade_cost(
    entry_price: float,
    exit_price: float,
    size: float,
    symbol: str,
    atr: Optional[float] = None,
    config: Optional[CostConfig] = None
) -> TradeCost:
    """
    Convenience function to estimate trade cost.
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        symbol: Trading symbol
        atr: Average True Range
        config: Cost configuration
    
    Returns:
        TradeCost breakdown
    """
    model = CostModel(config)
    return model.estimate_cost(entry_price, exit_price, size, symbol, atr)


def calculate_net_R(
    raw_R: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    size: float,
    symbol: str,
    atr: Optional[float] = None,
    config: Optional[CostConfig] = None
) -> float:
    """
    Convenience function to calculate net R-multiple.
    
    Args:
        raw_R: Raw R-multiple
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        size: Position size
        symbol: Trading symbol
        atr: Average True Range
        config: Cost configuration
    
    Returns:
        Net R-multiple after costs
    """
    model = CostModel(config)
    return model.net_R_after_costs(raw_R, entry_price, stop_loss, take_profit, size, symbol, atr)
