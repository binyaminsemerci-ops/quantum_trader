"""
Exchange Simulator - Simulates order execution during replay
"""

import logging
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of simulated order execution"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    filled_price: float
    size: float
    fee: float
    slippage: float
    executed: bool
    reason: Optional[str] = None
    
    @property
    def total_cost(self) -> float:
        """Total cost including fees and slippage"""
        return self.fee + self.slippage


class ExchangeSimulator:
    """
    Simulates realistic order execution for replay sessions.
    
    Models slippage, commissions, partial fills, and execution failures.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1%
        slippage_model: str = "realistic",
        base_slippage_bps: float = 2.0,  # 2 basis points
        max_slippage_bps: float = 50.0,  # 50 basis points
        failure_rate: float = 0.001,  # 0.1% of orders fail
    ):
        """
        Args:
            commission_rate: Commission as fraction (0.001 = 0.1%)
            slippage_model: "none", "realistic", "pessimistic"
            base_slippage_bps: Base slippage in basis points
            max_slippage_bps: Maximum slippage in basis points
            failure_rate: Probability of execution failure
        """
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.base_slippage_bps = base_slippage_bps
        self.max_slippage_bps = max_slippage_bps
        self.failure_rate = failure_rate
        self.logger = logging.getLogger(f"{__name__}.ExchangeSimulator")
    
    def execute(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        volume: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Simulate execution at given price.
        
        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            size: Position size (USD or contracts)
            price: Market price
            volume: Market volume (for slippage calculation)
        
        Returns:
            ExecutionResult with execution details
        """
        # Random execution failure
        if random.random() < self.failure_rate:
            self.logger.warning(f"Execution failed for {symbol} {side}")
            return ExecutionResult(
                symbol=symbol,
                side=side,
                filled_price=0.0,
                size=0.0,
                fee=0.0,
                slippage=0.0,
                executed=False,
                reason="RANDOM_FAILURE"
            )
        
        # Calculate slippage
        slippage_bps = self._calculate_slippage(size, price, volume)
        slippage_pct = slippage_bps / 10000
        
        # Filled price (worse for buyer, better for seller)
        if side == "LONG":
            filled_price = price * (1 + slippage_pct)
        else:  # SHORT
            filled_price = price * (1 - slippage_pct)
        
        # Calculate costs
        notional_value = size * filled_price
        commission = notional_value * self.commission_rate
        slippage_cost = abs(filled_price - price) * size
        
        self.logger.debug(
            f"Executed {symbol} {side}: "
            f"size={size:.4f}, price={price:.4f}, "
            f"filled={filled_price:.4f}, slip={slippage_bps:.2f}bps, "
            f"fee={commission:.4f}"
        )
        
        return ExecutionResult(
            symbol=symbol,
            side=side,
            filled_price=filled_price,
            size=size,
            fee=commission,
            slippage=slippage_cost,
            executed=True,
            reason="FILLED"
        )
    
    def _calculate_slippage(
        self,
        size: float,
        price: float,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate slippage in basis points.
        
        Args:
            size: Order size
            price: Market price
            volume: Market volume
        
        Returns:
            Slippage in basis points
        """
        if self.slippage_model == "none":
            return 0.0
        
        # Base slippage
        slippage = self.base_slippage_bps
        
        # Add volume-based slippage
        if volume is not None and volume > 0:
            notional = size * price
            volume_impact = (notional / volume) * 100  # Percentage of volume
            
            # Higher impact = more slippage (square root to dampen)
            slippage += min(
                volume_impact ** 0.5 * 5,  # Scale factor
                self.max_slippage_bps - self.base_slippage_bps
            )
        
        # Pessimistic model adds extra
        if self.slippage_model == "pessimistic":
            slippage *= 2.0
        
        # Random variation (±20%)
        slippage *= random.uniform(0.8, 1.2)
        
        return min(slippage, self.max_slippage_bps)
    
    def can_execute(
        self,
        size: float,
        available_balance: float,
        price: float
    ) -> tuple[bool, str]:
        """
        Check if order can be executed given available balance.
        
        Args:
            size: Order size
            available_balance: Available cash
            price: Market price
        
        Returns:
            (can_execute, reason)
        """
        required = size * price
        required_with_costs = required * (1 + self.commission_rate + 0.01)  # Add buffer
        
        if required_with_costs > available_balance:
            return False, f"Insufficient balance: need {required_with_costs:.2f}, have {available_balance:.2f}"
        
        return True, "OK"
