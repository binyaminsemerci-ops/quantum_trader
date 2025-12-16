"""
Exchange Simulator - Simulates order execution with realistic conditions
"""

import logging
import numpy as np
from datetime import datetime
from typing import Protocol

from .scenario_models import Scenario, ScenarioType, ExecutionResult

logger = logging.getLogger(__name__)


class TradeDecision(Protocol):
    """Interface for trade decisions from Orchestrator"""
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    stop_loss: float | None
    take_profit: float | None


class ExchangeSimulator:
    """
    Simulates realistic order execution with:
    - Slippage based on volatility and liquidity
    - Spread costs
    - Latency
    - Execution failures
    - Partial fills
    
    Adapts behavior based on scenario stress conditions.
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.001,  # 0.1% base slippage
        base_latency_ms: float = 50.0,
        failure_rate: float = 0.001  # 0.1% base failure rate
    ):
        """
        Initialize exchange simulator.
        
        Args:
            base_slippage_pct: Normal market slippage
            base_latency_ms: Normal execution latency
            failure_rate: Base probability of execution failure
        """
        self.base_slippage_pct = base_slippage_pct
        self.base_latency_ms = base_latency_ms
        self.base_failure_rate = failure_rate
        logger.info("[SST] ExchangeSimulator initialized")
    
    def execute_order(
        self,
        decision: TradeDecision,
        current_bar: dict,
        scenario: Scenario
    ) -> ExecutionResult:
        """
        Simulate order execution.
        
        Args:
            decision: Trade decision to execute
            current_bar: Current market bar (OHLCV + stress metadata)
            scenario: Current scenario context
            
        Returns:
            ExecutionResult with fill details or failure
        """
        # Check for stress conditions
        is_stressed = current_bar.get("stressed", False)
        stress_type = current_bar.get("stress_type", "")
        
        # Calculate failure probability
        failure_prob = self._calculate_failure_rate(
            is_stressed, stress_type, scenario
        )
        
        # Simulate execution failure
        if np.random.random() < failure_prob:
            return ExecutionResult(
                success=False,
                filled_price=0.0,
                filled_qty=0.0,
                slippage_pct=0.0,
                latency_ms=self.base_latency_ms,
                error_reason=self._get_failure_reason(stress_type),
                timestamp=datetime.utcnow()
            )
        
        # Calculate slippage
        slippage_pct = self._calculate_slippage(
            decision, current_bar, is_stressed, stress_type
        )
        
        # Calculate spread cost
        spread_pct = self._calculate_spread(current_bar)
        
        # Determine fill price
        market_price = current_bar["close"]
        
        if decision.side == "BUY":
            # Pay spread + slippage on buy
            filled_price = market_price * (1 + spread_pct + slippage_pct)
        else:  # SELL
            # Lose spread + slippage on sell
            filled_price = market_price * (1 - spread_pct - slippage_pct)
        
        # Calculate latency
        latency_ms = self._calculate_latency(is_stressed, stress_type)
        
        # Determine filled quantity (partial fills possible)
        filled_qty = self._calculate_fill_quantity(
            decision.quantity, current_bar, is_stressed
        )
        
        return ExecutionResult(
            success=True,
            filled_price=filled_price,
            filled_qty=filled_qty,
            slippage_pct=slippage_pct + spread_pct,
            latency_ms=latency_ms,
            error_reason=None,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_failure_rate(
        self,
        is_stressed: bool,
        stress_type: str,
        scenario: Scenario
    ) -> float:
        """Calculate probability of execution failure"""
        
        if scenario.type == ScenarioType.EXECUTION_FAILURE:
            # Scenario explicitly tests execution failures
            return scenario.parameters.get("failure_rate", 0.20)
        
        if not is_stressed:
            return self.base_failure_rate
        
        # Increase failure rate based on stress type
        multipliers = {
            "flash_crash": 10.0,
            "liquidity_drop": 15.0,
            "data_corruption": 20.0,
            "spread_explosion": 5.0,
            "volatility_spike": 3.0
        }
        
        multiplier = multipliers.get(stress_type, 5.0)
        return min(self.base_failure_rate * multiplier, 0.30)
    
    def _calculate_slippage(
        self,
        decision: TradeDecision,
        current_bar: dict,
        is_stressed: bool,
        stress_type: str
    ) -> float:
        """Calculate slippage percentage"""
        
        # Base slippage
        slippage = self.base_slippage_pct
        
        # Increase with low liquidity (volume)
        volume = current_bar.get("volume", 1000.0)
        if volume < 100:
            slippage *= 10.0
        elif volume < 500:
            slippage *= 3.0
        
        # Increase with volatility
        high = current_bar.get("high", 0)
        low = current_bar.get("low", 0)
        close = current_bar.get("close", 1)
        
        if close > 0:
            bar_range_pct = (high - low) / close
            slippage *= (1 + bar_range_pct * 10)
        
        # Stress multipliers
        if is_stressed:
            stress_multipliers = {
                "flash_crash": 20.0,
                "flash_crash_recovery": 10.0,
                "volatility_spike": 5.0,
                "liquidity_drop": 15.0,
                "spread_explosion": 8.0,
                "pump": 4.0,
                "dump": 6.0
            }
            multiplier = stress_multipliers.get(stress_type, 3.0)
            slippage *= multiplier
        
        # Cap slippage at reasonable maximum
        return min(slippage, 0.10)  # Max 10%
    
    def _calculate_spread(self, current_bar: dict) -> float:
        """Calculate bid-ask spread percentage"""
        
        # Base spread
        base_spread = 0.0005  # 0.05% (5 bps)
        
        # Check for spread explosion stress
        spread_multiplier = current_bar.get("spread_multiplier", 1.0)
        
        # Increase spread with low liquidity
        volume = current_bar.get("volume", 1000.0)
        if volume < 100:
            volume_mult = 5.0
        elif volume < 500:
            volume_mult = 2.0
        else:
            volume_mult = 1.0
        
        total_spread = base_spread * spread_multiplier * volume_mult
        
        # Cap spread
        return min(total_spread, 0.05)  # Max 5%
    
    def _calculate_latency(self, is_stressed: bool, stress_type: str) -> float:
        """Calculate execution latency in milliseconds"""
        
        latency = self.base_latency_ms
        
        if is_stressed:
            stress_latency = {
                "flash_crash": 500.0,
                "liquidity_drop": 300.0,
                "volatility_spike": 200.0,
                "data_corruption": 1000.0
            }
            latency = stress_latency.get(stress_type, latency * 3)
        
        # Add random jitter
        jitter = np.random.uniform(0.8, 1.2)
        return latency * jitter
    
    def _calculate_fill_quantity(
        self,
        requested_qty: float,
        current_bar: dict,
        is_stressed: bool
    ) -> float:
        """Calculate filled quantity (may be partial)"""
        
        # Check liquidity
        volume = current_bar.get("volume", 1000.0)
        
        # Under stress or low liquidity, partial fills possible
        if is_stressed or volume < 200:
            fill_pct = np.random.uniform(0.70, 1.0)
            return requested_qty * fill_pct
        
        # Normal conditions: full fill
        return requested_qty
    
    def _get_failure_reason(self, stress_type: str) -> str:
        """Get human-readable failure reason"""
        
        reasons = {
            "flash_crash": "Exchange circuit breaker triggered",
            "liquidity_drop": "Insufficient liquidity",
            "data_corruption": "Market data unavailable",
            "spread_explosion": "Excessive spread - order rejected",
            "volatility_spike": "Volatility halt",
            "": "Network timeout"
        }
        
        return reasons.get(stress_type, "Unknown execution error")
