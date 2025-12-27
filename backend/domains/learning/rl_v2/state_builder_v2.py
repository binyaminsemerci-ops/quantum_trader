"""
State Builder v2 - Advanced State Representation System
========================================================

Builds sophisticated state representations for:
- Meta Strategy RL (regime-aware, market pressure)
- Position Sizing RL (portfolio exposure, equity curve)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, List, Optional
import structlog

from backend.utils.regime_detector_v2 import RegimeDetectorV2
from backend.utils.volatility_tools_v2 import VolatilityToolsV2
from backend.utils.winrate_tracker_v2 import WinRateTrackerV2
from backend.utils.equity_curve_tools_v2 import EquityCurveToolsV2

logger = structlog.get_logger(__name__)


class StateBuilderV2:
    """
    Advanced state builder for RL v2.
    
    Builds states with:
    - Trailing win rates
    - Volatility metrics
    - Equity curve slopes
    - Market pressure
    - Regime labels
    - Account health
    """
    
    def __init__(self):
        """Initialize State Builder v2."""
        self.regime_detector = RegimeDetectorV2()
        self.volatility_tools = VolatilityToolsV2()
        self.winrate_tracker = WinRateTrackerV2()
        self.equity_curve_tools = EquityCurveToolsV2()
        
        logger.info("[State Builder v2] Initialized")
    
    def build_meta_strategy_state(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build Meta Strategy state representation v2.
        
        State includes:
        - regime: Market regime label
        - volatility: Market volatility
        - market_pressure: Buy/sell pressure
        - confidence: Signal confidence
        - previous_winrate: Trailing win rate
        - account_health: Account health score
        
        Args:
            data: Dictionary containing:
                - regime: Current market regime (optional, will detect if not provided)
                - confidence: Signal confidence
                - market_price: Current market price
                - account_balance: Current account balance
                - price_history: Recent price history (optional)
                - volume_history: Recent volume history (optional)
                - trace_id: Trace ID for logging (optional)
            
        Returns:
            Meta strategy state dictionary
        """
        # Extract parameters from data
        regime = data.get("regime")
        confidence = data.get("confidence", 0.5)
        market_price = data.get("market_price", 0.0)
        account_balance = data.get("account_balance", 10000.0)
        price_history = data.get("price_history")
        volume_history = data.get("volume_history")
        trace_id = data.get("trace_id", "")
        
        # Detect regime if not provided
        if not regime and price_history:
            regime = self.regime_detector.detect_regime(price_history, volume_history)
        elif not regime:
            regime = "UNKNOWN"
        
        # Calculate volatility
        if price_history and len(price_history) > 3:
            volatility = self.volatility_tools.calculate_volatility(price_history)
        else:
            volatility = 0.02  # Default
        
        # Calculate market pressure
        if price_history and len(price_history) > 5:
            market_pressure = self.volatility_tools.calculate_market_pressure(price_history)
        else:
            market_pressure = 0.0
        
        # Get trailing winrate
        trailing_winrate = self.winrate_tracker.get_trailing_winrate()
        
        # Calculate account health
        account_health = self.equity_curve_tools.calculate_account_health(account_balance)
        
        state = {
            "regime": regime,
            "volatility": volatility,
            "market_pressure": market_pressure,
            "confidence": confidence,
            "previous_winrate": trailing_winrate,
            "account_health": account_health
        }
        
        logger.debug(
            "[State Builder v2] Meta strategy state built",
            trace_id=trace_id,
            state=state
        )
        
        return state
    
    def build_position_sizing_state(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build Position Sizing state representation v2.
        
        State includes:
        - signal_confidence: Signal confidence
        - portfolio_exposure: Current portfolio exposure
        - recent_winrate: Trailing win rate
        - volatility: Market volatility
        - equity_curve_slope: Equity curve slope
        
        Args:
            data: Dictionary containing:
                - confidence: Signal confidence (alias: signal_confidence)
                - portfolio_exposure: Portfolio exposure (0-1)
                - volatility: Market volatility (alias: market_volatility)
                - account_balance: Current account balance
                - equity_history: Equity history (optional)
                - trace_id: Trace ID for logging (optional)
            
        Returns:
            Position sizing state dictionary
        """
        # Extract parameters from data
        signal_confidence = data.get("confidence", data.get("signal_confidence", 0.5))
        portfolio_exposure = data.get("portfolio_exposure", 0.0)
        market_volatility = data.get("volatility", data.get("market_volatility", 0.02))
        account_balance = data.get("account_balance", 10000.0)
        equity_history = data.get("equity_history")
        recent_trades = data.get("recent_trades")
        trace_id = data.get("trace_id", "")
        
        # Record equity history if provided
        if equity_history:
            for equity in equity_history:
                self.equity_curve_tools.record_equity_point(equity)
        
        # Record recent trades if provided
        if recent_trades:
            for trade in recent_trades:
                result = trade.get("result", "unknown")
                self.winrate_tracker.record_trade_outcome(result == "win")
        
        # Get trailing winrate
        trailing_winrate = self.winrate_tracker.get_trailing_winrate()
        
        # Calculate equity curve slope
        equity_slope = self.equity_curve_tools.calculate_equity_curve_slope()
        
        state = {
            "signal_confidence": signal_confidence,
            "portfolio_exposure": portfolio_exposure,
            "recent_winrate": trailing_winrate,
            "volatility": market_volatility,
            "equity_curve_slope": equity_slope
        }
        
        logger.debug(
            "[State Builder v2] Position sizing state built",
            trace_id=trace_id,
            state=state
        )
        
        return state
    
    def record_trade_outcome(self, is_win: bool, pnl_pct: float):
        """
        Record trade outcome for win rate calculation.
        
        Args:
            is_win: True if trade was profitable
            pnl_pct: P&L percentage
        """
        self.winrate_tracker.record_trade_outcome(is_win)
    
    def record_equity_point(self, equity: float):
        """
        Record equity point for equity curve tracking.
        
        Args:
            equity: Current equity/balance
        """
        self.equity_curve_tools.record_equity_point(equity)
    
    def reset(self):
        """Reset all tracking buffers."""
        self.winrate_tracker.reset()
        self.equity_curve_tools.reset()
        logger.info("[State Builder v2] Reset complete")
