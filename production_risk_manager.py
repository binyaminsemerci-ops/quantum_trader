#!/usr/bin/env python3
"""
Production Risk Management System

This module implements comprehensive risk management features including:
- Position sizing based on account equity
- Stop-loss and take-profit mechanisms
- Maximum drawdown protection
- Portfolio-level risk controls
- Real-time position monitoring
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from config.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters for trading."""

    max_position_size_pct: float = 2.0  # Max 2% of equity per position
    stop_loss_pct: float = 2.0  # 2% stop loss
    take_profit_pct: float = 4.0  # 4% take profit (2:1 ratio)
    max_portfolio_risk_pct: float = 10.0  # Max 10% portfolio risk
    max_daily_loss_pct: float = 5.0  # Max 5% daily loss
    max_drawdown_pct: float = 15.0  # Max 15% total drawdown
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1 risk/reward
    max_correlation_exposure: float = 0.3  # Max 30% in correlated assets


@dataclass
class Position:
    """Individual position tracking."""

    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    risk_amount: float = 0.0


@dataclass
class PortfolioState:
    """Current portfolio state for risk monitoring."""

    total_equity: float
    available_equity: float
    positions: List[Position]
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    daily_trades: int = 0
    risk_utilization: float = 0.0


class RiskManager:
    """Production-ready risk management system."""

    def __init__(self, parameters: Optional[RiskParameters] = None):
        self.params = parameters or RiskParameters()
        self.portfolio = PortfolioState(
            total_equity=settings.starting_equity,
            available_equity=settings.starting_equity,
            positions=[],
        )
        self.equity_peak = settings.starting_equity
        self.daily_start_equity = settings.starting_equity
        self.log_file = Path("logs/risk_management.log")
        self.log_file.parent.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup risk management logging."""
        risk_logger = logging.getLogger("risk_manager")
        if not risk_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            risk_logger.addHandler(handler)
            risk_logger.setLevel(logging.INFO)

    def calculate_position_size(
        self, entry_price: float, stop_loss_price: float, signal_confidence: float = 1.0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            signal_confidence: AI model confidence (0.0 to 1.0)

        Returns:
            Tuple of (position_size, risk_info)
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share == 0:
            return 0.0, {"error": "No risk defined - entry equals stop loss"}

        # Calculate maximum risk amount
        max_risk_amount = self.portfolio.available_equity * (
            self.params.max_position_size_pct / 100
        )

        # Adjust for signal confidence
        adjusted_risk_amount = max_risk_amount * signal_confidence

        # Calculate position size
        position_size = adjusted_risk_amount / risk_per_share

        # Apply additional constraints
        max_equity_position = (
            self.portfolio.available_equity
            * (self.params.max_position_size_pct / 100)
            / entry_price
        )
        position_size = min(position_size, max_equity_position)

        # Check portfolio risk limits
        total_position_value = position_size * entry_price
        portfolio_risk_pct = (total_position_value / self.portfolio.total_equity) * 100

        if portfolio_risk_pct > self.params.max_portfolio_risk_pct:
            position_size *= self.params.max_portfolio_risk_pct / portfolio_risk_pct

        risk_info = {
            "position_size": position_size,
            "position_value": position_size * entry_price,
            "risk_amount": adjusted_risk_amount,
            "risk_per_share": risk_per_share,
            "portfolio_risk_pct": (
                position_size * entry_price / self.portfolio.total_equity
            )
            * 100,
            "signal_confidence": signal_confidence,
        }

        return position_size, risk_info

    def calculate_stop_loss_take_profit(
        self, entry_price: float, side: str, atr: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.

        Args:
            entry_price: Entry price
            side: 'long' or 'short'
            atr: Average True Range for dynamic stops (optional)

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if atr and atr > 0:
            # Dynamic stops based on volatility (ATR)
            stop_distance = atr * 2  # 2x ATR for stop loss
            profit_distance = stop_distance * self.params.min_risk_reward_ratio
        else:
            # Fixed percentage stops
            stop_distance = entry_price * (self.params.stop_loss_pct / 100)
            profit_distance = entry_price * (self.params.take_profit_pct / 100)

        if side == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # short
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance

        return stop_loss, take_profit

    def validate_trade_signal(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        signal_strength: float = 1.0,
        market_data: Optional[Dict] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if a trade signal should be executed based on risk rules.

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "signal_strength": signal_strength,
            "checks": {},
        }

        # Check 1: Daily loss limit
        daily_loss_pct = (self.portfolio.daily_pnl / self.daily_start_equity) * 100
        if daily_loss_pct < -self.params.max_daily_loss_pct:
            validation_info["checks"]["daily_loss"] = {
                "passed": False,
                "current": daily_loss_pct,
                "limit": -self.params.max_daily_loss_pct,
            }
            return False, validation_info
        validation_info["checks"]["daily_loss"] = {"passed": True}

        # Check 2: Maximum drawdown
        current_drawdown = (
            (self.equity_peak - self.portfolio.total_equity) / self.equity_peak
        ) * 100
        if current_drawdown > self.params.max_drawdown_pct:
            validation_info["checks"]["max_drawdown"] = {
                "passed": False,
                "current": current_drawdown,
                "limit": self.params.max_drawdown_pct,
            }
            return False, validation_info
        validation_info["checks"]["max_drawdown"] = {"passed": True}

        # Check 3: Portfolio risk utilization
        if self.portfolio.risk_utilization > self.params.max_portfolio_risk_pct:
            validation_info["checks"]["portfolio_risk"] = {
                "passed": False,
                "current": self.portfolio.risk_utilization,
                "limit": self.params.max_portfolio_risk_pct,
            }
            return False, validation_info
        validation_info["checks"]["portfolio_risk"] = {"passed": True}

        # Check 4: Position limits per symbol
        existing_positions = [p for p in self.portfolio.positions if p.symbol == symbol]
        if len(existing_positions) >= 1:  # Max 1 position per symbol
            validation_info["checks"]["position_limit"] = {
                "passed": False,
                "existing_positions": len(existing_positions),
            }
            return False, validation_info
        validation_info["checks"]["position_limit"] = {"passed": True}

        # Check 5: Minimum signal strength
        min_signal_strength = 0.6  # Require at least 60% confidence
        if signal_strength < min_signal_strength:
            validation_info["checks"]["signal_strength"] = {
                "passed": False,
                "current": signal_strength,
                "minimum": min_signal_strength,
            }
            return False, validation_info
        validation_info["checks"]["signal_strength"] = {"passed": True}

        # All checks passed
        logger.info(f"Trade signal validated for {symbol} {side} @ {entry_price}")
        return True, validation_info

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Add a new position to the portfolio."""
        try:
            # Calculate stop loss and take profit if not provided
            if stop_loss is None or take_profit is None:
                calc_stop, calc_tp = self.calculate_stop_loss_take_profit(
                    entry_price, side
                )
                stop_loss = stop_loss or calc_stop
                take_profit = take_profit or calc_tp

            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss) * quantity

            position = Position(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=datetime.now(timezone.utc),
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_amount=risk_amount,
            )

            self.portfolio.positions.append(position)

            # Update available equity
            position_value = entry_price * quantity
            self.portfolio.available_equity -= position_value

            # Log the position
            logger.info(
                f"Added position: {symbol} {side} {quantity}@{entry_price} SL:{stop_loss} TP:{take_profit}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False

    def update_portfolio(self, current_prices: Dict[str, float]):
        """Update portfolio with current market prices."""
        total_unrealized_pnl = 0.0

        for position in self.portfolio.positions:
            if position.symbol in current_prices:
                current_price = current_prices[position.symbol]

                if position.side == "long":
                    unrealized_pnl = (
                        current_price - position.entry_price
                    ) * position.quantity
                else:  # short
                    unrealized_pnl = (
                        position.entry_price - current_price
                    ) * position.quantity

                position.unrealized_pnl = unrealized_pnl
                total_unrealized_pnl += unrealized_pnl

        # Update portfolio equity
        self.portfolio.total_equity = (
            self.portfolio.available_equity + total_unrealized_pnl
        )

        # Update drawdown tracking
        if self.portfolio.total_equity > self.equity_peak:
            self.equity_peak = self.portfolio.total_equity

        current_drawdown = (
            (self.equity_peak - self.portfolio.total_equity) / self.equity_peak
        ) * 100
        self.portfolio.max_drawdown = max(self.portfolio.max_drawdown, current_drawdown)

    def check_exit_conditions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Check if any positions should be closed based on stop loss/take profit."""
        exit_signals = []

        for position in self.portfolio.positions:
            if position.symbol not in current_prices:
                continue

            current_price = current_prices[position.symbol]
            should_exit = False
            exit_reason = ""

            if position.side == "long":
                if current_price <= position.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price >= position.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
            else:  # short
                if current_price >= position.stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif current_price <= position.take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

            if should_exit:
                exit_signals.append(
                    {
                        "position": position,
                        "exit_price": current_price,
                        "exit_reason": exit_reason,
                        "pnl": position.unrealized_pnl,
                    }
                )

        return exit_signals

    def close_position(self, symbol: str, exit_price: float, reason: str = "manual"):
        """Close a position and update portfolio."""
        position_to_close = None
        for i, position in enumerate(self.portfolio.positions):
            if position.symbol == symbol:
                position_to_close = position
                break

        if not position_to_close:
            logger.warning(f"No position found for {symbol}")
            return False

        # Calculate final PnL
        if position_to_close.side == "long":
            pnl = (
                exit_price - position_to_close.entry_price
            ) * position_to_close.quantity
        else:
            pnl = (
                position_to_close.entry_price - exit_price
            ) * position_to_close.quantity

        # Update portfolio
        self.portfolio.available_equity += (
            position_to_close.entry_price * position_to_close.quantity
        ) + pnl
        self.portfolio.daily_pnl += pnl

        # Remove position
        self.portfolio.positions.remove(position_to_close)

        logger.info(
            f"Closed position: {symbol} @ {exit_price} PnL: {pnl:.2f} Reason: {reason}"
        )
        return True

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        return {
            "total_equity": self.portfolio.total_equity,
            "available_equity": self.portfolio.available_equity,
            "positions_count": len(self.portfolio.positions),
            "daily_pnl": self.portfolio.daily_pnl,
            "daily_pnl_pct": (self.portfolio.daily_pnl / self.daily_start_equity) * 100,
            "max_drawdown": self.portfolio.max_drawdown,
            "positions": [asdict(p) for p in self.portfolio.positions],
            "risk_parameters": asdict(self.params),
        }

    def save_state(self, filepath: Optional[str] = None):
        """Save current portfolio state to file."""
        if filepath is None:
            filepath = (
                f"logs/portfolio_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        state = self.get_portfolio_summary()
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Portfolio state saved to {filepath}")


def test_risk_manager():
    """Test the risk management system."""
    print("🧪 Testing Risk Management System")

    # Initialize risk manager
    risk_params = RiskParameters(
        max_position_size_pct=1.0,  # Conservative for testing
        stop_loss_pct=2.0,
        take_profit_pct=4.0,
    )

    rm = RiskManager(risk_params)

    # Test position sizing
    print("\n📊 Testing Position Sizing:")
    entry_price = 50000.0
    stop_loss = 49000.0

    position_size, risk_info = rm.calculate_position_size(
        entry_price, stop_loss, signal_confidence=0.8
    )
    print(f"Position size: {position_size:.6f}")
    print(f"Risk info: {risk_info}")

    # Test trade validation
    print("\n✅ Testing Trade Validation:")
    is_valid, validation_info = rm.validate_trade_signal(
        "BTCUSDT", "long", entry_price, signal_strength=0.8
    )
    print(f"Trade valid: {is_valid}")
    print(f"Validation: {validation_info['checks']}")

    # Test adding position
    print("\n📈 Testing Position Management:")
    success = rm.add_position("BTCUSDT", "long", entry_price, position_size)
    print(f"Position added: {success}")

    # Test portfolio update
    print("\n💰 Testing Portfolio Updates:")
    current_prices = {"BTCUSDT": 51000.0}  # Price moved up
    rm.update_portfolio(current_prices)

    summary = rm.get_portfolio_summary()
    print(f"Portfolio equity: ${summary['total_equity']:.2f}")
    print(f"Daily PnL: ${summary['daily_pnl']:.2f}")

    print("\n✅ Risk Management System Test Complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production Risk Management System")
    parser.add_argument("--test", action="store_true", help="Run risk management tests")

    args = parser.parse_args()

    if args.test:
        test_risk_manager()
    else:
        print("Risk Management System - Use --test to run tests")
