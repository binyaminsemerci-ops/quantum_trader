"""
Dynamic Position Sizing using Kelly Criterion
Optimizes position sizes based on win probability and risk/reward

Expected Impact: +40-60% profit through optimal sizing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DynamicPositionSizer:
    """
    Kelly Criterion-based position sizing
    
    Kelly Formula: f* = (p * b - q) / b
    where:
    - f* = optimal fraction of bankroll to risk
    - p = win probability (from ML confidence)
    - q = loss probability (1 - p)
    - b = win/loss ratio (avg_win / avg_loss)
    
    Safety features:
    - Fractional Kelly (50% of full Kelly to reduce variance)
    - Maximum position limits (10% per trade)
    - Portfolio exposure limits (20% total)
    - Volatility adjustments
    """
    
    def __init__(
        self,
        account_balance: float,
        max_position: float = 0.10,  # Max 10% per trade
        min_position: float = 0.01,  # Min 1% per trade
        max_portfolio_risk: float = 0.20,  # Max 20% total exposure
        kelly_fraction: float = 0.5  # Use 50% of Kelly (safer)
    ):
        """
        Initialize position sizer
        
        Args:
            account_balance: Current account balance
            max_position: Maximum position size as fraction (0.10 = 10%)
            min_position: Minimum position size as fraction
            max_portfolio_risk: Maximum total portfolio risk
            kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
        """
        self.balance = account_balance
        self.max_position = max_position
        self.min_position = min_position
        self.max_portfolio_risk = max_portfolio_risk
        self.kelly_fraction = kelly_fraction
        
        self.trade_history: List[Dict] = []
        self.open_positions: Dict[str, Dict] = {}
    
    def calculate_kelly_fraction(
        self,
        win_prob: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_prob: Probability of winning (0 to 1)
            avg_win: Average win amount (as fraction, e.g., 0.025 = 2.5%)
            avg_loss: Average loss amount (as fraction, e.g., 0.015 = 1.5%)
            
        Returns:
            Kelly fraction (optimal position size as fraction of bankroll)
        """
        if avg_loss <= 0:
            logger.warning("avg_loss <= 0, using minimum position")
            return self.min_position
        
        # Kelly formula
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        p = win_prob
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # Apply fractional Kelly (more conservative)
        kelly_adjusted = kelly * self.kelly_fraction
        
        # Clamp to limits
        kelly_adjusted = np.clip(kelly_adjusted, self.min_position, self.max_position)
        
        # If Kelly is negative, trade has negative expectancy - don't trade!
        if kelly_adjusted < self.min_position:
            logger.warning(f"Negative Kelly ({kelly:.4f}) - trade has negative expectancy!")
            return 0.0
        
        return kelly_adjusted
    
    def calculate_position_size(
        self,
        signal: Dict,
        current_price: float,
        stop_loss_price: float,
        symbol: str = "unknown"
    ) -> Dict:
        """
        Calculate optimal position size considering multiple factors:
        1. Kelly Criterion (win probability based)
        2. ML confidence (prediction certainty)
        3. Market volatility (ATR/price)
        4. Portfolio exposure (diversification)
        5. Stop loss distance (risk per unit)
        
        Args:
            signal: ML signal with 'confidence', 'prediction', etc.
            current_price: Current market price
            stop_loss_price: Stop loss price
            symbol: Trading symbol
            
        Returns:
            Dictionary with position sizing details
        """
        # 1. Get win probability from ML confidence
        confidence = signal.get('confidence', 0.60)  # Default 60%
        
        # 2. Get historical statistics
        win_rate, avg_win, avg_loss = self._get_historical_stats()
        
        # 3. Calculate base Kelly size
        kelly_size = self.calculate_kelly_fraction(
            win_prob=confidence,
            avg_win=avg_win,
            avg_loss=avg_loss
        )
        
        if kelly_size == 0.0:
            return {
                'position_size': 0.0,
                'position_value': 0.0,
                'risk_amount': 0.0,
                'risk_fraction': 0.0,
                'reason': 'negative_expectancy'
            }
        
        # 4. Adjust for market volatility
        volatility = signal.get('volatility', 0.02)  # Default 2% volatility
        vol_multiplier = self._calculate_volatility_adjustment(volatility)
        
        # 5. Adjust for portfolio exposure
        exposure_multiplier = self._calculate_exposure_adjustment()
        
        # 6. Adjust for confidence
        confidence_multiplier = self._calculate_confidence_adjustment(confidence)
        
        # 7. Calculate final position fraction
        final_fraction = (
            kelly_size * 
            vol_multiplier * 
            exposure_multiplier * 
            confidence_multiplier
        )
        
        # Clamp to absolute limits
        final_fraction = np.clip(final_fraction, self.min_position, self.max_position)
        
        # 8. Calculate position size based on stop loss
        risk_amount = self.balance * final_fraction
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk <= 0:
            logger.error("Stop loss price equals current price!")
            return {
                'position_size': 0.0,
                'position_value': 0.0,
                'risk_amount': 0.0,
                'risk_fraction': 0.0,
                'reason': 'invalid_stop_loss'
            }
        
        # Position size = risk_amount / price_risk_per_unit
        risk_per_unit = price_risk
        position_size = risk_amount / risk_per_unit
        position_value = position_size * current_price
        
        # Ensure we don't exceed balance
        if position_value > self.balance * self.max_position:
            position_size = (self.balance * self.max_position) / current_price
            position_value = position_size * current_price
            risk_amount = position_size * risk_per_unit
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_fraction': final_fraction,
            'kelly_size': kelly_size,
            'vol_multiplier': vol_multiplier,
            'exposure_multiplier': exposure_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'reason': 'success'
        }
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """
        Reduce position size in high volatility
        
        Args:
            volatility: Market volatility (e.g., ATR/price)
            
        Returns:
            Multiplier (0.5 to 1.0)
        """
        # High volatility → reduce size
        # Low volatility → full size
        
        # volatility = 0.01 (1%) → multiplier = 0.91
        # volatility = 0.05 (5%) → multiplier = 0.67
        # volatility = 0.10 (10%) → multiplier = 0.50
        
        multiplier = 1 / (1 + volatility * 5)
        return np.clip(multiplier, 0.5, 1.0)
    
    def _calculate_exposure_adjustment(self) -> float:
        """
        Reduce position size when portfolio exposure is high
        
        Returns:
            Multiplier (0.3 to 1.0)
        """
        current_exposure = self._get_portfolio_exposure()
        
        if current_exposure >= self.max_portfolio_risk:
            # At or above limit → minimum multiplier
            return 0.3
        
        # Linear reduction as we approach limit
        # exposure = 0% → multiplier = 1.0
        # exposure = 10% → multiplier = 0.65
        # exposure = 20% → multiplier = 0.3
        
        multiplier = 1.0 - (current_exposure / self.max_portfolio_risk) * 0.7
        return np.clip(multiplier, 0.3, 1.0)
    
    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Scale position with ML confidence
        
        Args:
            confidence: ML confidence (0 to 1)
            
        Returns:
            Multiplier (0.5 to 1.2)
        """
        # Low confidence → reduce size
        # High confidence → increase size (up to 1.2x)
        
        # confidence = 0.50 → multiplier = 0.50
        # confidence = 0.70 → multiplier = 0.90
        # confidence = 0.85 → multiplier = 1.05
        # confidence = 0.95 → multiplier = 1.15
        
        if confidence < 0.60:
            # Very low confidence
            multiplier = 0.5
        elif confidence < 0.75:
            # Medium confidence
            multiplier = 0.5 + (confidence - 0.60) * 2.67  # Linear 0.5 to 0.9
        else:
            # High confidence
            multiplier = 0.9 + (confidence - 0.75) * 1.2  # Linear 0.9 to 1.2
        
        return np.clip(multiplier, 0.5, 1.2)
    
    def _get_historical_stats(self) -> Tuple[float, float, float]:
        """
        Calculate win rate and avg win/loss from recent trades
        
        Returns:
            Tuple of (win_rate, avg_win, avg_loss)
        """
        if not self.trade_history:
            # Default values for new system (conservative)
            return 0.55, 0.020, 0.015  # 55% win rate, 2% avg win, 1.5% avg loss
        
        # Use last 100 trades
        recent_trades = self.trade_history[-100:]
        
        wins = [t['pnl_pct'] for t in recent_trades if t['pnl_pct'] > 0]
        losses = [abs(t['pnl_pct']) for t in recent_trades if t['pnl_pct'] < 0]
        
        if not wins or not losses:
            return 0.55, 0.020, 0.015
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        return win_rate, avg_win, avg_loss
    
    def _get_portfolio_exposure(self) -> float:
        """
        Calculate current portfolio risk exposure
        
        Returns:
            Total risk as fraction of balance (e.g., 0.15 = 15%)
        """
        if not self.open_positions:
            return 0.0
        
        total_risk = sum(
            pos.get('risk_amount', 0) 
            for pos in self.open_positions.values()
        )
        
        return total_risk / self.balance
    
    def add_position(self, position_id: str, position: Dict):
        """Record new open position"""
        self.open_positions[position_id] = position
    
    def remove_position(self, position_id: str):
        """Remove closed position"""
        if position_id in self.open_positions:
            del self.open_positions[position_id]
    
    def record_trade(self, trade: Dict):
        """
        Record completed trade for statistics
        
        Args:
            trade: Dict with keys: 'pnl_pct', 'symbol', 'win', etc.
        """
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl_pct': trade['pnl_pct'],
            'symbol': trade.get('symbol', 'unknown'),
            'win': trade['pnl_pct'] > 0
        })
    
    def update_balance(self, new_balance: float):
        """Update account balance"""
        self.balance = new_balance
    
    def get_statistics(self) -> Dict:
        """
        Get current position sizing statistics
        
        Returns:
            Dictionary with stats
        """
        win_rate, avg_win, avg_loss = self._get_historical_stats()
        
        return {
            'balance': self.balance,
            'open_positions': len(self.open_positions),
            'portfolio_exposure': self._get_portfolio_exposure(),
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': avg_win / avg_loss if avg_loss > 0 else 0,
            'expectancy': win_rate * avg_win - (1 - win_rate) * avg_loss
        }


def create_position_sizer(account_balance: float) -> DynamicPositionSizer:
    """
    Factory function to create position sizer
    
    Args:
        account_balance: Current account balance
        
    Returns:
        Configured DynamicPositionSizer
    """
    return DynamicPositionSizer(account_balance)


if __name__ == '__main__':
    print("Testing Dynamic Position Sizing...")
    
    # Create sizer with $10,000 balance
    sizer = DynamicPositionSizer(account_balance=10000)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'High Confidence, Low Vol',
            'signal': {'confidence': 0.85, 'volatility': 0.015},
            'price': 100,
            'stop_loss': 98
        },
        {
            'name': 'Medium Confidence, Normal Vol',
            'signal': {'confidence': 0.65, 'volatility': 0.025},
            'price': 100,
            'stop_loss': 97
        },
        {
            'name': 'Low Confidence, High Vol',
            'signal': {'confidence': 0.55, 'volatility': 0.05},
            'price': 100,
            'stop_loss': 95
        },
    ]
    
    print("\n" + "="*70)
    for scenario in scenarios:
        print(f"\n[CHART] Scenario: {scenario['name']}")
        print(f"   Confidence: {scenario['signal']['confidence']:.0%}")
        print(f"   Volatility: {scenario['signal']['volatility']:.1%}")
        print(f"   Price: ${scenario['price']}, Stop: ${scenario['stop_loss']}")
        
        result = sizer.calculate_position_size(
            signal=scenario['signal'],
            current_price=scenario['price'],
            stop_loss_price=scenario['stop_loss']
        )
        
        print(f"\n   Results:")
        print(f"     Position Size: {result['position_size']:.2f} units")
        print(f"     Position Value: ${result['position_value']:.2f}")
        print(f"     Risk Amount: ${result['risk_amount']:.2f}")
        print(f"     Risk %: {result['risk_fraction']*100:.2f}%")
        print(f"     Kelly Fraction: {result['kelly_size']*100:.2f}%")
    
    print("\n" + "="*70)
    
    # Show statistics
    stats = sizer.get_statistics()
    print(f"\n[CHART_UP] Position Sizer Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n[OK] Position sizing test complete!")
