"""
PnL Tracker - Track trade performance for adaptive optimization
"""
from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PnLTracker:
    """Track PnL history for adaptive adjustments"""
    
    def __init__(self, max_history: int = 20):
        """
        Initialize PnL tracker
        
        Args:
            max_history: Number of trades to track (default 20)
        """
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
        logger.info(f"✅ PnLTracker initialized (max_history={max_history})")
    
    def add_trade(self, pnl: float, symbol: str, leverage: float):
        """
        Add trade result to history
        
        Args:
            pnl: Trade PnL as percentage (e.g., 0.05 = +5%)
            symbol: Trading pair
            leverage: Leverage used
        """
        self.history.append({
            'pnl': pnl,
            'symbol': symbol,
            'leverage': leverage
        })
        logger.debug(f"Added trade: {symbol} {leverage}x → PnL {pnl:+.2%}")
    
    def avg_last_20(self) -> float:
        """Get average PnL of last 20 trades"""
        if not self.history:
            return 0.0
        
        avg = sum(t['pnl'] for t in self.history) / len(self.history)
        logger.debug(f"Avg PnL (last {len(self.history)} trades): {avg:+.2%}")
        return avg
    
    def avg_by_leverage(self, leverage_range: tuple) -> float:
        """
        Get average PnL for specific leverage range
        
        Args:
            leverage_range: (min, max) leverage range
            
        Returns:
            Average PnL for trades in this range
        """
        min_lev, max_lev = leverage_range
        filtered = [
            t['pnl'] for t in self.history 
            if min_lev <= t['leverage'] <= max_lev
        ]
        
        if not filtered:
            return 0.0
        
        avg = sum(filtered) / len(filtered)
        logger.debug(
            f"Avg PnL for {min_lev}-{max_lev}x leverage "
            f"({len(filtered)} trades): {avg:+.2%}"
        )
        return avg
    
    def win_rate(self) -> float:
        """Get win rate (percentage of profitable trades)"""
        if not self.history:
            return 0.0
        
        wins = sum(1 for t in self.history if t['pnl'] > 0)
        rate = wins / len(self.history)
        logger.debug(f"Win rate: {rate:.1%} ({wins}/{len(self.history)})")
        return rate
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.history:
            return {
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        pnls = [t['pnl'] for t in self.history]
        
        return {
            'avg_pnl': sum(pnls) / len(pnls),
            'win_rate': self.win_rate(),
            'total_trades': len(self.history),
            'best_trade': max(pnls),
            'worst_trade': min(pnls)
        }
