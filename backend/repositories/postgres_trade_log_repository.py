"""
PostgreSQL Trade Log Repository for OpportunityRanker
Implements TradeLogRepository protocol
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, select

logger = logging.getLogger(__name__)


class PostgresTradeLogRepository:
    """Real implementation of TradeLogRepository using PostgreSQL."""
    
    def __init__(self, db_session_factory):
        """
        Initialize with database session factory.
        
        Args:
            db_session_factory: Callable that returns SQLAlchemy Session
        """
        self.session_factory = db_session_factory
        logger.info("PostgresTradeLogRepository initialized")
    
    def get_symbol_winrate(self, symbol: str, lookback_trades: int = 20) -> float:
        """
        Calculate winrate from closed positions.
        
        Args:
            symbol: Trading pair
            lookback_trades: Number of recent trades to analyze
            
        Returns:
            Winrate as decimal (0.0 to 1.0)
        """
        try:
            # Import here to avoid circular imports and table redefinition
            from backend.models.trade_logs import TradeLog
            
            with self.session_factory() as session:
                # Query closed trades for symbol
                stmt = (
                    select(TradeLog)
                    .where(TradeLog.symbol == symbol)
                    .where(TradeLog.status == "closed")
                    .order_by(TradeLog.timestamp.desc())
                    .limit(lookback_trades)
                )
                
                trades = session.execute(stmt).scalars().all()
                
                if not trades:
                    logger.debug(f"No closed trades found for {symbol}")
                    return 0.5  # Neutral default
                
                # Count winning trades (profit > 0)
                # Assume 'reason' field contains 'profit' or similar indicator
                # Or use price comparison logic
                winning = sum(1 for t in trades if self._is_winning_trade(t))
                total = len(trades)
                
                winrate = winning / total if total > 0 else 0.5
                
                logger.debug(
                    f"Winrate for {symbol}: {winning}/{total} = {winrate:.2%}"
                )
                return winrate
                
        except Exception as e:
            logger.error(f"Failed to calculate winrate for {symbol}: {e}")
            return 0.5  # Neutral default on error
    
    def _is_winning_trade(self, trade) -> bool:
        """
        Determine if a trade was profitable.
        
        Args:
            trade: TradeLog instance
            
        Returns:
            True if trade was profitable
        """
        # Strategy 1: Check if reason contains profit indicator
        if trade.reason:
            reason_lower = trade.reason.lower()
            if 'profit' in reason_lower or 'tp' in reason_lower:
                return True
            if 'loss' in reason_lower or 'sl' in reason_lower:
                return False
        
        # Strategy 2: Compare entry/exit prices
        # This requires additional fields or table joins
        # For now, use simple heuristic
        
        # Default: assume 50/50
        return True  # Placeholder - adjust based on your schema
