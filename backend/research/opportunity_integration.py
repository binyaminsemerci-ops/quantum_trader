"""
OpportunityRanker integration for Strategy Generator AI.

This module connects the OpportunityRanker to the Strategy Generator,
filtering symbols based on opportunity scores.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpportunityFilteredSymbols:
    """
    Provides top-ranked symbols from OpportunityRanker for strategy generation.
    
    Strategy Generator will only create strategies for symbols with high opportunity scores.
    """
    
    def __init__(self, opportunity_ranker=None, top_n: int = 10, min_score: float = 0.65):
        """
        Initialize opportunity-filtered symbol provider.
        
        Args:
            opportunity_ranker: OpportunityRanker instance from main app state
            top_n: Maximum number of top symbols to return
            min_score: Minimum opportunity score (0-1) to consider
        """
        self.opportunity_ranker = opportunity_ranker
        self.top_n = top_n
        self.min_score = min_score
        self.fallback_symbols = ["BTCUSDT", "ETHUSDT"]  # Fallback if ranker unavailable
    
    def get_top_symbols(self) -> List[str]:
        """
        Get top-ranked symbols for strategy generation.
        
        Returns:
            List of symbol names (e.g., ["BTCUSDT", "ETHUSDT"])
            Falls back to default symbols if OpportunityRanker is unavailable.
        """
        if not self.opportunity_ranker:
            logger.warning("[SG AI] OpportunityRanker not available, using fallback symbols")
            return self.fallback_symbols
        
        try:
            # Get current rankings
            rankings = self.opportunity_ranker.get_rankings()
            
            if not rankings:
                logger.warning("[SG AI] No opportunity rankings available, using fallback")
                return self.fallback_symbols
            
            # Filter by minimum score
            filtered = [
                rank.symbol for rank in rankings 
                if rank.composite_score >= self.min_score
            ][:self.top_n]
            
            if not filtered:
                logger.warning(
                    f"[SG AI] No symbols above min_score={self.min_score}, "
                    f"using top {self.top_n} regardless"
                )
                filtered = [rank.symbol for rank in rankings[:self.top_n]]
            
            logger.info(
                f"[SG AI] Selected {len(filtered)} symbols from OpportunityRanker: "
                f"{filtered[:5]}{'...' if len(filtered) > 5 else ''}"
            )
            
            # Log scores for transparency
            for rank in rankings[:len(filtered)]:
                logger.debug(
                    f"  {rank.symbol}: score={rank.composite_score:.2f}, "
                    f"trend={rank.trend_score:.2f}, vol={rank.volatility_score:.2f}"
                )
            
            return filtered if filtered else self.fallback_symbols
            
        except Exception as e:
            logger.error(f"[SG AI] Error getting opportunity symbols: {e}", exc_info=True)
            return self.fallback_symbols
    
    def get_symbol_score(self, symbol: str) -> Optional[float]:
        """
        Get opportunity score for a specific symbol.
        
        Args:
            symbol: Symbol name (e.g., "BTCUSDT")
            
        Returns:
            Opportunity score (0-1) or None if not found
        """
        if not self.opportunity_ranker:
            return None
        
        try:
            rankings = self.opportunity_ranker.get_rankings()
            for rank in rankings:
                if rank.symbol == symbol:
                    return rank.composite_score
            return None
        except Exception as e:
            logger.error(f"[SG AI] Error getting score for {symbol}: {e}")
            return None
    
    def should_generate_strategy(self, symbol: str) -> bool:
        """
        Check if strategy generation is recommended for this symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            True if symbol has good opportunity score
        """
        score = self.get_symbol_score(symbol)
        
        if score is None:
            # If ranker unavailable, allow fallback symbols
            return symbol in self.fallback_symbols
        
        return score >= self.min_score
