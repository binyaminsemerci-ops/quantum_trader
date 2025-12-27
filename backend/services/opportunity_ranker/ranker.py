"""
Market Opportunity Ranker - identifies and ranks best trading opportunities.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services.eventbus import InMemoryEventBus, OpportunitiesUpdatedEvent
from backend.services.opportunity_ranker.models import SymbolScore, RankingCriteria


logger = logging.getLogger(__name__)


class MarketOpportunityRanker:
    """
    Analyzes symbols and ranks them by opportunity score.
    
    Publishes OpportunitiesUpdatedEvent when rankings change.
    """
    
    def __init__(
        self,
        eventbus: InMemoryEventBus,
        criteria: Optional[RankingCriteria] = None,
    ):
        self.eventbus = eventbus
        self.criteria = criteria or RankingCriteria()
        self.current_rankings: List[SymbolScore] = []
        self._running = False
        
    async def score_symbol(
        self,
        symbol: str,
        market_data: Dict,
    ) -> Optional[SymbolScore]:
        """
        Calculate opportunity score for a single symbol.
        
        Returns None if symbol doesn't meet minimum criteria.
        """
        try:
            # Extract market data
            volume_24h = market_data.get("volume_24h", 0)
            atr = market_data.get("atr", 0)
            trend_strength = market_data.get("trend_strength", 0)
            volatility = market_data.get("volatility", 0)
            spread = market_data.get("spread", 0)
            recent_return = market_data.get("recent_return", 0)
            
            # Check minimum volume
            if volume_24h < self.criteria.min_volume:
                return None
            
            # Calculate component scores (0-1)
            trend_score = self._calculate_trend_score(trend_strength)
            volatility_score = self._calculate_volatility_score(volatility, atr)
            liquidity_score = self._calculate_liquidity_score(volume_24h, spread)
            performance_score = self._calculate_performance_score(recent_return)
            
            # Check minimum liquidity
            if liquidity_score < self.criteria.min_liquidity_score:
                return None
            
            # Calculate composite score
            score = SymbolScore.calculate(
                symbol=symbol,
                trend_score=trend_score,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                performance_score=performance_score,
                criteria=self.criteria,
                volume_24h=volume_24h,
                atr=atr,
                trend_strength=trend_strength,
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            return None
    
    def _calculate_trend_score(self, trend_strength: float) -> float:
        """
        Score trend strength (0-1).
        
        Strong trends (>0.7 or <-0.7) get high scores.
        """
        abs_strength = abs(trend_strength)
        if abs_strength >= 0.7:
            return 1.0
        elif abs_strength >= 0.5:
            return 0.7
        elif abs_strength >= 0.3:
            return 0.4
        else:
            return 0.1
    
    def _calculate_volatility_score(self, volatility: float, atr: float) -> float:
        """
        Score volatility (0-1).
        
        Moderate volatility (1-3% ATR) is ideal.
        """
        if atr <= 0:
            return 0.5
        
        if 1 <= atr <= 3:
            return 1.0
        elif 0.5 <= atr <= 5:
            return 0.7
        elif atr < 0.5 or atr > 7:
            return 0.3
        else:
            return 0.5
    
    def _calculate_liquidity_score(self, volume_24h: float, spread: float) -> float:
        """
        Score liquidity (0-1).
        
        High volume + low spread = high liquidity.
        """
        # Volume component (0-0.7)
        if volume_24h >= 1e10:  # 10B+
            vol_score = 0.7
        elif volume_24h >= 5e9:  # 5B+
            vol_score = 0.6
        elif volume_24h >= 1e9:  # 1B+
            vol_score = 0.5
        else:
            vol_score = 0.3
        
        # Spread component (0-0.3)
        if spread <= 0.001:  # 0.1%
            spread_score = 0.3
        elif spread <= 0.005:  # 0.5%
            spread_score = 0.2
        else:
            spread_score = 0.1
        
        return vol_score + spread_score
    
    def _calculate_performance_score(self, recent_return: float) -> float:
        """
        Score recent performance (0-1).
        
        Strong positive momentum gets high scores.
        """
        if recent_return >= 0.1:  # 10%+
            return 1.0
        elif recent_return >= 0.05:  # 5%+
            return 0.8
        elif recent_return >= 0.02:  # 2%+
            return 0.6
        elif recent_return >= 0:
            return 0.4
        elif recent_return >= -0.02:
            return 0.3
        else:
            return 0.1
    
    async def rank_all_symbols(
        self,
        symbols_data: Dict[str, Dict],
    ) -> List[SymbolScore]:
        """
        Rank all symbols by opportunity score.
        
        Returns sorted list (highest score first).
        """
        scores = []
        
        for symbol, data in symbols_data.items():
            score = await self.score_symbol(symbol, data)
            if score:
                scores.append(score)
        
        # Sort by total_score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        self.current_rankings = scores
        return scores
    
    def get_top_n_opportunities(self, n: int = 10) -> List[SymbolScore]:
        """Get top N ranked symbols."""
        return self.current_rankings[:n]
    
    async def publish_rankings(self):
        """Publish current rankings to EventBus."""
        if not self.current_rankings:
            logger.warning("No rankings to publish")
            return
        
        # Extract top symbols and scores
        top_opportunities = self.get_top_n_opportunities(10)
        top_symbols = [s.symbol for s in top_opportunities]
        scores = {s.symbol: s.total_score for s in top_opportunities}
        
        event = OpportunitiesUpdatedEvent.create(
            top_symbols=top_symbols,
            scores=scores,
            criteria=self.criteria.to_dict(),
            excluded_count=0,
        )
        
        await self.eventbus.publish(event)
        logger.info(
            f"Published top {len(top_opportunities)} opportunities"
        )
    
    async def run_forever(
        self,
        symbols_data_provider,
        interval_seconds: int = 300,
    ):
        """
        Continuously rank symbols and publish updates.
        
        Args:
            symbols_data_provider: Async callable that returns Dict[symbol, data]
            interval_seconds: Update interval (default 5 minutes)
        """
        self._running = True
        logger.info("MarketOpportunityRanker started")
        
        while self._running:
            try:
                # Get fresh market data
                symbols_data = await symbols_data_provider()
                
                # Rank all symbols
                scores = await self.rank_all_symbols(symbols_data)
                
                logger.info(f"Ranked {len(scores)} symbols")
                
                # Publish update
                await self.publish_rankings()
                
                # Wait for next update
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("MarketOpportunityRanker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in ranking loop: {e}")
                await asyncio.sleep(interval_seconds)
        
        logger.info("MarketOpportunityRanker stopped")
    
    def stop(self):
        """Stop the ranking loop."""
        self._running = False
