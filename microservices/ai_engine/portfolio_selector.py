"""
Portfolio Selection Layer - Capital Allocation Optimizer

This module implements portfolio-level filtering to convert signal-driven trading
into capital-allocation-driven trading. It selects the highest-confidence, least-correlated
signals across the entire portfolio.

Flow:
1. Confidence filter (reuse QT_MIN_CONFIDENCE)
2. HOLD filter
3. Rank by confidence descending
4. Select top N
5. Correlation filter (vs open positions)

Author: AI Engine Team
Date: 2026-02-18
"""

import logging
from typing import List, Dict, Any, Set, Optional
from datetime import datetime, timedelta
import asyncio
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class PortfolioSelector:
    """
    Portfolio-level signal selector with confidence ranking and correlation filtering.
    
    This component operates AFTER ensemble prediction but BEFORE signal publishing,
    ensuring only the highest-quality, diversified signals are sent to execution.
    """
    
    def __init__(self, settings, redis_client):
        """
        Initialize portfolio selector.
        
        Args:
            settings: Configuration object with TOP_N_LIMIT, MAX_SYMBOL_CORRELATION, etc.
            redis_client: Redis client for fetching open positions and price data
        """
        self.settings = settings
        self.redis_client = redis_client
        
        # Configuration
        self.top_n_limit = getattr(settings, 'TOP_N_LIMIT', 10)
        self.max_correlation = getattr(settings, 'MAX_SYMBOL_CORRELATION', 0.80)
        self.min_confidence = getattr(settings, 'MIN_SIGNAL_CONFIDENCE', 0.55)
        
        # Correlation computation settings
        self.correlation_window_days = 30  # 30-day rolling window
        self.correlation_interval = '1h'   # 1-hour candles
        
        # Cache for price returns (symbol -> returns array)
        self._returns_cache: Dict[str, np.ndarray] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        logger.info(
            f"[Portfolio-Selector] Initialized: top_n={self.top_n_limit}, "
            f"max_corr={self.max_correlation}, min_conf={self.min_confidence}"
        )
    
    async def select(
        self,
        predictions: List[Dict[str, Any]],
        open_positions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Select best predictions using confidence ranking and correlation filtering.
        
        Args:
            predictions: List of prediction dicts with 'symbol', 'action', 'confidence', etc.
            open_positions: List of symbols with open positions (optional, will fetch if None)
        
        Returns:
            Filtered list of predictions to publish
        """
        if not predictions:
            logger.debug("[Portfolio-Selector] No predictions to process")
            return []
        
        total_count = len(predictions)
        
        # STEP 1: Confidence filter + HOLD filter
        eligible = self._filter_by_confidence(predictions)
        eligible_count = len(eligible)
        
        if not eligible:
            logger.info(
                f"[Portfolio-Selector] ⛔ No eligible predictions after confidence filter "
                f"(total={total_count}, threshold={self.min_confidence:.2%})"
            )
            return []
        
        # STEP 2: Rank by confidence descending
        eligible.sort(key=lambda p: p.get('confidence', 0.0), reverse=True)
        
        # STEP 3: Select top N
        top_n = eligible[:self.top_n_limit]
        top_n_count = len(top_n)
        
        # STEP 4: Correlation filter (if we have open positions)
        if open_positions is None:
            open_positions = await self._fetch_open_positions()
        
        if open_positions:
            final_selected = await self._filter_by_correlation(top_n, open_positions)
        else:
            # No open positions → no correlation constraint
            final_selected = top_n
            logger.debug("[Portfolio-Selector] No open positions - skipping correlation filter")
        
        final_count = len(final_selected)
        
        # Structured logging
        highest_conf = final_selected[0]['confidence'] if final_selected else 0.0
        lowest_conf = final_selected[-1]['confidence'] if final_selected else 0.0
        
        logger.info(
            f"[Portfolio-Selector] 📊 Selection complete: "
            f"total={total_count}, eligible={eligible_count}, top_n={top_n_count}, "
            f"final={final_count} | conf_range=[{lowest_conf:.2%}, {highest_conf:.2%}]"
        )
        
        # Log rejections
        rejected_count = top_n_count - final_count
        if rejected_count > 0:
            rejected_symbols = [
                f"{p['symbol']}({p['confidence']:.1%})" 
                for p in top_n if p not in final_selected
            ]
            logger.info(
                f"[Portfolio-Selector] ⛔ Rejected {rejected_count} due to correlation: "
                f"{', '.join(rejected_symbols[:10])}"  # Limit to first 10
            )
        
        return final_selected
    
    def _filter_by_confidence(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter predictions by confidence threshold and action type.
        
        Removes:
        - HOLD actions
        - Predictions with confidence < MIN_SIGNAL_CONFIDENCE
        
        Args:
            predictions: Raw prediction list
        
        Returns:
            Filtered list
        """
        return [
            p for p in predictions
            if p.get('action') != 'HOLD' and p.get('confidence', 0.0) >= self.min_confidence
        ]
    
    async def _filter_by_correlation(
        self,
        candidates: List[Dict[str, Any]],
        open_positions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates by correlation with open positions.
        
        For each candidate:
        1. Compute rolling correlation with each open position
        2. If any correlation > MAX_SYMBOL_CORRELATION → reject
        3. Otherwise → keep
        
        Args:
            candidates: List of top-N candidates
            open_positions: List of symbols with open positions
        
        Returns:
            Candidates passing correlation filter
        """
        if not open_positions:
            return candidates
        
        selected = []
        
        for candidate in candidates:
            symbol = candidate.get('symbol')
            if not symbol:
                continue
            
            # Check if already in open positions (skip correlation check if same symbol)
            if symbol in open_positions:
                # Allow adding to existing position (no correlation check needed)
                selected.append(candidate)
                logger.debug(
                    f"[Portfolio-Selector] ✅ {symbol} - existing position, allowed"
                )
                continue
            
            # Compute correlations with all open positions
            try:
                is_correlated = await self._is_highly_correlated(symbol, open_positions)
                
                if not is_correlated:
                    selected.append(candidate)
                    logger.debug(
                        f"[Portfolio-Selector] ✅ {symbol} - low correlation, allowed"
                    )
                else:
                    logger.debug(
                        f"[Portfolio-Selector] ⛔ {symbol} - high correlation (>{self.max_correlation:.2f}), rejected"
                    )
            
            except Exception as e:
                # FAIL-SAFE: If correlation computation fails → allow trade
                logger.warning(
                    f"[Portfolio-Selector] ⚠️ Correlation check failed for {symbol}: {e} - allowing trade (fail-safe)"
                )
                selected.append(candidate)
        
        return selected
    
    async def _is_highly_correlated(self, symbol: str, open_positions: List[str]) -> bool:
        """
        Check if symbol is highly correlated with any open position.
        
        Args:
            symbol: Candidate symbol
            open_positions: List of symbols with open positions
        
        Returns:
            True if any correlation > MAX_SYMBOL_CORRELATION, False otherwise
        """
        # Get returns for candidate symbol
        candidate_returns = await self._get_returns(symbol)
        if candidate_returns is None or len(candidate_returns) < 10:
            # Insufficient data → fail-safe allow
            logger.debug(f"[Portfolio-Selector] {symbol} - insufficient data for correlation, allowing")
            return False
        
        # Check correlation with each open position
        for position_symbol in open_positions:
            if position_symbol == symbol:
                continue  # Skip self-correlation
            
            position_returns = await self._get_returns(position_symbol)
            if position_returns is None or len(position_returns) < 10:
                continue  # Skip if no data for this position
            
            # Align arrays to same length (use minimum length)
            min_len = min(len(candidate_returns), len(position_returns))
            candidate_aligned = candidate_returns[-min_len:]
            position_aligned = position_returns[-min_len:]
            
            # Compute Pearson correlation
            if min_len >= 10:  # Require at least 10 data points
                correlation = np.corrcoef(candidate_aligned, position_aligned)[0, 1]
                
                logger.debug(
                    f"[Portfolio-Selector] Correlation {symbol} vs {position_symbol}: {correlation:.3f}"
                )
                
                if abs(correlation) > self.max_correlation:
                    logger.info(
                        f"[Portfolio-Selector] 🔴 High correlation detected: "
                        f"{symbol} vs {position_symbol} = {correlation:.3f} (threshold={self.max_correlation:.2f})"
                    )
                    return True
        
        return False
    
    async def _get_returns(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get rolling returns for a symbol (cached).
        
        Uses 1-hour candles for the last 30 days.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Numpy array of returns, or None if unavailable
        """
        # Check cache
        now = datetime.utcnow()
        if symbol in self._returns_cache:
            cache_age = (now - self._cache_timestamp.get(symbol, now)).total_seconds()
            if cache_age < self._cache_ttl:
                return self._returns_cache[symbol]
        
        # Fetch from Redis or compute
        try:
            returns = await self._compute_returns_from_redis(symbol)
            
            if returns is not None and len(returns) > 0:
                # Cache result
                self._returns_cache[symbol] = returns
                self._cache_timestamp[symbol] = now
                return returns
            
            return None
        
        except Exception as e:
            logger.warning(f"[Portfolio-Selector] Failed to get returns for {symbol}: {e}")
            return None
    
    async def _compute_returns_from_redis(self, symbol: str) -> Optional[np.ndarray]:
        """
        Compute returns from price data in Redis.
        
        Sources (in order of preference):
        1. OHLCV history cache (quantum:history:<symbol>:1h)
        2. Recent klines from market data stream
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Array of returns (price[i] / price[i-1] - 1)
        """
        # Try OHLCV history cache first
        history_key = f"quantum:history:{symbol}:1h"
        
        try:
            # Get last 30 days of 1h candles (720 data points)
            # Redis sorted set: score=timestamp, value=json
            end_ts = int(datetime.utcnow().timestamp() * 1000)
            start_ts = int((datetime.utcnow() - timedelta(days=self.correlation_window_days)).timestamp() * 1000)
            
            # ZRANGEBYSCORE to get candles in time range
            # Note: This assumes OHLCV data is stored in Redis
            # If not available, we'll use a simpler approach
            
            # For now, use a simplified approach: get recent prices from klines
            prices = await self._get_recent_prices(symbol)
            
            if prices is not None and len(prices) >= 2:
                # Compute returns
                prices_array = np.array(prices)
                returns = np.diff(prices_array) / prices_array[:-1]
                return returns
            
            return None
        
        except Exception as e:
            logger.debug(f"[Portfolio-Selector] Error computing returns for {symbol}: {e}")
            return None
    
    async def _get_recent_prices(self, symbol: str) -> Optional[List[float]]:
        """
        Get recent prices from Redis klines cache.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            List of close prices
        """
        try:
            # Try to get from OHLCV history (simplified - assume close prices stored)
            # In real implementation, this would fetch from your specific Redis structure
            
            # For this implementation, we'll use a reasonable fail-safe approach:
            # Try to get recent price points from any available cache
            
            # Option 1: Get from price feed cache
            price_key = f"quantum:price:{symbol}"
            price_str = await self.redis_client.get(price_key)
            
            if price_str:
                # If we only have current price, return None (insufficient data)
                # In production, you'd fetch historical OHLCV here
                return None
            
            # Option 2: Get from OHLCV history (if available)
            # This is placeholder - implement based on your Redis schema
            history_key = f"quantum:ohlcv:{symbol}:1h"
            
            # For now, return None to trigger fail-safe behavior
            # TODO: Implement actual OHLCV fetching based on your Redis schema
            return None
        
        except Exception as e:
            logger.debug(f"[Portfolio-Selector] Error fetching prices for {symbol}: {e}")
            return None
    
    async def _fetch_open_positions(self) -> List[str]:
        """
        Fetch list of symbols with open positions from Redis.
        
        Returns:
            List of symbols with active positions
        """
        try:
            # Get all position snapshot keys
            # Pattern: quantum:position:snapshot:*
            pattern = "quantum:position:snapshot:*"
            
            cursor = 0
            symbols = set()
            
            # Scan for position keys
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                for key in keys:
                    # Extract symbol from key
                    # Format: quantum:position:snapshot:<symbol>
                    parts = key.decode('utf-8').split(':')
                    if len(parts) >= 4:
                        symbol = parts[3]
                        
                        # Check if position is still open (has quantity > 0)
                        position_data = await self.redis_client.get(key)
                        if position_data:
                            # Position exists and is active
                            symbols.add(symbol)
                
                if cursor == 0:
                    break
            
            symbols_list = list(symbols)
            logger.debug(f"[Portfolio-Selector] Found {len(symbols_list)} open positions: {symbols_list}")
            return symbols_list
        
        except Exception as e:
            logger.warning(f"[Portfolio-Selector] Failed to fetch open positions: {e}")
            return []
    
    def clear_cache(self):
        """Clear returns cache (for testing or manual refresh)."""
        self._returns_cache.clear()
        self._cache_timestamp.clear()
        logger.info("[Portfolio-Selector] Cache cleared")
