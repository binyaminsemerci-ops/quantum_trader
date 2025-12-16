"""
Opportunity Ranker (OppRank) Module

Evaluates and ranks all available trading symbols by computing multiple
market-quality metrics and producing a weighted opportunity score (0.0-1.0).

This module identifies high-edge trading opportunities by analyzing:
- Trend strength and direction
- Volatility quality (optimal range)
- Liquidity and depth
- Spread and fee costs
- Historical symbol performance
- Regime compatibility
- Market noise levels

The ranked output guides strategy selection, orchestrator decisions,
and MSC AI policy adjustments.
"""

from typing import Protocol, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# PROTOCOLS (Dependency Interfaces)
# ============================================================================

class MarketDataClient(Protocol):
    """Interface for retrieving market data."""
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch recent OHLCV candles.
        
        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...
    
    def get_spread(self, symbol: str) -> float:
        """Get current bid-ask spread percentage."""
        ...
    
    def get_liquidity(self, symbol: str) -> float:
        """Get liquidity score (e.g., 24h volume in USD)."""
        ...


class TradeLogRepository(Protocol):
    """Interface for accessing historical trade performance."""
    
    def get_symbol_winrate(self, symbol: str, last_n: int = 200) -> float:
        """
        Calculate winrate for symbol from last N trades.
        
        Returns float between 0.0 (0%) and 1.0 (100%).
        """
        ...


class RegimeDetector(Protocol):
    """Interface for market regime detection."""
    
    def get_global_regime(self) -> str:
        """
        Get current global market regime.
        
        Returns: 'BULL', 'BEAR', 'CHOPPY', 'RANGING', etc.
        """
        ...


class OpportunityStore(Protocol):
    """Interface for storing/retrieving opportunity rankings."""
    
    def update(self, rankings: Dict[str, float]) -> None:
        """Store symbol rankings with timestamp."""
        ...
    
    def get(self) -> Dict[str, float]:
        """Retrieve latest rankings."""
        ...


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SymbolMetrics:
    """Complete set of computed metrics for a symbol."""
    symbol: str
    trend_strength: float  # 0-1
    volatility_quality: float  # 0-1
    liquidity_score: float  # 0-1
    spread_score: float  # 0-1
    symbol_winrate_score: float  # 0-1
    regime_score: float  # 0-1
    noise_score: float  # 0-1 (inverse of noise)
    final_score: float  # 0-1 weighted aggregate
    timestamp: datetime


@dataclass
class OpportunityRanking:
    """Ranked opportunity for a symbol."""
    symbol: str
    overall_score: float
    rank: int
    metric_scores: Dict[str, float]
    metadata: Dict[str, any]
    timestamp: datetime


class TrendDirection(Enum):
    """Trend direction classification."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# ============================================================================
# OPPORTUNITY RANKER
# ============================================================================

class OpportunityRanker:
    """
    Main Opportunity Ranker class.
    
    Evaluates all configured symbols and produces ranked opportunity scores
    based on multiple market-quality metrics.
    """
    
    # Default metric weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        'trend_strength': 0.25,
        'volatility_quality': 0.20,
        'liquidity_score': 0.15,
        'regime_score': 0.15,
        'symbol_winrate_score': 0.10,
        'spread_score': 0.10,
        'noise_score': 0.05,
    }
    
    # Volatility quality thresholds (as % of price)
    OPTIMAL_VOLATILITY_MIN = 0.015  # 1.5% daily ATR
    OPTIMAL_VOLATILITY_MAX = 0.08   # 8% daily ATR
    
    # Liquidity thresholds (USD volume)
    MIN_LIQUIDITY_USD = 1_000_000    # $1M
    OPTIMAL_LIQUIDITY_USD = 100_000_000  # $100M
    
    # Spread thresholds (percentage)
    OPTIMAL_SPREAD_MAX = 0.0005  # 0.05% or 5 basis points
    POOR_SPREAD_MIN = 0.002      # 0.2%
    
    def __init__(
        self,
        market_data: MarketDataClient,
        trade_logs: TradeLogRepository,
        regime_detector: RegimeDetector,
        opportunity_store: OpportunityStore,
        *,
        symbols: List[str],
        timeframe: str = "1h",
        candle_limit: int = 200,
        weights: Optional[Dict[str, float]] = None,
        min_score_threshold: float = 0.3,
    ):
        """
        Initialize Opportunity Ranker.
        
        Args:
            market_data: Client for fetching market data
            trade_logs: Repository for trade history
            regime_detector: Global regime detection service
            opportunity_store: Storage for rankings
            symbols: List of symbols to evaluate
            timeframe: Candle timeframe for analysis (default: 1h)
            candle_limit: Number of candles to analyze (default: 200)
            weights: Custom metric weights (optional)
            min_score_threshold: Minimum score to include in rankings
        """
        self.market_data = market_data
        self.trade_logs = trade_logs
        self.regime_detector = regime_detector
        self.opportunity_store = opportunity_store
        
        self.symbols = symbols
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.min_score_threshold = min_score_threshold
        
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._validate_weights()
        
        logger.info(
            f"OpportunityRanker initialized: {len(symbols)} symbols, "
            f"timeframe={timeframe}, threshold={min_score_threshold}"
        )
    
    def _validate_weights(self) -> None:
        """Ensure weights are valid and sum to 1.0."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def compute_symbol_scores(self) -> Dict[str, SymbolMetrics]:
        """
        Compute all metrics and final scores for each symbol.
        
        Returns:
            Dictionary mapping symbol to SymbolMetrics
        """
        global_regime = self.regime_detector.get_global_regime()
        logger.info(f"Computing scores for {len(self.symbols)} symbols (regime: {global_regime})")
        
        results = {}
        
        for symbol in self.symbols:
            try:
                metrics = self._compute_symbol_metrics(symbol, global_regime)
                results[symbol] = metrics
                logger.debug(f"{symbol}: final_score={metrics.final_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to compute metrics for {symbol}: {e}")
                continue
        
        logger.info(f"Successfully computed metrics for {len(results)}/{len(self.symbols)} symbols")
        return results
    
    def update_rankings(self) -> Dict[str, float]:
        """
        Compute scores, filter by threshold, sort, store, and return rankings.
        
        Returns:
            Dictionary of {symbol: score} sorted by score descending
        """
        metrics_dict = self.compute_symbol_scores()
        
        # Extract scores and filter by threshold
        rankings = {
            symbol: metrics.final_score
            for symbol, metrics in metrics_dict.items()
            if metrics.final_score >= self.min_score_threshold
        }
        
        # Sort by score descending
        rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
        
        # Store results in Redis
        self.opportunity_store.update(rankings)
        
        # Write to PolicyStore if available
        if hasattr(self, 'policy_store') and self.policy_store:
            try:
                self.policy_store.patch({'opp_rankings': rankings})
                logger.info(f"[OpportunityRanker] ✅ Rankings written to PolicyStore ({len(rankings)} symbols)")
            except Exception as e:
                logger.error(f"[OpportunityRanker] ❌ Failed to write to PolicyStore: {e}")
        
        logger.info(
            f"Rankings updated: {len(rankings)} symbols passed threshold "
            f"(top: {list(rankings.keys())[:5]})"
        )
        
        return rankings
    
    def get_top_n(self, n: int = 10) -> List[str]:
        """
        Get top N symbols by opportunity score.
        
        Args:
            n: Number of top symbols to return
            
        Returns:
            List of symbols sorted by score descending
        """
        rankings = self.opportunity_store.get()
        return list(rankings.keys())[:n]
    
    # ========================================================================
    # SYMBOL METRICS COMPUTATION
    # ========================================================================
    
    def _compute_symbol_metrics(self, symbol: str, global_regime: str) -> SymbolMetrics:
        """Compute all metrics for a single symbol."""
        # Fetch market data
        df = self.market_data.get_latest_candles(symbol, self.timeframe, self.candle_limit)
        
        if df is None or len(df) < 50:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Compute individual metrics
        trend_strength = self._compute_trend_strength(df)
        volatility_quality = self._compute_volatility_quality(df)
        liquidity_score = self._compute_liquidity_score(symbol)
        spread_score = self._compute_spread_score(symbol)
        symbol_winrate_score = self._compute_symbol_winrate(symbol)
        regime_score = self._compute_regime_score(df, global_regime)
        noise_score = self._compute_noise_score(df)
        
        # Aggregate final score
        final_score = self._aggregate_scores({
            'trend_strength': trend_strength,
            'volatility_quality': volatility_quality,
            'liquidity_score': liquidity_score,
            'spread_score': spread_score,
            'symbol_winrate_score': symbol_winrate_score,
            'regime_score': regime_score,
            'noise_score': noise_score,
        })
        
        return SymbolMetrics(
            symbol=symbol,
            trend_strength=trend_strength,
            volatility_quality=volatility_quality,
            liquidity_score=liquidity_score,
            spread_score=spread_score,
            symbol_winrate_score=symbol_winrate_score,
            regime_score=regime_score,
            noise_score=noise_score,
            final_score=final_score,
            timestamp=datetime.utcnow(),
        )
    
    # ========================================================================
    # METRIC CALCULATORS (each returns 0.0 - 1.0)
    # ========================================================================
    
    def _compute_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Compute trend strength score.
        
        Based on:
        - EMA slope (50 vs 200)
        - Consistency of higher highs/higher lows
        - Price position relative to EMAs
        
        Returns:
            Float 0.0-1.0 (higher = stronger trend)
        """
        if len(df) < 200:
            return 0.5  # Neutral if insufficient data
        
        # Calculate EMAs
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        ema_200 = df['close'].ewm(span=200, adjust=False).mean()
        
        # Current values
        current_price = df['close'].iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        current_ema200 = ema_200.iloc[-1]
        
        # EMA slope (normalized)
        ema50_slope = (ema_50.iloc[-1] - ema_50.iloc[-20]) / ema_50.iloc[-20]
        ema200_slope = (ema_200.iloc[-1] - ema_200.iloc[-20]) / ema_200.iloc[-20]
        
        # Trend alignment score
        if current_price > current_ema50 > current_ema200:
            alignment_score = 1.0  # Strong uptrend
        elif current_price < current_ema50 < current_ema200:
            alignment_score = 1.0  # Strong downtrend
        elif current_price > current_ema50 or current_price > current_ema200:
            alignment_score = 0.6  # Partial uptrend
        elif current_price < current_ema50 or current_price < current_ema200:
            alignment_score = 0.6  # Partial downtrend
        else:
            alignment_score = 0.3  # Choppy
        
        # Slope strength (use absolute value for trend strength)
        slope_strength = min(abs(ema50_slope) * 20, 1.0)  # Scale and cap at 1.0
        
        # Higher highs / higher lows consistency (last 20 periods)
        highs = df['high'].iloc[-20:].values
        lows = df['low'].iloc[-20:].values
        
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        hl_count = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        lh_count = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        
        consistency = max(
            (hh_count + hl_count) / 38,  # Uptrend consistency
            (lh_count + ll_count) / 38   # Downtrend consistency
        )
        
        # Combined score
        trend_strength = (alignment_score * 0.5 + slope_strength * 0.3 + consistency * 0.2)
        
        return np.clip(trend_strength, 0.0, 1.0)
    
    def _compute_volatility_quality(self, df: pd.DataFrame) -> float:
        """
        Compute volatility quality score.
        
        Optimal volatility = enough movement to profit, but not too chaotic.
        
        Returns:
            Float 0.0-1.0 (higher = better volatility for trading)
        """
        if len(df) < 14:
            return 0.5
        
        # Calculate ATR (14-period)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().iloc[-1]
        
        # Normalize ATR as percentage of price
        current_price = df['close'].iloc[-1]
        atr_pct = atr / current_price
        
        # Score based on optimal range
        if atr_pct < self.OPTIMAL_VOLATILITY_MIN:
            # Too low - not enough movement
            score = atr_pct / self.OPTIMAL_VOLATILITY_MIN
        elif atr_pct <= self.OPTIMAL_VOLATILITY_MAX:
            # Optimal range - full score
            score = 1.0
        else:
            # Too high - too chaotic
            excess = atr_pct - self.OPTIMAL_VOLATILITY_MAX
            score = max(0.0, 1.0 - (excess / self.OPTIMAL_VOLATILITY_MAX))
        
        # Volatility stability bonus (low variance = more predictable)
        recent_atr = true_range.rolling(window=14).mean().iloc[-20:]
        atr_stability = 1.0 - min(recent_atr.std() / recent_atr.mean(), 1.0)
        
        final_score = score * 0.8 + atr_stability * 0.2
        
        return np.clip(final_score, 0.0, 1.0)
    
    def _compute_liquidity_score(self, symbol: str) -> float:
        """
        Compute liquidity score based on volume and depth.
        
        Returns:
            Float 0.0-1.0 (higher = better liquidity)
        """
        try:
            liquidity_usd = self.market_data.get_liquidity(symbol)
            
            if liquidity_usd < self.MIN_LIQUIDITY_USD:
                # Poor liquidity
                score = liquidity_usd / self.MIN_LIQUIDITY_USD * 0.3
            elif liquidity_usd >= self.OPTIMAL_LIQUIDITY_USD:
                # Excellent liquidity
                score = 1.0
            else:
                # Linear interpolation between min and optimal
                score = 0.3 + (liquidity_usd - self.MIN_LIQUIDITY_USD) / \
                        (self.OPTIMAL_LIQUIDITY_USD - self.MIN_LIQUIDITY_USD) * 0.7
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to get liquidity for {symbol}: {e}")
            return 0.5  # Neutral default
    
    def _compute_spread_score(self, symbol: str) -> float:
        """
        Compute spread/fee score.
        
        Lower spread = higher score.
        
        Returns:
            Float 0.0-1.0 (higher = lower spread/better for trading)
        """
        try:
            spread_pct = self.market_data.get_spread(symbol)
            
            if spread_pct <= self.OPTIMAL_SPREAD_MAX:
                score = 1.0
            elif spread_pct >= self.POOR_SPREAD_MIN:
                score = 0.2
            else:
                # Linear decay between optimal and poor
                score = 1.0 - (spread_pct - self.OPTIMAL_SPREAD_MAX) / \
                        (self.POOR_SPREAD_MIN - self.OPTIMAL_SPREAD_MAX) * 0.8
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to get spread for {symbol}: {e}")
            return 0.5
    
    def _compute_symbol_winrate(self, symbol: str) -> float:
        """
        Compute historical winrate score for this symbol.
        
        Returns:
            Float 0.0-1.0 (winrate from trade history)
        """
        try:
            winrate = self.trade_logs.get_symbol_winrate(symbol, last_n=200)
            return np.clip(winrate, 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Failed to get winrate for {symbol}: {e}")
            return 0.5  # Neutral default (50% winrate)
    
    def _compute_regime_score(self, df: pd.DataFrame, global_regime: str) -> float:
        """
        Compute regime compatibility score.
        
        Rewards symbols whose trend aligns with global regime.
        
        Returns:
            Float 0.0-1.0 (higher = better regime alignment)
        """
        trend_direction = self._detect_trend_direction(df)
        
        # Regime compatibility matrix
        if global_regime == "BULL":
            if trend_direction == TrendDirection.BULLISH:
                return 1.0
            elif trend_direction == TrendDirection.NEUTRAL:
                return 0.6
            else:
                return 0.3
                
        elif global_regime == "BEAR":
            if trend_direction == TrendDirection.BEARISH:
                return 1.0
            elif trend_direction == TrendDirection.NEUTRAL:
                return 0.6
            else:
                return 0.3
                
        elif global_regime in ["CHOPPY", "RANGING"]:
            if trend_direction == TrendDirection.NEUTRAL:
                return 0.8
            else:
                return 0.5
        
        # Unknown regime
        return 0.5
    
    def _detect_trend_direction(self, df: pd.DataFrame) -> TrendDirection:
        """Detect symbol's trend direction."""
        if len(df) < 50:
            return TrendDirection.NEUTRAL
        
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        
        if current_price > current_ema20 > current_ema50:
            return TrendDirection.BULLISH
        elif current_price < current_ema20 < current_ema50:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL
    
    def _compute_noise_score(self, df: pd.DataFrame) -> float:
        """
        Compute noise score (inverse of noise level).
        
        Low noise = clean, predictable movement = high score
        High noise = chaotic, unpredictable = low score
        
        Based on:
        - Wick ratio (body vs shadow)
        - High-low variance
        - Close-to-close vs high-low ratio
        
        Returns:
            Float 0.0-1.0 (higher = less noise)
        """
        if len(df) < 20:
            return 0.5
        
        # Wick ratio analysis (last 20 candles)
        recent = df.iloc[-20:]
        
        body = np.abs(recent['close'] - recent['open'])
        upper_wick = recent['high'] - np.maximum(recent['close'], recent['open'])
        lower_wick = np.minimum(recent['close'], recent['open']) - recent['low']
        total_wick = upper_wick + lower_wick
        
        # Avoid division by zero
        total_range = recent['high'] - recent['low']
        total_range = total_range.replace(0, np.nan)
        
        wick_ratio = (total_wick / total_range).mean()
        body_ratio = (body / total_range).mean()
        
        # Lower wick ratio = cleaner movement
        wick_score = 1.0 - min(wick_ratio, 1.0)
        
        # Higher body ratio = stronger directional moves
        body_score = min(body_ratio * 1.5, 1.0)
        
        # Variance analysis
        close_changes = recent['close'].pct_change().dropna()
        range_sizes = (recent['high'] - recent['low']) / recent['close']
        
        # Lower variance = more predictable
        variance_score = 1.0 - min(close_changes.std() / 0.05, 1.0)  # Normalize by 5%
        
        # Combined noise score
        noise_score = (wick_score * 0.3 + body_score * 0.4 + variance_score * 0.3)
        
        return np.clip(noise_score, 0.0, 1.0)
    
    # ========================================================================
    # SCORE AGGREGATION
    # ========================================================================
    
    def _aggregate_scores(self, metric_dict: Dict[str, float]) -> float:
        """
        Aggregate individual metric scores into final opportunity score.
        
        Args:
            metric_dict: Dictionary of metric_name -> score (0-1)
            
        Returns:
            Final weighted score 0.0-1.0
        """
        final_score = sum(
            metric_dict[metric] * self.weights[metric]
            for metric in self.weights.keys()
        )
        
        return np.clip(final_score, 0.0, 1.0)
