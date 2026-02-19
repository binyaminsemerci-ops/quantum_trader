"""
Meta-Strategy Selector Integration Module

Integrates Meta-Strategy Selector with EventDrivenExecutor to enable
AI-driven dynamic strategy selection based on market regimes.

This module bridges the gap between:
- Market Context (from symbols, technical indicators, liquidity)
- Regime Detection (trending, ranging, volatile, etc.)
- Strategy Selection (via RL-based Meta-Strategy Selector)
- Trading Profile (TP/SL parameters applied to trades)

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from backend.services.ai.strategy_profiles import (
    StrategyID,
    StrategyProfile,
    get_strategy_profile,
    get_default_strategy,
)
from backend.services.ai.regime_detector import (
    RegimeDetector,
    MarketRegime,
    MarketContext,
    calculate_trend_strength,
    calculate_adx_from_highs_lows,
)
from backend.services.ai.meta_strategy_selector import (
    MetaStrategySelector,
    StrategyDecision,
    get_meta_strategy_selector,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategySelectionResult:
    """
    Complete result of strategy selection for a symbol.
    
    Contains:
    - Selected strategy
    - Market regime
    - Market context
    - Decision metadata
    """
    
    symbol: str
    strategy: StrategyProfile
    regime: MarketRegime
    context: MarketContext
    decision: StrategyDecision
    tpsl_config: Dict[str, float]  # Ready-to-use TP/SL parameters
    
    def __str__(self) -> str:
        return (
            f"{self.symbol}: {self.strategy.name} "
            f"(regime={self.regime.value}, confidence={self.decision.confidence:.2%})"
        )


class MetaStrategyIntegration:
    """
    Integration layer between EventDrivenExecutor and Meta-Strategy Selector.
    
    Responsibilities:
    - Build market context from available data
    - Detect market regime
    - Select optimal strategy via RL
    - Provide TP/SL configuration
    - Update RL rewards after trade close
    """
    
    def __init__(
        self,
        enabled: bool = True,
        epsilon: float = 0.10,
        alpha: float = 0.20,
        state_file: Optional[Path] = None,
    ):
        """
        Initialize Meta-Strategy Integration.
        
        Args:
            enabled: Whether meta-strategy selection is enabled
            epsilon: Exploration rate for RL (0-1)
            alpha: EMA smoothing for rewards (0-1)
            state_file: Path to RL state persistence file
        """
        self.enabled = enabled
        
        if not self.enabled:
            logger.info("[META-STRATEGY] Disabled - using default strategy")
            return
        
        # Initialize components
        self.regime_detector = RegimeDetector()
        self.meta_selector = get_meta_strategy_selector(
            epsilon=epsilon,
            alpha=alpha,
            state_file=state_file or Path("data/meta_strategy_state.json"),
        )
        
        # Track active strategies per symbol
        self._active_strategies: Dict[str, StrategySelectionResult] = {}
        
        # Metrics
        self.total_selections = 0
        self.total_regime_detections = 0
        self.total_reward_updates = 0
        
        logger.info(
            f"[OK] Meta-Strategy Integration initialized: "
            f"enabled={enabled}, epsilon={epsilon:.2%}, alpha={alpha:.2%}"
        )
    
    async def select_strategy_for_signal(
        self,
        symbol: str,
        signal: Dict,
        market_data: Optional[Dict] = None,
    ) -> StrategySelectionResult:
        """
        Select optimal strategy for a trading signal.
        
        Args:
            symbol: Trading symbol
            signal: AI signal dict with action, confidence, etc.
            market_data: Optional market data (OHLCV, indicators, etc.)
            
        Returns:
            StrategySelectionResult with selected strategy and context
            
        Example:
            >>> integration = MetaStrategyIntegration()
            >>> result = await integration.select_strategy_for_signal(
            ...     symbol="BTCUSDT",
            ...     signal={"action": "BUY", "confidence": 0.75},
            ...     market_data=market_data
            ... )
            >>> print(result.strategy.name, result.regime.value)
            Ultra Aggressive trend_up
        """
        self.total_selections += 1
        
        # If disabled, return default strategy
        if not self.enabled:
            default = get_default_strategy()
            context = MarketContext(symbol=symbol)
            return StrategySelectionResult(
                symbol=symbol,
                strategy=default,
                regime=MarketRegime.UNKNOWN,
                context=context,
                decision=StrategyDecision(
                    strategy_id=default.strategy_id,
                    strategy_profile=default,
                    regime=MarketRegime.UNKNOWN,
                    is_exploration=False,
                    confidence=1.0,
                    reasoning="Meta-strategy disabled - using default",
                    q_values={},
                    timestamp="",
                ),
                tpsl_config=default.to_tpsl_config(),
            )
        
        # Build market context
        context = await self._build_market_context(symbol, signal, market_data)
        
        # Detect regime
        self.total_regime_detections += 1
        regime_result = self.regime_detector.detect_regime(context)
        
        logger.info(
            f"[REGIME] {symbol}: {regime_result.regime.value.upper()} "
            f"(conf={regime_result.confidence:.1%}) - {regime_result.reasoning}"
        )
        
        # Select strategy via RL
        decision = self.meta_selector.choose_strategy(
            symbol=symbol,
            regime=regime_result.regime,
            context=context,
        )
        
        logger.info(
            f"[STRATEGY] {symbol}: {decision.strategy_profile.name} "
            f"(explore={decision.is_exploration}, conf={decision.confidence:.1%}) - {decision.reasoning}"
        )
        
        # Build result
        result = StrategySelectionResult(
            symbol=symbol,
            strategy=decision.strategy_profile,
            regime=regime_result.regime,
            context=context,
            decision=decision,
            tpsl_config=decision.strategy_profile.to_tpsl_config(),
        )
        
        # Store as active
        self._active_strategies[symbol] = result
        
        return result
    
    async def update_strategy_reward(
        self,
        symbol: str,
        realized_r: float,
        trade_meta: Optional[Dict] = None,
    ):
        """
        Update RL reward after trade closes.
        
        Args:
            symbol: Trading symbol
            realized_r: Realized R (e.g., +3.5 for +3.5R win, -1.0 for SL hit)
            trade_meta: Optional trade metadata (pnl, duration, etc.)
            
        Example:
            >>> await integration.update_strategy_reward(
            ...     symbol="BTCUSDT",
            ...     realized_r=3.5,  # +3.5R trade
            ...     trade_meta={"pnl": 85.0, "duration_hours": 4.2}
            ... )
        """
        if not self.enabled:
            return
        
        self.total_reward_updates += 1
        
        # Get active strategy for this symbol
        if symbol not in self._active_strategies:
            logger.warning(
                f"[META-STRATEGY] Cannot update reward for {symbol}: no active strategy tracked"
            )
            return
        
        result = self._active_strategies[symbol]
        
        # Update RL
        self.meta_selector.update_reward(
            symbol=symbol,
            regime=result.regime,
            strategy_id=result.strategy.strategy_id,
            reward=realized_r,
            meta_info=trade_meta,
        )
        
        logger.info(
            f"[RL UPDATE] {symbol} {result.regime.value} {result.strategy.name}: "
            f"R={realized_r:+.2f}"
        )
        
        # Remove from active
        del self._active_strategies[symbol]
    
    async def _build_market_context(
        self,
        symbol: str,
        signal: Dict,
        market_data: Optional[Dict],
    ) -> MarketContext:
        """
        Build MarketContext from available data.
        
        Pulls data from:
        - Signal dict (confidence, model scores)
        - Market data (if provided)
        - Trading Profile / binance data (fallback)
        """
        context = MarketContext(symbol=symbol)
        
        # Extract from signal
        if signal:
            context.current_price = signal.get("price", 0.0)
        
        # Extract from market_data if provided
        if market_data:
            # Technical indicators
            context.atr = market_data.get("atr", 0.0)
            context.atr_pct = market_data.get("atr_pct", 0.0)
            context.sma_20 = market_data.get("sma_20")
            context.sma_50 = market_data.get("sma_50")
            context.ema_12 = market_data.get("ema_12")
            context.ema_26 = market_data.get("ema_26")
            context.adx = market_data.get("adx")
            context.bb_width = market_data.get("bb_width")
            
            # Volume & liquidity
            context.volume_24h = market_data.get("volume_24h", 0.0)
            context.avg_volume_24h = market_data.get("avg_volume_24h", 0.0)
            context.depth_5bps = market_data.get("depth_5bps", 0.0)
            context.spread_bps = market_data.get("spread_bps", 0.0)
            
            # Market structure
            context.funding_rate = market_data.get("funding_rate", 0.0)
            context.open_interest = market_data.get("open_interest", 0.0)
        
        # Calculate trend strength if we have MAs
        if context.current_price > 0:
            context.trend_strength = calculate_trend_strength(
                context.current_price,
                context.sma_20,
                context.sma_50,
                context.ema_12,
                context.ema_26,
            )
        
        # Get symbol-specific performance history if available
        try:
            from backend.services.symbol_performance import get_symbol_performance_manager
            perf_mgr = get_symbol_performance_manager()
            stats = perf_mgr.get_stats(symbol)
            
            if stats and stats.trade_count >= 3:
                context.recent_win_rate = stats.win_rate
                context.recent_avg_r = stats.avg_R
                context.trade_count = stats.trade_count
        except Exception as e:
            logger.debug(f"Could not fetch symbol performance: {e}")
        
        return context
    
    def get_active_strategy(self, symbol: str) -> Optional[StrategySelectionResult]:
        """Get currently active strategy for symbol."""
        return self._active_strategies.get(symbol)
    
    def get_metrics(self) -> Dict:
        """Get integration metrics."""
        return {
            "enabled": self.enabled,
            "total_selections": self.total_selections,
            "total_regime_detections": self.total_regime_detections,
            "total_reward_updates": self.total_reward_updates,
            "active_strategies": len(self._active_strategies),
            "epsilon": self.meta_selector.epsilon if self.enabled else None,
            "alpha": self.meta_selector.alpha if self.enabled else None,
        }
    
    def get_performance_summary(self) -> Dict:
        """Get RL performance summary."""
        if not self.enabled:
            return {"message": "Meta-strategy disabled"}
        
        return self.meta_selector.get_performance_summary()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_integration: Optional[MetaStrategyIntegration] = None


def get_meta_strategy_integration(
    enabled: Optional[bool] = None,
    epsilon: float = 0.10,
    alpha: float = 0.20,
) -> MetaStrategyIntegration:
    """
    Get or create global Meta-Strategy Integration instance.
    
    Args:
        enabled: Override enabled status (None = use env var)
        epsilon: Exploration rate
        alpha: EMA smoothing
        
    Returns:
        MetaStrategyIntegration instance
    """
    global _global_integration
    
    if _global_integration is None:
        # Default: enabled unless explicitly disabled via env
        if enabled is None:
            enabled = os.getenv("META_STRATEGY_ENABLED", "true").lower() == "true"
        
        _global_integration = MetaStrategyIntegration(
            enabled=enabled,
            epsilon=epsilon,
            alpha=alpha,
        )
    
    return _global_integration


if __name__ == "__main__":
    # Demo
    import asyncio
    
    async def demo():
        print("🧠 META-STRATEGY INTEGRATION DEMO\n")
        
        integration = MetaStrategyIntegration(enabled=True, epsilon=0.15)
        
        # Simulate signal with market data
        signal = {
            "symbol": "BTCUSDT",
            "action": "BUY",
            "confidence": 0.75,
            "price": 100000.0,
        }
        
        market_data = {
            "atr": 500.0,
            "atr_pct": 0.005,
            "volume_24h": 50_000_000,
            "spread_bps": 2.5,
            "depth_5bps": 500_000,
            "sma_20": 99000.0,
            "sma_50": 97000.0,
            "adx": 35.0,
        }
        
        # Select strategy
        result = await integration.select_strategy_for_signal(
            symbol="BTCUSDT",
            signal=signal,
            market_data=market_data,
        )
        
        print(f"Selected: {result}")
        print(f"Regime: {result.regime.value}")
        print(f"Strategy: {result.strategy.name}")
        print(f"TP/SL Config: {result.tpsl_config}")
        print()
        
        # Simulate trade close with reward
        await integration.update_strategy_reward(
            symbol="BTCUSDT",
            realized_r=3.5,
            trade_meta={"pnl": 85.0},
        )
        
        print("\nMetrics:")
        print(integration.get_metrics())
    
    asyncio.run(demo())
