"""
Smart Position Sizer - Rule-based position sizing with instant logic

Provides intelligent position sizing based on:
1. Volatility-based sizing
2. Trend-strength filter
3. Win rate adjustment
4. Market regime detection
5. Correlation filter

NO TRAINING REQUIRED - Works from day 1!
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import math

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result from position sizing calculation."""
    size_usd: float
    size_pct: float
    leverage: float
    tp_pct: float
    sl_pct: float
    confidence: float
    reasoning: str
    adjustments: List[str]


class SmartPositionSizer:
    """
    Smart Position Sizer with rule-based logic.
    
    Replaces random RL decisions with intelligent sizing based on:
    - Market volatility
    - Trend strength
    - Recent win rate
    - Market regime
    - Position correlations
    """
    
    def __init__(
        self,
        base_position_size: float = 300.0,
        max_leverage: float = 5.0,
        track_last_n_trades: int = 10
    ):
        """
        Initialize Smart Position Sizer.
        
        Args:
            base_position_size: Base position size in USD
            max_leverage: Maximum leverage allowed
            track_last_n_trades: Number of recent trades to track for win rate
        """
        self.base_position_size = base_position_size
        self.max_leverage = max_leverage
        self.track_last_n_trades = track_last_n_trades
        
        # Track recent trades for win rate calculation
        self.recent_trades = deque(maxlen=track_last_n_trades)
        
        # Track open positions for correlation check
        self.open_positions: Dict[str, str] = {}  # symbol -> side (LONG/SHORT)
        
        logger.info(
            f"✅ SmartPositionSizer initialized: "
            f"base_size=${base_position_size}, max_lev={max_leverage}x, "
            f"tracking last {track_last_n_trades} trades"
        )
    
    def calculate_optimal_size(
        self,
        symbol: str,
        side: str,
        volatility: float,
        trend_strength: float,
        regime: str = "UNKNOWN",
        market_data: Optional[Dict] = None
    ) -> SizingResult:
        """
        Calculate optimal position size using smart rules.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            side: "LONG" or "SHORT"
            volatility: Current volatility (0.0-1.0, e.g., 0.05 = 5%)
            trend_strength: Trend strength (0.0-1.0)
            regime: Market regime ("TRENDING", "RANGING", "HIGH_VOLATILITY", "BREAKOUT")
            market_data: Optional additional market data
        
        Returns:
            SizingResult with size, TP, SL, and reasoning
        """
        adjustments = []
        reasoning_parts = []
        
        # Start with base size
        size_multiplier = 1.0
        confidence = 0.5  # Base confidence
        
        # ===================================================================
        # RULE 1: VOLATILITY-BASED SIZING
        # ===================================================================
        if volatility > 0.05:  # High volatility (>5%)
            vol_multiplier = max(0.3, 1.0 - (volatility - 0.05) * 5)
            size_multiplier *= vol_multiplier
            confidence -= 0.2
            adjustments.append(f"High volatility ({volatility*100:.1f}%) → reduce to {vol_multiplier*100:.0f}%")
            reasoning_parts.append("high volatility")
        elif volatility < 0.02:  # Low volatility (<2%)
            size_multiplier *= 1.0
            confidence += 0.1
            adjustments.append(f"Low volatility ({volatility*100:.1f}%) → full size OK")
            reasoning_parts.append("low volatility")
        else:  # Medium volatility (2-5%)
            vol_multiplier = 0.75
            size_multiplier *= vol_multiplier
            adjustments.append(f"Medium volatility ({volatility*100:.1f}%) → reduce to 75%")
        
        # ===================================================================
        # RULE 2: TREND-STRENGTH FILTER
        # ===================================================================
        if trend_strength > 0.7:  # Strong trend
            trend_bonus = 1.25
            size_multiplier *= trend_bonus
            confidence += 0.2
            adjustments.append(f"Strong trend ({trend_strength:.2f}) → +25% size bonus")
            reasoning_parts.append("strong trend")
        elif trend_strength < 0.3:  # Weak/ranging
            trend_penalty = 0.5
            size_multiplier *= trend_penalty
            confidence -= 0.2
            adjustments.append(f"Weak trend ({trend_strength:.2f}) → -50% penalty")
            reasoning_parts.append("weak trend")
        else:  # Moderate trend
            adjustments.append(f"Moderate trend ({trend_strength:.2f}) → no adjustment")
        
        # ===================================================================
        # RULE 3: WIN RATE ADJUSTMENT
        # ===================================================================
        recent_win_rate = self._calculate_recent_win_rate()
        
        if recent_win_rate is not None:
            if recent_win_rate < 0.3:  # <30% - STOP TRADING!
                logger.warning(
                    f"⚠️ Win rate critically low ({recent_win_rate*100:.1f}%) - "
                    f"BLOCKING trade to prevent further losses!"
                )
                return SizingResult(
                    size_usd=0.0,
                    size_pct=0.0,
                    leverage=0.0,
                    tp_pct=0.0,
                    sl_pct=0.0,
                    confidence=0.0,
                    reasoning=f"Win rate too low ({recent_win_rate*100:.0f}%) - emergency stop",
                    adjustments=["🚫 TRADE BLOCKED - Win rate < 30%"]
                )
            elif recent_win_rate < 0.4:  # 30-40% - Defensive
                wr_penalty = 0.5
                size_multiplier *= wr_penalty
                confidence -= 0.2
                adjustments.append(f"Losing streak ({recent_win_rate*100:.0f}%) → -50% defensive sizing")
                reasoning_parts.append("losing streak")
            elif recent_win_rate > 0.6:  # >60% - Aggressive
                wr_bonus = 1.25
                size_multiplier *= wr_bonus
                confidence += 0.2
                adjustments.append(f"Hot streak ({recent_win_rate*100:.0f}%) → +25% aggressive sizing")
                reasoning_parts.append("hot streak")
            else:  # 40-60% - Normal
                adjustments.append(f"Normal win rate ({recent_win_rate*100:.0f}%) → no adjustment")
        else:
            adjustments.append("No recent trades → using base sizing")
        
        # ===================================================================
        # RULE 4: MARKET REGIME DETECTION
        # ===================================================================
        tp_pct = 0.06  # Default TP
        sl_pct = 0.025  # Default SL
        
        if regime == "TRENDING":
            tp_pct = 0.10
            sl_pct = 0.02
            confidence += 0.1
            adjustments.append("TRENDING regime → TP=10%, SL=2% (let it run)")
        elif regime == "RANGING":
            tp_pct = 0.03
            sl_pct = 0.015
            size_multiplier *= 0.7  # Reduce size in ranging markets
            confidence -= 0.1
            adjustments.append("RANGING regime → TP=3%, SL=1.5% (scalp mode)")
            reasoning_parts.append("ranging market")
        elif regime == "HIGH_VOLATILITY":
            tp_pct = 0.08
            sl_pct = 0.04
            adjustments.append("HIGH_VOLATILITY regime → TP=8%, SL=4% (wider stops)")
        elif regime == "BREAKOUT":
            tp_pct = 0.12
            sl_pct = 0.03
            size_multiplier *= 1.1
            confidence += 0.15
            adjustments.append("BREAKOUT regime → TP=12%, SL=3% (momentum play)")
            reasoning_parts.append("breakout")
        else:
            adjustments.append(f"Unknown regime ({regime}) → using default TP/SL")
        
        # ===================================================================
        # RULE 5: CORRELATION FILTER
        # ===================================================================
        correlation_penalty = self._check_position_correlation(symbol, side)
        if correlation_penalty < 1.0:
            size_multiplier *= correlation_penalty
            confidence -= 0.1
            adjustments.append(
                f"Correlation risk → reduce by {(1-correlation_penalty)*100:.0f}% "
                f"(avoid overexposure)"
            )
            reasoning_parts.append("correlation risk")
        
        # ===================================================================
        # CALCULATE FINAL SIZE
        # ===================================================================
        final_size_usd = self.base_position_size * size_multiplier
        final_size_pct = size_multiplier
        
        # Cap at reasonable limits
        final_size_usd = max(10.0, min(final_size_usd, self.base_position_size * 2.0))
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Build reasoning string
        if reasoning_parts:
            reasoning = "Adjustments: " + ", ".join(reasoning_parts)
        else:
            reasoning = "Standard sizing"
        
        result = SizingResult(
            size_usd=final_size_usd,
            size_pct=final_size_pct,
            leverage=self.max_leverage,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            confidence=confidence,
            reasoning=reasoning,
            adjustments=adjustments
        )
        
        logger.info(
            f"💡 SMART SIZING: {symbol} {side}\n"
            f"   Size: ${final_size_usd:.2f} ({final_size_pct*100:.0f}%), "
            f"TP={tp_pct*100:.1f}%, SL={sl_pct*100:.2f}%\n"
            f"   Confidence: {confidence*100:.0f}%\n"
            f"   Reasoning: {reasoning}\n"
            f"   Adjustments:\n" +
            "\n".join(f"     • {adj}" for adj in adjustments)
        )
        
        return result
    
    def _calculate_recent_win_rate(self) -> Optional[float]:
        """
        Calculate win rate from recent trades.
        
        Returns:
            Win rate (0.0-1.0) or None if no recent trades
        """
        if not self.recent_trades:
            return None
        
        wins = sum(1 for trade in self.recent_trades if trade['win'])
        return wins / len(self.recent_trades)
    
    def _check_position_correlation(self, symbol: str, side: str) -> float:
        """
        Check if opening this position would create correlation risk.
        
        Returns:
            Multiplier (0.5-1.0) where lower means higher correlation risk
        """
        if not self.open_positions:
            return 1.0
        
        # Define correlation groups (simplified)
        correlation_groups = {
            'BTC': ['BTCUSDT', 'BTCUSD', 'BTCBUSD'],
            'ETH': ['ETHUSDT', 'ETHUSD', 'ETHBUSD'],
            'BNB': ['BNBUSDT', 'BNBUSD', 'BNBBUSD'],
            'MAJORS': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'ALTS_LAYER1': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSD'],
            'ALTS_DEFI': ['UNIUSDT', 'AAVEUSDT', 'LINKUSDT'],
            'MEME': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'],
        }
        
        # Find which group this symbol belongs to
        symbol_group = None
        for group_name, symbols in correlation_groups.items():
            if symbol in symbols:
                symbol_group = group_name
                break
        
        if not symbol_group:
            return 1.0  # Unknown symbol, no penalty
        
        # Count positions in same group with same direction
        same_group_same_direction = 0
        same_group_opposite_direction = 0
        
        for open_symbol, open_side in self.open_positions.items():
            for group_name, symbols in correlation_groups.items():
                if open_symbol in symbols and group_name == symbol_group:
                    if open_side == side:
                        same_group_same_direction += 1
                    else:
                        same_group_opposite_direction += 1
        
        # Calculate penalty
        if same_group_same_direction >= 3:
            return 0.3  # Severe overexposure - 70% reduction
        elif same_group_same_direction == 2:
            return 0.5  # High correlation - 50% reduction
        elif same_group_same_direction == 1:
            return 0.75  # Moderate correlation - 25% reduction
        elif same_group_opposite_direction >= 1:
            return 0.8  # Hedge position but still some correlation
        else:
            return 1.0  # No correlation issue
    
    def update_trade_outcome(self, symbol: str, win: bool, pnl_usd: float):
        """
        Update trade history for win rate tracking.
        
        Args:
            symbol: Trading symbol
            win: True if trade was profitable
            pnl_usd: PnL in USD
        """
        self.recent_trades.append({
            'symbol': symbol,
            'win': win,
            'pnl_usd': pnl_usd
        })
        
        win_rate = self._calculate_recent_win_rate()
        logger.info(
            f"📊 Trade outcome recorded: {symbol} {'WIN' if win else 'LOSS'} "
            f"(${pnl_usd:+.2f}) | "
            f"Recent win rate: {win_rate*100:.0f}% "
            f"({len(self.recent_trades)} trades)"
        )
    
    def add_open_position(self, symbol: str, side: str):
        """Track a new open position for correlation checks."""
        self.open_positions[symbol] = side
        logger.debug(f"Position opened: {symbol} {side} (total: {len(self.open_positions)})")
    
    def remove_open_position(self, symbol: str):
        """Remove a closed position from tracking."""
        if symbol in self.open_positions:
            side = self.open_positions.pop(symbol)
            logger.debug(f"Position closed: {symbol} {side} (remaining: {len(self.open_positions)})")
    
    def get_stats(self) -> Dict:
        """
        Get current statistics.
        
        Returns:
            Dict with recent win rate, open positions count, etc.
        """
        win_rate = self._calculate_recent_win_rate()
        
        return {
            'recent_win_rate': win_rate,
            'recent_trades_count': len(self.recent_trades),
            'open_positions_count': len(self.open_positions),
            'open_positions': list(self.open_positions.items()),
            'base_size': self.base_position_size,
            'max_leverage': self.max_leverage,
        }


# Global instance
_smart_sizer_instance: Optional[SmartPositionSizer] = None


def get_smart_position_sizer() -> SmartPositionSizer:
    """
    Get or create global SmartPositionSizer instance.
    
    Returns:
        SmartPositionSizer instance
    """
    global _smart_sizer_instance
    
    if _smart_sizer_instance is None:
        _smart_sizer_instance = SmartPositionSizer()
    
    return _smart_sizer_instance
