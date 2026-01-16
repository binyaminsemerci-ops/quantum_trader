#!/usr/bin/env python3
"""
🤖 Reinforcement Learning Position Sizing Agent

Learns optimal position sizes and leverage based on:
- Market regime (volatility, trend, liquidity)
- Signal quality (confidence, consensus, model agreement)
- Portfolio state (exposure, correlation, drawdown)
- Historical performance (win rate, profit factor, Sharpe ratio)

Uses Q-learning with continuous state discretization.

🧮 ENHANCED WITH TRADING MATHEMATICIAN AI:
- Automatic optimal parameter calculation
- Kelly Criterion for position sizing
- ATR-based TP/SL optimization
- No manual adjustments needed!

🔧 SPRINT 1 - D1: PolicyStore Integration
- Reads risk limits from PolicyStore (single source of truth)
- No more hardcoded ENV variables for leverage/risk
"""
import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

# Import Trading Mathematician
try:
    from backend.services.ai.trading_mathematician import (
        TradingMathematician,
        AccountState,
        MarketConditions,
        PerformanceMetrics,
    )
    MATH_AI_AVAILABLE = True
    logger.info("Trading Mathematician AI loaded - AUTONOMOUS MODE ENABLED")
except ImportError as e:
    MATH_AI_AVAILABLE = False
    logger.warning(f"⚠️  Trading Mathematician not available: {e}")


class MarketRegime(str, Enum):
    """Market regime classification"""
    HIGH_VOL_TRENDING = "high_vol_trending"      # High volatility + strong trend
    LOW_VOL_TRENDING = "low_vol_trending"        # Low volatility + strong trend
    HIGH_VOL_RANGING = "high_vol_ranging"        # High volatility + ranging
    LOW_VOL_RANGING = "low_vol_ranging"          # Low volatility + ranging
    CHOPPY = "choppy"                            # Unstable, whipsaw conditions
    UNKNOWN = "unknown"


class ConfidenceBucket(str, Enum):
    """Signal confidence buckets"""
    VERY_HIGH = "very_high"  # >= 85%
    HIGH = "high"            # 70-85%
    MEDIUM = "medium"        # 55-70%
    LOW = "low"              # 45-55%
    VERY_LOW = "very_low"    # < 45%


class PortfolioState(str, Enum):
    """Portfolio exposure state"""
    LIGHT = "light"          # < 30% exposure
    MODERATE = "moderate"    # 30-60% exposure
    HEAVY = "heavy"          # 60-80% exposure
    MAX = "max"              # >= 80% exposure


@dataclass
class SizingDecision:
    """Position sizing decision from RL agent with TP/SL management"""
    position_size_usd: float
    leverage: float
    risk_pct: float
    confidence: float
    reasoning: str
    state_key: str
    action_key: str  # 🎯 CRITICAL: Must be included for RL learning
    q_value: float
    # 🔥 NEW: TP/SL Management
    tp_percent: float  # Take profit as percentage (e.g., 0.06 = 6%)
    sl_percent: float  # Stop loss as percentage (e.g., 0.025 = 2.5%)
    partial_tp_enabled: bool  # Whether to use partial TP
    partial_tp_percent: float  # First partial TP target (e.g., 0.03 = 3%)
    partial_tp_size: float  # Size to close at first TP (e.g., 0.5 = 50%)


@dataclass
class SizingOutcome:
    """Outcome of a sizing decision (for learning)"""
    state_key: str
    action_key: str
    reward: float
    pnl_pct: float
    duration_hours: float
    max_drawdown_pct: float
    timestamp: datetime


class RLPositionSizingAgent:
    """
    Reinforcement Learning agent for position sizing AND TP/SL management
    
    🔥 UNIFIED SYSTEM - All trade parameters decided by RL:
    - Position size
    - Leverage
    - Take Profit levels (full + partial)
    - Stop Loss
    
    State Space:
    - Market regime (5 buckets)
    - Signal confidence (5 buckets)
    - Portfolio exposure (4 buckets)
    - Recent performance (3 buckets: good/neutral/bad)
    
    Action Space:
    - Position size: [10%, 25%, 50%, 75%, 100%] of max
    - Leverage: [1x, 2x, 3x, 4x, 5x]
    - TP Strategy: [Conservative (5%/10%), Balanced (6%/12%), Aggressive (8%/15%)]
    - SL Strategy: [Tight (1.5%), Medium (2.5%), Wide (3.5%)]
    
    Total: 5 × 5 × 4 × 3 = 300 states × (25 size/lev × 3 TP × 3 SL) = 67,500 state-action pairs
    """
    
    def __init__(
        self,
        policy_store=None,  # 🔧 SPRINT 1 - D1: PolicyStore injection
        state_file: str = "data/rl_position_sizing_state.json",
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        exploration_rate: float = 0.10,  # 🎯 REDUCED from 0.50 to 0.10 - 90% exploit (full size), 10% explore
        min_position_usd: float = 10.0,
        max_position_usd: float = 8000.0,  # 🔥 INCREASED from $1000 to $8000 (80% of $10K balance)
        min_leverage: float = 15.0,        # 🔥 Minimum 15x
        max_leverage: float = 25.0,        # 🔥 Maximum 25x (fallback if PolicyStore unavailable)
        use_math_ai: bool = True,  # 🧮 NEW: Enable Trading Mathematician
    ):
        self.policy_store = policy_store  # 🔧 SPRINT 1 - D1
        self.state_file = Path(state_file)
        self.learning_rate = learning_rate  # alpha
        self.discount_factor = discount_factor  # gamma
        self.exploration_rate = exploration_rate  # epsilon
        
        self.min_position_usd = min_position_usd
        self.max_position_usd = max_position_usd
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage  # Fallback default
        
        # 🧮 Initialize Trading Mathematician if enabled
        self.use_math_ai = use_math_ai and MATH_AI_AVAILABLE
        if self.use_math_ai:
            # 🔧 SPRINT 1 - D1: Read from PolicyStore if available, else ENV fallback
            if self.policy_store:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    env_max_leverage = loop.run_until_complete(
                        self.policy_store.get_value("risk.max_leverage", 5.0)
                    )
                    env_risk_per_trade = loop.run_until_complete(
                        self.policy_store.get_value("risk.max_risk_pct_per_trade", 0.02)
                    )
                    logger.info(
                        f"🔧 PolicyStore Config: MAX_LEVERAGE={env_max_leverage}x, "
                        f"RISK={env_risk_per_trade*100:.1f}%"
                    )
                except Exception as e:
                    logger.warning(f"Failed to read from PolicyStore, using ENV: {e}")
                    env_max_leverage = float(os.getenv("RM_MAX_LEVERAGE", str(self.max_leverage)))
                    env_risk_per_trade = float(os.getenv("RM_RISK_PER_TRADE_PCT", "0.02"))
            else:
                # Fallback to ENV if no PolicyStore
                env_max_leverage = float(os.getenv("RM_MAX_LEVERAGE", str(self.max_leverage)))
                env_risk_per_trade = float(os.getenv("RM_RISK_PER_TRADE_PCT", "0.02"))
                logger.info(
                    f"📄 ENV Config (no PolicyStore): MAX_LEVERAGE={env_max_leverage}x, "
                    f"RISK={env_risk_per_trade*100:.1f}%"
                )
            
            # 🔧 SPRINT 1 - D1: Update max_leverage from PolicyStore/ENV
            self.max_leverage = env_max_leverage
            
            self.math_ai = TradingMathematician(
                risk_per_trade_pct=env_risk_per_trade,
                target_profit_pct=0.20,    # 20% daily target
                min_risk_reward=2.0,
                safety_cap=env_max_leverage,  # 🎯 Kelly safety cap FROM POLICYSTORE/ENV!
                conservative_mode=False,
            )
            logger.info(
                f"🧮 Math AI ENABLED - Kelly Criterion with {env_max_leverage}x SAFETY CAP "
                f"({env_risk_per_trade*100:.1f}% capital)"
            )
        else:
            # 🔧 SPRINT 1 - D1: Even without Math AI, read max_leverage from PolicyStore
            if self.policy_store:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    self.max_leverage = loop.run_until_complete(
                        self.policy_store.get_value("risk.max_leverage", self.max_leverage)
                    )
                    logger.info(f"🔧 PolicyStore (no Math AI): MAX_LEVERAGE={self.max_leverage}x")
                except Exception as e:
                    logger.warning(f"Failed PolicyStore read: {e}, using default {self.max_leverage}x")
            
            self.math_ai = None
            if not MATH_AI_AVAILABLE:
                logger.warning("⚠️  Math AI not available, using RL-only mode")
            else:
                logger.info("ℹ️  Math AI disabled, using RL-only mode")
        
        # Initialize Q-table for all state-action pairs
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.outcomes: list[SizingOutcome] = []
        self.recent_win_rate = 0.5  # Start neutral
        
        # 🆕 Bootstrap with estimated historical performance
        self._bootstrap_trade_history()
        
        # Action space
        # Calculate min multiplier based on min/max position
        min_mult = self.min_position_usd / self.max_position_usd  # ~0.01 for 10/1000
        self.size_multipliers = [
            min_mult,      # Minimum position ($10)
            0.1,           # 10% of max ($100)
            0.25,          # 25% of max ($250)
            0.5,           # 50% of max ($500)
            1.0            # 100% of max ($1000)
        ]
        self.leverages = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # 🔥 NEW: TP/SL Action Space
        # Format: (full_tp_pct, partial_tp_pct, sl_pct, partial_enabled)
        self.tpsl_strategies = {
            'conservative': {
                'full_tp': 0.025,     # 2.5% full TP (tighter for faster closes)
                'partial_tp': 0.0125, # 1.25% first TP (take 50% profit)
                'sl': 0.01,           # 1% stop loss
                'partial_size': 0.5,  # Close 50% at first TP
                'partial_enabled': True
            },
            'balanced': {
                'full_tp': 0.03,      # 3% full TP (reduced from 6%)
                'partial_tp': 0.015,  # 1.5% first TP
                'sl': 0.015,          # 1.5% stop loss (reduced from 2.5%)
                'partial_size': 0.5,
                'partial_enabled': True
            },
            'aggressive': {
                'full_tp': 0.04,      # 4% full TP (reduced from 8%)
                'partial_tp': 0.02,   # 2% first TP (reduced from 4%)
                'sl': 0.02,           # 2% stop loss (reduced from 3.5%)
                'partial_size': 0.5,
                'partial_enabled': True
            }
        }
        
        # Load state first (this may override exploration_rate)
        self._load_state()
        
        # Log AFTER loading state to show actual values used
        logger.info(
            f"[RL-SIZING] 🤖 Initialized: "
            f"alpha={self.learning_rate}, gamma={self.discount_factor}, epsilon={self.exploration_rate}, "
            f"position range=${self.min_position_usd}-${self.max_position_usd}, leverage {self.min_leverage}x-{self.max_leverage}x"
        )
    
    def _load_state(self):
        """Load Q-table and outcomes from disk"""
        # 🎯 CRITICAL: ALWAYS force exploration_rate to 0.10
        # Do this BEFORE loading state file to ensure it's always applied
        self.exploration_rate = 0.10
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                self.q_table = data.get('q_table', {})
                
                # Load outcomes
                outcomes_data = data.get('outcomes', [])
                self.outcomes = [
                    SizingOutcome(
                        state_key=o['state_key'],
                        action_key=o['action_key'],
                        reward=o['reward'],
                        pnl_pct=o['pnl_pct'],
                        duration_hours=o['duration_hours'],
                        max_drawdown_pct=o['max_drawdown_pct'],
                        timestamp=datetime.fromisoformat(o['timestamp'])
                    )
                    for o in outcomes_data
                ]
                
                # Calculate recent win rate
                if len(self.outcomes) >= 10:
                    recent_outcomes = self.outcomes[-20:]
                    wins = sum(1 for o in recent_outcomes if o.reward > 0)
                    self.recent_win_rate = wins / len(recent_outcomes)
                
                logger.info(
                    f"[RL-SIZING] 📂 Loaded state: "
                    f"{len(self.q_table)} states, {len(self.outcomes)} outcomes, "
                    f"win_rate={self.recent_win_rate:.1%}, epsilon=0.10 (forced)"
                )
            except Exception as e:
                logger.warning(f"[RL-SIZING] Failed to load state: {e}")
                self.q_table = {}
                self.outcomes = []
        
        # 🔥 BOOST Q-values for larger sizes to encourage aggressive trading
        self._boost_large_size_q_values()
    
    def _boost_large_size_q_values(self):
        """Manually boost Q-values for actions with larger position sizes.
        
        This encourages the RL agent to explore larger positions more often.
        Only boosts actions with size_mult >= 0.25 (i.e., $75+ positions).
        """
        boost_count = 0
        boost_amount = 0.15  # Add +0.15 to Q-value
        
        # Iterate all states and actions in Q-table
        for state_key in list(self.q_table.keys()):
            for action_key in list(self.q_table[state_key].keys()):
                # Parse action_key: "size_0.25_lev_2.0"
                try:
                    parts = action_key.split('_')
                    size_mult = float(parts[1])
                    
                    # Boost actions with size >= 0.25 ($75+)
                    if size_mult >= 0.25:
                        old_q = self.q_table[state_key][action_key]
                        self.q_table[state_key][action_key] += boost_amount
                        boost_count += 1
                except (IndexError, ValueError):
                    pass  # Skip invalid action keys
        
        if boost_count > 0:
            logger.info(
                f"[RL-SIZING] 🚀 Boosted {boost_count} Q-values by +{boost_amount} "
                f"for larger positions (size >= 0.25)"
            )
    
    def _save_state(self):
        """Save Q-table and outcomes to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'q_table': self.q_table,
                'outcomes': [
                    {
                        'state_key': o.state_key,
                        'action_key': o.action_key,
                        'reward': o.reward,
                        'pnl_pct': o.pnl_pct,
                        'duration_hours': o.duration_hours,
                        'max_drawdown_pct': o.max_drawdown_pct,
                        'timestamp': o.timestamp.isoformat()
                    }
                    for o in self.outcomes[-1000:]  # Keep last 1000
                ],
                'metadata': {
                    'learning_rate': self.learning_rate,
                    'exploration_rate': self.exploration_rate,
                    'recent_win_rate': self.recent_win_rate,
                    'total_updates': len(self.outcomes),
                    'last_updated': datetime.now(timezone.utc).isoformat()
                }
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[RL-SIZING] 💾 Saved state: {len(self.outcomes)} outcomes")
        except Exception as e:
            logger.error(f"[RL-SIZING] Failed to save state: {e}")
    
    def _classify_regime(
        self,
        atr_pct: float,
        adx: Optional[float] = None,
        trend_strength: Optional[float] = None
    ) -> MarketRegime:
        """Classify market regime based on volatility and trend"""
        # High volatility threshold
        high_vol = atr_pct > 0.03  # 3%+ ATR
        
        # Trend strength (use ADX if available, else use trend_strength)
        if adx is not None:
            strong_trend = adx > 25
        elif trend_strength is not None:
            strong_trend = abs(trend_strength) > 0.5
        else:
            strong_trend = False
        
        if high_vol and strong_trend:
            return MarketRegime.HIGH_VOL_TRENDING
        elif not high_vol and strong_trend:
            return MarketRegime.LOW_VOL_TRENDING
        elif high_vol and not strong_trend:
            return MarketRegime.HIGH_VOL_RANGING
        elif not high_vol and not strong_trend:
            return MarketRegime.LOW_VOL_RANGING
        else:
            return MarketRegime.CHOPPY
    
    def _classify_confidence(self, confidence: float) -> ConfidenceBucket:
        """Classify signal confidence into buckets"""
        if confidence >= 0.85:
            return ConfidenceBucket.VERY_HIGH
        elif confidence >= 0.70:
            return ConfidenceBucket.HIGH
        elif confidence >= 0.55:
            return ConfidenceBucket.MEDIUM
        elif confidence >= 0.45:
            return ConfidenceBucket.LOW
        else:
            return ConfidenceBucket.VERY_LOW
    
    def _classify_portfolio(self, exposure_pct: float) -> PortfolioState:
        """Classify portfolio exposure"""
        if exposure_pct >= 0.80:
            return PortfolioState.MAX
        elif exposure_pct >= 0.60:
            return PortfolioState.HEAVY
        elif exposure_pct >= 0.30:
            return PortfolioState.MODERATE
        else:
            return PortfolioState.LIGHT
    
    def _classify_performance(self) -> str:
        """Classify recent performance"""
        if self.recent_win_rate >= 0.60:
            return "good"
        elif self.recent_win_rate >= 0.45:
            return "neutral"
        else:
            return "bad"
    
    def _get_state_key(
        self,
        regime: MarketRegime,
        confidence: ConfidenceBucket,
        portfolio: PortfolioState,
        performance: str
    ) -> str:
        """Generate state key"""
        return f"{regime.value}|{confidence.value}|{portfolio.value}|{performance}"
    
    def _get_action_key(self, size_mult: float, leverage: float, tpsl_strategy: str) -> str:
        """Generate action key including TP/SL strategy"""
        return f"size_{size_mult:.2f}|lev_{leverage:.1f}|tpsl_{tpsl_strategy}"
    
    def _get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for state-action pair"""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            # Initialize with optimistic values based on action
            parts = action_key.split('|')
            size_mult = float(parts[0].split('_')[1])
            leverage = float(parts[1].split('_')[1])
            tpsl_strategy = parts[2].split('_')[1] if len(parts) > 2 else 'balanced'
            
            # 🔥 OPTIMISTIC INITIALIZATION
            init_q = 0.0
            init_q += size_mult * 0.5              # Reward larger sizes (0 to +0.5)
            init_q += (leverage - 1.0) * 0.1       # Small bonus for leverage (0 to +0.4)
            init_q += 0.1 if 0.25 <= size_mult <= 0.75 else 0.0  # Bonus for balanced sizes
            
            # 🔥 NEW: TP/SL strategy bonus
            if tpsl_strategy == 'balanced':
                init_q += 0.2  # Prefer balanced by default
            elif tpsl_strategy == 'aggressive':
                init_q += 0.1  # Slight preference for aggressive
            
            self.q_table[state_key][action_key] = init_q
        
        return self.q_table[state_key][action_key]
    
    def _select_action(self, state_key: str) -> Tuple[float, float, str]:
        """
        Select action using epsilon-greedy policy.
        
        PATCH-P0-05: Conservative fallback for HIGH_VOL when Q≈0 (untrained state).
        
        Returns:
            (size_mult, leverage, tpsl_strategy)
        """
        import random
        
        # PATCH-P0-05: Check for HIGH_VOL regime with untrained Q-values
        is_high_vol = "high_vol" in state_key.lower()
        
        # Check if state is mostly untrained (avg Q-value near 0)
        if is_high_vol and state_key in self.q_table:
            q_values = list(self.q_table[state_key].values())
            if q_values:
                avg_q = sum(q_values) / len(q_values)
                
                # If Q-values are near zero (untrained), use conservative fallback
                if abs(avg_q) < 0.05:
                    logger.warning(
                        f"[RL-FALLBACK-HIGH-VOL] State={state_key}, avg_Q={avg_q:.3f} "
                        f"(untrained) → forcing CONSERVATIVE fallback (0.3x size, 1.0x lev)"
                    )
                    return (0.3, 1.0, 'conservative')  # Safe defaults for HIGH_VOL
        
        # Exploration: random action
        if random.random() < self.exploration_rate:
            size_mult = random.choice(self.size_multipliers)
            leverage = random.choice(self.leverages)
            tpsl_strategy = random.choice(list(self.tpsl_strategies.keys()))
            logger.debug(
                f"[RL-SIZING] 🎲 Exploring: size={size_mult:.1%}, lev={leverage:.1f}x, tpsl={tpsl_strategy}"
            )
            return size_mult, leverage, tpsl_strategy
        
        # Exploitation: best action
        best_q = float('-inf')
        best_action = (0.5, 2.0, 'balanced')  # Default: moderate
        
        for size_mult in self.size_multipliers:
            for leverage in self.leverages:
                for tpsl_strategy in self.tpsl_strategies.keys():
                    action_key = self._get_action_key(size_mult, leverage, tpsl_strategy)
                    q_value = self._get_q_value(state_key, action_key)
                    
                    if q_value > best_q:
                        best_q = q_value
                        best_action = (size_mult, leverage, tpsl_strategy)
        
        # PATCH-P0-05: Additional safety check - if best Q-value is near zero in HIGH_VOL, use conservative
        if is_high_vol and abs(best_q) < 0.05:
            logger.warning(
                f"[RL-FALLBACK-HIGH-VOL] Best Q={best_q:.3f} near zero in HIGH_VOL → "
                f"forcing CONSERVATIVE (0.3x size, 1.0x lev)"
            )
            return (0.3, 1.0, 'conservative')
        
        logger.debug(
            f"[RL-SIZING] 🎯 Exploiting: size={best_action[0]:.1%}, lev={best_action[1]:.1f}x, "
            f"tpsl={best_action[2]} (Q={best_q:.3f})"
        )
        return best_action
    
    def _bootstrap_trade_history(self):
        """
        Bootstrap trade history with estimated outcomes.
        Uses assumed 60% win rate to create synthetic history for Kelly Criterion.
        """
        # Create 25 synthetic trades (20 more than minimum for Kelly)
        # Assume 60% win rate, 3% avg win, 2% avg loss
        for i in range(25):
            is_win = i % 10 < 6  # 60% win rate
            reward = 0.03 if is_win else -0.02
            pnl_pct = 0.03 if is_win else -0.02
            
            outcome = SizingOutcome(
                state_key="bootstrap",
                action_key=f"bootstrap_{i}",
                reward=reward,
                pnl_pct=pnl_pct,
                duration_hours=2.0,
                max_drawdown_pct=0.01,
                timestamp=datetime.utcnow(),
            )
            self.outcomes.append(outcome)
        
        logger.info(f"📊 Bootstrapped {len(self.outcomes)} trades for Kelly Criterion (60% WR estimated)")
    
    def decide_sizing(
        self,
        symbol: str,
        confidence: float,
        atr_pct: float,
        current_exposure_pct: float,
        equity_usd: float,
        adx: Optional[float] = None,
        trend_strength: Optional[float] = None
    ) -> SizingDecision:
        """
        Decide optimal position size, leverage, AND TP/SL levels
        
        🧮 MATH AI MODE: Calculates everything automatically!
        🔥 RL MODE: Q-learning based decisions
        
        Args:
            symbol: Trading symbol
            confidence: Signal confidence (0-1)
            atr_pct: ATR as % of price
            current_exposure_pct: Current portfolio exposure (0-1)
            equity_usd: Total equity in USD
            adx: Average Directional Index (trend strength)
            trend_strength: Alternative trend strength metric
        
        Returns:
            SizingDecision with position size, leverage, TP/SL, and reasoning
        """
        # 🧮 MATH AI MODE: Calculate optimal parameters
        if self.use_math_ai and self.math_ai:
            try:
                logger.info(f"🧮 Math AI calculating optimal parameters for {symbol}...")
                
                # Get current open positions count
                # TODO: Query from database
                open_positions = int(current_exposure_pct * 15)  # Estimate based on exposure
                
                # Create account state
                account = AccountState(
                    balance=equity_usd,
                    equity=equity_usd,
                    margin_used=equity_usd * current_exposure_pct,
                    open_positions=open_positions,
                    max_positions=15,
                )
                
                # Create market conditions
                market = MarketConditions(
                    symbol=symbol,
                    atr_pct=atr_pct,
                    daily_volatility=atr_pct * 2.5,  # Rough estimate: daily vol ~2.5x ATR
                    trend_strength=trend_strength if trend_strength else 0.7,
                    liquidity_score=0.9,  # Default high liquidity for major pairs
                )
                
                # Calculate performance metrics from RL history
                total_trades = len(self.outcomes)
                if total_trades > 0:
                    wins = sum(1 for o in self.outcomes if o.reward > 0)
                    win_rate = wins / total_trades
                    
                    # Calculate average win/loss percentages
                    winning_outcomes = [o for o in self.outcomes if o.reward > 0]
                    losing_outcomes = [o for o in self.outcomes if o.reward < 0]
                    
                    avg_win_pct = sum(o.reward for o in winning_outcomes) / len(winning_outcomes) if winning_outcomes else 0.03
                    avg_loss_pct = abs(sum(o.reward for o in losing_outcomes) / len(losing_outcomes)) if losing_outcomes else 0.015
                    
                    # Profit factor
                    total_wins = sum(o.reward for o in winning_outcomes)
                    total_losses = abs(sum(o.reward for o in losing_outcomes))
                    profit_factor = total_wins / total_losses if total_losses > 0 else 1.5
                else:
                    # Defaults for new agent
                    win_rate = 0.55
                    avg_win_pct = 0.03
                    avg_loss_pct = 0.015
                    profit_factor = 1.5
                
                performance = PerformanceMetrics(
                    total_trades=total_trades,
                    win_rate=win_rate,
                    avg_win_pct=avg_win_pct,
                    avg_loss_pct=avg_loss_pct,
                    profit_factor=profit_factor,
                    sharpe_ratio=1.0,  # TODO: Calculate from outcomes
                )
                
                # Calculate optimal parameters WITH signal confidence
                optimal = self.math_ai.calculate_optimal_parameters(
                    account, market, performance, confidence  # PASS CONFIDENCE!
                )
                
                # Apply Kelly Criterion if enough trade history
                if total_trades >= 20:
                    optimal = self.math_ai.adjust_for_kelly_criterion(optimal, performance)
                
                # 🔥 NO CAPS! Use Math AI values directly!
                position_size_usd = optimal.margin_usd  # Use full Math AI calculation
                leverage = optimal.leverage  # Use full Math AI leverage (25x)
                
                # Calculate risk %
                risk_pct = (position_size_usd / equity_usd) * (leverage / 5.0) * 0.01
                
                # Generate reasoning
                reasoning = (
                    f"🧮 MATH AI: ${optimal.margin_usd:.0f}@{optimal.leverage:.1f}x | "
                    f"TP={optimal.tp_pct*100:.2f}% SL={optimal.sl_pct*100:.2f}% | "
                    f"Expected: ${optimal.expected_profit_usd:.2f} | "
                    f"Confidence: {optimal.confidence_score:.2f} | "
                    f"History: {total_trades} trades, {win_rate*100:.1f}% WR"
                )
                
                logger.info(
                    f"[MATH-AI] 🧮 {symbol}: ${position_size_usd:.0f} @ {leverage:.1f}x "
                    f"| TP={optimal.tp_pct*100:.2f}% (partial@{optimal.partial_tp_pct*100:.2f}%), "
                    f"SL={optimal.sl_pct*100:.2f}% | Expected: ${optimal.expected_profit_usd:.2f} | "
                    f"{total_trades} trades, WR={win_rate*100:.1f}%"
                )
                
                # Store last state/action for RL learning (use dummy values for Math AI)
                self._last_state_key = "math_ai_mode"
                self._last_action_key = f"math_ai_{position_size_usd:.0f}_{leverage:.1f}x"
                
                # Return Math AI decision
                decision = SizingDecision(
                    position_size_usd=position_size_usd,
                    leverage=leverage,
                    risk_pct=risk_pct,
                    confidence=optimal.confidence_score,
                    reasoning=reasoning,
                    state_key="math_ai_mode",
                    action_key=f"math_ai_{position_size_usd:.0f}_{leverage:.1f}x",
                    q_value=0.0,  # Math AI doesn't use Q-values
                    tp_percent=optimal.tp_pct,
                    sl_percent=optimal.sl_pct,
                    partial_tp_enabled=True,
                    partial_tp_percent=optimal.partial_tp_pct,
                    partial_tp_size=0.5,  # Always take 50% at partial TP
                )
                logger.info(f"🔍 [DEBUG] Returning SizingDecision: leverage={decision.leverage}, position_size={decision.position_size_usd}")
                return decision
                
            except Exception as e:
                logger.error(f"❌ Math AI failed: {e}, falling back to RL mode", exc_info=True)
                # Fall through to RL mode
        
        # 🔥 STANDARD RL MODE: Q-learning based decisions
        # Classify state
        regime = self._classify_regime(atr_pct, adx, trend_strength)
        confidence_bucket = self._classify_confidence(confidence)
        portfolio_state = self._classify_portfolio(current_exposure_pct)
        performance = self._classify_performance()
        
        state_key = self._get_state_key(regime, confidence_bucket, portfolio_state, performance)
        
        # Select action (size multiplier, leverage, AND TP/SL strategy)
        size_mult, leverage, tpsl_strategy = self._select_action(state_key)
        
        # Calculate actual position size
        position_size_usd = self.max_position_usd * size_mult
        position_size_usd = max(self.min_position_usd, min(position_size_usd, self.max_position_usd))
        
        # Cap leverage
        leverage = max(self.min_leverage, min(leverage, self.max_leverage))
        
        # Calculate risk %
        risk_pct = (position_size_usd / equity_usd) * (leverage / 5.0) * 0.01
        
        # Get TP/SL parameters from selected strategy
        tpsl_params = self.tpsl_strategies[tpsl_strategy]
        
        # Get Q-value for logging
        action_key = self._get_action_key(size_mult, leverage, tpsl_strategy)
        q_value = self._get_q_value(state_key, action_key)
        
        # Generate reasoning
        reasoning = (
            f"Regime={regime.value}, Conf={confidence_bucket.value}, "
            f"Exposure={portfolio_state.value}, Perf={performance}, "
            f"Strategy={tpsl_strategy}, Q={q_value:.3f}"
        )
        
        logger.info(
            f"[RL-TPSL] 🤖 {symbol}: ${position_size_usd:.0f} @ {leverage:.1f}x "
            f"| TP={tpsl_params['full_tp']*100:.1f}% (partial@{tpsl_params['partial_tp']*100:.1f}%), "
            f"SL={tpsl_params['sl']*100:.1f}% | {tpsl_strategy.upper()} | Q={q_value:.3f}"
        )
        
        # Store last state/action for outcome tracking
        self._last_state_key = state_key
        self._last_action_key = action_key
        
        return SizingDecision(
            position_size_usd=position_size_usd,
            leverage=leverage,
            risk_pct=risk_pct,
            confidence=confidence,
            reasoning=reasoning,
            state_key=state_key,
            action_key=action_key,  # 🎯 CRITICAL: Include action_key for RL learning
            q_value=q_value,
            # 🔥 TP/SL Management
            tp_percent=tpsl_params['full_tp'],
            sl_percent=tpsl_params['sl'],
            partial_tp_enabled=tpsl_params['partial_enabled'],
            partial_tp_percent=tpsl_params['partial_tp'],
            partial_tp_size=tpsl_params['partial_size']
        )
    
    def update_from_outcome(
        self,
        state_key: str,
        action_key: str,
        pnl_pct: float,
        duration_hours: float,
        max_drawdown_pct: float
    ):
        """
        Update Q-table based on trade outcome
        
        Reward function:
        - Base reward: PnL %
        - Time penalty: -0.01 per 24h (encourage quick wins)
        - Drawdown penalty: -max_drawdown_pct (discourage risky trades)
        - Win bonus: +0.1 if profitable
        """
        # Calculate reward
        reward = pnl_pct
        
        # Time penalty (encourage efficiency)
        time_penalty = -(duration_hours / 24.0) * 0.01
        reward += time_penalty
        
        # Drawdown penalty (discourage risky drawdowns)
        drawdown_penalty = -abs(max_drawdown_pct) * 0.5
        reward += drawdown_penalty
        
        # Win bonus
        if pnl_pct > 0:
            reward += 0.1
        
        # Get current Q-value
        current_q = self._get_q_value(state_key, action_key)
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[R + γ*max(Q(s',a')) - Q(s,a)]
        # Simplified (no next state): Q(s,a) = Q(s,a) + α[R - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward - current_q)
        
        # Update Q-table
        self.q_table[state_key][action_key] = new_q
        
        # Store outcome
        outcome = SizingOutcome(
            state_key=state_key,
            action_key=action_key,
            reward=reward,
            pnl_pct=pnl_pct,
            duration_hours=duration_hours,
            max_drawdown_pct=max_drawdown_pct,
            timestamp=datetime.now(timezone.utc)
        )
        self.outcomes.append(outcome)
        
        # Update win rate
        if len(self.outcomes) >= 10:
            recent_outcomes = self.outcomes[-20:]
            wins = sum(1 for o in recent_outcomes if o.reward > 0)
            self.recent_win_rate = wins / len(recent_outcomes)
        
        logger.info(
            f"[RL-SIZING] 📈 Update: Q={current_q:.3f}→{new_q:.3f}, "
            f"R={reward:.3f} (PnL={pnl_pct:+.2%}, DD={max_drawdown_pct:.2%}, T={duration_hours:.1f}h), "
            f"WinRate={self.recent_win_rate:.1%}"
        )
        
        # Save state
        self._save_state()
    
    def get_statistics(self) -> Dict:
        """Get agent statistics"""
        if not self.outcomes:
            return {
                'total_updates': 0,
                'win_rate': 0.5,
                'avg_reward': 0.0,
                'avg_pnl_pct': 0.0,
                'states_explored': 0,
                'actions_explored': 0
            }
        
        recent = self.outcomes[-100:]
        
        wins = sum(1 for o in recent if o.pnl_pct > 0)
        avg_reward = sum(o.reward for o in recent) / len(recent)
        avg_pnl = sum(o.pnl_pct for o in recent) / len(recent)
        
        return {
            'total_updates': len(self.outcomes),
            'win_rate': wins / len(recent),
            'avg_reward': avg_reward,
            'avg_pnl_pct': avg_pnl,
            'states_explored': len(self.q_table),
            'actions_explored': sum(len(actions) for actions in self.q_table.values()),
            'recent_win_rate': self.recent_win_rate
        }


# Singleton instance
_rl_sizing_agent: Optional[RLPositionSizingAgent] = None


def get_rl_sizing_agent(
    enabled: bool = True,
    **kwargs
) -> Optional[RLPositionSizingAgent]:
    """Get or create RL sizing agent singleton"""
    global _rl_sizing_agent
    
    if not enabled:
        return None
    
    if _rl_sizing_agent is None:
        _rl_sizing_agent = RLPositionSizingAgent(**kwargs)
    
    return _rl_sizing_agent
