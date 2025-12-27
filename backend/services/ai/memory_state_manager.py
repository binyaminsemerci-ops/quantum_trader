"""
MEMORY STATE MANAGER - AI System Memory and Context Tracking

This module provides persistent memory for the AI trading system, tracking:
- Market regime states and transitions
- Performance metrics across regimes and symbols
- Pattern recognition and outcome history
- Confidence calibration and adjustment factors

Author: Quantum Trader AI Team
Date: 2025-11-26
Version: 1.0
"""

import json
import logging
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & TYPES
# ============================================================================

class MarketRegime(str, Enum):
    """Market regime classifications"""
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    UNKNOWN = "UNKNOWN"


class MemoryLevel(str, Enum):
    """Memory confidence levels"""
    HIGH = "HIGH"          # 200+ samples
    MEDIUM = "MEDIUM"      # 50-200 samples
    LOW = "LOW"            # 10-50 samples
    COLD_START = "COLD_START"  # < 10 samples


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RegimeState:
    """Current market regime state"""
    current_regime: MarketRegime
    regime_duration: int  # seconds
    regime_stability: float  # 0-1
    previous_regime: Optional[MarketRegime]
    regime_confidence: float  # 0-1
    last_transition: Optional[datetime]
    transition_count: int  # transitions in last 5 min
    
    def to_dict(self) -> Dict:
        return {
            'current_regime': self.current_regime.value,
            'regime_duration': self.regime_duration,
            'regime_stability': self.regime_stability,
            'previous_regime': self.previous_regime.value if self.previous_regime else None,
            'regime_confidence': self.regime_confidence,
            'last_transition': self.last_transition.isoformat() if self.last_transition else None,
            'transition_count': self.transition_count
        }


@dataclass
class PerformanceMemory:
    """Performance tracking across dimensions"""
    symbol_win_rates: Dict[str, float] = field(default_factory=dict)
    symbol_sample_counts: Dict[str, int] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confidence_buckets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'symbol_win_rates': self.symbol_win_rates,
            'symbol_sample_counts': self.symbol_sample_counts,
            'regime_performance': self.regime_performance,
            'confidence_buckets': self.confidence_buckets,
            'recent_pnl': list(self.recent_pnl),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }


@dataclass
class PatternMemory:
    """Pattern recognition and outcome tracking"""
    failed_patterns: Dict[str, int] = field(default_factory=dict)
    successful_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pattern_outcomes: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'failed_patterns': self.failed_patterns,
            'successful_patterns': self.successful_patterns,
            'pattern_outcomes': {k: list(v) for k, v in self.pattern_outcomes.items()}
        }


@dataclass
class MemoryContext:
    """
    Synthesized memory context for decision making
    
    This is the output that Orchestrator Policy uses to adjust trading parameters
    """
    confidence_adjustment: float  # Add to base threshold (-0.2 to +0.2)
    risk_multiplier: float  # Multiply position size (0.1 to 2.0)
    pattern_reliability: float  # 0-1 score for pattern memory quality
    regime_stability: float  # 0-1 score for regime confidence
    allow_new_entries: bool  # Emergency stop if False
    memory_level: MemoryLevel  # Quality of memory data
    recent_performance_score: float  # -1 to +1 score
    symbol_blacklist: List[str]  # Symbols to avoid
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'confidence_adjustment': self.confidence_adjustment,
            'risk_multiplier': self.risk_multiplier,
            'pattern_reliability': self.pattern_reliability,
            'regime_stability': self.regime_stability,
            'allow_new_entries': self.allow_new_entries,
            'memory_level': self.memory_level.value,
            'recent_performance_score': self.recent_performance_score,
            'symbol_blacklist': self.symbol_blacklist,
            'metadata': self.metadata
        }


# ============================================================================
# MEMORY STATE MANAGER
# ============================================================================

class MemoryStateManager:
    """
    Central memory management for AI trading system
    
    Responsibilities:
    1. Track market regime states and transitions
    2. Maintain performance statistics (EWMA-based)
    3. Pattern recognition and outcome memory
    4. Confidence calibration and adjustment
    5. Generate MemoryContext for policy decisions
    
    Integration Points:
    - AITradingEngine: Provides signal outcomes
    - OrchestratorPolicy: Consumes MemoryContext
    - EnsembleManager: Adjusted confidence scores
    - Risk OS: Dynamic risk parameters
    """
    
    def __init__(
        self,
        ewma_alpha: float = 0.3,
        min_samples_for_memory: int = 10,
        regime_lock_duration: int = 120,  # seconds
        checkpoint_path: str = "/app/data/memory_state.json",
        checkpoint_interval: int = 60  # seconds
    ):
        """
        Initialize Memory State Manager
        
        Args:
            ewma_alpha: Decay factor for exponential moving averages (0.1-0.5)
                       0.3 = 70% old + 30% new (half-life ~2 observations)
            min_samples_for_memory: Minimum samples before trusting memory
            regime_lock_duration: Seconds to lock regime after oscillation
            checkpoint_path: Path for state persistence
            checkpoint_interval: Seconds between checkpoints
        """
        self.ewma_alpha = ewma_alpha
        self.min_samples = min_samples_for_memory
        self.regime_lock_duration = regime_lock_duration
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_interval = checkpoint_interval
        
        # State containers
        self.regime_state = self._init_regime_state()
        self.performance_memory = PerformanceMemory()
        self.pattern_memory = PatternMemory()
        
        # Internal tracking
        self.regime_lock_until: Optional[datetime] = None
        self.last_checkpoint: datetime = datetime.now(timezone.utc)
        self.total_trades: int = 0
        
        # Load persisted state if exists
        self._load_checkpoint()
        
        logger.info(f"[MEMORY] Initialized with α={ewma_alpha}, min_samples={self.min_samples}")
    
    def _init_regime_state(self) -> RegimeState:
        """Initialize regime state with defaults"""
        return RegimeState(
            current_regime=MarketRegime.UNKNOWN,
            regime_duration=0,
            regime_stability=0.0,
            previous_regime=None,
            regime_confidence=0.0,
            last_transition=None,
            transition_count=0
        )
    
    # ========================================================================
    # CORE UPDATE METHODS
    # ========================================================================
    
    def update_regime(
        self,
        new_regime: MarketRegime,
        regime_confidence: float,
        market_features: Dict[str, float]
    ) -> None:
        """
        Update market regime state with transition detection
        
        Args:
            new_regime: Detected market regime
            regime_confidence: Confidence in detection (0-1)
            market_features: Current market features for stability calc
        """
        now = datetime.now(timezone.utc)
        
        # Check if regime is locked (oscillation protection)
        if self.regime_lock_until and now < self.regime_lock_until:
            logger.debug(f"[MEMORY] Regime locked until {self.regime_lock_until}")
            return
        
        # Detect transition
        if new_regime != self.regime_state.current_regime:
            self._handle_regime_transition(new_regime, now)
        else:
            # Same regime - increase duration
            if self.regime_state.last_transition:
                self.regime_state.regime_duration = int(
                    (now - self.regime_state.last_transition).total_seconds()
                )
        
        # Update confidence and stability
        self.regime_state.regime_confidence = regime_confidence
        self.regime_state.regime_stability = self._calculate_regime_stability(
            market_features
        )
        
        logger.debug(
            f"[MEMORY] Regime: {self.regime_state.current_regime.value}, "
            f"Duration: {self.regime_state.regime_duration}s, "
            f"Stability: {self.regime_state.regime_stability:.2f}"
        )
    
    def _handle_regime_transition(self, new_regime: MarketRegime, now: datetime) -> None:
        """Handle regime state transition"""
        # Check for oscillation (too many transitions)
        recent_window = now - timedelta(minutes=5)
        if self.regime_state.last_transition and self.regime_state.last_transition > recent_window:
            self.regime_state.transition_count += 1
        else:
            self.regime_state.transition_count = 1
        
        # Oscillation detection
        if self.regime_state.transition_count > 3:
            logger.warning(
                f"[MEMORY] Regime oscillation detected! "
                f"{self.regime_state.transition_count} transitions in 5 min. "
                f"Locking regime for {self.regime_lock_duration}s"
            )
            self.regime_lock_until = now + timedelta(seconds=self.regime_lock_duration)
            return
        
        # Valid transition
        logger.info(
            f"[MEMORY] Regime transition: "
            f"{self.regime_state.current_regime.value} → {new_regime.value}"
        )
        
        self.regime_state.previous_regime = self.regime_state.current_regime
        self.regime_state.current_regime = new_regime
        self.regime_state.last_transition = now
        self.regime_state.regime_duration = 0
    
    def _calculate_regime_stability(self, market_features: Dict[str, float]) -> float:
        """
        Calculate regime stability score based on market features
        
        High stability = features consistent with regime
        Low stability = features conflicting with regime
        """
        regime = self.regime_state.current_regime
        
        # Extract key features
        volatility = market_features.get('atr_pct', 0.02)
        momentum = abs(market_features.get('momentum', 0.0))
        trend_strength = abs(market_features.get('trend_strength', 0.0))
        
        stability = 1.0
        
        if regime == MarketRegime.TRENDING:
            # Expect high momentum, strong trend, moderate volatility
            if momentum < 0.02:
                stability *= 0.7
            if trend_strength < 0.5:
                stability *= 0.6
            if volatility > 0.05:
                stability *= 0.8
        
        elif regime == MarketRegime.RANGING:
            # Expect low momentum, weak trend, low volatility
            if momentum > 0.05:
                stability *= 0.7
            if trend_strength > 0.3:
                stability *= 0.6
        
        elif regime == MarketRegime.VOLATILE:
            # Expect high volatility
            if volatility < 0.04:
                stability *= 0.5
        
        return max(0.0, min(1.0, stability))
    
    def record_trade_outcome(
        self,
        symbol: str,
        action: str,
        confidence: float,
        pnl: float,
        regime: MarketRegime,
        setup_hash: Optional[str] = None
    ) -> None:
        """
        Record trade outcome and update all memory layers
        
        Args:
            symbol: Trading symbol
            action: BUY/SELL
            confidence: AI confidence at entry
            pnl: Trade PnL (can be negative)
            regime: Market regime during trade
            setup_hash: Optional hash of market setup/pattern
        """
        self.total_trades += 1
        is_win = pnl > 0
        
        # 1. Update symbol win rates (EWMA)
        current_wr = self.performance_memory.symbol_win_rates.get(symbol, 0.5)
        new_wr = self.ewma_alpha * (1.0 if is_win else 0.0) + (1 - self.ewma_alpha) * current_wr
        self.performance_memory.symbol_win_rates[symbol] = new_wr
        
        # Track sample count
        self.performance_memory.symbol_sample_counts[symbol] = \
            self.performance_memory.symbol_sample_counts.get(symbol, 0) + 1
        
        # 2. Update regime performance
        regime_key = regime.value
        if regime_key not in self.performance_memory.regime_performance:
            self.performance_memory.regime_performance[regime_key] = {
                'win_rate': 0.5,
                'avg_pnl': 0.0,
                'sample_count': 0,
                'total_pnl': 0.0
            }
        
        regime_perf = self.performance_memory.regime_performance[regime_key]
        regime_perf['sample_count'] += 1
        regime_perf['total_pnl'] += pnl
        regime_perf['avg_pnl'] = regime_perf['total_pnl'] / regime_perf['sample_count']
        
        # EWMA on regime win rate
        current_regime_wr = regime_perf['win_rate']
        regime_perf['win_rate'] = (
            self.ewma_alpha * (1.0 if is_win else 0.0) +
            (1 - self.ewma_alpha) * current_regime_wr
        )
        
        # 3. Update confidence calibration buckets
        conf_bucket = self._get_confidence_bucket(confidence)
        if conf_bucket not in self.performance_memory.confidence_buckets:
            self.performance_memory.confidence_buckets[conf_bucket] = {
                'predicted_wr': confidence,
                'actual_wins': 0,
                'total_trades': 0,
                'actual_wr': 0.0
            }
        
        bucket = self.performance_memory.confidence_buckets[conf_bucket]
        bucket['total_trades'] += 1
        if is_win:
            bucket['actual_wins'] += 1
        bucket['actual_wr'] = bucket['actual_wins'] / bucket['total_trades']
        
        # 4. Update recent PnL sequence
        self.performance_memory.recent_pnl.append(pnl)
        
        # 5. Update consecutive win/loss tracking
        if is_win:
            self.performance_memory.consecutive_wins += 1
            self.performance_memory.consecutive_losses = 0
        else:
            self.performance_memory.consecutive_losses += 1
            self.performance_memory.consecutive_wins = 0
        
        # 6. Update pattern memory if hash provided
        if setup_hash:
            if setup_hash not in self.pattern_memory.pattern_outcomes:
                self.pattern_memory.pattern_outcomes[setup_hash] = []
            
            self.pattern_memory.pattern_outcomes[setup_hash].append(pnl)
            
            if is_win:
                if setup_hash not in self.pattern_memory.successful_patterns:
                    self.pattern_memory.successful_patterns[setup_hash] = {
                        'count': 0,
                        'total_pnl': 0.0,
                        'avg_pnl': 0.0
                    }
                
                pattern = self.pattern_memory.successful_patterns[setup_hash]
                pattern['count'] += 1
                pattern['total_pnl'] += pnl
                pattern['avg_pnl'] = pattern['total_pnl'] / pattern['count']
            else:
                self.pattern_memory.failed_patterns[setup_hash] = \
                    self.pattern_memory.failed_patterns.get(setup_hash, 0) + 1
        
        logger.info(
            f"[MEMORY] Recorded trade: {symbol} {action} "
            f"PnL=${pnl:.2f} (Win={is_win}) "
            f"Confidence={confidence:.2f} Regime={regime.value}"
        )
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Bin confidence into buckets for calibration tracking"""
        if confidence < 0.5:
            return "0.0-0.5"
        elif confidence < 0.6:
            return "0.5-0.6"
        elif confidence < 0.7:
            return "0.6-0.7"
        elif confidence < 0.8:
            return "0.7-0.8"
        elif confidence < 0.9:
            return "0.8-0.9"
        else:
            return "0.9-1.0"
    
    # ========================================================================
    # MEMORY CONTEXT GENERATION
    # ========================================================================
    
    def get_memory_context(self, symbol: Optional[str] = None) -> MemoryContext:
        """
        Generate MemoryContext for trading decisions
        
        This is the main output consumed by OrchestratorPolicy
        
        Args:
            symbol: Optional specific symbol to focus context on
            
        Returns:
            MemoryContext with adjustment factors
        """
        # Determine memory level
        memory_level = self._get_memory_level()
        
        # Cold start protection
        if memory_level == MemoryLevel.COLD_START:
            return self._cold_start_context()
        
        # Calculate components
        confidence_adj = self._calculate_confidence_adjustment()
        risk_mult = self._calculate_risk_multiplier()
        pattern_rel = self._calculate_pattern_reliability()
        perf_score = self._calculate_performance_score()
        allow_entries = self._check_allow_entries()
        blacklist = self._get_symbol_blacklist()
        
        context = MemoryContext(
            confidence_adjustment=confidence_adj,
            risk_multiplier=risk_mult,
            pattern_reliability=pattern_rel,
            regime_stability=self.regime_state.regime_stability,
            allow_new_entries=allow_entries,
            memory_level=memory_level,
            recent_performance_score=perf_score,
            symbol_blacklist=blacklist,
            metadata={
                'total_trades': self.total_trades,
                'current_regime': self.regime_state.current_regime.value,
                'regime_duration': self.regime_state.regime_duration,
                'consecutive_losses': self.performance_memory.consecutive_losses,
                'consecutive_wins': self.performance_memory.consecutive_wins
            }
        )
        
        logger.debug(
            f"[MEMORY] Generated context: "
            f"ConfAdj={confidence_adj:+.2f}, RiskMult={risk_mult:.2f}, "
            f"PerfScore={perf_score:+.2f}, AllowEntries={allow_entries}"
        )
        
        return context
    
    def _get_memory_level(self) -> MemoryLevel:
        """Determine quality level of memory data"""
        if self.total_trades < 10:
            return MemoryLevel.COLD_START
        elif self.total_trades < 50:
            return MemoryLevel.LOW
        elif self.total_trades < 200:
            return MemoryLevel.MEDIUM
        else:
            return MemoryLevel.HIGH
    
    def _cold_start_context(self) -> MemoryContext:
        """Conservative context for cold start"""
        logger.warning(
            f"[MEMORY] Cold start mode - only {self.total_trades} trades recorded"
        )
        return MemoryContext(
            confidence_adjustment=+0.10,  # Raise threshold
            risk_multiplier=0.5,          # Half normal risk
            pattern_reliability=0.0,
            regime_stability=0.0,
            allow_new_entries=True,
            memory_level=MemoryLevel.COLD_START,
            recent_performance_score=0.0,
            symbol_blacklist=[],
            metadata={'total_trades': self.total_trades}
        )
    
    def _calculate_confidence_adjustment(self) -> float:
        """
        Calculate confidence threshold adjustment based on recent performance
        
        Returns:
            Float in range [-0.20, +0.20]
            Negative = lower threshold (more trades)
            Positive = raise threshold (fewer trades)
        """
        adjustment = 0.0
        
        # 1. Recent PnL trend
        if len(self.performance_memory.recent_pnl) >= 20:
            recent_sum = sum(list(self.performance_memory.recent_pnl)[-20:])
            
            if recent_sum < -300:  # Lost $300 in last 20 trades
                adjustment += 0.15  # Significantly raise threshold
            elif recent_sum < -150:
                adjustment += 0.10
            elif recent_sum < -50:
                adjustment += 0.05
            elif recent_sum > 300:  # Made $300
                adjustment -= 0.05  # Slightly lower threshold
        
        # 2. Consecutive losses
        if self.performance_memory.consecutive_losses >= 5:
            adjustment += 0.20  # Max penalty
        elif self.performance_memory.consecutive_losses >= 3:
            adjustment += 0.10
        
        # 3. Regime-specific performance
        current_regime = self.regime_state.current_regime.value
        if current_regime in self.performance_memory.regime_performance:
            regime_perf = self.performance_memory.regime_performance[current_regime]
            if regime_perf['sample_count'] >= 10:
                regime_wr = regime_perf['win_rate']
                
                if regime_wr < 0.40:  # Poor win rate in this regime
                    adjustment += 0.10
                elif regime_wr > 0.65:  # Strong win rate
                    adjustment -= 0.05
        
        # 4. Regime stability factor
        if self.regime_state.regime_stability < 0.5:
            adjustment += 0.05  # Less stable = higher threshold
        
        # Clamp to [-0.20, +0.20]
        return max(-0.20, min(0.20, adjustment))
    
    def _calculate_risk_multiplier(self) -> float:
        """
        Calculate position size multiplier based on memory
        
        Returns:
            Float in range [0.1, 2.0]
            < 1.0 = reduce position size
            > 1.0 = increase position size
        """
        multiplier = 1.0
        
        # 1. Consecutive losses - aggressive reduction
        if self.performance_memory.consecutive_losses >= 5:
            multiplier = 0.2  # 80% reduction
        elif self.performance_memory.consecutive_losses >= 3:
            multiplier = 0.5
        elif self.performance_memory.consecutive_losses >= 2:
            multiplier = 0.7
        
        # 2. Recent PnL trend
        if len(self.performance_memory.recent_pnl) >= 50:
            recent_50_sum = sum(list(self.performance_memory.recent_pnl)[-50:])
            
            if recent_50_sum < -500:  # Large recent loss
                multiplier *= 0.3
            elif recent_50_sum < -200:
                multiplier *= 0.5
            elif recent_50_sum > 500:  # Large recent profit
                multiplier *= 1.3
        
        # 3. Regime performance
        current_regime = self.regime_state.current_regime.value
        if current_regime in self.performance_memory.regime_performance:
            regime_perf = self.performance_memory.regime_performance[current_regime]
            if regime_perf['sample_count'] >= 10:
                if regime_perf['avg_pnl'] < -5:  # Negative avg PnL
                    multiplier *= 0.6
        
        # 4. Memory level confidence
        memory_level = self._get_memory_level()
        if memory_level == MemoryLevel.LOW:
            multiplier *= 0.7  # Less confident = smaller size
        
        # Clamp to [0.1, 2.0]
        return max(0.1, min(2.0, multiplier))
    
    def _calculate_pattern_reliability(self) -> float:
        """
        Calculate overall pattern memory reliability score
        
        Returns:
            Float in range [0.0, 1.0]
        """
        total_patterns = len(self.pattern_memory.pattern_outcomes)
        
        if total_patterns < 10:
            return 0.0  # Not enough pattern data
        
        # Calculate success rate across patterns
        successful = len(self.pattern_memory.successful_patterns)
        failed = sum(self.pattern_memory.failed_patterns.values())
        total = successful + failed
        
        if total == 0:
            return 0.0
        
        success_rate = successful / total
        
        # Adjust for sample size
        reliability = success_rate * min(1.0, total / 100)
        
        return reliability
    
    def _calculate_performance_score(self) -> float:
        """
        Calculate recent performance score
        
        Returns:
            Float in range [-1.0, +1.0]
            Negative = poor recent performance
            Positive = strong recent performance
        """
        if len(self.performance_memory.recent_pnl) < 10:
            return 0.0
        
        recent_pnl = list(self.performance_memory.recent_pnl)[-50:]
        
        # Calculate metrics
        total_pnl = sum(recent_pnl)
        wins = sum(1 for pnl in recent_pnl if pnl > 0)
        win_rate = wins / len(recent_pnl)
        
        # Normalize PnL to [-1, +1]
        # Assume $500 over 50 trades = excellent (+1.0)
        # Assume -$500 over 50 trades = terrible (-1.0)
        pnl_score = total_pnl / 500
        pnl_score = max(-1.0, min(1.0, pnl_score))
        
        # Combine PnL and win rate
        # Win rate 0.6 = +0.2, 0.4 = -0.2
        wr_score = (win_rate - 0.5) * 2
        
        score = 0.6 * pnl_score + 0.4 * wr_score
        
        return max(-1.0, min(1.0, score))
    
    def _check_allow_entries(self) -> bool:
        """
        Emergency check - should new entries be allowed?
        
        Returns:
            False if memory indicates severe issues
        """
        # Check 1: Consecutive losses
        if self.performance_memory.consecutive_losses >= 7:
            logger.error("[MEMORY] EMERGENCY STOP: 7+ consecutive losses")
            return False
        
        # Check 2: Recent massive loss
        if len(self.performance_memory.recent_pnl) >= 20:
            recent_20_sum = sum(list(self.performance_memory.recent_pnl)[-20:])
            if recent_20_sum < -800:  # Lost $800 in 20 trades
                logger.error("[MEMORY] EMERGENCY STOP: $800 loss in 20 trades")
                return False
        
        # Check 3: Win rate collapse
        if len(self.performance_memory.recent_pnl) >= 30:
            recent_30 = list(self.performance_memory.recent_pnl)[-30:]
            wins = sum(1 for pnl in recent_30 if pnl > 0)
            wr = wins / len(recent_30)
            
            if wr < 0.25:  # < 25% win rate
                logger.error(f"[MEMORY] EMERGENCY STOP: Win rate = {wr:.1%}")
                return False
        
        return True
    
    def _get_symbol_blacklist(self) -> List[str]:
        """
        Get list of symbols to avoid based on poor performance
        
        Returns:
            List of symbol strings
        """
        blacklist = []
        
        for symbol, win_rate in self.performance_memory.symbol_win_rates.items():
            sample_count = self.performance_memory.symbol_sample_counts.get(symbol, 0)
            
            # Only blacklist if sufficient samples
            if sample_count >= 15:
                if win_rate < 0.30:  # < 30% win rate
                    blacklist.append(symbol)
                    logger.warning(
                        f"[MEMORY] Blacklisting {symbol}: "
                        f"WR={win_rate:.1%} over {sample_count} trades"
                    )
        
        return blacklist
    
    # ========================================================================
    # PATTERN HASHING
    # ========================================================================
    
    @staticmethod
    def hash_market_setup(
        symbol: str,
        regime: MarketRegime,
        volatility_bucket: str,
        momentum_bucket: str,
        trend_strength_bucket: str
    ) -> str:
        """
        Create hash for current market setup/pattern
        
        Args:
            symbol: Trading symbol
            regime: Market regime
            volatility_bucket: "LOW" / "MEDIUM" / "HIGH"
            momentum_bucket: "LOW" / "MEDIUM" / "HIGH"
            trend_strength_bucket: "WEAK" / "MODERATE" / "STRONG"
            
        Returns:
            Hash string for pattern matching
        """
        pattern_str = (
            f"{symbol}|{regime.value}|"
            f"{volatility_bucket}|{momentum_bucket}|{trend_strength_bucket}"
        )
        
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def query_pattern_memory(self, setup_hash: str) -> Optional[Dict[str, Any]]:
        """
        Query pattern memory for historical outcomes
        
        Args:
            setup_hash: Hash of current market setup
            
        Returns:
            Dict with pattern statistics or None if not found
        """
        if setup_hash not in self.pattern_memory.pattern_outcomes:
            return None
        
        outcomes = self.pattern_memory.pattern_outcomes[setup_hash]
        
        if len(outcomes) < 5:
            return None  # Not enough data
        
        wins = sum(1 for pnl in outcomes if pnl > 0)
        win_rate = wins / len(outcomes)
        avg_pnl = sum(outcomes) / len(outcomes)
        
        return {
            'sample_count': len(outcomes),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': sum(outcomes),
            'max_win': max(outcomes),
            'max_loss': min(outcomes)
        }
    
    # ========================================================================
    # CALIBRATION & DIAGNOSTICS
    # ========================================================================
    
    def get_calibration_curve(self) -> Dict[str, Dict[str, float]]:
        """
        Get confidence calibration curve
        
        Returns:
            Dict mapping confidence buckets to calibration data
        """
        return self.performance_memory.confidence_buckets.copy()
    
    def calculate_brier_score(self) -> float:
        """
        Calculate Brier score for confidence calibration quality
        
        Lower is better (0.0 = perfect calibration)
        
        Returns:
            Brier score (0.0 to 1.0)
        """
        if not self.performance_memory.confidence_buckets:
            return 1.0  # No data = worst score
        
        squared_errors = []
        
        for bucket_data in self.performance_memory.confidence_buckets.values():
            predicted = bucket_data['predicted_wr']
            actual = bucket_data['actual_wr']
            squared_errors.append((predicted - actual) ** 2)
        
        if not squared_errors:
            return 1.0
        
        brier_score = sum(squared_errors) / len(squared_errors)
        
        return brier_score
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive memory diagnostics
        
        Returns:
            Dict with all memory state information
        """
        return {
            'memory_level': self._get_memory_level().value,
            'total_trades': self.total_trades,
            'regime_state': self.regime_state.to_dict(),
            'performance_memory': {
                'symbols_tracked': len(self.performance_memory.symbol_win_rates),
                'consecutive_wins': self.performance_memory.consecutive_wins,
                'consecutive_losses': self.performance_memory.consecutive_losses,
                'recent_pnl_sum': sum(self.performance_memory.recent_pnl),
                'regime_performance': self.performance_memory.regime_performance
            },
            'pattern_memory': {
                'patterns_tracked': len(self.pattern_memory.pattern_outcomes),
                'successful_patterns': len(self.pattern_memory.successful_patterns),
                'failed_patterns': sum(self.pattern_memory.failed_patterns.values())
            },
            'calibration': {
                'brier_score': self.calculate_brier_score(),
                'buckets': self.get_calibration_curve()
            }
        }
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def checkpoint(self) -> None:
        """Save current state to disk"""
        try:
            # Create checkpoint directory if needed
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_trades': self.total_trades,
                'regime_state': self.regime_state.to_dict(),
                'performance_memory': self.performance_memory.to_dict(),
                'pattern_memory': self.pattern_memory.to_dict()
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.last_checkpoint = datetime.now(timezone.utc)
            logger.info(f"[MEMORY] Checkpoint saved: {self.checkpoint_path}")
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> None:
        """Load state from disk if exists"""
        if not self.checkpoint_path.exists():
            logger.info("[MEMORY] No checkpoint found - starting fresh")
            return
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            self.total_trades = state.get('total_trades', 0)
            
            # Restore regime state
            regime_data = state.get('regime_state', {})
            self.regime_state = RegimeState(
                current_regime=MarketRegime(regime_data.get('current_regime', 'UNKNOWN')),
                regime_duration=regime_data.get('regime_duration', 0),
                regime_stability=regime_data.get('regime_stability', 0.0),
                previous_regime=MarketRegime(regime_data['previous_regime']) if regime_data.get('previous_regime') else None,
                regime_confidence=regime_data.get('regime_confidence', 0.0),
                last_transition=datetime.fromisoformat(regime_data['last_transition']) if regime_data.get('last_transition') else None,
                transition_count=regime_data.get('transition_count', 0)
            )
            
            # Restore performance memory
            perf_data = state.get('performance_memory', {})
            self.performance_memory = PerformanceMemory(
                symbol_win_rates=perf_data.get('symbol_win_rates', {}),
                symbol_sample_counts=perf_data.get('symbol_sample_counts', {}),
                regime_performance=perf_data.get('regime_performance', {}),
                confidence_buckets=perf_data.get('confidence_buckets', {}),
                recent_pnl=deque(perf_data.get('recent_pnl', []), maxlen=100),
                consecutive_wins=perf_data.get('consecutive_wins', 0),
                consecutive_losses=perf_data.get('consecutive_losses', 0)
            )
            
            # Restore pattern memory
            pattern_data = state.get('pattern_memory', {})
            self.pattern_memory = PatternMemory(
                failed_patterns=pattern_data.get('failed_patterns', {}),
                successful_patterns=pattern_data.get('successful_patterns', {}),
                pattern_outcomes={
                    k: deque(v, maxlen=100) 
                    for k, v in pattern_data.get('pattern_outcomes', {}).items()
                }
            )
            
            logger.info(
                f"[MEMORY] Checkpoint loaded: {self.total_trades} trades, "
                f"regime={self.regime_state.current_regime.value}"
            )
            
        except Exception as e:
            logger.error(f"[MEMORY] Failed to load checkpoint: {e}")
    
    def auto_checkpoint_check(self) -> None:
        """Check if it's time for automatic checkpoint"""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_checkpoint).total_seconds()
        
        if elapsed >= self.checkpoint_interval:
            self.checkpoint()
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def reset_performance_memory(self) -> None:
        """Emergency reset of performance memory (use with caution)"""
        logger.warning("[MEMORY] RESETTING PERFORMANCE MEMORY")
        self.performance_memory = PerformanceMemory()
        self.total_trades = 0
    
    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed statistics for specific symbol"""
        win_rate = self.performance_memory.symbol_win_rates.get(symbol, 0.5)
        sample_count = self.performance_memory.symbol_sample_counts.get(symbol, 0)
        
        return {
            'symbol': symbol,
            'win_rate': win_rate,
            'sample_count': sample_count,
            'is_blacklisted': symbol in self._get_symbol_blacklist(),
            'memory_quality': 'HIGH' if sample_count >= 30 else 'MEDIUM' if sample_count >= 10 else 'LOW'
        }
