#!/usr/bin/env python3
"""
ACTIVE POSITIONS CONTROLLER - DYNAMIC SLOT MANAGEMENT
====================================================

Dynamically manages active position slots based on regime and portfolio risk.
Does NOT use hardcoded symbols - reads universe from PolicyStore.

Key Features:
1. Dynamic slot allocation (3-6 slots based on regime, hard cap 10)
2. Capital rotation (close weakest if new candidate is significantly better)
3. Correlation caps (prevent >80% correlation with portfolio)
4. Margin usage caps (prevent exceeding 60-65% global margin)
5. Policy-driven (reads universe from Redis quantum:policy:current)
6. Fail-closed (blocks opens if no policy available)

Author: Quantum Trader AI Team
Date: February 3, 2026
Version: 1.0
"""

import json
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from backend.core.policy_store import PolicyStore
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class RegimeType(Enum):
    """Market regime types"""
    TREND_STRONG = "TREND_STRONG"
    TREND_MODERATE = "TREND_MODERATE"
    RANGE = "RANGE"
    CHOP = "CHOP"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    UNKNOWN = "UNKNOWN"


class SlotDecision(Enum):
    """Slot controller decisions"""
    ALLOW = "ALLOW"
    BLOCKED_SLOTS_FULL = "BLOCKED_SLOTS_FULL"
    BLOCKED_NO_POLICY = "BLOCKED_NO_POLICY"
    BLOCKED_CORRELATION = "BLOCKED_CORRELATION"
    BLOCKED_MARGIN = "BLOCKED_MARGIN"
    ROTATION_TRIGGERED = "ROTATION_TRIGGERED"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SlotConfig:
    """Configuration for slot controller"""
    base_slots: int = 4
    regime_slots_trend_strong: int = 6
    regime_slots_range_chop: int = 3
    regime_slots_volatility_spike: int = 3
    hard_cap_slots: int = 10
    
    # Rotation settings
    rotation_switch_threshold: float = 0.15  # 15% better score
    
    # Risk caps
    max_correlation: float = 0.80
    max_margin_usage_pct: float = 65.0
    
    # Regime detection thresholds
    trend_strong_threshold: float = 0.75
    trend_moderate_threshold: float = 0.45
    volatility_spike_threshold: float = 2.5  # ATR multiplier


@dataclass
class PositionSnapshot:
    """Snapshot of an active position"""
    symbol: str
    size_usd: float
    leverage: float
    unrealized_pnl_usd: float
    entry_price: float
    current_price: float
    age_seconds: float
    score: float  # Current quality score
    returns_series: Optional[List[float]] = None  # For correlation


@dataclass
class SlotControllerState:
    """Current state of slot controller"""
    timestamp: str
    
    # Slot allocation
    desired_slots: int
    open_positions_count: int
    available_slots: int
    
    # Regime
    regime: RegimeType
    regime_confidence: float
    
    # Risk metrics
    total_margin_usage_pct: float
    portfolio_correlation_avg: float
    
    # Policy
    policy_universe: List[str]
    policy_available: bool


@dataclass
class SlotDecisionRecord:
    """Record of a slot controller decision"""
    timestamp: str
    decision: SlotDecision
    
    # Request
    symbol: str
    action: str
    candidate_score: float
    
    # Context
    desired_slots: int
    open_positions_count: int
    regime: RegimeType
    
    # Rotation details (if applicable)
    rotation_triggered: bool = False
    weakest_symbol: Optional[str] = None
    weakest_score: Optional[float] = None
    
    # Block details (if applicable)
    block_reason: str = ""
    correlation_with_portfolio: Optional[float] = None
    margin_usage_pct: Optional[float] = None


# ============================================================================
# ACTIVE POSITIONS CONTROLLER
# ============================================================================

class ActivePositionsController:
    """
    Dynamic slot management for active positions.
    
    Enforces:
    - Dynamic slot allocation based on regime
    - Capital rotation (close weakest, open better)
    - Correlation caps
    - Margin usage caps
    - Policy-driven universe (no hardcoding)
    """
    
    def __init__(
        self,
        policy_store: PolicyStore,
        config: Optional[SlotConfig] = None,
    ):
        self.policy_store = policy_store
        self.config = config or SlotConfig()
        
        # State
        self.current_state: Optional[SlotControllerState] = None
        self.decision_history: List[SlotDecisionRecord] = []
        self.max_history_size = 1000
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "slots_full_blocks": 0,
            "no_policy_blocks": 0,
            "correlation_blocks": 0,
            "margin_blocks": 0,
            "rotations_triggered": 0,
            "allows": 0,
        }
        
        logger.info(
            "active_positions_controller_initialized",
            base_slots=self.config.base_slots,
            hard_cap=self.config.hard_cap_slots,
            rotation_threshold=self.config.rotation_switch_threshold,
            max_correlation=self.config.max_correlation,
            max_margin_pct=self.config.max_margin_usage_pct,
        )
    
    async def load_policy_universe(self) -> Tuple[bool, List[str]]:
        """
        Load universe from PolicyStore.
        
        Returns:
            (success, universe_list)
        """
        try:
            # Read from Redis: HGET quantum:policy:current universe_symbols
            policy_data = await self.policy_store.redis.hgetall("quantum:policy:current")
            
            if not policy_data or b"universe_symbols" not in policy_data:
                logger.error(
                    "active_slots_no_policy",
                    message="quantum:policy:current missing universe_symbols field",
                )
                return False, []
            
            universe_json = policy_data[b"universe_symbols"].decode("utf-8")
            universe = json.loads(universe_json)
            
            if not isinstance(universe, list) or len(universe) == 0:
                logger.error(
                    "active_slots_invalid_policy",
                    universe_type=type(universe).__name__,
                    universe_len=len(universe) if isinstance(universe, list) else 0,
                )
                return False, []
            
            logger.debug(
                "active_slots_policy_loaded",
                universe_count=len(universe),
                symbols=universe[:5],  # Log first 5
            )
            
            return True, universe
        
        except Exception as e:
            logger.error(
                "active_slots_policy_load_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False, []
    
    def detect_regime(
        self,
        market_features: Dict[str, float],
    ) -> Tuple[RegimeType, float]:
        """
        Detect market regime from features.
        
        Args:
            market_features: Dict with keys like:
                - trend_strength: float [0..1]
                - atr_pct: float (volatility %)
                - momentum_consistency: float [0..1]
        
        Returns:
            (regime, confidence)
        """
        trend_strength = market_features.get("trend_strength", 0.5)
        atr_pct = market_features.get("atr_pct", 1.5)
        momentum_consistency = market_features.get("momentum_consistency", 0.5)
        
        # Volatility spike check (high priority)
        if atr_pct > self.config.volatility_spike_threshold:
            return RegimeType.VOLATILITY_SPIKE, 0.9
        
        # Trend strength analysis
        if trend_strength >= self.config.trend_strong_threshold and momentum_consistency >= 0.6:
            return RegimeType.TREND_STRONG, 0.85
        
        elif trend_strength >= self.config.trend_moderate_threshold:
            return RegimeType.TREND_MODERATE, 0.7
        
        # Range/chop detection
        elif trend_strength < 0.3 and momentum_consistency < 0.4:
            return RegimeType.CHOP, 0.75
        
        elif trend_strength < 0.4:
            return RegimeType.RANGE, 0.65
        
        else:
            return RegimeType.UNKNOWN, 0.5
    
    def compute_desired_slots(
        self,
        regime: RegimeType,
    ) -> int:
        """
        Compute desired slot count based on regime.
        
        Args:
            regime: Detected market regime
        
        Returns:
            Desired slot count [1..hard_cap]
        """
        if regime == RegimeType.TREND_STRONG:
            slots = self.config.regime_slots_trend_strong
        
        elif regime in [RegimeType.RANGE, RegimeType.CHOP, RegimeType.VOLATILITY_SPIKE]:
            slots = self.config.regime_slots_range_chop
        
        elif regime == RegimeType.TREND_MODERATE:
            slots = self.config.base_slots
        
        else:  # UNKNOWN
            slots = self.config.base_slots
        
        # Apply hard cap
        slots = min(slots, self.config.hard_cap_slots)
        
        return slots
    
    def compute_correlation_with_portfolio(
        self,
        candidate_returns: List[float],
        portfolio_positions: List[PositionSnapshot],
    ) -> float:
        """
        Compute correlation between candidate and portfolio basket.
        
        Args:
            candidate_returns: Log-returns series for candidate
            portfolio_positions: List of active positions with returns_series
        
        Returns:
            Max correlation with any position in portfolio [0..1]
        """
        if not portfolio_positions:
            return 0.0
        
        max_corr = 0.0
        
        for position in portfolio_positions:
            if position.returns_series is None or len(position.returns_series) < 20:
                continue
            
            # Align lengths
            min_len = min(len(candidate_returns), len(position.returns_series))
            if min_len < 20:
                continue
            
            cand_aligned = candidate_returns[-min_len:]
            pos_aligned = position.returns_series[-min_len:]
            
            # Pearson correlation
            corr_matrix = np.corrcoef(cand_aligned, pos_aligned)
            corr = abs(corr_matrix[0, 1])  # Absolute correlation
            
            if np.isnan(corr):
                corr = 0.0
            
            max_corr = max(max_corr, corr)
        
        return max_corr
    
    async def evaluate_position_request(
        self,
        symbol: str,
        action: str,
        candidate_score: float,
        candidate_returns: Optional[List[float]],
        market_features: Dict[str, float],
        portfolio_positions: List[PositionSnapshot],
        total_margin_usage_pct: float,
    ) -> Tuple[SlotDecision, SlotDecisionRecord]:
        """
        Evaluate if position should be opened/expanded.
        
        Args:
            symbol: Trading symbol
            action: NEW_TRADE or EXPAND_POSITION
            candidate_score: Quality score for candidate [0..100]
            candidate_returns: Log-returns series for correlation check
            market_features: Features for regime detection
            portfolio_positions: List of current positions
            total_margin_usage_pct: Current margin usage %
        
        Returns:
            (decision, decision_record)
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Step 1: Load policy universe
        policy_ok, policy_universe = await self.load_policy_universe()
        
        if not policy_ok:
            decision = SlotDecision.BLOCKED_NO_POLICY
            record = SlotDecisionRecord(
                timestamp=timestamp,
                decision=decision,
                symbol=symbol,
                action=action,
                candidate_score=candidate_score,
                desired_slots=0,
                open_positions_count=len(portfolio_positions),
                regime=RegimeType.UNKNOWN,
                block_reason="PolicyStore unavailable or invalid",
            )
            
            self.stats["total_decisions"] += 1
            self.stats["no_policy_blocks"] += 1
            
            logger.error(
                "BLOCKED_NO_POLICY",
                symbol=symbol,
                action=action,
                reason="quantum:policy:current missing or invalid",
            )
            
            return decision, record
        
        # Step 2: Check if symbol is in policy universe
        if symbol not in policy_universe:
            decision = SlotDecision.BLOCKED_NO_POLICY
            record = SlotDecisionRecord(
                timestamp=timestamp,
                decision=decision,
                symbol=symbol,
                action=action,
                candidate_score=candidate_score,
                desired_slots=0,
                open_positions_count=len(portfolio_positions),
                regime=RegimeType.UNKNOWN,
                block_reason=f"Symbol not in policy universe ({len(policy_universe)} symbols)",
            )
            
            self.stats["total_decisions"] += 1
            self.stats["no_policy_blocks"] += 1
            
            logger.warning(
                "BLOCKED_NO_POLICY",
                symbol=symbol,
                action=action,
                reason="symbol_not_in_universe",
                universe_count=len(policy_universe),
            )
            
            return decision, record
        
        # Step 3: Detect regime
        regime, regime_confidence = self.detect_regime(market_features)
        
        # Step 4: Compute desired slots
        desired_slots = self.compute_desired_slots(regime)
        
        open_positions_count = len(portfolio_positions)
        available_slots = max(0, desired_slots - open_positions_count)
        
        # Update state
        self.current_state = SlotControllerState(
            timestamp=timestamp,
            desired_slots=desired_slots,
            open_positions_count=open_positions_count,
            available_slots=available_slots,
            regime=regime,
            regime_confidence=regime_confidence,
            total_margin_usage_pct=total_margin_usage_pct,
            portfolio_correlation_avg=0.0,  # Will compute if needed
            policy_universe=policy_universe,
            policy_available=True,
        )
        
        logger.info(
            "ACTIVE_SLOTS",
            desired=desired_slots,
            open=open_positions_count,
            available=available_slots,
            regime=regime.value,
            regime_confidence=f"{regime_confidence:.2f}",
            symbol=symbol,
        )
        
        # Step 5: Check margin usage cap
        if total_margin_usage_pct >= self.config.max_margin_usage_pct:
            decision = SlotDecision.BLOCKED_MARGIN
            record = SlotDecisionRecord(
                timestamp=timestamp,
                decision=decision,
                symbol=symbol,
                action=action,
                candidate_score=candidate_score,
                desired_slots=desired_slots,
                open_positions_count=open_positions_count,
                regime=regime,
                block_reason=f"Margin usage {total_margin_usage_pct:.1f}% >= {self.config.max_margin_usage_pct:.1f}%",
                margin_usage_pct=total_margin_usage_pct,
            )
            
            self.stats["total_decisions"] += 1
            self.stats["margin_blocks"] += 1
            
            logger.warning(
                "BLOCKED_MARGIN",
                symbol=symbol,
                margin_usage_pct=f"{total_margin_usage_pct:.1f}",
                max_margin_pct=f"{self.config.max_margin_usage_pct:.1f}",
            )
            
            return decision, record
        
        # Step 6: Check correlation (if returns provided)
        if candidate_returns and len(candidate_returns) >= 20 and portfolio_positions:
            max_corr = self.compute_correlation_with_portfolio(
                candidate_returns,
                portfolio_positions,
            )
            
            if max_corr > self.config.max_correlation:
                decision = SlotDecision.BLOCKED_CORRELATION
                record = SlotDecisionRecord(
                    timestamp=timestamp,
                    decision=decision,
                    symbol=symbol,
                    action=action,
                    candidate_score=candidate_score,
                    desired_slots=desired_slots,
                    open_positions_count=open_positions_count,
                    regime=regime,
                    block_reason=f"Correlation {max_corr:.2f} > {self.config.max_correlation:.2f}",
                    correlation_with_portfolio=max_corr,
                )
                
                self.stats["total_decisions"] += 1
                self.stats["correlation_blocks"] += 1
                
                logger.warning(
                    "CORR_BLOCK",
                    symbol=symbol,
                    corr=f"{max_corr:.3f}",
                    threshold=f"{self.config.max_correlation:.2f}",
                )
                
                return decision, record
        
        # Step 7: Check if slots are full
        if open_positions_count >= desired_slots:
            # Check if rotation is possible
            if portfolio_positions:
                # Find weakest position
                weakest = min(portfolio_positions, key=lambda p: p.score)
                
                # Check if candidate is significantly better
                score_improvement = (candidate_score - weakest.score) / weakest.score
                
                if score_improvement > self.config.rotation_switch_threshold:
                    # Rotation triggered
                    decision = SlotDecision.ROTATION_TRIGGERED
                    record = SlotDecisionRecord(
                        timestamp=timestamp,
                        decision=decision,
                        symbol=symbol,
                        action=action,
                        candidate_score=candidate_score,
                        desired_slots=desired_slots,
                        open_positions_count=open_positions_count,
                        regime=regime,
                        rotation_triggered=True,
                        weakest_symbol=weakest.symbol,
                        weakest_score=weakest.score,
                    )
                    
                    self.stats["total_decisions"] += 1
                    self.stats["rotations_triggered"] += 1
                    
                    logger.info(
                        "ROTATION_CLOSE",
                        weakest=weakest.symbol,
                        weakest_score=f"{weakest.score:.2f}",
                        new_symbol=symbol,
                        new_score=f"{candidate_score:.2f}",
                        threshold=f"{self.config.rotation_switch_threshold:.2%}",
                        improvement=f"{score_improvement:.2%}",
                    )
                    
                    return decision, record
            
            # Slots full, no rotation
            decision = SlotDecision.BLOCKED_SLOTS_FULL
            record = SlotDecisionRecord(
                timestamp=timestamp,
                decision=decision,
                symbol=symbol,
                action=action,
                candidate_score=candidate_score,
                desired_slots=desired_slots,
                open_positions_count=open_positions_count,
                regime=regime,
                block_reason=f"Slots full ({open_positions_count}/{desired_slots}), candidate not better enough",
            )
            
            self.stats["total_decisions"] += 1
            self.stats["slots_full_blocks"] += 1
            
            logger.info(
                "BLOCKED_SLOTS_FULL",
                symbol=symbol,
                score=f"{candidate_score:.2f}",
                slots=f"{open_positions_count}/{desired_slots}",
            )
            
            return decision, record
        
        # Step 8: Allow (slots available)
        decision = SlotDecision.ALLOW
        record = SlotDecisionRecord(
            timestamp=timestamp,
            decision=decision,
            symbol=symbol,
            action=action,
            candidate_score=candidate_score,
            desired_slots=desired_slots,
            open_positions_count=open_positions_count,
            regime=regime,
        )
        
        self.stats["total_decisions"] += 1
        self.stats["allows"] += 1
        
        logger.info(
            "ACTIVE_SLOTS_ALLOW",
            symbol=symbol,
            score=f"{candidate_score:.2f}",
            slots=f"{open_positions_count + 1}/{desired_slots}",
        )
        
        return decision, record
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics"""
        return {
            **self.stats,
            "current_state": self.current_state.__dict__ if self.current_state else None,
        }
