#!/usr/bin/env python3
"""
P1-B: Execution Policy & Capital Deployment
Hard controls on WHEN and HOW MUCH capital is deployed.

Policy Decision Flow:
    trade.intent → allow_new_entry() → compute_order_size() → place_order()

All entry decisions centralized. No silent blocks.
"""
import os
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Execution policy decisions - all must be logged explicitly"""
    ALLOW_NEW_ENTRY = "ALLOW_NEW_ENTRY"
    ALLOW_SCALE_IN = "ALLOW_SCALE_IN"
    BLOCK_MAX_POSITIONS = "BLOCK_MAX_POSITIONS"
    BLOCK_MAX_EXPOSURE = "BLOCK_MAX_EXPOSURE"
    BLOCK_COOLDOWN = "BLOCK_COOLDOWN"
    BLOCK_EXISTING_POSITION = "BLOCK_EXISTING_POSITION"
    BLOCK_LOW_CONFIDENCE = "BLOCK_LOW_CONFIDENCE"
    BLOCK_REGIME_RULE = "BLOCK_REGIME_RULE"
    BLOCK_SCALE_IN_RULE = "BLOCK_SCALE_IN_RULE"
    BLOCK_INSUFFICIENT_CAPITAL = "BLOCK_INSUFFICIENT_CAPITAL"


@dataclass
class PolicyConfig:
    """Execution policy configuration - all limits in one place"""
    
    # Position limits
    max_open_positions_total: int
    max_open_positions_per_symbol: int
    max_open_positions_per_regime: int
    
    # Capital limits (USDT)
    max_total_exposure_usdt: float
    max_exposure_per_symbol_usdt: float
    
    # Scale-in controls
    allow_scale_in: bool
    scale_in_max_count: int
    scale_in_confidence_delta: float  # New confidence must exceed old by this
    
    # Cooldown controls (seconds)
    cooldown_seconds_per_symbol: int
    cooldown_seconds_global: int
    
    # Confidence threshold
    min_confidence: float
    
    @classmethod
    def from_env(cls) -> 'PolicyConfig':
        """Load configuration from environment variables"""
        return cls(
            max_open_positions_total=int(os.getenv("POLICY_MAX_POSITIONS_TOTAL", "10")),
            max_open_positions_per_symbol=int(os.getenv("POLICY_MAX_PER_SYMBOL", "3")),
            max_open_positions_per_regime=int(os.getenv("POLICY_MAX_PER_REGIME", "5")),
            
            max_total_exposure_usdt=float(os.getenv("POLICY_MAX_EXPOSURE_TOTAL", "5000.0")),
            max_exposure_per_symbol_usdt=float(os.getenv("POLICY_MAX_EXPOSURE_PER_SYMBOL", "1000.0")),
            
            allow_scale_in=os.getenv("POLICY_ALLOW_SCALE_IN", "true").lower() == "true",
            scale_in_max_count=int(os.getenv("POLICY_SCALE_IN_MAX", "2")),
            scale_in_confidence_delta=float(os.getenv("POLICY_SCALE_IN_CONF_DELTA", "0.05")),
            
            cooldown_seconds_per_symbol=int(os.getenv("POLICY_COOLDOWN_SYMBOL", "300")),  # 5 min
            cooldown_seconds_global=int(os.getenv("POLICY_COOLDOWN_GLOBAL", "60")),  # 1 min
            
            min_confidence=float(os.getenv("POLICY_MIN_CONFIDENCE", "0.7"))
        )


@dataclass
class PortfolioState:
    """Current portfolio state for policy decisions"""
    total_positions: int
    positions_by_symbol: Dict[str, List[Dict]]  # symbol -> [positions]
    positions_by_regime: Dict[str, int]  # regime -> count
    total_exposure_usdt: float
    exposure_by_symbol: Dict[str, float]  # symbol -> USDT exposure
    available_capital_usdt: float
    last_trade_time: float  # Global last trade timestamp
    last_trade_by_symbol: Dict[str, float]  # symbol -> last trade timestamp


class ExecutionPolicy:
    """
    Central execution policy - all entry decisions go through here.
    
    Purpose:
        - Control WHEN system opens new positions
        - Control HOW MUCH capital is deployed
        - Prioritize existing positions
        
    Does NOT:
        - Modify AI signals
        - Change Exit Brain logic
        - Handle observability
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig.from_env()
        logger.info(f"🛡️ Execution Policy initialized")
        logger.info(f"   Max positions: {self.config.max_open_positions_total} total, "
                   f"{self.config.max_open_positions_per_symbol} per symbol")
        logger.info(f"   Max exposure: ${self.config.max_total_exposure_usdt:.0f} total, "
                   f"${self.config.max_exposure_per_symbol_usdt:.0f} per symbol")
        logger.info(f"   Scale-in: {'ENABLED' if self.config.allow_scale_in else 'DISABLED'} "
                   f"(max {self.config.scale_in_max_count} times)")
        logger.info(f"   Cooldown: {self.config.cooldown_seconds_per_symbol}s per symbol, "
                   f"{self.config.cooldown_seconds_global}s global")
    
    def allow_new_entry(
        self,
        intent: Dict,
        portfolio: PortfolioState
    ) -> Tuple[PolicyDecision, str]:
        """
        Decide if new entry is allowed.
        
        Args:
            intent: Trade intent with symbol, side, confidence, price, qty, etc.
            portfolio: Current portfolio state
            
        Returns:
            (decision, reason_detail)
            
        Decision priority (first match wins):
            1. Confidence check
            2. Global cooldown
            3. Symbol cooldown
            4. Max total positions
            5. Max total exposure
            6. Existing position check (scale-in logic)
            7. Max exposure per symbol
            8. Regime limits
            9. ALLOW_NEW_ENTRY
        """
        symbol = intent.get("symbol", "").upper()
        confidence = intent.get("confidence", 0.0)
        side = intent.get("side", "").upper()
        price = intent.get("entry_price", intent.get("price", 0.0))
        qty = intent.get("quantity", 0.0)
        regime = intent.get("regime", "unknown")
        
        # 1. Confidence threshold
        if confidence < self.config.min_confidence:
            reason = (f"confidence={confidence:.2f} < {self.config.min_confidence:.2f}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_LOW_CONFIDENCE | {reason}")
            return PolicyDecision.BLOCK_LOW_CONFIDENCE, reason
        
        # 2. Global cooldown
        now = time.time()
        global_cooldown_remaining = self.config.cooldown_seconds_global - (now - portfolio.last_trade_time)
        if global_cooldown_remaining > 0:
            reason = f"global_cooldown={global_cooldown_remaining:.0f}s remaining"
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_COOLDOWN | {reason}")
            return PolicyDecision.BLOCK_COOLDOWN, reason
        
        # 3. Symbol-specific cooldown
        last_symbol_trade = portfolio.last_trade_by_symbol.get(symbol, 0)
        symbol_cooldown_remaining = self.config.cooldown_seconds_per_symbol - (now - last_symbol_trade)
        if symbol_cooldown_remaining > 0:
            reason = f"symbol_cooldown={symbol_cooldown_remaining:.0f}s remaining"
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_COOLDOWN | {reason}")
            return PolicyDecision.BLOCK_COOLDOWN, reason
        
        # 4. Max total positions
        if portfolio.total_positions >= self.config.max_open_positions_total:
            reason = (f"total_positions={portfolio.total_positions} >= "
                     f"max={self.config.max_open_positions_total}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_MAX_POSITIONS | {reason}")
            return PolicyDecision.BLOCK_MAX_POSITIONS, reason
        
        # 5. Max total exposure
        estimated_exposure = price * qty if price and qty else 0
        if portfolio.total_exposure_usdt + estimated_exposure > self.config.max_total_exposure_usdt:
            reason = (f"total_exposure=${portfolio.total_exposure_usdt:.0f} + "
                     f"${estimated_exposure:.0f} > max=${self.config.max_total_exposure_usdt:.0f}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_MAX_EXPOSURE | {reason}")
            return PolicyDecision.BLOCK_MAX_EXPOSURE, reason
        
        # 6. Existing position check - SCALE-IN LOGIC
        existing_positions = portfolio.positions_by_symbol.get(symbol, [])
        if existing_positions:
            # Check if scale-in is allowed
            if not self.config.allow_scale_in:
                reason = f"existing_position=true, scale_in=DISABLED"
                logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                           f"decision=BLOCK_EXISTING_POSITION | {reason}")
                return PolicyDecision.BLOCK_EXISTING_POSITION, reason
            
            # Check scale-in count limit
            position_count = len(existing_positions)
            if position_count >= self.config.scale_in_max_count:
                reason = (f"scale_in_count={position_count} >= "
                         f"max={self.config.scale_in_max_count}")
                logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                           f"decision=BLOCK_SCALE_IN_RULE | {reason}")
                return PolicyDecision.BLOCK_SCALE_IN_RULE, reason
            
            # Check if same direction
            same_direction = any(
                (pos.get("side") == "long" and side == "BUY") or
                (pos.get("side") == "short" and side == "SELL")
                for pos in existing_positions
            )
            if not same_direction:
                reason = f"existing_direction={existing_positions[0].get('side')}, new_side={side}"
                logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                           f"decision=BLOCK_SCALE_IN_RULE | opposite_direction | {reason}")
                return PolicyDecision.BLOCK_SCALE_IN_RULE, reason
            
            # Check confidence improvement requirement
            existing_confidence = max(pos.get("confidence", 0) for pos in existing_positions)
            if confidence <= existing_confidence + self.config.scale_in_confidence_delta:
                reason = (f"new_confidence={confidence:.2f} <= "
                         f"existing={existing_confidence:.2f} + "
                         f"delta={self.config.scale_in_confidence_delta:.2f}")
                logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                           f"decision=BLOCK_SCALE_IN_RULE | insufficient_confidence | {reason}")
                return PolicyDecision.BLOCK_SCALE_IN_RULE, reason
        
        # 7. Max exposure per symbol (after scale-in)
        current_symbol_exposure = portfolio.exposure_by_symbol.get(symbol, 0.0)
        if current_symbol_exposure + estimated_exposure > self.config.max_exposure_per_symbol_usdt:
            reason = (f"symbol_exposure=${current_symbol_exposure:.0f} + "
                     f"${estimated_exposure:.0f} > "
                     f"max=${self.config.max_exposure_per_symbol_usdt:.0f}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_MAX_EXPOSURE | per_symbol | {reason}")
            return PolicyDecision.BLOCK_MAX_EXPOSURE, reason
        
        # 8. Max positions per symbol
        if len(existing_positions) + 1 > self.config.max_open_positions_per_symbol:
            reason = (f"symbol_positions={len(existing_positions) + 1} > "
                     f"max={self.config.max_open_positions_per_symbol}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_MAX_POSITIONS | per_symbol | {reason}")
            return PolicyDecision.BLOCK_MAX_POSITIONS, reason
        
        # 9. Regime limits
        current_regime_count = portfolio.positions_by_regime.get(regime, 0)
        if current_regime_count >= self.config.max_open_positions_per_regime:
            reason = (f"regime={regime}, positions={current_regime_count} >= "
                     f"max={self.config.max_open_positions_per_regime}")
            logger.info(f"🚫 ENTRY_POLICY_DECISION | {symbol} | "
                       f"decision=BLOCK_REGIME_RULE | {reason}")
            return PolicyDecision.BLOCK_REGIME_RULE, reason
        
        # ✅ ALL CHECKS PASSED
        decision = PolicyDecision.ALLOW_SCALE_IN if existing_positions else PolicyDecision.ALLOW_NEW_ENTRY
        reason = (f"confidence={confidence:.2f}, "
                 f"exposure_after=${portfolio.total_exposure_usdt + estimated_exposure:.0f}, "
                 f"positions_after={portfolio.total_positions + 1}")
        
        logger.info(f"✅ ENTRY_POLICY_DECISION | {symbol} | "
                   f"decision={decision.value} | {reason}")
        
        return decision, reason
    
    def compute_order_size(
        self,
        intent: Dict,
        portfolio: PortfolioState,
        risk_score: float = 1.0
    ) -> float:
        """
        Calculate order size with hard capital controls.
        
        Args:
            intent: Trade intent
            portfolio: Current portfolio state
            risk_score: Risk multiplier from Risk Brain (0.0-2.0, default 1.0)
            
        Returns:
            Order quantity (0 if insufficient capital)
        """
        symbol = intent.get("symbol", "").upper()
        price = intent.get("entry_price", intent.get("price", 0.0))
        confidence = intent.get("confidence", 0.0)
        leverage = intent.get("leverage", 1)
        
        if not price or price <= 0:
            logger.error(f"❌ CAPITAL_ALLOCATION | {symbol} | invalid_price={price}")
            return 0.0
        
        # 1. Calculate available capital (after all existing exposures)
        remaining_capital = portfolio.available_capital_usdt
        
        # 2. Calculate capital allocation for this trade
        # Base allocation: percentage of remaining capital
        base_allocation_pct = 0.20  # 20% of remaining capital per trade
        
        # Adjust by confidence (0.7-1.0 conf → 0.85-1.15 multiplier)
        confidence_multiplier = 0.5 + (confidence * 0.65)
        
        # Adjust by risk score from Risk Brain
        risk_multiplier = risk_score
        
        # Calculate USDT allocation
        allocation_usdt = remaining_capital * base_allocation_pct * confidence_multiplier * risk_multiplier
        
        # 3. Apply hard caps
        # Cap at max exposure per symbol
        current_symbol_exposure = portfolio.exposure_by_symbol.get(symbol, 0.0)
        max_additional_exposure = self.config.max_exposure_per_symbol_usdt - current_symbol_exposure
        allocation_usdt = min(allocation_usdt, max_additional_exposure)
        
        # Cap at remaining total exposure capacity
        total_capacity_remaining = self.config.max_total_exposure_usdt - portfolio.total_exposure_usdt
        allocation_usdt = min(allocation_usdt, total_capacity_remaining)
        
        # 4. Convert USDT to quantity (accounting for leverage)
        # Notional value = allocation_usdt * leverage
        notional_value = allocation_usdt * leverage
        quantity = notional_value / price
        
        # 5. Validate minimum allocation
        min_allocation_usdt = 10.0  # $10 minimum trade
        if allocation_usdt < min_allocation_usdt:
            logger.warning(
                f"⚠️ CAPITAL_ALLOCATION | {symbol} | "
                f"allocation=${allocation_usdt:.2f} < min=${min_allocation_usdt:.2f} | "
                f"BLOCK_INSUFFICIENT_CAPITAL"
            )
            return 0.0
        
        logger.info(
            f"💰 CAPITAL_ALLOCATION | {symbol} | "
            f"qty={quantity:.4f} | "
            f"allocation=${allocation_usdt:.2f} | "
            f"notional=${notional_value:.2f} | "
            f"leverage={leverage}x | "
            f"price=${price:.4f} | "
            f"confidence={confidence:.2f} | "
            f"risk_score={risk_score:.2f} | "
            f"remaining_capital=${remaining_capital:.0f} | "
            f"exposure_after=${portfolio.total_exposure_usdt + allocation_usdt:.0f}"
        )
        
        return quantity
