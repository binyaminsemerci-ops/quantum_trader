"""
Exit Brain v3 Planner - Core orchestration logic for exit strategies.
Phase 4M+ Integration: Uses Cross-Exchange Intelligence for adaptive TP/SL
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from backend.domains.exits.exit_brain_v3.models import (
    ExitContext,
    ExitLeg,
    ExitPlan,
    ExitKind
)
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
    MarketRegime,
    TPProfile,
    TrailingProfile,
    get_tp_and_trailing_profile,
    calculate_tp_price,
    get_trailing_callback_for_profit
)
from backend.domains.exits.exit_brain_v3.dynamic_tp_calculator import (
    DynamicTPCalculator,
    calculate_dynamic_tp_levels
)
from backend.domains.exits.exit_brain_v3.cross_exchange_adapter import (
    CrossExchangeAdapter,
    get_cross_exchange_adapter
)

logger = logging.getLogger(__name__)


class ExitBrainV3:
    """
    Central orchestrator for ALL exit logic.
    
    Responsibilities:
    - Aggregate context from RL, Risk v3, market conditions
    - Decide optimal exit strategy (TP/SL/trailing/partial)
    - Produce unified ExitPlan
    - Coordinate with existing execution layers
    
    Does NOT:
    - Place orders directly (delegates to executor/position_monitor)
    - Override emergency risk controls (respects Risk v3)
    - Replace existing modules (thin coordination layer)
    """
    
    def __init__(self, config: Optional[Dict] = None, redis_client=None):
        """
        Initialize Exit Brain v3.
        
        Args:
            config: Optional configuration overrides
            redis_client: Redis client for cross-exchange adapter
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Profile system configuration
        self.use_profiles = self.config.get("use_profiles", True)  # NEW: Enable profile system
        self.strategy_id = self.config.get("strategy_id", "RL_V3")  # Default strategy ID
        
        # Dynamic TP calculator
        self.use_dynamic_tp = self.config.get("use_dynamic_tp", True)  # AI-driven TP sizing
        self.dynamic_tp_calculator = DynamicTPCalculator() if self.use_dynamic_tp else None
        
        # Phase 4M+: Cross-Exchange Intelligence adapter
        self.use_cross_exchange = self.config.get("use_cross_exchange", True)
        self.cross_exchange_adapter: Optional[CrossExchangeAdapter] = None
        if self.use_cross_exchange and redis_client:
            self.cross_exchange_adapter = get_cross_exchange_adapter(redis_client=redis_client)
            self.logger.info("[EXIT BRAIN] ✓ Cross-Exchange Intelligence enabled")
        else:
            self.logger.info("[EXIT BRAIN] Cross-Exchange Intelligence disabled")
        
        # Legacy config (for backward compatibility if use_profiles=False)
        self.default_tp_pct = self.config.get("default_tp_pct", 0.03)  # 3%
        self.default_sl_pct = self.config.get("default_sl_pct", 0.025)  # 2.5%
        self.default_trail_pct = self.config.get("default_trail_pct", 0.015)  # 1.5%
        
        # Partial exit ladder (legacy template)
        self.partial_template = self.config.get("partial_template", {
            "tp1": {"size_pct": 0.25, "multiplier": 0.5},  # 25% at 0.5R (1.5%)
            "tp2": {"size_pct": 0.25, "multiplier": 1.0},  # 25% at 1.0R (3.0%)
            "tp3": {"size_pct": 0.50, "multiplier": 2.0},  # 50% trailing at 2.0R (6.0%)
        })
        
        # Risk mode adjustments
        self.risk_mode_multipliers = {
            "NORMAL": 1.0,
            "CONSERVATIVE": 0.7,  # Tighter TP/SL
            "CRITICAL": 0.5,      # Very tight exits
            "ESS_ACTIVE": 0.3     # Emergency mode
        }
        
        # Market regime adjustments (legacy, used if use_profiles=False)
        self.regime_adjustments = {
            "NORMAL": {"tp_mult": 1.0, "sl_mult": 1.0, "trail": True},
            "VOLATILE": {"tp_mult": 1.3, "sl_mult": 1.5, "trail": False},  # Wider SL, closer TP
            "TRENDING": {"tp_mult": 1.5, "sl_mult": 0.8, "trail": True},   # Let profits run
            "RANGE_BOUND": {"tp_mult": 0.8, "sl_mult": 1.2, "trail": False}  # Quick exits
        }
    
    async def build_exit_plan(self, ctx: ExitContext) -> ExitPlan:
        """
        Build unified exit strategy from context.
        
        Decision logic:
        1. Check risk mode → adjust targets if CRITICAL/ESS
        2. Check RL hints → incorporate if confidence > threshold
        3. Check market regime → adjust TP/SL width
        4. Check unrealized PnL → may tighten if already profitable
        5. Build leg structure (TP ladder, SL, trailing)
        
        Args:
            ctx: Complete exit context
            
        Returns:
            ExitPlan with coordinated legs
        """
        self.logger.info(
            f"[EXIT BRAIN] Building plan for {ctx.symbol}: "
            f"Side={ctx.side}, PnL={ctx.unrealized_pnl_pct:.2f}%, "
            f"Risk={ctx.risk_mode}, Regime={ctx.market_regime}"
        )
        
        # Step 1: Determine base targets from RL or defaults
        base_tp_pct, base_sl_pct = self._get_base_targets(ctx)
        
        # Step 1.5: Phase 4M+ - Apply cross-exchange adjustments
        cross_exchange_adjustments = None
        if self.cross_exchange_adapter:
            try:
                # Get global volatility state from cross-exchange data
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, use await
                    state = asyncio.create_task(self.cross_exchange_adapter.get_global_volatility_state())
                else:
                    # Sync context, run in event loop
                    state = loop.run_until_complete(self.cross_exchange_adapter.get_global_volatility_state())
                
                # Calculate ATR/TP/SL adjustments
                base_atr = 0.02  # 2% baseline ATR (can be overridden by ctx if available)
                cross_exchange_adjustments = self.cross_exchange_adapter.calculate_adjustments(
                    state=state,
                    base_atr=base_atr,
                    base_tp=base_tp_pct,
                    base_sl=base_sl_pct
                )
                
                # Apply cross-exchange multipliers
                base_tp_pct *= cross_exchange_adjustments.tp_multiplier
                base_sl_pct *= cross_exchange_adjustments.sl_multiplier
                
                self.logger.info(
                    f"[EXIT BRAIN] 🌐 Cross-Exchange adjustments applied: {cross_exchange_adjustments.reasoning}"
                )
                
                # Publish status to Redis
                asyncio.create_task(self.cross_exchange_adapter.publish_status(cross_exchange_adjustments, state))
                
            except Exception as e:
                self.logger.warning(f"[EXIT BRAIN] Cross-Exchange adjustment failed: {e}, using base values")
        
        # Step 2: Apply risk mode adjustments
        risk_mult = self.risk_mode_multipliers.get(ctx.risk_mode, 1.0)
        tp_pct = base_tp_pct * risk_mult
        # CRITICAL mode: tighter SL (smaller loss), NORMAL: default SL
        # Risk mult: CRITICAL=0.5, NORMAL=1.0 → SL should DECREASE with lower mult
        sl_pct = base_sl_pct * risk_mult  # Lower risk_mult = tighter SL
        
        # Step 3: Apply market regime adjustments
        regime_adj = self.regime_adjustments.get(ctx.market_regime, self.regime_adjustments["NORMAL"])
        tp_pct *= regime_adj["tp_mult"]
        sl_pct *= regime_adj["sl_mult"]
        trail_enabled = regime_adj["trail"] and ctx.trail_enabled
        
        # Override trailing if cross-exchange says no
        if cross_exchange_adjustments and not cross_exchange_adjustments.use_trailing:
            trail_enabled = False
            self.logger.info("[EXIT BRAIN] Trailing disabled by cross-exchange volatility")
        
        # Step 4: Enforce min/max bounds
        tp_pct = max(0.015, min(0.15, tp_pct))  # 1.5% - 15%
        sl_pct = max(0.01, min(0.05, sl_pct))   # 1.0% - 5.0%
        
        # Step 5: Build leg structure
        legs = []
        
        # Check if emergency exit needed
        if ctx.risk_mode == "ESS_ACTIVE":
            legs.append(self._build_emergency_leg(ctx))
            strategy_id = "EMERGENCY_EXIT"
            reason = f"ESS ACTIVE - immediate exit required"
            profile_name = None
        
        # Check if position already in significant profit (may adjust to lock profit)
        elif ctx.unrealized_pnl_pct > 10.0:
            legs = self._build_profit_lock_legs(ctx, tp_pct, sl_pct)
            strategy_id = "PROFIT_LOCK"
            reason = f"Lock profit at {ctx.unrealized_pnl_pct:.1f}% PnL"
            profile_name = None
        
        # Standard exit strategy: partial ladder + trailing
        else:
            legs = self._build_standard_legs(ctx, tp_pct, sl_pct, trail_enabled)
            strategy_id = "STANDARD_LADDER"
            reason = f"3-leg strategy: TP={tp_pct:.1f}%, SL={sl_pct:.1f}%"
            # Extract profile name from legs metadata if using profiles
            profile_name = None
            if legs and "profile_name" in legs[0].metadata:
                profile_name = legs[0].metadata["profile_name"]
        
        # Build plan
        plan = ExitPlan(
            symbol=ctx.symbol,
            legs=legs,
            strategy_id=strategy_id,
            source="EXIT_BRAIN_V3",
            reason=reason,
            confidence=ctx.rl_confidence or 0.75,
            profile_name=profile_name,
            market_regime=ctx.market_regime,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        
        self.logger.info(
            f"[EXIT BRAIN] Plan created for {ctx.symbol}: {len(legs)} legs, "
            f"Strategy={strategy_id}, TP={plan.total_tp_pct:.0%}, SL={plan.total_sl_pct:.0%}, "
            f"Profile={profile_name or 'legacy'}"
        )

        return plan
    
    def _get_base_targets(self, ctx: ExitContext) -> tuple[float, float]:
        """Determine base TP/SL from RL hints or defaults"""
        tp_pct = ctx.rl_tp_hint or self.default_tp_pct
        sl_pct = ctx.rl_sl_hint or self.default_sl_pct
        
        # Validate RL hints are reasonable
        if ctx.rl_tp_hint:
            if ctx.rl_confidence and ctx.rl_confidence < 0.5:
                # Low confidence → use defaults
                tp_pct = self.default_tp_pct
                self.logger.debug(f"[EXIT BRAIN] Low RL confidence, using default TP")
        
        return tp_pct, sl_pct
    
    def _build_standard_legs(
        self,
        ctx: ExitContext,
        tp_pct: float,
        sl_pct: float,
        trail_enabled: bool
    ) -> List[ExitLeg]:
        """
        Build standard exit strategy legs.
        
        NEW: Uses TP Profile system if use_profiles=True
        LEGACY: Falls back to old partial_template if use_profiles=False
        """
        if self.use_profiles:
            return self._build_profile_based_legs(ctx, sl_pct)
        else:
            return self._build_legacy_legs(ctx, tp_pct, sl_pct, trail_enabled)
    
    def _build_profile_based_legs(
        self,
        ctx: ExitContext,
        sl_pct: float
    ) -> List[ExitLeg]:
        """Build legs using TP Profile system with optional dynamic TP sizing"""
        legs = []
        
        # Map context regime to MarketRegime enum
        regime_map = {
            "NORMAL": MarketRegime.NORMAL,
            "VOLATILE": MarketRegime.VOLATILE,
            "TRENDING": MarketRegime.TREND,
            "RANGE_BOUND": MarketRegime.RANGE,
            "CHOP": MarketRegime.CHOP
        }
        regime = regime_map.get(ctx.market_regime, MarketRegime.NORMAL)
        
        # Try dynamic TP first if enabled
        tp_profile = None
        trailing_profile = None
        
        if self.use_dynamic_tp and self.dynamic_tp_calculator:
            from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import build_dynamic_tp_profile
            tp_profile = build_dynamic_tp_profile(ctx)
            
            if tp_profile:
                position_size_usd = ctx.size * ctx.entry_price
                self.logger.info(
                    f"[EXIT BRAIN] Using DYNAMIC TP for {ctx.symbol}: "
                    f"Profile='{tp_profile.name}', Size=${position_size_usd:.0f}, "
                    f"Leverage={ctx.leverage}x, Volatility={ctx.volatility if ctx.volatility else 0.025:.2%}"
                )
                self.logger.info(
                    f"[EXIT BRAIN] DYNAMIC TP: {tp_profile.description}"
                )
        
        # Fallback to static profile if dynamic failed or disabled
        if tp_profile is None:
            tp_profile, trailing_profile = get_tp_and_trailing_profile(
                symbol=ctx.symbol,
                strategy_id=self.strategy_id,
                regime=regime
            )
            self.logger.info(
                f"[EXIT BRAIN] Using STATIC profile '{tp_profile.name}' for {ctx.symbol} "
                f"(regime={regime.value}, SL={sl_pct:.1%})"
            )
        
        # Build TP legs from profile (works for both dynamic and static)
        for idx, profile_leg in enumerate(tp_profile.tp_legs):
            # Calculate TP price from R multiple
            r_multiple = profile_leg.r_multiple
            tp_pct = r_multiple * sl_pct  # R = SL distance
            
            is_dynamic = "DYNAMIC" in tp_profile.name
            legs.append(ExitLeg(
                kind=ExitKind.TP,
                size_pct=profile_leg.size_fraction,
                trigger_pct=tp_pct,
                priority=idx,
                reason=f"TP{idx+1}: {r_multiple:.1f}R @ {tp_pct:.1%} ({'DYNAMIC' if is_dynamic else profile_leg.kind.value})",
                r_multiple=r_multiple,
                profile_leg_index=idx,
                metadata={
                    "profile_name": tp_profile.name,
                    "kind": profile_leg.kind.value,
                    "dynamic": is_dynamic
                }
            ))
        
        # Add trailing leg if profile has trailing configured
        if trailing_profile:
            # Check if profile already uses remaining size for trailing
            # (some profiles include trailing in tp_legs)
            used_size = sum(leg.size_pct for leg in legs)
            trailing_size = 1.0 - used_size
            
            if trailing_size > 0.01:  # Only add if significant size remains
                # Determine activation point
                activation_r = trailing_profile.activation_r
                activation_pct = activation_r * sl_pct
                
                legs.append(ExitLeg(
                    kind=ExitKind.TRAIL,
                    size_pct=trailing_size,
                    trigger_pct=activation_pct,  # Start trailing at this profit level
                    trail_callback=trailing_profile.callback_pct,
                    priority=len(legs),
                    reason=f"TRAIL: {trailing_size:.0%} @ {trailing_profile.callback_pct:.1%} callback",
                    metadata={
                        "profile_name": tp_profile.name,
                        "activation_r": activation_r,
                        "tightening_curve": trailing_profile.tightening_curve
                    }
                ))
        
        # Add SL leg
        legs.append(ExitLeg(
            kind=ExitKind.SL,
            size_pct=1.0,
            trigger_pct=-sl_pct,
            priority=0,  # Highest priority
            reason=f"SL: Risk limit at {sl_pct:.1%}",
            metadata={"profile_name": tp_profile.name}
        ))
        
        return legs
    
    def _build_legacy_legs(
        self,
        ctx: ExitContext,
        tp_pct: float,
        sl_pct: float,
        trail_enabled: bool
    ) -> List[ExitLeg]:
        """Build legs using legacy partial_template (BACKWARD COMPATIBLE)"""
        legs = []
        
        # TP1: 25% at 50% of target (early profit lock)
        tp1_pct = tp_pct * self.partial_template["tp1"]["multiplier"]
        legs.append(ExitLeg(
            kind=ExitKind.TP,
            size_pct=self.partial_template["tp1"]["size_pct"],
            trigger_pct=tp1_pct,
            priority=0,
            reason=f"TP1: Quick profit at {tp1_pct:.1f}%"
        ))
        
        # TP2: 25% at 100% of target
        tp2_pct = tp_pct * self.partial_template["tp2"]["multiplier"]
        legs.append(ExitLeg(
            kind=ExitKind.TP,
            size_pct=self.partial_template["tp2"]["size_pct"],
            trigger_pct=tp2_pct,
            priority=1,
            reason=f"TP2: Core target at {tp2_pct:.1f}%"
        ))
        
        # TP3: 50% with trailing (if enabled) or static TP
        if trail_enabled:
            trail_callback = ctx.trail_hint or self.default_trail_pct
            legs.append(ExitLeg(
                kind=ExitKind.TRAIL,
                size_pct=self.partial_template["tp3"]["size_pct"],
                trail_callback=trail_callback,
                priority=2,
                reason=f"TP3: Trailing stop {trail_callback:.1f}%"
            ))
        else:
            tp3_pct = tp_pct * self.partial_template["tp3"]["multiplier"]
            legs.append(ExitLeg(
                kind=ExitKind.TP,
                size_pct=self.partial_template["tp3"]["size_pct"],
                trigger_pct=tp3_pct,
                priority=2,
                reason=f"TP3: Extended target at {tp3_pct:.1f}%"
            ))
        
        # SL: 100% at stop loss
        legs.append(ExitLeg(
            kind=ExitKind.SL,
            size_pct=1.0,
            trigger_pct=-sl_pct,  # Negative = below entry
            priority=0,  # Highest priority
            reason=f"SL: Risk limit at {sl_pct:.1f}%"
        ))
        
        return legs
    
    def _build_profit_lock_legs(
        self,
        ctx: ExitContext,
        tp_pct: float,
        sl_pct: float
    ) -> List[ExitLeg]:
        """Build strategy to lock existing profit"""
        legs = []
        
        # Tighten SL to breakeven + fees (lock profit)
        breakeven_pct = 0.002  # 0.2% above entry (covers fees)
        legs.append(ExitLeg(
            kind=ExitKind.SL,
            size_pct=1.0,
            trigger_pct=breakeven_pct,
            priority=0,
            reason=f"SL: Lock profit at breakeven+fees"
        ))
        
        # Single TP at extension
        extended_tp = ctx.unrealized_pnl_pct * 1.5  # 50% above current
        legs.append(ExitLeg(
            kind=ExitKind.TP,
            size_pct=1.0,
            trigger_pct=extended_tp,
            priority=1,
            reason=f"TP: Extended target {extended_tp:.1f}%"
        ))
        
        return legs
    
    def _build_emergency_leg(self, ctx: ExitContext) -> ExitLeg:
        """Build emergency exit leg for ESS"""
        return ExitLeg(
            kind=ExitKind.EMERGENCY,
            size_pct=1.0,  # Close entire position
            trigger_price=None,  # Market order
            priority=0,
            condition="IMMEDIATE",
            reason="ESS ACTIVE - emergency system shutdown"
        )
