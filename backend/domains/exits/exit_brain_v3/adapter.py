"""
Exit Brain v3 - Adapter for Dynamic AI Exit Decisions

Bridges Exit Brain v3 planner with dynamic executor.
Pure BRAIN interface - no order placement.

Phase 2A: Makes decisions, logs them (shadow mode).
Phase 2B: Same decisions will translate to orders via exit_order_gateway.
"""
import logging
from typing import Optional
from datetime import datetime

from .types import PositionContext, ExitDecision, ExitDecisionType
from .models import ExitContext, ExitKind
from .planner import ExitBrainV3
from .router import ExitRouter
from .integration import ExitPlan

logger = logging.getLogger(__name__)


class ExitBrainAdapter:
    """
    Adapter that translates Exit Brain v3 plans into dynamic exit decisions.
    
    Responsibilities:
    - Given PositionContext, fetch/build ExitPlan via Exit Brain v3
    - Interpret plan + current market state to decide what to do NOW
    - Return structured ExitDecision
    - Pure BRAIN - no order placement
    
    Design:
    - Dynamic, not hardcoded TP/SL thresholds
    - Uses AI hints (RL outputs, regime, volatility) when available
    - Extensible for future AI improvements
    """
    
    def __init__(
        self,
        planner: Optional[ExitBrainV3] = None,
        router: Optional[ExitRouter] = None
    ):
        """
        Initialize adapter.
        
        Args:
            planner: Exit Brain v3 planner (creates plans)
            router: Exit Brain v3 router (caches plans)
        """
        # Initialize planner with dynamic TP enabled
        planner_config = {
            "use_profiles": True,
            "use_dynamic_tp": True,  # 🎯 Enable AI-driven TP sizing
            "strategy_id": "RL_V3"
        }
        self.planner = planner or ExitBrainV3(config=planner_config)
        self.router = router or ExitRouter()  # Singleton via __new__
        self.logger = logging.getLogger(__name__ + ".adapter")
    
    async def decide(self, ctx: PositionContext) -> ExitDecision:
        """
        Make AI-driven exit decision for position.
        
        Phase 2A: Returns decision for logging (shadow mode).
        Phase 2B: Same decision will translate to orders.
        
        Args:
            ctx: Complete position context
            
        Returns:
            ExitDecision with concrete action to take
        """
        try:
            # Get or create Exit Brain plan for this position
            plan = await self._get_or_create_plan(ctx)
            
            if not plan:
                self.logger.warning(
                    f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: No plan available, "
                    f"returning NO_CHANGE"
                )
                return ExitDecision(
                    decision_type=ExitDecisionType.NO_CHANGE,
                    symbol=ctx.symbol,
                    reason="No Exit Brain plan available",
                    current_price=ctx.current_price,
                    unrealized_pnl=ctx.unrealized_pnl
                )
            
            # Interpret plan based on current state
            decision = self._interpret_plan(ctx, plan)
            
            self.logger.debug(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: {decision.summary()}"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Error making decision: {e}",
                exc_info=True
            )
            # Safe fallback: no change
            return ExitDecision(
                decision_type=ExitDecisionType.NO_CHANGE,
                symbol=ctx.symbol,
                reason=f"Error in adapter: {e}",
                current_price=ctx.current_price,
                unrealized_pnl=ctx.unrealized_pnl
            )
    
    async def _get_or_create_plan(self, ctx: PositionContext) -> Optional[ExitPlan]:
        """
        Get existing plan or create new one.
        
        Args:
            ctx: Position context
            
        Returns:
            ExitPlan or None
        """
        # Check for existing plan
        existing_plan = self.router.get_active_plan(ctx.symbol)
        
        if existing_plan:
            self.logger.debug(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Using existing plan "
                f"(strategy={existing_plan.strategy_id})"
            )
            return existing_plan
        
        # Create new plan
        self.logger.info(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Creating new Exit Brain plan"
        )
        
        # Build plan using Exit Brain v3 planner directly
        # (Router's async method not compatible with sync decide())
        # Create ExitContext from PositionContext
        exit_ctx = ExitContext(
            symbol=ctx.symbol,
            side=ctx.side,
            entry_price=ctx.entry_price,
            current_price=ctx.current_price,
            size=ctx.size,
            unrealized_pnl_pct=ctx.unrealized_pnl,  # PositionContext.unrealized_pnl is % terms
            unrealized_pnl_usd=ctx.unrealized_pnl_abs,  # Use property for USD value
            signal_confidence=ctx.meta.get("confidence", 0.7),
            market_regime=ctx.regime or "NORMAL",
            volatility=ctx.meta.get("volatility", 0.02),
            trend_strength=ctx.meta.get("momentum", 0.0),
            leverage=ctx.leverage or 1.0
        )
        
        # Generate plan using planner (async method build_exit_plan)
        plan = await self.planner.build_exit_plan(exit_ctx)
        
        # 🔧 FIX: Cache plan in router so it persists between cycles
        self.router.active_plans[ctx.symbol] = plan
        self.logger.info(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Cached new plan - "
            f"Router now has {len(self.router.active_plans)} plans"
        )
        
        return plan
    
    def _interpret_plan(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """
        Interpret Exit Brain plan based on current market state.
        
        This is where dynamic AI decision-making happens.
        
        Strategy:
        1. Check if we should take partial profit (based on plan + current PnL)
        2. Check if we should move SL (trailing, breakeven, tighter)
        3. Check if we should exit fully (hit max loss, regime change, etc.)
        4. Default: no change (plan is working as expected)
        
        Args:
            ctx: Current position context
            plan: Exit Brain plan
            
        Returns:
            ExitDecision with concrete action
        """
        symbol = ctx.symbol
        current_pnl_pct = ctx.unrealized_pnl
        
        # Emergency full exit conditions
        if self._should_full_exit(ctx, plan):
            return self._decide_full_exit(ctx, plan)
        
        # Partial profit taking
        if self._should_take_partial_profit(ctx, plan):
            return self._decide_partial_close(ctx, plan)
        
        # Take profit limit placement/updates (place TP first if missing)
        # Priority: TP before SL, since TP defines profit targets
        if self._should_update_tp_limits(ctx, plan):
            return self._decide_update_tp_limits(ctx, plan)
        
        # Initial SL placement (CRITICAL: place hard SL if missing)
        if self._should_place_initial_sl(ctx, plan):
            return self._decide_place_initial_sl(ctx, plan)
        
        # Stop loss management (trailing, breakeven, tighter)
        if self._should_move_sl(ctx, plan):
            return self._decide_move_sl(ctx, plan)
        
        # Default: plan is working, no changes needed
        return ExitDecision(
            decision_type=ExitDecisionType.NO_CHANGE,
            symbol=symbol,
            reason="Plan working as expected, no action needed",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            meta={"plan_strategy": plan.strategy_id}
        )
    
    def _should_full_exit(self, ctx: PositionContext, plan: ExitPlan) -> bool:
        """Should we exit entire position immediately?"""
        # Emergency conditions:
        # 1. Hit maximum acceptable loss
        # 2. Extreme regime change (trend -> crash)
        # 3. Position duration exceeded safe limits
        
        # Example: If PnL < -5% and regime changed to high_vol/crash
        if ctx.unrealized_pnl < -5.0 and ctx.regime in ["high_vol", "crash"]:
            return True
        
        # Example: Position open too long in losing state
        if ctx.duration_hours and ctx.duration_hours > 48 and ctx.unrealized_pnl < -2.0:
            return True
        
        return False
    
    def _decide_full_exit(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """Create full exit decision."""
        return ExitDecision(
            decision_type=ExitDecisionType.FULL_EXIT_NOW,
            symbol=ctx.symbol,
            reason=f"Emergency exit: pnl={ctx.unrealized_pnl:.2f}%, regime={ctx.regime}",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            confidence=0.95,  # High confidence in emergency exits
            meta={"plan_strategy": plan.strategy_id, "emergency": True}
        )
    
    def _should_take_partial_profit(self, ctx: PositionContext, plan: ExitPlan) -> bool:
        """Should we take partial profit now?"""
        # Look at PARTIAL leg in plan
        if not plan.legs:
            return False
        
        partial_tp_leg = next((leg for leg in plan.legs if leg.kind == ExitKind.PARTIAL), None)
        if not partial_tp_leg:
            return False
        
        # Example: If current price reached or exceeded partial TP level
        if ctx.is_long:
            # For long: TP is above entry
            if ctx.current_price >= partial_tp_leg.trigger_price:
                return True
        else:
            # For short: TP is below entry
            if ctx.current_price <= partial_tp_leg.trigger_price:
                return True
        
        return False
    
    def _decide_partial_close(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """Create partial close decision."""
        # Get fraction from PARTIAL leg
        partial_tp_leg = next((leg for leg in plan.legs if leg.kind == ExitKind.PARTIAL), None)
        fraction = partial_tp_leg.size_pct if partial_tp_leg else 0.5  # Default 50%
        
        return ExitDecision(
            decision_type=ExitDecisionType.PARTIAL_CLOSE,
            symbol=ctx.symbol,
            fraction_to_close=fraction,
            reason=f"Partial TP hit: pnl={ctx.unrealized_pnl:.2f}%, taking {fraction:.0%} profit",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            confidence=0.85,
            meta={
                "plan_strategy": plan.strategy_id,
                "tp_price": partial_tp_leg.trigger_price if partial_tp_leg else None
            }
        )
    
    def _should_move_sl(self, ctx: PositionContext, plan: ExitPlan) -> bool:
        """
        Should we move stop loss?
        
        SL Movement Rules:
        1. Breakeven: When PnL > 1.5%, move SL to entry price (lock in zero loss)
        2. Trailing: When PnL > 4%, trail SL to lock in 25-50% of profit
        3. Always tighten: Never move SL further from price (increase risk)
        """
        # Get current active SL (if any) from executor state
        active_sl = ctx.meta.get("active_sl")
        
        if active_sl is None:
            # No SL set yet - this should be handled by _should_place_initial_sl
            return False
        
        # Rule 1: BREAKEVEN - Move SL to entry when position profitable
        # Trigger: PnL > 1.5% (configurable threshold)
        breakeven_threshold_pct = 1.5
        
        if ctx.unrealized_pnl > breakeven_threshold_pct:
            # Check if current SL is still below/above entry (not yet at breakeven)
            if ctx.is_long:
                # For LONG: SL below entry is worse, breakeven is better
                # Move SL UP to entry price
                if active_sl < ctx.entry_price * 0.999:  # Allow 0.1% buffer
                    self.logger.debug(
                        f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Should move SL to breakeven "
                        f"(PnL={ctx.unrealized_pnl:.2f}% > {breakeven_threshold_pct}%, "
                        f"current_sl={active_sl:.4f} < entry={ctx.entry_price:.4f})"
                    )
                    return True
            else:
                # For SHORT: SL above entry is worse, breakeven is better
                # Move SL DOWN to entry price
                if active_sl > ctx.entry_price * 1.001:  # Allow 0.1% buffer
                    self.logger.debug(
                        f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Should move SL to breakeven "
                        f"(PnL={ctx.unrealized_pnl:.2f}% > {breakeven_threshold_pct}%, "
                        f"current_sl={active_sl:.4f} > entry={ctx.entry_price:.4f})"
                    )
                    return True
        
        # Rule 2: TRAILING SL - Lock in profit when position deep in profit
        # Trigger: PnL > 4% (configurable threshold)
        trailing_threshold_pct = 4.0
        trail_distance_pct = 0.015  # Trail 1.5% from current price
        
        if ctx.unrealized_pnl > trailing_threshold_pct:
            # Calculate proposed trailing SL
            if ctx.is_long:
                proposed_trail_sl = ctx.current_price * (1 - trail_distance_pct)
                # Only trail if it TIGHTENS (moves UP for long)
                # This locks in profit without increasing risk
                if proposed_trail_sl > active_sl * 1.001:  # At least 0.1% improvement
                    # Calculate locked profit
                    locked_profit_pct = ((proposed_trail_sl - ctx.entry_price) / ctx.entry_price) * 100
                    self.logger.debug(
                        f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Should trail SL "
                        f"(PnL={ctx.unrealized_pnl:.2f}% > {trailing_threshold_pct}%, "
                        f"new_sl={proposed_trail_sl:.4f} > current_sl={active_sl:.4f}, "
                        f"locks_in={locked_profit_pct:.2f}% profit)"
                    )
                    return True
            else:
                proposed_trail_sl = ctx.current_price * (1 + trail_distance_pct)
                # Only trail if it TIGHTENS (moves DOWN for short)
                if proposed_trail_sl < active_sl * 0.999:  # At least 0.1% improvement
                    # Calculate locked profit
                    locked_profit_pct = ((ctx.entry_price - proposed_trail_sl) / ctx.entry_price) * 100
                    self.logger.debug(
                        f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Should trail SL "
                        f"(PnL={ctx.unrealized_pnl:.2f}% > {trailing_threshold_pct}%, "
                        f"new_sl={proposed_trail_sl:.4f} < current_sl={active_sl:.4f}, "
                        f"locks_in={locked_profit_pct:.2f}% profit)"
                    )
                    return True
        
        # No SL movement needed
        return False
    
    def _decide_move_sl(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """
        Create move SL decision.
        
        Three strategies:
        1. Breakeven: Move SL to entry price (lock in zero loss)
        2. Trailing: Trail SL behind price to lock in profit
        3. Never loosen: Only tighten SL, never move it away from price
        """
        # Get current active SL from executor state
        active_sl = ctx.meta.get("active_sl")
        
        # Thresholds
        breakeven_threshold = 1.5  # Move to breakeven at 1.5% profit
        trailing_threshold = 4.0   # Start trailing at 4% profit
        trail_distance_pct = 0.015  # Trail 1.5% from current price
        
        # Strategy 1: BREAKEVEN (1.5% <= PnL < 4%)
        if breakeven_threshold <= ctx.unrealized_pnl < trailing_threshold:
            # Move to entry price with small buffer for fees
            if ctx.is_long:
                new_sl = ctx.entry_price * 1.001  # 0.1% above entry (covers fees)
            else:
                new_sl = ctx.entry_price * 0.999  # 0.1% below entry
            
            sl_reason = "breakeven"
            
            self.logger.warning(
                f"[EXIT_BRAIN_ADAPTER] 🎯 {ctx.symbol} {ctx.side.upper()}: "
                f"MOVE_SL to BREAKEVEN: {active_sl:.4f} → {new_sl:.4f} "
                f"(PnL={ctx.unrealized_pnl:.2f}%, entry={ctx.entry_price:.4f})"
            )
        
        # Strategy 2: TRAILING (PnL >= 4%)
        elif ctx.unrealized_pnl >= trailing_threshold:
            # Trail SL behind current price to lock in profit
            if ctx.is_long:
                new_sl = ctx.current_price * (1 - trail_distance_pct)
                locked_profit_pct = ((new_sl - ctx.entry_price) / ctx.entry_price) * 100
            else:
                new_sl = ctx.current_price * (1 + trail_distance_pct)
                locked_profit_pct = ((ctx.entry_price - new_sl) / ctx.entry_price) * 100
            
            # Calculate what % of current profit we're locking in
            profit_lock_ratio = (locked_profit_pct / ctx.unrealized_pnl) * 100 if ctx.unrealized_pnl > 0 else 0
            
            sl_reason = "trail_lockin"
            
            self.logger.warning(
                f"[EXIT_BRAIN_ADAPTER] 📈 {ctx.symbol} {ctx.side.upper()}: "
                f"TRAILING SL: {active_sl:.4f} → {new_sl:.4f} "
                f"(PnL={ctx.unrealized_pnl:.2f}%, locks_in={locked_profit_pct:.2f}% profit, "
                f"{profit_lock_ratio:.0f}% of current gain)"
            )
        
        else:
            # Fallback: shouldn't reach here (caught by _should_move_sl)
            new_sl = active_sl
            sl_reason = "no_change"
            locked_profit_pct = None
            
            self.logger.warning(
                f"[EXIT_BRAIN_ADAPTER] ⚠️ {ctx.symbol}: _decide_move_sl called "
                f"but PnL={ctx.unrealized_pnl:.2f}% doesn't meet criteria, "
                f"keeping SL at {active_sl:.4f}"
            )
        
        # Validate: New SL must be tighter (not looser)
        if ctx.is_long and new_sl < active_sl * 0.999:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] ❌ {ctx.symbol}: Rejected MOVE_SL - "
                f"would LOOSEN SL for LONG (new={new_sl:.4f} < old={active_sl:.4f})"
            )
            # Fallback to no change
            new_sl = active_sl
            sl_reason = "rejected_loosening"
        
        if not ctx.is_long and new_sl > active_sl * 1.001:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] ❌ {ctx.symbol}: Rejected MOVE_SL - "
                f"would LOOSEN SL for SHORT (new={new_sl:.4f} > old={active_sl:.4f})"
            )
            # Fallback to no change
            new_sl = active_sl
            sl_reason = "rejected_loosening"
        
        return ExitDecision(
            decision_type=ExitDecisionType.MOVE_SL,
            symbol=ctx.symbol,
            new_sl_price=new_sl,
            sl_reason=sl_reason,
            reason=f"SL {sl_reason}: pnl={ctx.unrealized_pnl:.2f}%, {active_sl:.4f} → {new_sl:.4f}",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            confidence=0.90,  # High confidence for SL moves
            meta={
                "plan_strategy": plan.strategy_id,
                "old_sl": active_sl,
                "new_sl": new_sl,
                "locked_profit_pct": locked_profit_pct if ctx.unrealized_pnl >= trailing_threshold else None,
                "breakeven_threshold": breakeven_threshold,
                "trailing_threshold": trailing_threshold
            }
        )
    
    def _calculate_tighter_sl(self, ctx: PositionContext, plan: ExitPlan) -> float:
        """Calculate tighter SL based on regime/volatility."""
        # Get HARD_SL from plan as baseline
        hard_sl_leg = next((leg for leg in plan.legs if leg.kind == ExitKind.SL), None)
        
        if hard_sl_leg:
            return hard_sl_leg.trigger_price
        
        # Fallback: 2% from entry
        if ctx.is_long:
            return ctx.entry_price * 0.98
        else:
            return ctx.entry_price * 1.02
    
    def _should_place_initial_sl(self, ctx: PositionContext, plan: ExitPlan) -> bool:
        """
        Should we set initial SL level?
        
        Rule: ALWAYS set initial SL if none exists.
        This is critical for risk protection - no position should be unprotected.
        """
        # Check if active SL already set
        active_sl = ctx.meta.get("active_sl")
        
        if active_sl is not None:
            # SL already exists, don't place initial SL again
            self.logger.debug(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: SL already set at {active_sl:.4f}, "
                f"skipping initial SL placement"
            )
            return False
        
        # No SL set yet - CRITICAL: Must set initial SL for risk protection
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: No SL found, will place initial SL"
        )
        return True
    
    def _decide_place_initial_sl(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """
        Create decision to place initial hard SL.
        
        Initial SL Rules:
        1. Use plan's SL leg if available (AI-determined)
        2. Otherwise calculate based on volatility
        3. Default: 1.5-2% SL for risk protection
        4. For LONG: SL below entry, for SHORT: SL above entry
        """
        # Get SL leg from plan (if exists)
        sl_leg = next((leg for leg in plan.legs if leg.kind == ExitKind.SL), None)
        
        if sl_leg:
            # Use SL percentage from plan (AI-determined)
            sl_pct = abs(sl_leg.trigger_pct)
            source = "plan"
            
            self.logger.debug(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Using plan SL: {sl_pct:.2%}"
            )
        else:
            # Fallback: Calculate SL based on volatility
            # Higher volatility = wider SL to avoid noise stops
            volatility = ctx.meta.get("volatility", 0.02)
            
            # Base SL formula: min 1%, max 2.5%, scaled by volatility
            # Low vol (1%) → 1% SL
            # Normal vol (2%) → 1.5% SL
            # High vol (3%+) → 2.5% SL
            sl_pct = max(0.01, min(0.025, volatility * 1.0))
            source = "volatility"
            
            self.logger.info(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: No SL leg in plan, "
                f"using volatility-based SL: {sl_pct:.2%} (volatility={volatility:.2%})"
            )
        
        # Calculate SL price from entry
        # LONG: SL below entry (protect against downside)
        # SHORT: SL above entry (protect against upside)
        if ctx.is_long:
            sl_price = ctx.entry_price * (1 - sl_pct)
        else:
            sl_price = ctx.entry_price * (1 + sl_pct)
        
        # Validate SL price is sensible (must be positive)
        if sl_price <= 0:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] ❌ {ctx.symbol}: Invalid SL price {sl_price:.4f}, "
                f"falling back to 2% default"
            )
            sl_pct = 0.02
            sl_price = ctx.entry_price * (1 - sl_pct if ctx.is_long else 1 + sl_pct)
            source = "fallback"
        
        # Validate SL is on correct side of entry
        if ctx.is_long and sl_price >= ctx.entry_price:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] ❌ {ctx.symbol} LONG: SL {sl_price:.4f} "
                f"not below entry {ctx.entry_price:.4f}, forcing 2% below"
            )
            sl_pct = 0.02
            sl_price = ctx.entry_price * 0.98
            source = "corrected"
        
        if not ctx.is_long and sl_price <= ctx.entry_price:
            self.logger.error(
                f"[EXIT_BRAIN_ADAPTER] ❌ {ctx.symbol} SHORT: SL {sl_price:.4f} "
                f"not above entry {ctx.entry_price:.4f}, forcing 2% above"
            )
            sl_pct = 0.02
            sl_price = ctx.entry_price * 1.02
            source = "corrected"
        
        # Log initial SL placement
        self.logger.warning(
            f"[EXIT_BRAIN_ADAPTER] 🛡️ {ctx.symbol} {ctx.side.upper()}: "
            f"INITIAL SL @ {sl_price:.4f} ({sl_pct:.2%} from entry {ctx.entry_price:.4f}, "
            f"source={source}, current_price={ctx.current_price:.4f})"
        )
        
        return ExitDecision(
            decision_type=ExitDecisionType.MOVE_SL,
            symbol=ctx.symbol,
            new_sl_price=sl_price,
            sl_reason="initial_sl",
            reason=f"Initial SL: {sl_pct:.2%} from entry (source={source})",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            confidence=0.95,  # High confidence for initial SL
            meta={
                "plan_strategy": plan.strategy_id,
                "sl_pct": sl_pct,
                "initial_sl": True,
                "entry_price": ctx.entry_price,
                "sl_source": source
            }
        )
    
    def _should_update_tp_limits(self, ctx: PositionContext, plan: ExitPlan) -> bool:
        """Should we update TP levels?"""
        # Check if active TP levels have been set yet
        active_tp_levels = ctx.meta.get("active_tp_levels", [])
        
        # DEBUG: Log what we see
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: _should_update_tp_limits check: "
            f"active_tp_levels={len(active_tp_levels)} entries, "
            f"type={type(active_tp_levels)}"
        )
        
        # If no TP levels set yet (empty list or None), we need to set them
        # Note: active_tp_levels is list of (price, pct) tuples
        if not active_tp_levels or len(active_tp_levels) == 0:
            # Check if we have TP legs in the plan to set
            tp_legs = [leg for leg in plan.legs if leg.kind == ExitKind.TP]
            if tp_legs:
                self.logger.debug(
                    f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: TP not set yet, will set {len(tp_legs)} levels"
                )
                return True
            else:
                self.logger.debug(f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: No TP legs in plan")
                return False
        
        # TP already set - log this
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: TP already set ({len(active_tp_levels)} levels), skipping UPDATE_TP_LIMITS"
        )
        
        # If regime changed significantly, TP levels should adapt
        # BUT: Only if we have a valid regime (not "unknown")
        current_regime = ctx.regime
        last_regime = ctx.meta.get("last_regime")
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Regime check: current={current_regime}, last={last_regime}"
        )
        
        # Only update TP on regime change if both regimes are known
        if (current_regime and current_regime != "unknown" and 
            last_regime and last_regime != "unknown" and 
            last_regime != current_regime):
            self.logger.debug(f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Regime changed, will update TP")
            return True
        
        # If volatility spiked, widen TP levels
        volatility_spike = ctx.meta.get("volatility_spike")
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Volatility spike check: {volatility_spike}"
        )
        if volatility_spike:
            self.logger.debug(f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Volatility spike, will update TP")
            return True
        
        # Otherwise, TP is already set - don't update
        self.logger.debug(
            f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: No regime change or volatility spike - returning False"
        )
        return False
    
    def _decide_update_tp_limits(self, ctx: PositionContext, plan: ExitPlan) -> ExitDecision:
        """
        Create update TP limits decision based on plan's TP legs.
        
        Uses the plan's ExitLeg objects with trigger_pct to calculate
        actual TP prices from entry price.
        """
        # Extract TP legs from plan
        tp_legs = [leg for leg in plan.legs if leg.kind == ExitKind.TP]
        
        if not tp_legs:
            # No TP legs in plan - shouldn't happen, but handle gracefully
            self.logger.warning(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Plan has no TP legs, cannot update TP limits"
            )
            return ExitDecision(
                decision_type=ExitDecisionType.NO_CHANGE,
                symbol=ctx.symbol,
                reason="No TP legs in plan",
                current_price=ctx.current_price,
                unrealized_pnl=ctx.unrealized_pnl,
                confidence=0.5
            )
        
        # Calculate TP prices from entry price and trigger_pct
        tp_levels = []
        tp_fractions = []
        
        for leg in tp_legs:
            # trigger_pct is POSITIVE for profit (e.g., +0.05 = 5% profit)
            # For LONG: TP = entry * (1 + trigger_pct)
            # For SHORT: TP = entry * (1 - trigger_pct)
            if ctx.is_long:
                tp_price = ctx.entry_price * (1 + leg.trigger_pct)
            else:
                tp_price = ctx.entry_price * (1 - leg.trigger_pct)
            
            # Skip TP levels that are "behind" current price (already achieved)
            # For LONG: TP must be above current price
            # For SHORT: TP must be below current price
            if ctx.is_long and tp_price <= ctx.current_price:
                self.logger.debug(
                    f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Skipping TP @ {tp_price:.4f} "
                    f"(below current {ctx.current_price:.4f}, already achieved)"
                )
                continue
            elif not ctx.is_long and tp_price >= ctx.current_price:
                self.logger.debug(
                    f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: Skipping TP @ {tp_price:.4f} "
                    f"(above current {ctx.current_price:.4f}, already achieved)"
                )
                continue
            
            tp_levels.append(tp_price)
            tp_fractions.append(leg.size_pct)
        
        # If all TP levels already achieved, no TPs to place
        if not tp_levels:
            self.logger.info(
                f"[EXIT_BRAIN_ADAPTER] {ctx.symbol}: All TP levels from plan already achieved "
                f"(current PnL: {ctx.unrealized_pnl:.2f}%), no TPs to place"
            )
            return ExitDecision(
                decision_type=ExitDecisionType.NO_CHANGE,
                symbol=ctx.symbol,
                reason="All TP levels already achieved",
                current_price=ctx.current_price,
                unrealized_pnl=ctx.unrealized_pnl,
                confidence=0.7
            )
        
        return ExitDecision(
            decision_type=ExitDecisionType.UPDATE_TP_LIMITS,
            symbol=ctx.symbol,
            new_tp_levels=tp_levels,
            tp_fractions=tp_fractions,
            reason=f"TP ladder: {len(tp_levels)} levels from plan ({plan.strategy_id})",
            current_price=ctx.current_price,
            unrealized_pnl=ctx.unrealized_pnl,
            confidence=0.85,  # Higher confidence since using plan
            meta={
                "plan_strategy": plan.strategy_id,
                "regime": ctx.regime,
                "tp_count": len(tp_levels),
                "original_tp_count": len(tp_legs)
            }
        )
