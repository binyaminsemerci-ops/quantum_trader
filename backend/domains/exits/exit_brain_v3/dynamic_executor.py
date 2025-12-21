"""
Exit Brain v3 - Dynamic Executor (Clean Implementation)

ARCHITECTURE: AI Planner + Active Monitoring + MARKET Exits + HYBRID STOP-LOSS

Flow:
1. ExitBrainV3 (planner) creates ExitPlan with exit legs (TP, SL, etc.)
2. ExitBrainAdapter translates plan to ExitDecision (MOVE_SL, UPDATE_TP_LIMITS, etc.)
3. Executor stores levels INTERNALLY in PositionExitState (NO exchange orders)
4. Monitoring loop checks price vs levels every N seconds
5. When level hit → execute MARKET reduce-only order immediately

HYBRID STOP-LOSS MODEL:
- Internal SL (active_sl): AI-driven, dynamic, optimizes exits (no exchange order)
- Hard SL (hard_sl_price): Binance STOP_MARKET order, static max-loss floor, survives crashes
- Hard SL placed when position state created, cancelled when position closes
- Hard SL ideally never triggers (internal SL exits earlier), acts as last-resort safety net

Key Design Principles:
- NO LIMIT/STOP/TAKE_PROFIT orders on exchange (except hard SL safety net)
- ALL exits are MARKET + reduceOnly for instant execution
- AI has full control - can adjust internal levels anytime without exchange interaction
- True dynamic SL/TP management without order replacement complexity

State Management:
- PositionExitState per position (key: "{symbol}:{side}")
- Tracks active_sl, tp_levels, triggered_legs internally
- Tracks hard_sl_price, hard_sl_order_id for Binance safety net
- State persists across cycles, cleared when position closes

Exit Execution:
- LONG: SL triggers when price <= active_sl, TP when price >= tp_price
- SHORT: SL triggers when price >= active_sl, TP when price <= tp_price
- SL closes full remaining position with MARKET order
- TP closes partial position (size_pct of remaining) with MARKET order
- All orders sent via exit_order_gateway with proper positionSide (hedge mode)
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from pathlib import Path

from .types import PositionContext, ExitDecision, ExitDecisionType, PositionExitState
from .adapter import ExitBrainAdapter
from .router import ExitRouter
from .precision import quantize_to_tick, quantize_to_step, get_binance_precision

logger = logging.getLogger(__name__)

# Maximum loss percentage for hard SL safety net
# This is a last-resort floor - internal AI SL should exit earlier
MAX_LOSS_PCT_HARD_SL = 0.02  # 2% max loss from entry

# ============================================================================
# RISK MANAGEMENT CONSTANTS
# ============================================================================
# Maximum margin loss per trade as a percentage (e.g., 0.10 = 10% of margin)
# This is the WORST CASE loss we will tolerate if SL is hit
MAX_MARGIN_LOSS_PER_TRADE_PCT = 0.10  # 10% max margin loss per trade

# Minimum practical stop distance as percentage of entry price
# If leverage is so high that allowed_move_pct < this, the trade is over-leveraged
MIN_PRICE_STOP_DISTANCE_PCT = 0.002  # 0.2% minimum stop distance

# ============================================================================
# DYNAMIC PARTIAL TP & LOSS GUARD CONSTANTS
# ============================================================================
# Maximum unrealized loss percentage per position before emergency exit
# This is independent of SL price - triggers on PnL percentage
# Example: 12.5 means -12.5% unrealized PnL triggers full close
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 12.5  # -12.5% max unrealized loss

# Default TP size distribution if Exit Plan doesn't specify
# Fractions apply to initial position size
DYNAMIC_TP_PROFILE_DEFAULT = [0.25, 0.25, 0.50]  # 25%, 25%, 50%

# Enable SL ratcheting after TP hits
# When True: SL tightens automatically after TPs trigger
RATCHET_SL_ENABLED = True

# Maximum number of TP levels to track (sanity cap)
MAX_TP_LEVELS = 4


class ExitBrainDynamicExecutor:
    """
    Dynamic AI-driven exit executor with active monitoring.
    
    Responsibilities:
    - Monitor open positions continuously (async loop)
    - Build PositionContext for each position
    - Get AI decisions from adapter
    - Store exit levels internally in PositionExitState
    - Check price vs levels every cycle
    - Execute MARKET reduce-only orders when levels hit
    
    Does NOT:
    - Place LIMIT/STOP/TAKE_PROFIT orders on exchange
    - Modify orders directly (uses exit_order_gateway)
    - Make exit decisions (delegates to adapter/planner)
    """
    
    def __init__(
        self,
        adapter: ExitBrainAdapter,
        exit_order_gateway,
        position_source,
        loop_interval_sec: float = 10.0,
        shadow_mode: bool = False
    ):
        """
        Initialize dynamic executor.
        
        Args:
            adapter: ExitBrainAdapter for AI decisions
            exit_order_gateway: Gateway for submitting orders
            position_source: Source for position data (BinanceClient)
            loop_interval_sec: Monitoring cycle interval
            shadow_mode: If True, only log decisions (no orders)
        """
        from backend.config.exit_mode import (
            get_exit_mode,
            get_exit_executor_mode,
            is_exit_brain_live_fully_enabled,
            is_challenge_100_profile
        )
        import os
        
        self.adapter = adapter
        self.router = ExitRouter()  # Singleton for plan cache invalidation
        self.exit_order_gateway = exit_order_gateway
        self.position_source = position_source
        self.loop_interval_sec = loop_interval_sec
        
        # Initialize logger FIRST (needed for all logging)
        self.logger = logging.getLogger(__name__ + ".executor")
        
        # Determine operating mode from config
        if is_exit_brain_live_fully_enabled():
            self.effective_mode = "LIVE"
            self.shadow_mode = False
        else:
            self.effective_mode = "SHADOW"
            self.shadow_mode = True
        
        # CHALLENGE_100 profile config
        # Note: Challenge is selected via EXIT_BRAIN_PROFILE, not EXIT_MODE
        self.challenge_mode = is_challenge_100_profile()
        if self.challenge_mode:
            self.challenge_risk_pct = float(os.getenv("CHALLENGE_RISK_PCT_PER_TRADE", "0.015"))
            self.challenge_max_risk_r = float(os.getenv("CHALLENGE_MAX_RISK_R", "1.5"))
            self.challenge_tp1_r = float(os.getenv("CHALLENGE_TP1_R", "1.0"))
            self.challenge_tp1_qty_pct = float(os.getenv("CHALLENGE_TP1_QTY_PCT", "0.30"))
            self.challenge_trail_atr_mult = float(os.getenv("CHALLENGE_TRAIL_ATR_MULT", "2.0"))
            self.challenge_time_stop_sec = float(os.getenv("CHALLENGE_TIME_STOP_SEC", "7200"))
            self.challenge_liq_buffer_pct = float(os.getenv("CHALLENGE_LIQ_BUFFER_PCT", "0.01"))
            self.challenge_hard_sl_enabled = os.getenv("CHALLENGE_HARD_SL_ENABLED", "true").lower() == "true"
            self.logger.warning(
                f"[CHALLENGE_100] Mode active - 1R={self.challenge_risk_pct:.2%}, "
                f"TP1={self.challenge_tp1_qty_pct:.0%} @ +{self.challenge_tp1_r}R, "
                f"time_stop={self.challenge_time_stop_sec}s, hard_sl={self.challenge_hard_sl_enabled}"
            )
        
        # CORE STATE: Internal exit state per position
        # Key format: "{symbol}:{side}" (e.g., "ETHUSDT:LONG")
        self._state: Dict[str, PositionExitState] = {}
        
        # Cache for regime/volatility (reduces API calls)
        # Format: {symbol: {"regime": str, "volatility": float, "timestamp": float}}
        self._market_data_cache: Dict[str, Dict] = {}
        self._cache_ttl_sec: float = 60.0  # Cache for 60 seconds
        
        # Binance hedge mode detection (dualSidePosition)
        self._hedge_mode: Optional[bool] = None  # Will detect on first cycle
        
        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Log final configuration
        exit_mode = get_exit_mode()
        executor_mode = get_exit_executor_mode()
        
        self.logger.info(
            f"[EXIT_BRAIN_EXECUTOR] Initialized in {self.effective_mode} MODE - "
            f"Config: EXIT_MODE={exit_mode}, EXIT_EXECUTOR_MODE={executor_mode}"
        )
        
        if self.effective_mode == "SHADOW":
            self.logger.info(
                "[EXIT_BRAIN_EXECUTOR] SHADOW mode: Will log AI decisions without placing orders"
            )
        else:
            self.logger.warning(
                "[EXIT_BRAIN_EXECUTOR] 🔴 LIVE MODE ACTIVE 🔴 - "
                "AI will place real MARKET orders via exit_order_gateway"
            )
    
    async def start(self):
        """Start monitoring loop."""
        if self._running:
            self.logger.warning("[EXIT_BRAIN_EXECUTOR] Already running")
            return
        
        self._running = True
        self.logger.info(
            f"[EXIT_BRAIN_EXECUTOR] Starting monitoring loop "
            f"(interval={self.loop_interval_sec}s, mode={self.effective_mode})"
        )
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def _detect_hedge_mode(self):
        """Detect if Binance account is in hedge mode (dualSidePosition)."""
        if self._hedge_mode is not None:
            return self._hedge_mode
        
        try:
            result = await asyncio.to_thread(
                self.position_source.futures_get_position_mode
            )
            self._hedge_mode = result.get('dualSidePosition', False)
            mode_str = "HEDGE (dual-side)" if self._hedge_mode else "ONE-WAY (single-side)"
            self.logger.warning(f"[EXIT_BRAIN_EXECUTOR] Binance position mode: {mode_str}")
            return self._hedge_mode
        except Exception as e:
            self.logger.error(f"[EXIT_BRAIN_EXECUTOR] Failed to detect hedge mode: {e}")
            # Default to False (one-way mode) to avoid order errors
            self._hedge_mode = False
            return False
    
    async def stop(self):
        """Stop monitoring loop."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("[EXIT_BRAIN_EXECUTOR] Stopping monitoring loop")
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring loop - runs continuously."""
        cycle = 0
        
        self.logger.warning(f"[EXIT_BRAIN_LOOP] ▶️  Monitoring loop STARTED (interval={self.loop_interval_sec}s)")
        
        while self._running:
            try:
                cycle += 1
                self.logger.info(f"[EXIT_BRAIN_LOOP] 🔄 Starting cycle {cycle}...")
                await self._monitoring_cycle(cycle)
                self.logger.info(f"[EXIT_BRAIN_LOOP] ✅ Cycle {cycle} complete")
                
            except asyncio.CancelledError:
                self.logger.warning(f"[EXIT_BRAIN_LOOP] ⏹️  Loop cancelled at cycle {cycle}")
                break
            except Exception as e:
                self.logger.error(
                    f"[EXIT_BRAIN_EXECUTOR] ❌ Error in cycle {cycle}: {e}",
                    exc_info=True
                )
            
            # Wait for next cycle
            self.logger.debug(f"[EXIT_BRAIN_LOOP] ⏳ Sleeping {self.loop_interval_sec}s before cycle {cycle+1}")
            await asyncio.sleep(self.loop_interval_sec)
    
    async def _monitoring_cycle(self, cycle: int):
        """
        Single monitoring cycle.
        
        Flow:
        1. Get open positions from Binance
        2. For each position:
           - Create/update PositionExitState
           - Build PositionContext
           - Get AI decision from adapter
           - Update state based on decision
        3. Check all states for triggered levels
        4. Execute MARKET orders where needed
        5. Clean up closed positions
        """
        # Detect hedge mode on first cycle
        if cycle == 1:
            await self._detect_hedge_mode()
        
        # Get open positions
        self.logger.info(f"[EXIT_BRAIN_CYCLE] 📡 Fetching positions...")
        positions = await self._get_open_positions()
        
        if not positions:
            self.logger.info(f"[EXIT_BRAIN_EXECUTOR] Cycle {cycle}: No open positions")
            return
        
        self.logger.warning(
            f"[EXIT_BRAIN_EXECUTOR] Cycle {cycle}: Processing {len(positions)} positions"
        )
        
        # Track active position keys for cleanup
        active_keys: Set[str] = set()
        
        # Process each position
        for pos_data in positions:
            try:
                # Extract position details
                symbol = pos_data.get('symbol')
                position_amt = float(pos_data.get('positionAmt', 0))
                
                # Skip zero positions
                if position_amt == 0:
                    continue
                
                # Determine side and size
                if position_amt > 0:
                    side = "LONG"
                    size = position_amt
                else:
                    side = "SHORT"
                    size = abs(position_amt)
                
                # State key
                state_key = f"{symbol}:{side}"
                active_keys.add(state_key)
                
                # Get or create state
                if state_key not in self._state:
                    entry_price = float(pos_data.get('entryPrice', 0))
                    import time
                    self._state[state_key] = PositionExitState(
                        symbol=symbol,
                        side=side,
                        position_size=size,
                        entry_price=entry_price,
                        initial_size=size,
                        remaining_size=size,
                        opened_at_ts=time.time(),  # For time stop
                        challenge_mode_active=self.challenge_mode
                    )
                    self.logger.info(
                        f"[EXIT_BRAIN_EXECUTOR] Created new state for {state_key} - "
                        f"entry=${entry_price:.4f}, initial_size={size}, "
                        f"challenge_mode={self.challenge_mode}"
                    )
                
                state = self._state[state_key]
                
                # Update state with current data
                state.position_size = size
                mark_price = float(pos_data.get('markPrice', 0))
                state.last_price = mark_price
                state.last_updated = datetime.now(timezone.utc).isoformat()
                
                # Update entry price if not set
                if state.entry_price is None:
                    state.entry_price = float(pos_data.get('entryPrice', 0))
                
                # Build context for AI (include current state for SL/TP tracking)
                ctx = self._build_position_context(pos_data, state)
                
                # CHECK LOSS GUARD FIRST - highest priority safety check
                # This runs before AI decisions and SL/TP checks
                loss_guard_triggered = await self._check_loss_guard(state, ctx)
                if loss_guard_triggered:
                    # Position closed by loss guard, skip further processing
                    continue
                
                # Get AI decision
                decision = await self.adapter.decide(ctx)
                
                # Update state based on decision
                await self._update_state_from_decision(state, ctx, decision)
                
                # CHALLENGE_100: Override AI logic with $100 Challenge rules
                if state.challenge_mode_active:
                    await self._apply_challenge_100_logic(state, pos_data, mark_price)
                
                # RISK ENFORCEMENT: Ensure position has SL that respects risk budget
                # If adapter/planner didn't set an SL, apply risk floor as initial SL
                if state.active_sl is None and not state.challenge_mode_active:
                    entry_price = float(pos_data.get('entryPrice', 0))
                    if entry_price > 0:
                        effective_leverage = self._compute_effective_leverage(pos_data)
                        risk_sl_price, allowed_move_pct = self._compute_risk_floor_sl(
                            entry_price,
                            state.side,
                            effective_leverage
                        )
                        
                        self.logger.info(
                            f"[EXIT_BRAIN_RISK] {state_key}: "
                            f"entry=${entry_price:.4f}, leverage={effective_leverage:.1f}x, "
                            f"max_margin_loss={MAX_MARGIN_LOSS_PER_TRADE_PCT:.1%}, "
                            f"allowed_move_pct={allowed_move_pct:.3%}, "
                            f"risk_sl_price=${risk_sl_price:.4f}"
                        )
                        
                        # Check for over-leverage
                        if allowed_move_pct < MIN_PRICE_STOP_DISTANCE_PCT:
                            self.logger.error(
                                f"[EXIT_BRAIN_RISK] {state_key}: "
                                f"allowed_move_pct={allowed_move_pct:.3%} < "
                                f"MIN_PRICE_STOP_DISTANCE_PCT={MIN_PRICE_STOP_DISTANCE_PCT:.3%} "
                                f"→ FULL_EXIT_NOW triggered due to over-leverage vs risk budget"
                            )
                            # Force immediate exit
                            dummy_decision = ExitDecision(
                                decision_type=ExitDecisionType.FULL_EXIT_NOW,
                                symbol=state.symbol,
                                reason="Over-leveraged vs risk budget"
                            )
                            await self._execute_emergency_exit(state, ctx, dummy_decision)
                        else:
                            # Set risk floor as initial SL
                            state.active_sl = risk_sl_price
                            self.logger.warning(
                                f"[EXIT_BRAIN_RISK] {state_key}: "
                                f"No strategic SL set by adapter, applying risk_floor=${risk_sl_price:.4f} "
                                f"as initial SL (allowed_move={allowed_move_pct:.3%})"
                            )
                
                # Hard SL placement: ENABLED in CHALLENGE_100 LIVE mode ONLY
                # Requires: EXIT_MODE=EXIT_BRAIN_V3 + EXIT_EXECUTOR_MODE=LIVE + EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
                if state.challenge_mode_active and self.challenge_hard_sl_enabled:
                    from backend.config.exit_mode import is_exit_brain_live_fully_enabled
                    
                    # Hard SL only in LIVE mode (same gate as executor LIVE behavior)
                    if (state.hard_sl_order_id is None and 
                        state.hard_sl_price is None and 
                        not hasattr(state, '_hard_sl_attempted') and
                        is_exit_brain_live_fully_enabled()):
                        
                        entry_price = float(pos_data.get('entryPrice', 0))
                        if entry_price > 0 and state.active_sl is not None:
                            self.logger.warning(
                                f"[CHALLENGE_100] {state_key}: Placing hard SL safety net "
                                f"(LIVE mode active, entry_price={entry_price}, soft_sl={state.active_sl:.4f})"
                            )
                            try:
                                await self._place_hard_sl_challenge(state, entry_price)
                                state._hard_sl_attempted = True
                            except Exception as hs_err:
                                self.logger.error(
                                    f"[CHALLENGE_100] {state_key}: Hard SL placement failed: {hs_err}",
                                    exc_info=True
                                )
                                state._hard_sl_attempted = True
                        else:
                            self.logger.warning(
                                f"[CHALLENGE_100] {state_key}: Skipping hard SL - "
                                f"invalid entry_price={entry_price} or active_sl={state.active_sl}"
                            )
                    elif not is_exit_brain_live_fully_enabled():
                        self.logger.debug(
                            f"[CHALLENGE_100] {state_key}: Hard SL disabled (SHADOW mode)"
                        )
                else:
                    # Standard mode: Hard SL disabled (conflicts with legacy modules)
                    self.logger.debug(
                        f"[EXIT_BRAIN_EXECUTOR] {state_key}: Hard SL placement disabled - "
                        f"using soft SL monitoring @ ${state.active_sl:.4f if state.active_sl else 'None'} + loss guard"
                    )
                
            except Exception as e:
                symbol = pos_data.get('symbol', 'UNKNOWN')
                self.logger.error(
                    f"[EXIT_BRAIN_EXECUTOR] Error processing {symbol}: {e}",
                    exc_info=True
                )
        
        # Clean up closed positions
        await self._cleanup_closed_positions(active_keys)
        
        # Check and execute triggered levels
        await self._check_and_execute_levels()
    
    async def _get_open_positions(self) -> List[Dict]:
        """Get open positions from Binance."""
        try:
            # Call Binance API directly via client
            # futures_position_information() returns all positions (including 0-size)
            from backend.integrations.binance.client_wrapper import BinanceClientWrapper
            
            wrapper = BinanceClientWrapper()
            positions = await wrapper.call_async(
                self.position_source.futures_position_information
            )
            
            # Filter to only open positions
            open_positions = [
                pos for pos in positions
                if float(pos.get('positionAmt', 0)) != 0
            ]
            
            return open_positions
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error fetching positions: {e}",
                exc_info=True
            )
            return []
    
    def _build_position_context(self, pos_data: Dict, state: PositionExitState) -> PositionContext:
        """
        Build PositionContext from Binance position data and current state.
        
        Args:
            pos_data: Raw position data from Binance
            state: Current PositionExitState for this position
            
        Returns:
            PositionContext for adapter
        """
        symbol = pos_data.get('symbol')
        position_amt = float(pos_data.get('positionAmt', 0))
        entry_price = float(pos_data.get('entryPrice', 0))
        mark_price = float(pos_data.get('markPrice', 0))
        unrealized_pnl = float(pos_data.get('unRealizedProfit', 0))
        leverage = int(pos_data.get('leverage', 1))
        
        # Determine side and size
        if position_amt > 0:
            side = "long"  # lowercase for PositionContext
            size = position_amt
        else:
            side = "short"
            size = abs(position_amt)
        
        # Calculate PnL %
        if entry_price > 0 and size > 0:
            notional = entry_price * size
            pnl_pct = (unrealized_pnl / notional) * 100 if notional > 0 else 0
        else:
            pnl_pct = 0
        
        # Detect regime
        regime = self._detect_regime(symbol, mark_price)
        
        # Assess risk state based on PnL
        risk_state = self._assess_risk_state(symbol, pnl_pct)
        
        # Build meta with current state info for adapter
        meta = {
            "position_data": pos_data,
            "unrealized_pnl_abs": unrealized_pnl,
            # Include state for adapter SL/TP logic
            "active_sl": state.active_sl,
            "active_tp_levels": [(price, pct) for price, pct in state.tp_levels],
            "triggered_legs": list(state.triggered_legs)
        }
        
        ctx = PositionContext(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=mark_price,
            size=size,
            unrealized_pnl=pnl_pct,
            leverage=leverage,
            exchange="binance",
            regime=regime,
            risk_state=risk_state,
            meta=meta
        )
        
        # Add volatility to meta (calculate from recent price action)
        volatility = self._calculate_volatility(symbol, mark_price)
        meta["volatility"] = volatility
        
        return ctx
    
    def _detect_regime(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Detect market regime using recent price action.
        Cached for 60s to reduce API calls (regime changes slowly).
        
        Regimes:
        - TRENDING: Strong directional move
        - RANGE: Sideways consolidation
        - VOLATILE: High volatility, choppy
        - unknown: Insufficient data
        """
        import time
        
        # Check cache first
        now = time.time()
        cached = self._market_data_cache.get(symbol)
        if cached and (now - cached.get("timestamp", 0)) < self._cache_ttl_sec:
            regime = cached.get("regime", "unknown")
            self.logger.debug(f"[EXIT_REGIME] {symbol}: Using cached regime={regime}")
            return regime
        
        try:
            # Fetch fresh data using position_source (python-binance Client directly)
            if not hasattr(self, 'position_source') or self.position_source is None:
                self.logger.warning(f"[EXIT_REGIME] {symbol}: No position_source available")
                return "unknown"
            
            klines = self.position_source.futures_klines(
                symbol=symbol,
                interval="15m",  # 15-min candles
                limit=20
            )
            
            if not klines or len(klines) < 15:
                return "unknown"
            
            # Extract closes
            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            
            # Calculate returns and volatility
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            
            # Calculate trend strength (price vs MA)
            ma = sum(closes) / len(closes)
            trend_strength = abs(current_price - ma) / ma
            
            # Calculate range (ATR-like)
            ranges = [(highs[i] - lows[i]) / closes[i] for i in range(len(closes))]
            avg_range = sum(ranges) / len(ranges)
            
            # Classify regime
            if trend_strength > 0.02 and volatility < 0.03:
                # Strong trend, low volatility → TRENDING
                regime = "TRENDING"
            elif avg_range < 0.015 and trend_strength < 0.01:
                # Tight range, no trend → RANGE
                regime = "RANGE"
            elif volatility > 0.04:
                # High volatility → VOLATILE
                regime = "VOLATILE"
            else:
                # Moderate conditions → NORMAL
                regime = "NORMAL"
            
            # Cache result
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {}
            self._market_data_cache[symbol]["regime"] = regime
            self._market_data_cache[symbol]["volatility"] = volatility
            self._market_data_cache[symbol]["timestamp"] = now
            
            self.logger.debug(f"[EXIT_REGIME] {symbol}: Detected regime={regime}, volatility={volatility:.4f} (cached for {self._cache_ttl_sec}s)")
            return regime
            
        except Exception as e:
            self.logger.warning(f"[EXIT_REGIME] {symbol}: Failed to detect regime: {e}")
            return "unknown"
    
    def _calculate_volatility(self, symbol: str, current_price: float) -> float:
        """
        Calculate recent volatility (last 20x 15min periods).
        Cached for 60s to reduce API calls (shares cache with regime detection).
        
        Returns:
            Volatility as fraction (e.g., 0.02 = 2%)
        """
        import time
        
        # Check cache first (volatility calculated together with regime)
        now = time.time()
        cached = self._market_data_cache.get(symbol)
        if cached and (now - cached.get("timestamp", 0)) < self._cache_ttl_sec:
            volatility = cached.get("volatility", 0.02)
            self.logger.debug(f"[EXIT_VOL] {symbol}: Using cached volatility={volatility:.4f}")
            return volatility
        
        # If regime detection hasn't run yet, fetch volatility separately
        # (normally regime detection will populate cache with both regime + volatility)
        try:
            # Use position_source (python-binance Client directly)
            if not hasattr(self, 'position_source') or self.position_source is None:
                return 0.02  # Default 2%
            
            klines = self.position_source.futures_klines(
                symbol=symbol,
                interval="15m",
                limit=20
            )
            
            if not klines or len(klines) < 10:
                return 0.02  # Default 2%
            
            # Extract closes and calculate returns
            closes = [float(k[4]) for k in klines]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            
            # Calculate volatility (standard deviation of returns)
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return)**2 for r in returns) / len(returns)
            volatility = variance ** 0.5
            
            volatility = min(0.10, max(0.005, volatility))  # Clamp 0.5% - 10%
            
            # Cache result
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {}
            self._market_data_cache[symbol]["volatility"] = volatility
            self._market_data_cache[symbol]["timestamp"] = now
            
            self.logger.debug(f"[EXIT_VOL] {symbol}: Calculated volatility={volatility:.4f} (cached for {self._cache_ttl_sec}s)")
            return volatility
            
        except Exception as e:
            self.logger.warning(f"[EXIT_VOL] {symbol}: Failed to calculate volatility: {e}")
            return 0.02  # Default 2%
    
    def _assess_risk_state(self, symbol: str, pnl_pct: float) -> Optional[str]:
        """Assess risk state based on PnL."""
        if pnl_pct < -3.0:
            return "high_risk"
        elif pnl_pct < -1.0:
            return "drawdown"
        else:
            return "normal"
    
    def _compute_effective_leverage(self, pos_data: Dict[str, Any]) -> float:
        """
        Compute effective leverage for a position.
        
        Priority order:
        1. Use 'actual_leverage' if provided (from AI system at order placement)
        2. Calculate from notional/margin if available (real leverage)
        3. Use 'leverage' field from Binance (often returns 1 for cross margin)
        
        Args:
            pos_data: Position data from Binance futures_position_information()
            
        Returns:
            Effective leverage (minimum 1.0)
        """
        symbol = pos_data.get('symbol', 'UNKNOWN')
        
        # PRIORITY 1: Use AI-determined leverage if provided
        if 'actual_leverage' in pos_data and pos_data['actual_leverage']:
            try:
                leverage = float(pos_data['actual_leverage'])
                if leverage >= 1.0:
                    self.logger.info(
                        f"[EXIT_BRAIN_LEVERAGE] {symbol}: Using AI-determined leverage: {leverage:.1f}x"
                    )
                    return leverage
            except (ValueError, TypeError):
                pass
        
        # PRIORITY 2: Calculate real leverage from position data
        try:
            notional = abs(float(pos_data.get('notional', 0)))
            margin = float(pos_data.get('initialMargin', 0)) or float(pos_data.get('positionInitialMargin', 0))
            
            if notional > 0 and margin > 0:
                calculated_leverage = notional / margin
                if calculated_leverage >= 1.0:
                    self.logger.info(
                        f"[EXIT_BRAIN_LEVERAGE] {symbol}: Calculated from notional/margin: {calculated_leverage:.1f}x"
                    )
                    return calculated_leverage
        except (ValueError, TypeError, KeyError) as e:
            self.logger.debug(f"[EXIT_BRAIN_LEVERAGE] {symbol}: Could not calculate leverage from notional/margin: {e}")
        
        # PRIORITY 3: Fallback to Binance 'leverage' field (often wrong for cross margin)
        leverage_str = pos_data.get('leverage', '1')
        try:
            leverage = float(leverage_str)
            self.logger.warning(
                f"[EXIT_BRAIN_LEVERAGE] {symbol}: Using Binance leverage field (may be inaccurate for cross margin): {leverage:.1f}x"
            )
        except (ValueError, TypeError):
            self.logger.warning(
                f"[EXIT_BRAIN_RISK] {symbol}: "
                f"Failed to parse leverage='{leverage_str}', defaulting to 1.0"
            )
            leverage = 1.0
        
        # Ensure minimum leverage of 1.0
        return max(1.0, leverage)
    
    def _compute_risk_floor_sl(
        self,
        entry_price: float,
        side: str,
        leverage: float
    ) -> tuple[float, float]:
        """
        Compute risk-floor SL price based on max margin loss per trade.
        
        Formula:
        - margin_loss_pct ≈ (price_move / entry_price) * leverage
        - To limit margin_loss_pct <= MAX_MARGIN_LOSS_PER_TRADE_PCT:
          allowed_move_pct = MAX_MARGIN_LOSS_PER_TRADE_PCT / max(leverage, 1)
        
        Args:
            entry_price: Position entry price
            side: "LONG" or "SHORT"
            leverage: Effective leverage
            
        Returns:
            Tuple of (risk_sl_price, allowed_move_pct)
        """
        # Calculate allowed price move percentage
        allowed_move_pct = MAX_MARGIN_LOSS_PER_TRADE_PCT / max(leverage, 1.0)
        
        # Calculate risk floor SL price
        if side == "LONG":
            # For LONG: SL must be at or above this price
            risk_sl_price = entry_price * (1.0 - allowed_move_pct)
        else:  # SHORT
            # For SHORT: SL must be at or below this price
            risk_sl_price = entry_price * (1.0 + allowed_move_pct)
        
        return risk_sl_price, allowed_move_pct
    
    def _apply_risk_floor_to_sl(
        self,
        strategic_sl: Optional[float],
        risk_sl_price: float,
        side: str,
        symbol: str
    ) -> Optional[float]:
        """
        Apply risk floor constraint to strategic SL.
        
        Logic:
        - LONG: final_sl = max(strategic_sl, risk_sl_price)
          (tighten SL if strategic is below risk floor)
        - SHORT: final_sl = min(strategic_sl, risk_sl_price)
          (tighten SL if strategic is above risk ceiling)
        
        Args:
            strategic_sl: SL price from AI/planner (can be None)
            risk_sl_price: Risk floor/ceiling SL price
            side: "LONG" or "SHORT"
            symbol: Symbol for logging
            
        Returns:
            Final SL price after risk constraints applied
        """
        # If no strategic SL, use risk floor as initial SL
        if strategic_sl is None:
            self.logger.info(
                f"[EXIT_BRAIN_RISK] {symbol} {side}: "
                f"No strategic SL, using risk_floor=${risk_sl_price:.4f}"
            )
            return risk_sl_price
        
        # Apply risk floor constraints
        if side == "LONG":
            # LONG: SL cannot be below risk floor (looser than max risk)
            final_sl = max(strategic_sl, risk_sl_price)
            
            if final_sl > strategic_sl:
                self.logger.warning(
                    f"[EXIT_BRAIN_RISK] {symbol} LONG: "
                    f"strategic_sl=${strategic_sl:.4f} below risk_floor=${risk_sl_price:.4f} "
                    f"→ tightening to final_sl=${final_sl:.4f}"
                )
            else:
                self.logger.debug(
                    f"[EXIT_BRAIN_RISK] {symbol} LONG: "
                    f"strategic_sl=${strategic_sl:.4f} >= risk_floor=${risk_sl_price:.4f} "
                    f"→ final_sl=${final_sl:.4f}"
                )
        else:  # SHORT
            # SHORT: SL cannot be above risk ceiling (looser than max risk)
            final_sl = min(strategic_sl, risk_sl_price)
            
            if final_sl < strategic_sl:
                self.logger.warning(
                    f"[EXIT_BRAIN_RISK] {symbol} SHORT: "
                    f"strategic_sl=${strategic_sl:.4f} above risk_ceiling=${risk_sl_price:.4f} "
                    f"→ tightening to final_sl=${final_sl:.4f}"
                )
            else:
                self.logger.debug(
                    f"[EXIT_BRAIN_RISK] {symbol} SHORT: "
                    f"strategic_sl=${strategic_sl:.4f} <= risk_ceiling=${risk_sl_price:.4f} "
                    f"→ final_sl=${final_sl:.4f}"
                )
        
        return final_sl
    
    async def _place_hard_sl(self, state: PositionExitState, entry_price: float):
        """
        Place hard STOP_MARKET order on Binance as safety net.
        
        This is a LAST-RESORT protection that survives backend crashes.
        The internal AI SL (active_sl) should exit earlier for better prices.
        
        Args:
            state: Position exit state to update
            entry_price: Position entry price for calculating max loss
        """
        # Calculate hard SL price (max loss floor)
        max_loss_pct = MAX_LOSS_PCT_HARD_SL
        
        if state.side == "LONG":
            hard_sl_price = entry_price * (1 - max_loss_pct)
            order_side = "SELL"
        else:  # SHORT
            hard_sl_price = entry_price * (1 + max_loss_pct)
            order_side = "BUY"
        
        # Get REAL precision from Binance API (not guessed)
        from .binance_precision_cache import get_binance_precision_from_api
        try:
            tick_size, step_size = get_binance_precision_from_api(state.symbol, self.position_source)
        except Exception as e:
            # Fallback to hardcoded if API fails
            self.logger.warning(f"[EXIT_BRAIN] Failed to get API precision for {state.symbol}: {e}")
            tick_size, step_size = get_binance_precision(state.symbol)
        
        # Round to tick size
        hard_sl_price = quantize_to_tick(hard_sl_price, tick_size)
        
        # Format stopPrice with correct decimal places for Binance
        from .precision import format_price_for_binance
        stop_price_str = format_price_for_binance(hard_sl_price, tick_size)
        
        # Build STOP_MARKET order
        order_params = {
            "symbol": state.symbol,
            "side": order_side,
            "type": "STOP_MARKET",
            "stopPrice": stop_price_str,  # Use formatted string
            "closePosition": True,
            "workingType": "MARK_PRICE",
        }
        
        # Only include positionSide if hedge mode is enabled
        if self._hedge_mode:
            order_params["positionSide"] = state.side
        
        try:
            # CRITICAL FIX: Cancel existing SL orders first to avoid -4130 error
            try:
                # Get all open orders for this symbol
                from backend.integrations.binance.client_wrapper import BinanceClientWrapper
                wrapper = BinanceClientWrapper()
                
                open_orders = await wrapper.call_async(
                    self.position_source.futures_get_open_orders,
                    symbol=state.symbol
                )
                
                # Find and cancel existing STOP_MARKET orders for this position side
                for order in open_orders:
                    if (order.get('type') == 'STOP_MARKET' and 
                        order.get('positionSide') == state.side):
                        
                        await wrapper.call_async(
                            self.position_source.futures_cancel_order,
                            symbol=state.symbol,
                            orderId=order['orderId']
                        )
                        
                        self.logger.info(
                            f"[EXIT_BRAIN_STATE] 🗑️ {state.symbol} {state.side}: "
                            f"Cancelled existing SL orderId={order['orderId']} @ ${order.get('stopPrice')} "
                            f"before placing new HARD SL"
                        )
            except Exception as cancel_err:
                # Log but continue - we still want to try placing new SL
                self.logger.warning(
                    f"[EXIT_BRAIN_STATE] ⚠️ {state.symbol} {state.side}: "
                    f"Could not cancel existing SL orders: {cancel_err}"
                )
            
            self.logger.warning(
                f"[EXIT_BRAIN_STATE] 🛡️ {state.symbol} {state.side}: "
                f"Placing HARD SL on Binance at ${hard_sl_price:.4f} "
                f"(max_loss={max_loss_pct:.2%} from entry ${entry_price:.4f})"
            )
            
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_brain_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="hard_sl",
                explanation=f"Hard SL safety net at {hard_sl_price}"
            )
            
            if resp and resp.get('orderId'):
                state.hard_sl_price = hard_sl_price
                state.hard_sl_order_id = str(resp['orderId'])
                self.logger.warning(
                    f"[EXIT_BRAIN_STATE] ✅ {state.symbol} {state.side}: "
                    f"HARD SL placed successfully - orderId={state.hard_sl_order_id}"
                )
            else:
                self.logger.error(
                    f"[EXIT_BRAIN_STATE] ❌ {state.symbol} {state.side}: "
                    f"HARD SL placement failed - no orderId in response"
                )
        
        except Exception as e:
            # Binance Testnet precision issues: Some symbols (XRPUSDT) fail algo orders
            # This is acceptable - soft SL monitoring + loss guard provide protection
            if "Precision is over the maximum" in str(e):
                self.logger.warning(
                    f"[EXIT_BRAIN_STATE] ⚠️ {state.symbol} {state.side}: "
                    f"HARD SL placement skipped (testnet precision issue) - "
                    f"soft SL @ ${state.active_sl:.5f} + loss guard (-12.5%) active"
                )
            else:
                self.logger.error(
                    f"[EXIT_BRAIN_STATE] ❌ {state.symbol} {state.side}: "
                    f"HARD SL placement failed: {e}",
                    exc_info=True
                )
    
    async def _check_loss_guard(
        self,
        state: PositionExitState,
        ctx: PositionContext
    ) -> bool:
        """
        Check and execute max unrealized loss guard.
        
        This is an independent safety mechanism that triggers emergency exit
        if unrealized PnL exceeds maximum acceptable loss percentage.
        
        Operates independently of SL price - uses PnL percentage directly.
        Prevents seeing -30%/-40% unrealized losses on positions.
        
        Args:
            state: Position exit state
            ctx: Position context with unrealized_pnl
            
        Returns:
            True if loss guard was triggered and position closed
        """
        # Skip if already triggered or position closed
        if state.loss_guard_triggered:
            return False
        
        if state.remaining_size is not None and state.remaining_size <= 0:
            return False
        
        # Get unrealized PnL percentage
        unrealized_pnl_pct = ctx.unrealized_pnl  # Already in percentage
        
        # Check if loss exceeds threshold
        if unrealized_pnl_pct <= -MAX_UNREALIZED_LOSS_PCT_PER_POSITION:
            self.logger.error(
                f"[EXIT_LOSS_GUARD] 🚨 {state.symbol} {state.side}: "
                f"LOSS GUARD TRIGGERED - unrealized_pnl={unrealized_pnl_pct:.2f}% "
                f"<= -{MAX_UNREALIZED_LOSS_PCT_PER_POSITION:.2f}%"
            )
            
            if self.shadow_mode:
                self.logger.info(
                    f"[EXIT_BRAIN_SHADOW] {state.symbol} {state.side}: "
                    f"LOSS GUARD would close full position (shadow mode)"
                )
                state.loss_guard_triggered = True
                return True
            
            try:
                # Calculate close quantity
                remaining_size = state.get_remaining_size()
                
                # Get precision and quantize
                tick_size, step_size = get_binance_precision(state.symbol)
                close_qty = quantize_to_step(remaining_size, step_size)
                
                if close_qty <= 0:
                    self.logger.warning(
                        f"[EXIT_LOSS_GUARD] {state.symbol} {state.side}: "
                        f"close_qty={close_qty} invalid, skipping"
                    )
                    return False
                
                # Build MARKET reduce-only order
                order_side = "SELL" if state.side == "LONG" else "BUY"
                
                order_params = {
                    "symbol": state.symbol,
                    "side": order_side,
                    "type": "MARKET",
                    "quantity": close_qty,
                    "reduceOnly": True,
                }
                
                # Only include positionSide if hedge mode
                if self._hedge_mode:
                    order_params["positionSide"] = state.side
                
                self.logger.error(
                    f"[EXIT_LOSS_GUARD] 🚨 {state.symbol} {state.side}: "
                    f"Closing FULL position {close_qty} with MARKET {order_side} "
                    f"(PnL={unrealized_pnl_pct:.2f}%)"
                )
                
                # Submit via gateway
                resp = await self.exit_order_gateway.submit_exit_order(
                    module_name="exit_brain_executor",
                    symbol=state.symbol,
                    order_params=order_params,
                    order_kind="loss_guard_emergency",
                    explanation=f"Loss guard @ {unrealized_pnl_pct:.2f}%"
                )
                
                if resp and resp.get('status') in ['NEW', 'FILLED']:
                    self.logger.error(
                        f"[EXIT_ORDER] ✅ LOSS GUARD MARKET {order_side} {state.symbol} "
                        f"{close_qty} executed - orderId={resp.get('orderId')}"
                    )
                    
                    # Mark as triggered and clear state
                    state.loss_guard_triggered = True
                    state.remaining_size = 0
                    state.position_size = 0
                    state.active_sl = None
                    state.tp_levels = []
                    state.triggered_legs.clear()
                    
                    # 🔧 Invalidate Router cache (position closed)
                    self.router.invalidate_plan(state.symbol)
                    self.logger.debug(
                        f"[EXIT_LOSS_GUARD] Invalidated Router cache for {state.symbol}"
                    )
                    
                    return True
            
            except Exception as e:
                self.logger.error(
                    f"[EXIT_LOSS_GUARD] ❌ {state.symbol} {state.side}: "
                    f"Failed to execute loss guard: {e}",
                    exc_info=True
                )
                return False
        
        return False
    
    def _recompute_dynamic_tp_and_sl(
        self,
        state: PositionExitState,
        ctx: PositionContext
    ) -> None:
        """
        Recompute dynamic TP and SL levels after TP trigger.
        
        Implements SL ratcheting logic based on number of TPs hit:
        - After TP1: Move SL to breakeven (or slightly above)
        - After TP2: Move SL to TP1 price (lock in profit)
        
        This ensures profits are protected as position moves favorably.
        
        Rules:
        - Only tightens SL (never moves away from price)
        - For LONG: SL can only move UP
        - For SHORT: SL can only move DOWN
        - Respects RATCHET_SL_ENABLED flag
        
        Args:
            state: Position exit state to update
            ctx: Position context with entry price
        """
        # Skip if position closed
        if state.remaining_size is not None and state.remaining_size <= 0:
            self.logger.debug(
                f"[EXIT_RATCHET_SL] {state.symbol} {state.side}: "
                f"Skipping - position closed (remaining_size={state.remaining_size})"
            )
            return
        
        # Skip if ratcheting disabled
        if not RATCHET_SL_ENABLED:
            return
        
        # Get entry price
        entry_price = state.entry_price or ctx.entry_price
        if entry_price is None or entry_price <= 0:
            self.logger.warning(
                f"[EXIT_RATCHET_SL] {state.symbol} {state.side}: "
                f"Cannot ratchet - invalid entry_price={entry_price}"
            )
            return
        
        # Current SL
        current_sl = state.active_sl
        new_sl = current_sl
        ratchet_reason = None
        
        # LONG position ratcheting rules
        if state.side == "LONG":
            # After first TP: move SL to breakeven (or slightly above)
            if state.tp_hits_count >= 1:
                breakeven_sl = entry_price  # Can add small buffer if desired
                
                if current_sl is None or breakeven_sl > current_sl:
                    new_sl = breakeven_sl
                    ratchet_reason = "breakeven after TP1"
            
            # After second TP: move SL to first TP price (if available)
            if state.tp_hits_count >= 2 and len(state.tp_levels) >= 2:
                # Find first triggered TP price
                first_tp_price = None
                for i, (tp_price, _) in enumerate(state.tp_levels):
                    if i in state.triggered_legs:
                        first_tp_price = tp_price
                        break
                
                if first_tp_price is not None:
                    if new_sl is None or first_tp_price > new_sl:
                        new_sl = first_tp_price
                        ratchet_reason = f"TP1 price after TP2 hit"
        
        # SHORT position ratcheting rules (mirror of LONG)
        else:  # SHORT
            # After first TP: move SL to breakeven (or slightly below)
            if state.tp_hits_count >= 1:
                breakeven_sl = entry_price
                
                if current_sl is None or breakeven_sl < current_sl:
                    new_sl = breakeven_sl
                    ratchet_reason = "breakeven after TP1"
            
            # After second TP: move SL to first TP price
            if state.tp_hits_count >= 2 and len(state.tp_levels) >= 2:
                # Find first triggered TP price
                first_tp_price = None
                for i, (tp_price, _) in enumerate(state.tp_levels):
                    if i in state.triggered_legs:
                        first_tp_price = tp_price
                        break
                
                if first_tp_price is not None:
                    if new_sl is None or first_tp_price < new_sl:
                        new_sl = first_tp_price
                        ratchet_reason = f"TP1 price after TP2 hit"
        
        # Apply new SL if changed
        if new_sl != current_sl and ratchet_reason is not None:
            old_sl_str = f"${current_sl:.4f}" if current_sl is not None else "None"
            self.logger.warning(
                f"[EXIT_RATCHET_SL] 🎯 {state.symbol} {state.side}: "
                f"SL ratcheted {old_sl_str} → ${new_sl:.4f} - {ratchet_reason} "
                f"(tp_hits={state.tp_hits_count})"
            )
            state.active_sl = new_sl
    
    async def _update_state_from_decision(
        self,
        state: PositionExitState,
        ctx: PositionContext,
        decision: ExitDecision
    ):
        """
        Update internal state based on AI decision.
        
        This method ONLY updates internal state - no exchange orders.
        Actual MARKET order execution happens in _check_and_execute_levels().
        
        Args:
            state: Position exit state to update
            ctx: Position context
            decision: AI decision from adapter
        """
        # Log decision if not NO_CHANGE
        if decision.decision_type != ExitDecisionType.NO_CHANGE:
            self.logger.info(
                f"[EXIT_BRAIN_DECISION] {state.symbol} {state.side}: "
                f"{decision.decision_type.value} - {decision.reason}"
            )
        
        # Handle each decision type
        if decision.decision_type == ExitDecisionType.NO_CHANGE:
            # No state updates needed
            pass
        
        elif decision.decision_type == ExitDecisionType.MOVE_SL:
            # Update SL level with risk floor enforcement
            if decision.new_sl_price:
                # Get position data for leverage calculation
                # ctx.meta contains position_data from Binance
                pos_data = ctx.meta.get('position_data', {})
                entry_price = ctx.entry_price
                
                # Compute effective leverage
                effective_leverage = self._compute_effective_leverage(pos_data)
                
                # Compute risk floor SL
                risk_sl_price, allowed_move_pct = self._compute_risk_floor_sl(
                    entry_price,
                    state.side,
                    effective_leverage
                )
                
                # Log risk calculation
                self.logger.info(
                    f"[EXIT_BRAIN_RISK] {state.symbol} {state.side}: "
                    f"entry=${entry_price:.4f}, leverage={effective_leverage:.1f}x, "
                    f"max_margin_loss={MAX_MARGIN_LOSS_PER_TRADE_PCT:.1%}, "
                    f"allowed_move_pct={allowed_move_pct:.3%}, "
                    f"risk_sl_price=${risk_sl_price:.4f}"
                )
                
                # Check if trade is over-leveraged (impossible to set realistic SL)
                if allowed_move_pct < MIN_PRICE_STOP_DISTANCE_PCT:
                    self.logger.error(
                        f"[EXIT_BRAIN_RISK] {state.symbol} {state.side}: "
                        f"allowed_move_pct={allowed_move_pct:.3%} < "
                        f"MIN_PRICE_STOP_DISTANCE_PCT={MIN_PRICE_STOP_DISTANCE_PCT:.3%} "
                        f"→ FULL_EXIT_NOW triggered due to over-leverage vs risk budget"
                    )
                    # Force immediate exit - trade is over-leveraged for risk budget
                    await self._execute_emergency_exit(state, ctx, decision)
                    return
                
                # Apply risk floor to strategic SL
                strategic_sl = decision.new_sl_price
                final_sl = self._apply_risk_floor_to_sl(
                    strategic_sl,
                    risk_sl_price,
                    state.side,
                    state.symbol
                )
                
                # Update state with risk-adjusted SL
                old_sl = state.active_sl
                state.active_sl = final_sl
                
                sl_reason = decision.sl_reason or decision.reason or "AI decision"
                
                if old_sl is None:
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"INITIAL SL set to ${state.active_sl:.4f} - {sl_reason} "
                        f"(risk-adjusted from ${strategic_sl:.4f})"
                    )
                else:
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"SL moved ${old_sl:.4f} → ${state.active_sl:.4f} - {sl_reason} "
                        f"(risk-adjusted from ${strategic_sl:.4f})"
                    )
        
        elif decision.decision_type == ExitDecisionType.UPDATE_TP_LIMITS:
            # Update TP levels
            if decision.new_tp_levels:
                # Cap at MAX_TP_LEVELS for sanity
                tp_prices = decision.new_tp_levels[:MAX_TP_LEVELS]
                
                # Build fractions
                fractions = decision.tp_fractions or [
                    1.0 / len(tp_prices)
                ] * len(tp_prices)
                
                # Ensure fractions match tp_prices length
                fractions = fractions[:len(tp_prices)]
                
                # Normalize fractions to sum to 1.0
                total_frac = sum(fractions)
                if total_frac > 1.0:
                    # Scale down to 1.0
                    fractions = [f / total_frac for f in fractions]
                    self.logger.warning(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"TP fractions sum={total_frac:.3f} > 1.0, normalized to 1.0"
                    )
                elif total_frac < 1.0:
                    # Allow runner remainder (position keeps running after all TPs)
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"TP fractions sum={total_frac:.3f} < 1.0, "
                        f"remaining {(1-total_frac)*100:.1f}% will be runner"
                    )
                
                state.tp_levels = list(zip(tp_prices, fractions))
                state.triggered_legs.clear()  # New plan resets executed legs
                # Reset remaining_size to current observed size so new plan fractions apply correctly
                state.remaining_size = state.position_size
                
                # Format for logging
                tp_str = ", ".join([
                    f"${price:.4f}({frac*100:.0f}%)"
                    for price, frac in state.tp_levels
                ])
                
                self.logger.info(
                    f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                    f"TP levels set: [{tp_str}] - {decision.reason}"
                )
        
        elif decision.decision_type == ExitDecisionType.FULL_EXIT_NOW:
            # Emergency exit - execute immediately
            await self._execute_emergency_exit(state, ctx, decision)
        
        elif decision.decision_type == ExitDecisionType.PARTIAL_CLOSE:
            # Partial close - execute immediately
            await self._execute_partial_close(state, ctx, decision)
    
    async def _execute_emergency_exit(
        self,
        state: PositionExitState,
        ctx: PositionContext,
        decision: ExitDecision
    ):
        """Execute emergency full exit with MARKET order."""
        if self.shadow_mode:
            self.logger.info(
                f"[EXIT_BRAIN_SHADOW] {state.symbol} {state.side}: "
                f"FULL_EXIT_NOW (shadow mode, no order) - {decision.reason}"
            )
            return
        
        try:
            # Build MARKET reduce-only order
            order_side = "SELL" if state.side == "LONG" else "BUY"
            
            # Get precision
            precision = get_binance_precision(state.symbol)
            qty = quantize_to_step(state.position_size, precision['stepSize'])
            
            if qty <= 0:
                self.logger.warning(
                    f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                    f"FULL_EXIT qty={qty} invalid after quantization, skipping"
                )
                return
            
            order_params = {
                "symbol": state.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": qty,
                "reduceOnly": True,
            }
            
            # Only include positionSide if hedge mode
            if self._hedge_mode:
                order_params["positionSide"] = state.side
            
            self.logger.warning(
                f"[EXIT_EMERGENCY] {state.symbol} {state.side}: "
                f"FULL EXIT @ ${state.last_price:.4f} - {decision.reason} - "
                f"Closing {qty} with MARKET {order_side}"
            )
            
            # Submit via gateway
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="emergency_exit",
                explanation=decision.reason
            )
            
            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.info(
                    f"[EXIT_ORDER] ✅ EMERGENCY MARKET {order_side} {state.symbol} "
                    f"{qty} executed successfully"
                )
                
                # Clear state after full exit
                state.active_sl = None
                state.tp_levels = []
                state.triggered_legs.clear()
                
                # 🔧 Invalidate Router cache (position closed)
                self.router.invalidate_plan(state.symbol)
                self.logger.debug(
                    f"[EXIT_BRAIN_EXECUTOR] Invalidated Router cache for {state.symbol}"
                )
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error executing emergency exit for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
            )
    
    async def _execute_partial_close(
        self,
        state: PositionExitState,
        ctx: PositionContext,
        decision: ExitDecision
    ):
        """Execute partial close with MARKET order."""
        if self.shadow_mode:
            self.logger.info(
                f"[EXIT_BRAIN_SHADOW] {state.symbol} {state.side}: "
                f"PARTIAL_CLOSE {decision.fraction_to_close*100:.0f}% "
                f"(shadow mode, no order) - {decision.reason}"
            )
            return
        
        try:
            fraction = decision.fraction_to_close or 0.0
            if fraction <= 0 or fraction > 1.0:
                self.logger.warning(
                    f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                    f"Invalid fraction {fraction}, skipping partial close"
                )
                return
            
            # Calculate close quantity
            close_qty = state.position_size * fraction
            
            # Get precision and quantize
            precision = get_binance_precision(state.symbol)
            close_qty = quantize_to_step(close_qty, precision['stepSize'])
            
            if close_qty <= 0:
                self.logger.warning(
                    f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                    f"PARTIAL_CLOSE qty={close_qty} invalid after quantization, skipping"
                )
                return
            
            # Build MARKET reduce-only order
            order_side = "SELL" if state.side == "LONG" else "BUY"
            
            order_params = {
                "symbol": state.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": close_qty,
                "reduceOnly": True,
            }
            
            # Only include positionSide if hedge mode
            if self._hedge_mode:
                order_params["positionSide"] = state.side
            
            self.logger.info(
                f"[EXIT_PARTIAL] {state.symbol} {state.side}: "
                f"PARTIAL CLOSE @ ${state.last_price:.4f} - {decision.reason} - "
                f"Closing {close_qty} ({fraction*100:.0f}%) with MARKET {order_side}"
            )
            
            # Submit via gateway
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="partial_close",
                explanation=decision.reason
            )
            
            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.info(
                    f"[EXIT_ORDER] ✅ PARTIAL MARKET {order_side} {state.symbol} "
                    f"{close_qty} executed successfully"
                )
                
                # Update position size (assume filled)
                state.position_size = max(0, state.position_size - close_qty)
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error executing partial close for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
            )
    
    async def _cleanup_closed_positions(self, active_keys: Set[str]):
        """
        Remove state for positions that are no longer open.
        
        Also cancels hard SL orders on Binance to prevent orphaned orders.
        """
        closed_keys = set(self._state.keys()) - active_keys
        
        for key in closed_keys:
            state = self._state[key]
            
            # Cancel hard SL if exists
            if state.hard_sl_order_id:
                try:
                    # Get client for cancellation
                    from backend.integrations.binance.client_wrapper import BinanceClientWrapper
                    wrapper = BinanceClientWrapper()
                    
                    await wrapper.call_async(
                        self.position_source.futures_cancel_order,
                        symbol=state.symbol,
                        orderId=state.hard_sl_order_id
                    )
                    
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] 🗑️ {state.symbol} {state.side}: "
                        f"HARD SL cancelled (position closed) - orderId={state.hard_sl_order_id}"
                    )
                except Exception as e:
                    # Log but don't fail - position already closed
                    self.logger.warning(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"Failed to cancel HARD SL (may already be cancelled): {e}"
                    )
            
            self.logger.info(
                f"[EXIT_BRAIN_EXECUTOR] Removed state for closed position: {key}"
            )
            del self._state[key]
    
    async def _check_and_execute_levels(self):
        """
        CORE MONITORING METHOD: Check all states for triggered levels.
        
        For each position state:
        1. Check if SL triggered → execute MARKET full close
        2. Check if any TP legs triggered → execute MARKET partial close
        
        Rules:
        - LONG: SL when price <= active_sl, TP when price >= tp_price
        - SHORT: SL when price >= active_sl, TP when price <= tp_price
        - SL closes full remaining position
        - TP closes partial (size_pct of remaining)
        - Only one TP per cycle (prevents race conditions)
        """
        for state_key, state in list(self._state.items()):
            try:
                # Skip if no price data
                if state.last_price is None:
                    continue
                
                current_price = state.last_price
                
                # Skip if no levels set
                if state.active_sl is None and not state.tp_levels:
                    continue
                
                # Debug log
                self.logger.debug(
                    f"[EXIT_MONITOR] {state_key}: "
                    f"price=${current_price:.4f}, "
                    f"SL={f'${state.active_sl:.4f}' if state.active_sl else 'None'}, "
                    f"TPs={len(state.tp_levels)}, "
                    f"triggered={len(state.triggered_legs)}"
                )
                
                # Check SL first (highest priority)
                should_trigger = state.should_trigger_sl(current_price)
                sl_str = f"{state.active_sl:.4f}" if state.active_sl else "None"
                self.logger.debug(
                    f"[EXIT_SL_CHECK] {state_key}: should_trigger_sl={should_trigger} "
                    f"(price={current_price:.4f}, SL={sl_str}, side={state.side})"
                )
                if should_trigger:
                    self.logger.warning(
                        f"[EXIT_SL_TRIGGER_START] {state_key}: SL TRIGGERING NOW!"
                    )
                    await self._execute_sl_trigger(state, current_price)
                    # After SL, position is closed - skip TP checks
                    continue
                
                # Check TP legs
                triggerable_tps = state.get_triggerable_tp_legs(current_price)
                
                # Log TP checks at WARNING level for visibility
                if state.tp_levels:
                    self.logger.warning(
                        f"[EXIT_TP_CHECK] {state_key}: price=${current_price:.5f}, "
                        f"triggerable={len(triggerable_tps)}/{len(state.tp_levels)} TPs"
                    )
                    for i, (tp_price, size_pct) in enumerate(state.tp_levels):
                        already_triggered = i in state.triggered_legs
                        should_trigger = (
                            (state.side == "LONG" and current_price >= tp_price) or
                            (state.side == "SHORT" and current_price <= tp_price)
                        )
                        self.logger.warning(
                            f"  TP{i}: price=${tp_price:.5f}, size={size_pct:.1%}, "
                            f"triggered={already_triggered}, should_trigger={should_trigger}"
                        )
                
                if triggerable_tps:
                    # Execute first triggered TP only (one per cycle for safety)
                    leg_index, tp_price, size_pct = triggerable_tps[0]
                    self.logger.warning(
                        f"[EXIT_TP_TRIGGER] 🎯 {state_key}: TP{leg_index} HIT @ ${current_price:.5f} "
                        f"(target=${tp_price:.5f}, size={size_pct:.1%})"
                    )
                    await self._execute_tp_trigger(
                        state, leg_index, tp_price, size_pct, current_price
                    )
                
            except Exception as e:
                self.logger.error(
                    f"[EXIT_BRAIN_EXECUTOR] Error checking levels for {state_key}: {e}",
                    exc_info=True
                )
    
    async def _execute_sl_trigger(self, state: PositionExitState, current_price: float):
        """Execute SL trigger with MARKET order."""
        if self.shadow_mode:
            self.logger.warning(
                f"[EXIT_BRAIN_SHADOW] {state.symbol} {state.side}: "
                f"SL TRIGGERED @ ${current_price:.4f} (SL=${state.active_sl:.4f}) "
                f"(shadow mode, no order)"
            )
            return
        
        order_params: Dict[str, Any] = {}
        try:
            # Calculate remaining size after TPs
            remaining_size = state.get_remaining_size()
            
            if remaining_size <= 0:
                self.logger.warning(
                    f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                    f"SL triggered but remaining_size={remaining_size}, skipping"
                )
                return
            
            # Get precision and quantize
            tick_size, step_size = get_binance_precision(state.symbol)
            close_qty = quantize_to_step(remaining_size, step_size)
            
            if close_qty <= 0:
                # Fallback to closePosition if we still have non-zero remaining size
                if remaining_size > 0:
                    order_side = "SELL" if state.side == "LONG" else "BUY"
                    order_params = {
                        "symbol": state.symbol,
                        "side": order_side,
                        "type": "MARKET",
                        "closePosition": True,
                    }
                    
                    # Only include positionSide if hedge mode
                    if self._hedge_mode:
                        order_params["positionSide"] = state.side
                    self.logger.warning(
                        f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                        f"SL quantized to 0, using closePosition MARKET fallback (remaining={remaining_size})"
                    )
                    await self._submit_and_finalize_sl(state, order_params, current_price)
                else:
                    self.logger.warning(
                        f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                        f"SL close_qty={close_qty} invalid after quantization, skipping"
                    )
                return
            
            # Build MARKET order
            order_side = "SELL" if state.side == "LONG" else "BUY"
            
            order_params = {
                "symbol": state.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": close_qty,
            }
            
            # Only include positionSide if hedge mode is enabled
            if self._hedge_mode:
                order_params["positionSide"] = state.side
            
            self.logger.warning(
                f"[EXIT_SL_TRIGGER] 🛑 {state.symbol} {state.side}: "
                f"SL HIT @ ${current_price:.4f} (SL=${state.active_sl:.4f}) - "
                f"Closing {close_qty} with MARKET {order_side}"
            )
            
            self.logger.warning(
                f"[EXIT_SL_ORDER] 📤 Submitting to Binance: {order_params}"
            )
            
            # Submit via gateway
            self.logger.info(f"[EXIT_SL_ORDER] Calling gateway.submit_exit_order...")
            
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="sl_market",
                explanation=f"SL triggered at ${current_price:.4f}"
            )
            
            self.logger.warning(
                f"[EXIT_SL_ORDER] 📥 Response from Binance: {resp}"
            )
            
            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.warning(
                    f"[EXIT_ORDER] ✅ SL MARKET {order_side} {state.symbol} "
                    f"{close_qty} executed successfully - orderId={resp.get('orderId')}"
                )
                
                await self._finalize_sl_state(state)
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] ❌ SL TRIGGER FAILED for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
            )
            self.logger.error(
                f"[EXIT_SL_ORDER] ❌ Order params were: {order_params}"
            )

    async def _submit_and_finalize_sl(self, state: PositionExitState, order_params: Dict[str, Any], current_price: float):
        """Submit SL order params and finalize state on success."""
        try:
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="sl_market",
                explanation=f"SL triggered at ${current_price:.4f}"
            )

            self.logger.warning(
                f"[EXIT_SL_ORDER] 📥 Response from Binance: {resp}"
            )

            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.warning(
                    f"[EXIT_ORDER] ✅ SL MARKET {order_params['side']} {state.symbol} "
                    f"closePosition order executed successfully - orderId={resp.get('orderId')}"
                )
                await self._finalize_sl_state(state)
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] ❌ SL fallback failed for {state.symbol} {state.side}: {e}",
                exc_info=True
            )

    async def _finalize_sl_state(self, state: PositionExitState):
        """Clear state after SL trigger and invalidate router cache."""
        # Clear state after SL trigger (position closed)
        state.active_sl = None
        state.tp_levels = []
        state.triggered_legs.clear()
        state.position_size = 0
        state.remaining_size = 0

        # 🔧 Invalidate Router cache (position closed)
        self.router.invalidate_plan(state.symbol)
        self.logger.debug(
            f"[EXIT_SL_TRIGGER] Invalidated Router cache for {state.symbol}"
        )
    
    async def _execute_tp_trigger(
        self,
        state: PositionExitState,
        leg_index: int,
        tp_price: float,
        size_pct: float,
        current_price: float
    ):
        """Execute TP trigger with MARKET order."""
        if self.shadow_mode:
            self.logger.info(
                f"[EXIT_BRAIN_SHADOW] {state.symbol} {state.side}: "
                f"TP{leg_index} TRIGGERED @ ${current_price:.4f} (TP=${tp_price:.4f}) "
                f"size_pct={size_pct*100:.0f}% (shadow mode, no order)"
            )
            return
        
        try:
            # Calculate close quantity (size_pct of REMAINING, not original)
            remaining_size = state.get_remaining_size()
            close_qty = remaining_size * size_pct
            
            # Get precision and quantize
            tick_size, step_size = get_binance_precision(state.symbol)
            pre_quant = close_qty
            close_qty = quantize_to_step(close_qty, step_size)
            self.logger.info(
                f"[EXIT_TP_QTY] {state.symbol} {state.side}: remaining={remaining_size:.6f}, "
                f"requested={pre_quant:.6f}, quantized={close_qty:.6f}, step={step_size}"
            )
            
            if close_qty <= 0:
                if remaining_size > 0:
                    order_side = "SELL" if state.side == "LONG" else "BUY"
                    order_params = {
                        "symbol": state.symbol,
                        "side": order_side,
                        "type": "MARKET",
                        "closePosition": True,
                    }
                    
                    # Only include positionSide if hedge mode
                    if self._hedge_mode:
                        order_params["positionSide"] = state.side
                    self.logger.warning(
                        f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                        f"TP{leg_index} quantized to 0, using closePosition MARKET fallback "
                        f"(remaining={remaining_size})"
                    )
                    await self._submit_tp_order_and_update_state(
                        state, leg_index, order_params, 0.0, current_price, close_position=True
                    )
                else:
                    self.logger.warning(
                        f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                        f"TP{leg_index} close_qty={close_qty} invalid after quantization, skipping"
                    )
                return
            
            # Ensure we don't exceed remaining size
            if close_qty > remaining_size:
                close_qty = remaining_size
            
            # Build MARKET reduce-only order
            order_side = "SELL" if state.side == "LONG" else "BUY"
            
            order_params = {
                "symbol": state.symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": close_qty,
            }
            
            # Only include positionSide if hedge mode
            if self._hedge_mode:
                order_params["positionSide"] = state.side
            
            self.logger.warning(
                f"[EXIT_TP_TRIGGER] 🎯 {state.symbol} {state.side}: "
                f"TP{leg_index} HIT @ ${current_price:.4f} (TP=${tp_price:.4f}) - "
                f"Closing {close_qty} ({size_pct*100:.0f}% of position) with MARKET {order_side}"
            )
            
            self.logger.warning(
                f"[EXIT_TP_ORDER] 📤 Submitting to Binance: {order_params}"
            )
            
            await self._submit_tp_order_and_update_state(
                state,
                leg_index,
                order_params,
                close_qty,
                current_price,
                close_position=False
            )
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error executing TP{leg_index} trigger for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
            )

    async def _submit_tp_order_and_update_state(
        self,
        state: PositionExitState,
        leg_index: int,
        order_params: Dict[str, Any],
        close_qty: float,
        current_price: float,
        close_position: bool
    ):
        """Submit TP order (qty or closePosition) and update state on success."""
        try:
            remaining_before = state.get_remaining_size()
            self.logger.info(f"[EXIT_TP_ORDER] Calling gateway.submit_exit_order...")
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind=f"tp_market_leg_{leg_index}",
                explanation=f"TP{leg_index} triggered at ${current_price:.4f}"
            )

            self.logger.warning(
                f"[EXIT_TP_ORDER] 📥 Response from Binance: {resp}"
            )

            if resp and resp.get('status') in ['NEW', 'FILLED']:
                executed_qty = close_qty if close_qty > 0 else remaining_before
                self.logger.warning(
                    f"[EXIT_ORDER] ✅ TP{leg_index} MARKET {order_params['side']} {state.symbol} "
                    f"{executed_qty} executed successfully - orderId={resp.get('orderId')}"
                )

                # Mark this leg as triggered
                state.triggered_legs.add(leg_index)

                # Determine actual size closed
                closed = executed_qty if not close_position else remaining_before

                # Update position size (assume filled)
                state.position_size = max(0, state.position_size - closed)

                # Update remaining_size for dynamic tracking
                if state.remaining_size is not None:
                    state.remaining_size = max(0, state.remaining_size - closed)
                else:
                    state.remaining_size = state.position_size

                # Increment TP hits counter
                state.tp_hits_count += 1

                self.logger.warning(
                    f"[EXIT_TP_TRIGGER] 📊 {state.symbol} {state.side}: "
                    f"Updated state after TP{leg_index} - "
                    f"remaining_size={state.remaining_size:.4f}, "
                    f"tp_hits_count={state.tp_hits_count}"
                )

                # DYNAMIC RECOMPUTATION: Adjust SL based on TP hits
                dummy_ctx = PositionContext(
                    symbol=state.symbol,
                    side=state.side.lower(),
                    entry_price=state.entry_price or 0,
                    current_price=current_price,
                    size=state.remaining_size or 0,
                    unrealized_pnl=0
                )
                self._recompute_dynamic_tp_and_sl(state, dummy_ctx)

        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error submitting TP order for {state.symbol} {state.side}: {e}",
                exc_info=True
            )
    
    # ========================================================================
    # CHALLENGE_100 MODE METHODS
    # ========================================================================
    
    async def _apply_challenge_100_logic(self, state: PositionExitState, pos_data: Dict, mark_price: float):
        """
        Apply $100 Challenge Mode exit logic.
        
        Rules:
        1. Initial SL: entry ± 2*ATR (clamped to max 1.5R loss)
        2. TP1 at +1R: Take 30% partial
        3. After TP1: Move SL to BE+fees
        4. Runner: 70% with 2*ATR trailing
        5. Time stop: Close if not TP1 within 7200s
        6. Liq safety: SL must be before liq price
        """
        import time
        
        state_key = f"{state.symbol}:{state.side}"
        entry_price = state.entry_price or float(pos_data.get('entryPrice', 0))
        
        if entry_price <= 0:
            self.logger.warning(f"[CHALLENGE_100] {state_key}: Invalid entry_price, skipping")
            return
        
        # Get account equity for 1R calculation
        equity_usdt = await self._get_account_equity()
        r_usdt = equity_usdt * self.challenge_risk_pct
        
        # Calculate ATR for this symbol
        atr = await self._calculate_atr(state.symbol)
        
        # 1. INITIAL SL: entry ± 2*ATR (clamped to 1.5R max loss)
        if state.active_sl is None:
            initial_sl = self._calculate_challenge_initial_sl(entry_price, state.side, atr, r_usdt, state.position_size)
            state.active_sl = initial_sl
            self.logger.warning(
                f"[CHALLENGE_100] {state_key}: Initial SL set @ ${initial_sl:.4f} "
                f"(2*ATR distance, clamped to {self.challenge_max_risk_r}R)"
            )
        
        # 2. LIQ SAFETY: Ensure SL is before liquidation price
        liq_price = float(pos_data.get('liquidationPrice', 0))
        if liq_price > 0:
            safe_sl = self._apply_liq_safety(state.side, state.active_sl, liq_price, entry_price)
            if safe_sl != state.active_sl:
                self.logger.error(
                    f"[CHALLENGE_100] {state_key}: SL adjusted for liq safety: "
                    f"${state.active_sl:.4f} → ${safe_sl:.4f} (liq=${liq_price:.4f})"
                )
                state.active_sl = safe_sl
        
        # 3. TP1 TRACKING: Check if +1R profit reached
        unrealized_pnl_usdt = (mark_price - entry_price) * state.position_size if state.side == "LONG" else (entry_price - mark_price) * state.position_size
        tp1_target_usdt = self.challenge_tp1_r * r_usdt
        
        if not state.tp1_taken and unrealized_pnl_usdt >= tp1_target_usdt:
            # TP1 triggered! Set up partial close
            tp1_price = mark_price  # Use current price as TP1 trigger
            tp1_qty_pct = self.challenge_tp1_qty_pct
            
            # Override tp_levels to trigger TP1 in next cycle
            state.tp_levels = [(tp1_price, tp1_qty_pct)]
            state.tp1_taken = True  # Mark as triggered (will execute in _check_and_execute_levels)
            
            self.logger.warning(
                f"[CHALLENGE_100] 🎯 {state_key}: TP1 REACHED @ ${mark_price:.4f} "
                f"(+{unrealized_pnl_usdt:.2f} USDT >= +{tp1_target_usdt:.2f} USDT = +{self.challenge_tp1_r}R) "
                f"- Will take {tp1_qty_pct:.0%} partial"
            )
        
        # 4. AFTER TP1: Move SL to BE+fees
        if state.tp1_taken and len(state.triggered_legs) > 0:
            # TP1 has been executed, move SL to breakeven + fees
            fee_rate = 0.0004  # 0.04% taker fee (approximate)
            notional = entry_price * state.initial_size
            total_fees = notional * fee_rate * 2  # Open + close fees
            fees_per_unit = total_fees / state.initial_size
            
            be_plus_fees = entry_price + fees_per_unit if state.side == "LONG" else entry_price - fees_per_unit
            
            # Only move SL up (LONG) or down (SHORT), never loosen
            should_update = False
            if state.side == "LONG" and (state.active_sl is None or be_plus_fees > state.active_sl):
                should_update = True
            elif state.side == "SHORT" and (state.active_sl is None or be_plus_fees < state.active_sl):
                should_update = True
            
            if should_update:
                state.active_sl = be_plus_fees
                self.logger.warning(
                    f"[CHALLENGE_100] 🛡️ {state_key}: SL moved to BE+fees @ ${be_plus_fees:.4f} "
                    f"(entry=${entry_price:.4f}, fees_per_unit=${fees_per_unit:.6f})"
                )
            
            # 5. RUNNER TRAILING: After TP1, trail runner with 2*ATR
            self._apply_challenge_trailing(state, mark_price, atr)
        
        # 6. TIME STOP: Close if not TP1 within time_stop_sec
        if not state.tp1_taken and state.opened_at_ts is not None:
            elapsed = time.time() - state.opened_at_ts
            if elapsed >= self.challenge_time_stop_sec:
                self.logger.error(
                    f"[CHALLENGE_100] ⏰ {state_key}: TIME STOP triggered "
                    f"(elapsed={elapsed:.0f}s >= {self.challenge_time_stop_sec:.0f}s) - "
                    f"TP1 not reached, closing position"
                )
                # Force full close via setting active_SL to current price
                state.active_sl = mark_price if state.side == "LONG" else mark_price
    
    def _calculate_challenge_initial_sl(self, entry_price: float, side: str, atr: float, r_usdt: float, position_size: float) -> float:
        """Calculate initial SL for CHALLENGE_100 based on 2*ATR, clamped to 1.5R max loss."""
        # Baseline: 2*ATR distance from entry
        if side == "LONG":
            sl_price = entry_price - (2 * atr)
        else:  # SHORT
            sl_price = entry_price + (2 * atr)
        
        # Clamp to max 1.5R loss
        max_loss_usdt = self.challenge_max_risk_r * r_usdt
        max_move_pct = max_loss_usdt / (entry_price * position_size)
        
        if side == "LONG":
            min_sl_price = entry_price * (1 - max_move_pct)
            sl_price = max(sl_price, min_sl_price)  # SL can't be lower than max loss
        else:  # SHORT
            max_sl_price = entry_price * (1 + max_move_pct)
            sl_price = min(sl_price, max_sl_price)  # SL can't be higher than max loss
        
        return sl_price
    
    def _apply_liq_safety(self, side: str, current_sl: float, liq_price: float, entry_price: float) -> float:
        """Ensure SL is before liquidation price with buffer."""
        if liq_price <= 0:
            return current_sl
        
        buffer = entry_price * self.challenge_liq_buffer_pct
        
        if side == "LONG":
            # Liq is below entry, SL must be above liq
            safe_sl = liq_price + buffer
            if current_sl < safe_sl:
                return safe_sl
        else:  # SHORT
            # Liq is above entry, SL must be below liq
            safe_sl = liq_price - buffer
            if current_sl > safe_sl:
                return safe_sl
        
        return current_sl
    
    def _apply_challenge_trailing(self, state: PositionExitState, current_price: float, atr: float):
        """Apply 2*ATR trailing to runner position after TP1."""
        # Track highest/lowest favorable price
        if state.side == "LONG":
            if state.highest_favorable_price is None or current_price > state.highest_favorable_price:
                state.highest_favorable_price = current_price
            
            # Trail SL: HFP - 2*ATR
            new_sl = state.highest_favorable_price - (2 * atr)
            
            # Only move SL up, never down
            if state.active_sl is None or new_sl > state.active_sl:
                old_sl = state.active_sl
                state.active_sl = new_sl
                self.logger.info(
                    f"[CHALLENGE_100] 📈 {state.symbol} LONG: Runner trailing updated - "
                    f"HFP=${state.highest_favorable_price:.4f}, SL ${old_sl:.4f if old_sl else 'None'} → ${new_sl:.4f} "
                    f"(2*ATR=${2*atr:.4f})"
                )
        else:  # SHORT
            if state.highest_favorable_price is None or current_price < state.highest_favorable_price:
                state.highest_favorable_price = current_price
            
            # Trail SL: LFP + 2*ATR
            new_sl = state.highest_favorable_price + (2 * atr)
            
            # Only move SL down, never up
            if state.active_sl is None or new_sl < state.active_sl:
                old_sl = state.active_sl
                state.active_sl = new_sl
                self.logger.info(
                    f"[CHALLENGE_100] 📉 {state.symbol} SHORT: Runner trailing updated - "
                    f"LFP=${state.highest_favorable_price:.4f}, SL ${old_sl:.4f if old_sl else 'None'} → ${new_sl:.4f} "
                    f"(2*ATR=${2*atr:.4f})"
                )
    
    async def _get_account_equity(self) -> float:
        """Get account equity in USDT for 1R calculation."""
        try:
            from backend.integrations.binance.client_wrapper import BinanceClientWrapper
            wrapper = BinanceClientWrapper()
            
            account_info = await wrapper.call_async(
                self.position_source.futures_account
            )
            
            # Get total wallet balance (USDT equity)
            equity = float(account_info.get('totalWalletBalance', 100.0))
            
            self.logger.debug(f"[CHALLENGE_100] Account equity: ${equity:.2f} USDT")
            return equity
        except Exception as e:
            self.logger.error(f"[CHALLENGE_100] Failed to get account equity: {e}, using default 100 USDT")
            return 100.0  # Fallback for $100 challenge
    
    async def _calculate_atr(self, symbol: str, periods: int = 14) -> float:
        """Calculate ATR (Average True Range) for symbol."""
        try:
            # Check cache first
            cached = self._market_data_cache.get(symbol)
            if cached and 'atr' in cached:
                cache_age = datetime.now(timezone.utc).timestamp() - cached['timestamp']
                if cache_age < self._cache_ttl_sec:
                    return cached['atr']
            
            # Fetch recent candles
            from backend.integrations.binance.client_wrapper import BinanceClientWrapper
            wrapper = BinanceClientWrapper()
            
            klines = await wrapper.call_async(
                self.position_source.futures_klines,
                symbol=symbol,
                interval="5m",
                limit=periods + 1
            )
            
            if not klines or len(klines) < 2:
                self.logger.warning(f"[CHALLENGE_100] No klines for {symbol}, using default ATR")
                return 0.01  # 1% default
            
            # Calculate True Range for each period
            trs = []
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                prev_close = float(klines[i-1][4])
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                trs.append(tr)
            
            # ATR = average of true ranges
            atr = sum(trs) / len(trs) if trs else 0.01
            
            # Cache result
            if symbol not in self._market_data_cache:
                self._market_data_cache[symbol] = {}
            self._market_data_cache[symbol]['atr'] = atr
            self._market_data_cache[symbol]['timestamp'] = datetime.now(timezone.utc).timestamp()
            
            self.logger.debug(f"[CHALLENGE_100] {symbol} ATR (14x5m): ${atr:.4f}")
            return atr
        
        except Exception as e:
            self.logger.error(f"[CHALLENGE_100] ATR calculation failed for {symbol}: {e}")
            return 0.01  # Fallback
    
    async def _place_hard_sl_challenge(self, state: PositionExitState, entry_price: float):
        """
        Place hard SL safety net for CHALLENGE_100 mode.
        
        **Critical Safety Net**: Hard SL placed 0.3R "behind" soft SL as last-resort protection.
        
        **LIVE Mode Only**: Only active when:
        - EXIT_MODE=EXIT_BRAIN_V3
        - EXIT_EXECUTOR_MODE=LIVE
        - EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
        
        **Gateway Compatibility**: Uses module_name="exit_brain_executor" to avoid blocking.
        """
        if state.active_sl is None:
            self.logger.error(
                f"[CHALLENGE_100_HARD_SL] {state.symbol} {state.side}: Cannot place hard SL - "
                f"soft SL not set yet"
            )
            return
        
        # Hard SL = soft SL + 0.3R margin (behind soft SL for safety buffer)
        equity = await self._get_account_equity()
        r_usdt = equity * self.challenge_risk_pct
        extra_r_distance = 0.3 * r_usdt / state.position_size  # Convert 0.3R to price
        
        if state.side == "LONG":
            hard_sl_price = state.active_sl - extra_r_distance
        else:  # SHORT
            hard_sl_price = state.active_sl + extra_r_distance
        
        # Quantize price to exchange precision
        tick_size, step_size = get_binance_precision(state.symbol)
        hard_sl_price = quantize_to_tick(hard_sl_price, tick_size)
        
        # Build Binance USDT-PERP STOP_MARKET order
        order_side = "SELL" if state.side == "LONG" else "BUY"
        order_params = {
            "symbol": state.symbol,
            "side": order_side,
            "type": "STOP_MARKET",
            "stopPrice": hard_sl_price,
            "closePosition": True,  # Close entire remaining position
        }
        
        # Only include positionSide if hedge mode
        if self._hedge_mode:
            order_params["positionSide"] = state.side  # LONG or SHORT
        
        self.logger.warning(
            f"[CHALLENGE_100_HARD_SL] 🛡️ {state.symbol} {state.side}: Attempting to place HARD SL safety net\n"
            f"  Soft SL: ${state.active_sl:.4f}\n"
            f"  Hard SL: ${hard_sl_price:.4f} (0.3R buffer = ${extra_r_distance:.4f})\n"
            f"  Order: {order_side} STOP_MARKET @ stopPrice={hard_sl_price}, reduceOnly=True, closePosition=True\n"
            f"  Module: exit_brain_executor (gateway-compatible)"
        )
        
        try:
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_brain_executor",  # Critical: Must match EXPECTED_EXIT_BRAIN_MODULES
                symbol=state.symbol,
                order_params=order_params,
                order_kind="hard_sl",  # Valid order kind from VALID_ORDER_KINDS
                explanation=f"CHALLENGE_100 hard SL safety net @ {hard_sl_price:.4f}"
            )
            
            if resp and resp.get('orderId'):
                state.hard_sl_price = hard_sl_price
                state.hard_sl_order_id = str(resp['orderId'])
                self.logger.warning(
                    f"[CHALLENGE_100_HARD_SL] ✅ Hard SL placed successfully\n"
                    f"  OrderID: {state.hard_sl_order_id}\n"
                    f"  Symbol: {state.symbol} {state.side}\n"
                    f"  StopPrice: ${hard_sl_price:.4f}\n"
                    f"  Type: STOP_MARKET, reduceOnly=True, closePosition=True"
                )
            elif resp is None:
                # Gateway returned None - likely blocked or failed
                self.logger.error(
                    f"[CHALLENGE_100_HARD_SL] ❌ BLOCKED: Gateway returned None for hard SL\n"
                    f"  Symbol: {state.symbol} {state.side}\n"
                    f"  StopPrice: ${hard_sl_price:.4f}\n"
                    f"  This may indicate: module_name rejection, LIVE mode not enabled, or exchange error\n"
                    f"  FALLBACK: Will rely on soft SL tracking only"
                )
            else:
                # Response but no orderId
                self.logger.error(
                    f"[CHALLENGE_100_HARD_SL] ⚠️ Unexpected response: {resp}\n"
                    f"  Symbol: {state.symbol} {state.side}\n"
                    f"  FALLBACK: Will rely on soft SL tracking only"
                )
        except Exception as e:
            self.logger.error(
                f"[CHALLENGE_100_HARD_SL] ❌ Exception during hard SL placement\n"
                f"  Symbol: {state.symbol} {state.side}\n"
                f"  Error: {e}\n"
                f"  FALLBACK: Will rely on soft SL tracking only",
                exc_info=True
            )
