"""
Exit Brain v3 - Dynamic Executor (Clean Implementation)

ARCHITECTURE: AI Planner + Active Monitoring + MARKET Exits

Flow:
1. ExitBrainV3 (planner) creates ExitPlan with exit legs (TP, SL, etc.)
2. ExitBrainAdapter translates plan to ExitDecision (MOVE_SL, UPDATE_TP_LIMITS, etc.)
3. Executor stores levels INTERNALLY in PositionExitState (NO exchange orders)
4. Monitoring loop checks price vs levels every N seconds
5. When level hit → execute MARKET reduce-only order immediately

Key Design Principles:
- NO LIMIT/STOP/TAKE_PROFIT orders on exchange
- ALL exits are MARKET + reduceOnly for instant execution
- AI has full control - can adjust levels anytime without exchange interaction
- True dynamic SL/TP management without order replacement complexity

State Management:
- PositionExitState per position (key: "{symbol}:{side}")
- Tracks active_sl, tp_levels, triggered_legs internally
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
from .precision import quantize_to_tick, quantize_to_step, get_binance_precision

logger = logging.getLogger(__name__)


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
            is_exit_brain_live_fully_enabled
        )
        
        self.adapter = adapter
        self.exit_order_gateway = exit_order_gateway
        self.position_source = position_source
        self.loop_interval_sec = loop_interval_sec
        
        # Determine operating mode from config
        if is_exit_brain_live_fully_enabled():
            self.effective_mode = "LIVE"
            self.shadow_mode = False
        else:
            self.effective_mode = "SHADOW"
            self.shadow_mode = True
        
        # CORE STATE: Internal exit state per position
        # Key format: "{symbol}:{side}" (e.g., "ETHUSDT:LONG")
        self._state: Dict[str, PositionExitState] = {}
        
        # Monitoring control
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Logging
        self.logger = logging.getLogger(__name__ + ".executor")
        
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
        
        while self._running:
            try:
                cycle += 1
                await self._monitoring_cycle(cycle)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"[EXIT_BRAIN_EXECUTOR] Error in cycle {cycle}: {e}",
                    exc_info=True
                )
            
            # Wait for next cycle
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
        # Get open positions
        positions = await self._get_open_positions()
        
        if not positions:
            self.logger.debug(f"[EXIT_BRAIN_EXECUTOR] Cycle {cycle}: No open positions")
            return
        
        self.logger.debug(
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
                    self._state[state_key] = PositionExitState(
                        symbol=symbol,
                        side=side,
                        position_size=size
                    )
                    self.logger.info(
                        f"[EXIT_BRAIN_EXECUTOR] Created new state for {state_key}"
                    )
                
                state = self._state[state_key]
                
                # Update state with current data
                state.position_size = size
                mark_price = float(pos_data.get('markPrice', 0))
                state.last_price = mark_price
                state.last_updated = datetime.now(timezone.utc).isoformat()
                
                # Build context for AI
                ctx = self._build_position_context(pos_data)
                
                # Get AI decision
                decision = await self.adapter.decide(ctx)
                
                # Update state based on decision
                await self._update_state_from_decision(state, ctx, decision)
                
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
            # Use position_source (BinanceClient)
            positions = await self.position_source.get_all_positions()
            
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
    
    def _build_position_context(self, pos_data: Dict) -> PositionContext:
        """
        Build PositionContext from Binance position data.
        
        Args:
            pos_data: Raw position data from Binance
            
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
        
        # Detect regime (simplified for now)
        regime = self._detect_regime(symbol, mark_price)
        
        # Assess risk state based on PnL
        risk_state = self._assess_risk_state(symbol, pnl_pct)
        
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
            meta={
                "position_data": pos_data,
                "unrealized_pnl_abs": unrealized_pnl
            }
        )
        
        return ctx
    
    def _detect_regime(self, symbol: str, current_price: float) -> Optional[str]:
        """Detect market regime (placeholder - extend with actual logic)."""
        return "unknown"
    
    def _assess_risk_state(self, symbol: str, pnl_pct: float) -> Optional[str]:
        """Assess risk state based on PnL."""
        if pnl_pct < -3.0:
            return "high_risk"
        elif pnl_pct < -1.0:
            return "drawdown"
        else:
            return "normal"
    
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
            # Update SL level
            if decision.new_sl_price:
                old_sl = state.active_sl
                state.active_sl = decision.new_sl_price
                
                sl_reason = decision.sl_reason or decision.reason or "AI decision"
                
                if old_sl is None:
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"INITIAL SL set to ${state.active_sl:.4f} - {sl_reason}"
                    )
                else:
                    self.logger.info(
                        f"[EXIT_BRAIN_STATE] {state.symbol} {state.side}: "
                        f"SL moved ${old_sl:.4f} → ${state.active_sl:.4f} - {sl_reason}"
                    )
        
        elif decision.decision_type == ExitDecisionType.UPDATE_TP_LIMITS:
            # Update TP levels
            if decision.new_tp_levels:
                # Build list of (price, fraction) tuples
                fractions = decision.tp_fractions or [
                    1.0 / len(decision.new_tp_levels)
                ] * len(decision.new_tp_levels)
                
                state.tp_levels = list(zip(decision.new_tp_levels, fractions))
                # Note: We don't reset triggered_legs - keeps track of already executed TPs
                
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
                "positionSide": state.side
            }
            
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
                "positionSide": state.side
            }
            
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
        """Remove state for positions that are no longer open."""
        closed_keys = set(self._state.keys()) - active_keys
        
        for key in closed_keys:
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
                if state.should_trigger_sl(current_price):
                    await self._execute_sl_trigger(state, current_price)
                    # After SL, position is closed - skip TP checks
                    continue
                
                # Check TP legs
                triggerable_tps = state.get_triggerable_tp_legs(current_price)
                
                if triggerable_tps:
                    # Execute first triggered TP only (one per cycle for safety)
                    leg_index, tp_price, size_pct = triggerable_tps[0]
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
            precision = get_binance_precision(state.symbol)
            close_qty = quantize_to_step(remaining_size, precision['stepSize'])
            
            if close_qty <= 0:
                self.logger.warning(
                    f"[EXIT_BRAIN_EXECUTOR] {state.symbol} {state.side}: "
                    f"SL close_qty={close_qty} invalid after quantization, skipping"
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
                "positionSide": state.side
            }
            
            self.logger.warning(
                f"[EXIT_SL_TRIGGER] 🛑 {state.symbol} {state.side}: "
                f"SL HIT @ ${current_price:.4f} (SL=${state.active_sl:.4f}) - "
                f"Closing {close_qty} with MARKET {order_side}"
            )
            
            # Submit via gateway
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind="sl_market",
                explanation=f"SL triggered at ${current_price:.4f}"
            )
            
            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.info(
                    f"[EXIT_ORDER] ✅ SL MARKET {order_side} {state.symbol} "
                    f"{close_qty} executed successfully"
                )
                
                # Clear state after SL trigger (position closed)
                state.active_sl = None
                state.tp_levels = []
                state.triggered_legs.clear()
                state.position_size = 0
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error executing SL trigger for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
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
            precision = get_binance_precision(state.symbol)
            close_qty = quantize_to_step(close_qty, precision['stepSize'])
            
            if close_qty <= 0:
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
                "reduceOnly": True,
                "positionSide": state.side
            }
            
            self.logger.info(
                f"[EXIT_TP_TRIGGER] 💰 {state.symbol} {state.side} TP{leg_index}: "
                f"HIT @ ${current_price:.4f} (TP=${tp_price:.4f}) - "
                f"Closing {close_qty} ({size_pct*100:.0f}%) with MARKET {order_side}"
            )
            
            # Submit via gateway
            resp = await self.exit_order_gateway.submit_exit_order(
                module_name="exit_executor",
                symbol=state.symbol,
                order_params=order_params,
                order_kind=f"tp_market_leg_{leg_index}",
                explanation=f"TP{leg_index} triggered at ${current_price:.4f}"
            )
            
            if resp and resp.get('status') in ['NEW', 'FILLED']:
                self.logger.info(
                    f"[EXIT_ORDER] ✅ TP{leg_index} MARKET {order_side} {state.symbol} "
                    f"{close_qty} executed successfully"
                )
                
                # Mark this leg as triggered
                state.triggered_legs.add(leg_index)
                
                # Update position size (assume filled)
                state.position_size = max(0, state.position_size - close_qty)
            
        except Exception as e:
            self.logger.error(
                f"[EXIT_BRAIN_EXECUTOR] Error executing TP{leg_index} trigger for "
                f"{state.symbol} {state.side}: {e}",
                exc_info=True
            )
