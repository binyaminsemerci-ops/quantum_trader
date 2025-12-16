"""
Trailing Stop Manager
Automatically updates stop losses as positions become profitable

[EXIT BRAIN V3] When enabled, reads trailing config from Exit Brain plans.
"""
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional
from binance.client import Client

logger = logging.getLogger(__name__)

# [NEW] EXIT BRAIN V3: Feature flag and imports
EXIT_BRAIN_V3_ENABLED = os.getenv("EXIT_BRAIN_V3_ENABLED", "false").lower() == "true"
EXIT_BRAIN_V3_AVAILABLE = False

if EXIT_BRAIN_V3_ENABLED:
    try:
        from backend.domains.exits.exit_brain_v3.router import ExitRouter
        from backend.domains.exits.exit_brain_v3.integration import to_trailing_config
        EXIT_BRAIN_V3_AVAILABLE = True
        logger.info("[EXIT BRAIN] Exit Brain v3 integration active in trailing_stop_manager")
    except ImportError as e:
        logger.warning(f"[EXIT BRAIN] Exit Brain v3 not available: {e}")

# [PHASE 1] Exit Order Gateway for observability
try:
    from backend.services.execution.exit_order_gateway import submit_exit_order
    EXIT_GATEWAY_AVAILABLE = True
except ImportError:
    EXIT_GATEWAY_AVAILABLE = False
    logger.warning("[EXIT_GATEWAY] Not available - will place orders directly")


class TrailingStopManager:
    """
    Manages trailing stops for all positions
    
    Features:
    - Updates peak/trough as price moves
    - Moves SL order on Binance when trailing activates
    - Respects AI-generated trail percentages
    - Only trails in profit direction (never loosens SL)
    """
    
    def __init__(
        self,
        trade_state_path: str = "/app/backend/data/trade_state.json",
        check_interval: int = 10,  # Check every 10 seconds
        min_profit_to_activate: float = 0.005  # Must be 0.5% in profit to activate trail
    ):
        self.trade_state_path = Path(trade_state_path)
        self.check_interval = check_interval
        self.min_profit_to_activate = min_profit_to_activate
        
        # Binance client - with testnet support
        use_testnet = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
        
        if use_testnet:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("Missing Binance TESTNET credentials (BINANCE_API_KEY, BINANCE_API_SECRET)")
            logger.info("[TESTNET] Trailing Stop Manager: Using Binance Testnet API")
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com'
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError("Missing Binance credentials")
            logger.info("[PRODUCTION] Trailing Stop Manager: Using Binance Live API")
            self.client = Client(api_key, api_secret)
        
        # Cache for exchange info
        self._precision_cache: Dict[str, int] = {}
        
        # [EXIT BRAIN V3] Initialize router if available
        self.exit_router = None
        if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE:
            self.exit_router = ExitRouter()
            logger.info("[EXIT BRAIN] Exit Router initialized for trailing config")
        
        logger.info("🔄 Trailing Stop Manager initialized")
    
    def _load_trade_state(self) -> Dict:
        """Load trade state from JSON file"""
        if not self.trade_state_path.exists():
            return {}
        try:
            return json.loads(self.trade_state_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load trade state: {e}")
            return {}
    
    def _save_trade_state(self, state: Dict) -> None:
        """Save trade state to JSON file"""
        try:
            self.trade_state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"Failed to save trade state: {e}")
    
    def _get_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]
        
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                filters = {f['filterType']: f for f in symbol_info['filters']}
                if 'PRICE_FILTER' in filters:
                    tick_size = float(filters['PRICE_FILTER']['tickSize'])
                    if tick_size >= 1:
                        precision = 0
                    else:
                        precision = len(str(tick_size).split('.')[-1])
                    self._precision_cache[symbol] = precision
                    return precision
        except Exception as e:
            logger.warning(f"Could not get precision for {symbol}: {e}")
        
        return 5  # Default precision
    
    async def _update_stop_loss_on_binance(
        self,
        symbol: str,
        side: str,
        new_sl_price: float
    ) -> bool:
        """
        Update SL order on Binance
        
        Process:
        1. Check existing SL orders
        2. Only update if new SL is BETTER (tighter) than existing
        3. Cancel old SL and place new one
        """
        try:
            # Get existing SL orders
            open_orders = await asyncio.to_thread(
                self.client.futures_get_open_orders,
                symbol=symbol
            )
            
            # Find existing SL orders and check if update is beneficial
            existing_sl_orders = []
            for order in open_orders:
                if order['type'] in ['STOP_MARKET', 'STOP_LOSS', 'STOP', 'TRAILING_STOP_MARKET']:
                    existing_sl_orders.append(order)
                    existing_sl_price = float(order.get('stopPrice', 0))
                    
                    # [COORDINATION] Only update if new SL is BETTER (tighter protection)
                    if side == 'LONG':
                        # For LONG: Higher SL = Better (tighter stop)
                        if existing_sl_price > 0 and new_sl_price <= existing_sl_price:
                            logger.debug(f"⏭️  {symbol} LONG: New SL ${new_sl_price:.6f} not better than existing ${existing_sl_price:.6f} - SKIP")
                            return False
                    else:  # SHORT
                        # For SHORT: Lower SL = Better (tighter stop)
                        if existing_sl_price > 0 and new_sl_price >= existing_sl_price:
                            logger.debug(f"⏭️  {symbol} SHORT: New SL ${new_sl_price:.6f} not better than existing ${existing_sl_price:.6f} - SKIP")
                            return False
            
            # If we get here, new SL is better - cancel existing and place new
            for order in existing_sl_orders:
                await asyncio.to_thread(
                    self.client.futures_cancel_order,
                    symbol=symbol,
                    orderId=order['orderId']
                )
                logger.info(f"🗑️  Cancelled old SL order {order['orderId']} for {symbol} (upgrading to tighter stop)")
            
            # Place new SL order
            sl_side = 'SELL' if side == 'LONG' else 'BUY'
            position_side = 'LONG' if side == 'LONG' else 'SHORT'
            precision = self._get_precision(symbol)
            sl_price_rounded = round(new_sl_price, precision)
            
            # [PHASE 1] Route through exit gateway for observability
            sl_order_params = {
                'symbol': symbol,
                'side': sl_side,
                'type': 'STOP_MARKET',
                'stopPrice': sl_price_rounded,
                'closePosition': True,
                'workingType': 'MARK_PRICE',
                'positionSide': position_side
            }
            
            if EXIT_GATEWAY_AVAILABLE:
                new_order = await submit_exit_order(
                    module_name="trailing_stop_manager",
                    symbol=symbol,
                    order_params=sl_order_params,
                    order_kind="trailing",
                    client=self.client,
                    explanation=f"Trailing SL update: ${sl_price_rounded} (profit lock)"
                )
            else:
                new_order = await asyncio.to_thread(
                    self.client.futures_create_order,
                    **sl_order_params
                )
            
            logger.info(
                f"[OK] Trailing SL updated for {symbol}: ${sl_price_rounded} "
                f"(order {new_order['orderId']})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update SL for {symbol}: {e}")
            return False
    
    async def _process_position(self, symbol: str, state: Dict, trade_states: Dict) -> None:
        """
        Process a single position for trailing stop
        
        [EXIT BRAIN V3] If enabled, reads trailing config from Exit Brain plan.
        Otherwise uses legacy ai_trail_pct from trade_state.
        """
        
        # [EXIT BRAIN V3] Try to get trailing config from Exit Brain plan
        trail_pct = None
        min_profit_threshold = self.min_profit_to_activate
        
        if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE and self.exit_router:
            try:
                # Check if Exit Brain plan exists for this symbol (cached only)
                plan = self.exit_router.get_active_plan(symbol)
                logger.debug(f"[EXIT BRAIN DEBUG] {symbol}: get_active_plan returned: {plan is not None}")
                if plan:
                    # Extract trailing config from plan
                    trail_config = to_trailing_config(plan, None)  # ctx not needed for config extraction
                    logger.debug(f"[EXIT BRAIN DEBUG] {symbol}: to_trailing_config returned: {trail_config}")
                    if trail_config and trail_config.get("enabled"):
                        trail_pct = trail_config.get("callback_pct", 0.015)
                        min_profit_threshold = trail_config.get("activation_pct", self.min_profit_to_activate)
                        logger.debug(
                            f"[EXIT BRAIN] {symbol}: Using plan-derived trailing "
                            f"(callback={trail_pct:.2%}, threshold={min_profit_threshold:.2%})"
                        )
            except Exception as e:
                logger.debug(f"[EXIT BRAIN] {symbol}: Could not get trailing config: {e}")
        
        # [LEGACY PATH] Fallback to trade_state ai_trail_pct
        if trail_pct is None:
            if 'ai_trail_pct' not in state:
                logger.debug(f"⏭️  {symbol}: No trail percentage set - SKIP")
                return
            trail_pct = state['ai_trail_pct']
            
            # Exit Brain v3 uses TAKE_PROFIT legs (handled by Dynamic Executor), not TRAIL legs
            # Legacy trailing only applies to positions without Exit Brain plans
            if EXIT_BRAIN_V3_ENABLED:
                logger.debug(
                    f"[EXIT_GUARD] {symbol}: Using legacy ai_trail_pct={trail_pct:.2%} "
                    f"(Exit Brain uses TP legs, not TRAIL legs)"
                )
        
        side = state['side']
        entry = state['avg_entry']
        
        try:
            # Get current position from Binance
            positions = await asyncio.to_thread(
                self.client.futures_position_information,
                symbol=symbol
            )
            
            if not positions:
                return
            
            pos = positions[0]
            pos_amt = float(pos['positionAmt'])
            
            # Skip if position closed
            if abs(pos_amt) < 0.0001:
                return
            
            current_price = float(pos['markPrice'])
            unrealized_pnl = float(pos['unRealizedProfit'])
            position_value = abs(pos_amt) * entry
            pnl_pct = (unrealized_pnl / position_value) if position_value > 0 else 0
            
            # NEW STRATEGY: Dynamic SL tightening based on profit level
            # Stage 1 (+1.5%): Move to breakeven
            # Stage 2 (+3.0%): Move to +1.5%
            # Stage 3 (+5.0%): Activate trailing 1% under peak
            
            breakeven_move_triggered = state.get('breakeven_move_triggered', False)
            stage2_move_triggered = state.get('stage2_move_triggered', False)
            trailing_activated = state.get('trailing_activated', False)
            
            # Stage 1: Move to breakeven at +1.5% profit
            if pnl_pct >= 0.015 and not breakeven_move_triggered:
                logger.info(f"🎯 {symbol}: +{pnl_pct*100:.2f}% profit → Moving SL to BREAKEVEN")
                success = await self._update_stop_loss_on_binance(symbol, side, entry)
                if success:
                    state['breakeven_move_triggered'] = True
                    state['trail_sl'] = entry
                    trade_states[symbol] = state
                    self._save_trade_state(trade_states)
                    logger.info(f"✅ {symbol}: SL moved to breakeven ${entry:.6f}")
                return
            
            # Stage 2: Move to +1.5% at +3.0% profit
            if pnl_pct >= 0.030 and breakeven_move_triggered and not stage2_move_triggered:
                target_sl = entry * (1.015 if side == "LONG" else 0.985)
                logger.info(f"🎯 {symbol}: +{pnl_pct*100:.2f}% profit → Moving SL to +1.5%")
                success = await self._update_stop_loss_on_binance(symbol, side, target_sl)
                if success:
                    state['stage2_move_triggered'] = True
                    state['trail_sl'] = target_sl
                    trade_states[symbol] = state
                    self._save_trade_state(trade_states)
                    logger.info(f"✅ {symbol}: SL moved to +1.5% ${target_sl:.6f}")
                return
            
            # Stage 3: Activate trailing at +5.0% profit
            if pnl_pct >= 0.050 and not trailing_activated:
                logger.info(f"🚀 {symbol}: +{pnl_pct*100:.2f}% profit → ACTIVATING TRAILING STOP (1% trail)")
                state['trailing_activated'] = True
                # Use tighter 1% trail for trailing stage
                state['ai_trail_pct'] = 0.010  # 1% trailing distance
                trail_pct = 0.010
                trade_states[symbol] = state
                self._save_trade_state(trade_states)
            
            # Only proceed with trailing if activated
            if pnl_pct < 0.050 or not trailing_activated:
                logger.debug(f"⏭️  {symbol}: PnL {pnl_pct*100:.2f}% - waiting for Stage 3 (+5%) to activate trailing")
                return
            
            # [COORDINATION] Log that we're trailing (Stage 3 active)
            logger.info(f"🔄 {symbol}: Trailing ACTIVE (PnL +{pnl_pct*100:.2f}%, trail {trail_pct*100:.1f}%)")
            
            # Update peak/trough
            updated = False
            
            if side == "LONG":
                old_peak = state.get('peak', entry)
                if current_price > old_peak:
                    state['peak'] = current_price
                    updated = True
                    logger.info(f"[CHART_UP] {symbol} new peak: ${current_price:.6f} (was ${old_peak:.6f})")
                
                # Calculate trailing stop from peak
                peak = state['peak']
                new_sl = peak * (1 - trail_pct)
                
                # Only update if new SL is higher than entry (tightening stop)
                if new_sl > entry:
                    # Check if we should update Binance SL
                    old_sl = state.get('trail_sl', entry)
                    if new_sl > old_sl * 1.001:  # Only update if >0.1% improvement
                        logger.info(f"🔄 {symbol} LONG: Attempting to tighten SL from ${old_sl:.6f} to ${new_sl:.6f} (peak=${peak:.6f})")
                        success = await self._update_stop_loss_on_binance(symbol, side, new_sl)
                        if success:
                            state['trail_sl'] = new_sl
                            updated = True
                            logger.info(
                                f"✅ [TARGET] {symbol} LONG trailing SL UPDATED: Peak=${peak:.6f}, "
                                f"SL=${new_sl:.6f} ({trail_pct*100:.1f}% trail) - OLD SL=${old_sl:.6f}"
                            )
                        else:
                            logger.warning(f"⚠️  {symbol} LONG: Failed to update SL (existing SL better or update rejected)")
                    else:
                        logger.debug(f"⏭️  {symbol} LONG: New SL ${new_sl:.6f} not enough improvement over ${old_sl:.6f}")
            
            else:  # SHORT
                old_trough = state.get('trough', entry)
                if current_price < old_trough:
                    state['trough'] = current_price
                    updated = True
                    logger.info(f"📉 {symbol} new trough: ${current_price:.6f} (was ${old_trough:.6f})")
                
                # Calculate trailing stop from trough
                trough = state['trough']
                new_sl = trough * (1 + trail_pct)
                
                # Only update if new SL is lower than entry (tightening stop)
                if new_sl < entry:
                    # Check if we should update Binance SL
                    old_sl = state.get('trail_sl', entry)
                    if new_sl < old_sl * 0.999:  # Only update if >0.1% improvement
                        logger.info(f"🔄 {symbol} SHORT: Attempting to tighten SL from ${old_sl:.6f} to ${new_sl:.6f} (trough=${trough:.6f})")
                        success = await self._update_stop_loss_on_binance(symbol, side, new_sl)
                        if success:
                            state['trail_sl'] = new_sl
                            updated = True
                            logger.info(
                                f"✅ [TARGET] {symbol} SHORT trailing SL UPDATED: Trough=${trough:.6f}, "
                                f"SL=${new_sl:.6f} ({trail_pct*100:.1f}% trail) - OLD SL=${old_sl:.6f}"
                            )
                        else:
                            logger.warning(f"⚠️  {symbol} SHORT: Failed to update SL (existing SL better or update rejected)")
                    else:
                        logger.debug(f"⏭️  {symbol} SHORT: New SL ${new_sl:.6f} not enough improvement over ${old_sl:.6f}")
            
            # Save updated state if changed
            if updated:
                trade_states[symbol] = state
                self._save_trade_state(trade_states)
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        logger.info(f"🔄 Starting trailing stop monitor (interval: {self.check_interval}s)")
        
        while True:
            try:
                # [NEW] Get open positions directly from Binance instead of relying on trade_states.json
                all_positions = await asyncio.to_thread(
                    self.client.futures_position_information
                )
                
                # Filter to only open positions
                open_positions = [
                    p for p in all_positions 
                    if float(p.get('positionAmt', 0)) != 0
                ]
                
                if not open_positions:
                    await asyncio.sleep(self.check_interval)
                    continue
                
                logger.debug(f"[TRAIL] Found {len(open_positions)} open positions to monitor")
                
                # [FALLBACK] Load trade_states for legacy trail config
                trade_states = self._load_trade_state()
                
                # Process each position
                tasks = []
                for pos in open_positions:
                    symbol = pos['symbol']
                    
                    # Build minimal state dict from position
                    pos_amt = float(pos['positionAmt'])
                    state = {
                        'side': 'LONG' if pos_amt > 0 else 'SHORT',
                        'avg_entry': float(pos['entryPrice']),
                        'positionAmt': abs(pos_amt)
                    }
                    
                    # Copy legacy trail config from trade_states if exists
                    if symbol in trade_states and isinstance(trade_states[symbol], dict):
                        legacy_state = trade_states[symbol]
                        state['ai_trail_pct'] = legacy_state.get('ai_trail_pct')
                        state['breakeven_move_triggered'] = legacy_state.get('breakeven_move_triggered', False)
                        state['stage2_move_triggered'] = legacy_state.get('stage2_move_triggered', False)
                        state['trailing_activated'] = legacy_state.get('trailing_activated', False)
                        state['peak_price'] = legacy_state.get('peak_price')
                        state['trough_price'] = legacy_state.get('trough_price')
                        state['trail_sl'] = legacy_state.get('trail_sl')
                    
                    task = self._process_position(symbol, state, trade_states)
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Error in trailing stop monitor: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def run(self) -> None:
        """Run the trailing stop manager"""
        asyncio.run(self.monitor_loop())


def main():
    """Entry point for standalone execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = TrailingStopManager()
    manager.run()


if __name__ == "__main__":
    main()
