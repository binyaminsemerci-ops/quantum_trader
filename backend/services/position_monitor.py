"""
Position Monitor - Automatic TP/SL Management
Monitors all open positions and ensures they have TP/SL protection
"""
import os
import asyncio
import logging
from typing import Dict, List
from binance.client import Client
from datetime import datetime

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Continuously monitors all open positions and ensures TP/SL protection.
    
    Features:
    - Detects positions without TP/SL orders
    - Automatically sets hybrid TP/SL strategy (partial exit + trailing)
    - Uses AI-generated TP/SL percentages when available
    - Runs independently from trade execution
    - 🚨 FIX #3: Re-evaluates AI sentiment and warns if changed
    """
    
    def __init__(
        self,
        check_interval: int = 10,  # 🎯 Check every 10 seconds for DYNAMIC TP/SL adjustment!
        ai_engine=None,  # 🚨 NEW: Accept AI engine for re-evaluation
    ):
        self.check_interval = check_interval
        self.ai_engine = ai_engine  # 🚨 NEW: Store AI engine reference
        
        # Binance client
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("Missing Binance credentials")
        self.client = Client(api_key, api_secret)
        
        # Configuration from environment
        self.tp_pct = float(os.getenv("QT_TP_PCT", "0.03"))
        self.sl_pct = float(os.getenv("QT_SL_PCT", "0.02"))
        self.trail_pct = float(os.getenv("QT_TRAIL_PCT", "0.015"))
        self.partial_tp = float(os.getenv("QT_PARTIAL_TP", "0.5"))
        
        logger.info(f"🔍 Position Monitor initialized: TP={self.tp_pct*100:.1f}% SL={self.sl_pct*100:.1f}% Trail={self.trail_pct*100:.1f}%")
    
    def _cancel_all_orders_for_symbol(self, symbol: str) -> int:
        """
        Cancel ALL open orders for a symbol (comprehensive cleanup).
        Returns number of orders cancelled.
        """
        cancelled_count = 0
        try:
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            if not open_orders:
                return 0
            
            logger.info(f"🗑️  Cancelling {len(open_orders)} open orders for {symbol}")
            
            for order in open_orders:
                try:
                    order_type = order.get('type', 'UNKNOWN')
                    order_id = order.get('orderId')
                    self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    logger.info(f"   ✓ Cancelled {order_type} order {order_id}")
                    cancelled_count += 1
                except Exception as e:
                    logger.warning(f"   ✗ Failed to cancel order {order_id}: {e}")
            
            logger.info(f"✅ Cancelled {cancelled_count}/{len(open_orders)} orders for {symbol}")
        except Exception as e:
            logger.error(f"❌ Error cancelling orders for {symbol}: {e}")
        
        return cancelled_count
    
    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    tick_str = f"{tick_size:.10f}".rstrip('0')
                    if '.' in tick_str:
                        return len(tick_str.split('.')[-1])
            return 5
        except:
            return 5
    
    def _get_quantity_precision(self, symbol: str) -> float:
        """Get quantity step size for symbol"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                lot_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_filter:
                    return float(lot_filter['stepSize'])
            return 0.01
        except:
            return 0.01
    
    def _round_quantity(self, qty: float, step_size: float) -> float:
        """Round quantity to step size"""
        if step_size >= 1:
            return int(qty)
        step_str = f"{step_size:.10f}".rstrip('0')
        if '.' in step_str:
            precision = len(step_str.split('.')[-1])
        else:
            precision = 0
        return round(qty, precision)
    
    async def _adjust_tpsl_dynamically(self, position: Dict, ai_signals: List[Dict] = None) -> bool:
        """🎯 DYNAMISK TP/SL JUSTERING basert på profit, sentiment og markedsbevegelse"""
        symbol = position['symbol']
        amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        mark_price = float(position['markPrice'])
        unrealized_pnl = float(position['unRealizedProfit'])
        
        # 🐛 DEBUG: Log position keys for first call
        if not hasattr(self, '_logged_keys'):
            logger.debug(f"Position keys for {symbol}: {list(position.keys())}")
            self._logged_keys = True
        
        leverage = float(position.get('leverage', 30))  # 🎯 Using 30x leverage
        
        # Beregn profit i prosent av margin (ikke position value!)
        margin = abs(amt * entry_price) / leverage
        pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0
        
        # Finn AI signal for dette symbolet
        ai_signal = None
        if ai_signals:
            ai_signal = next((s for s in ai_signals if s['symbol'] == symbol), None)
        
        # Sjekk om position er LONG eller SHORT
        is_long = amt > 0
        
        # 🎯 DYNAMISK LOGIKK FOR 30x LEVERAGE - Progressive partial profit taking:
        # Med 30x leverage beveger PnL seg MYE raskere!
        # Justerte thresholds for å matche høy leverage:
        # 1. I TAP: Hold original SL/TP
        # 2. Litt profit (5-10%): SL→breakeven, Partial TP #1 (25% @ 8%)
        # 3. God profit (10-20%): Lock 20%, Partial TP #2 (25% @ 12%)
        # 4. Veldig god (20-35%): Lock 40%, Partial TP #3 (25% @ 18%)
        # 5. Stor profit (35-60%): Lock 60%, Final TP (25% @ 30%+)
        # 6. Ekstrem (>60%): Lock 70%, Moon TP (remaining @ 50%)
        
        logger.debug(f"📊 {symbol}: PnL {pnl_pct:+.2f}% margin, Mark ${mark_price:.6f}, Entry ${entry_price:.6f}")
        
        # Beregn ny SL OG multiple TP levels basert på profit
        new_sl_price = None
        partial_tp_levels = []  # List of (price, qty_pct, reason)
        adjustment_reason = ""
        
        if pnl_pct < -3.0:
            # I TAP mer enn 3% - hold original SL, men advare
            logger.warning(f"⚠️ {symbol}: Losing {pnl_pct:.2f}% - holding SL/TP")
            if ai_signal and ai_signal.get('action') == 'HOLD' and ai_signal.get('confidence', 0) < 0.4:
                logger.warning(f"🚨 {symbol}: AI sentiment weak ({ai_signal.get('confidence', 0):.0%}) - consider closing!")
            return False
            
        elif -3.0 <= pnl_pct < 5.0:
            # Liten profit - la posisjonen vandre
            return False
            
        elif 5.0 <= pnl_pct < 10.0:
            # Moderat profit - SL til breakeven, første partial TP
            new_sl_price = entry_price
            # TP #1: 25% av position @ 8% margin profit
            tp1_pct = 0.08
            if is_long:
                tp1_price = entry_price * (1 + tp1_pct / leverage)
            else:
                tp1_price = entry_price * (1 - tp1_pct / leverage)
            partial_tp_levels.append((tp1_price, 0.25, "TP1@8%"))
            adjustment_reason = "breakeven + partial TP #1 (25% @ 8%)"
            
        elif 10.0 <= pnl_pct < 20.0:
            # God profit - lock 20%, andre partial TP
            profit_distance = abs(mark_price - entry_price)
            new_sl_price = entry_price + (lock_in_distance := profit_distance * 0.2) * (1 if is_long else -1)
            # TP #1: 25% @ 10%
            tp1_pct = 0.10
            tp1_price = entry_price * (1 + tp1_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp1_price, 0.25, "TP1@10%"))
            # TP #2: 25% @ 15%
            tp2_pct = 0.15
            tp2_price = entry_price * (1 + tp2_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp2_price, 0.25, "TP2@15%"))
            adjustment_reason = "lock 20%, partial TPs (25%+25%)"
            
        elif 20.0 <= pnl_pct < 35.0:
            # Veldig god - lock 40%, tredje partial TP
            profit_distance = abs(mark_price - entry_price)
            new_sl_price = entry_price + (lock_in_distance := profit_distance * 0.4) * (1 if is_long else -1)
            # TP #1: 20% @ 12%
            tp1_pct = 0.12
            tp1_price = entry_price * (1 + tp1_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp1_price, 0.20, "TP1@12%"))
            # TP #2: 20% @ 18%
            tp2_pct = 0.18
            tp2_price = entry_price * (1 + tp2_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp2_price, 0.20, "TP2@18%"))
            # TP #3: 30% @ 25%
            tp3_pct = 0.25
            tp3_price = entry_price * (1 + tp3_pct / leverage * (1 if is_long else -1))
            partial_tp_levels.append((tp3_price, 0.30, "TP3@25%"))
            adjustment_reason = "lock 40%, 3x partial TPs"
            
        elif 35.0 <= pnl_pct < 60.0:
            # Enorm - lock 60%, aggressive partial TPs
            profit_distance = abs(mark_price - entry_price)
            new_sl_price = entry_price + (lock_in_distance := profit_distance * 0.6) * (1 if is_long else -1)
            # TP #1: 20% @ 20%
            partial_tp_levels.append((entry_price * (1 + 0.20 / leverage * (1 if is_long else -1)), 0.20, "TP1@20%"))
            # TP #2: 20% @ 28%
            partial_tp_levels.append((entry_price * (1 + 0.28 / leverage * (1 if is_long else -1)), 0.20, "TP2@28%"))
            # TP #3: 30% @ 35%
            partial_tp_levels.append((entry_price * (1 + 0.35 / leverage * (1 if is_long else -1)), 0.30, "TP3@35%"))
            adjustment_reason = "lock 60%, aggressive partials"
            
        else:  # pnl_pct >= 60.0
            # EKSTREM - lock 70%, moon targets
            profit_distance = abs(mark_price - entry_price)
            new_sl_price = entry_price + (lock_in_distance := profit_distance * 0.7) * (1 if is_long else -1)
            # Let it ride to the moon with remaining position
            partial_tp_levels.append((entry_price * (1 + 0.30 / leverage * (1 if is_long else -1)), 0.15, "TP1@30%"))
            partial_tp_levels.append((entry_price * (1 + 0.40 / leverage * (1 if is_long else -1)), 0.15, "TP2@40%"))
            partial_tp_levels.append((entry_price * (1 + 0.50 / leverage * (1 if is_long else -1)), 0.20, "TP3@50%"))
            adjustment_reason = f"lock 70%, MOON targets! ({pnl_pct:.1f}%)"
        
        if not new_sl_price and not partial_tp_levels:
            return False
        
        # Hent price precision
        price_precision = self._get_price_precision(symbol)
        if new_sl_price:
            new_sl_price = round(new_sl_price, price_precision)
        
        # Round all TP prices
        for i in range(len(partial_tp_levels)):
            price, qty_pct, reason = partial_tp_levels[i]
            partial_tp_levels[i] = (round(price, price_precision), qty_pct, reason)
        
        # Sjekk eksisterende orders
        current_orders = self.client.futures_get_open_orders(symbol=symbol)
        existing_sl = None
        existing_tps = []
        
        for order in current_orders:
            if order['type'] == 'STOP_MARKET' and order.get('closePosition', False):
                existing_sl = float(order['stopPrice'])
            elif order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                existing_tps.append(float(order.get('stopPrice', 0)))
        
        # Bare oppdater SL hvis ny SL er bedre
        should_update_sl = False
        if new_sl_price:
            if not existing_sl:
                should_update_sl = True
            elif is_long and new_sl_price > existing_sl:
                should_update_sl = True
            elif not is_long and new_sl_price < existing_sl:
                should_update_sl = True
        
        # Oppdater TPs hvis vi har nye levels
        should_update_tps = len(partial_tp_levels) > 0
        
        if not should_update_sl and not should_update_tps:
            logger.debug(f"⏸️ {symbol}: SL/TP already optimal")
            return False
        
        # Oppdater SL og/eller TP!
        try:
            logger.info(f"🎯 {symbol}: ADJUSTING - {adjustment_reason}")
            
            # Oppdater SL
            if should_update_sl:
                if existing_sl:
                    logger.info(f"   SL: ${existing_sl:.6f} → ${new_sl_price:.6f}")
                else:
                    logger.info(f"   Setting SL: ${new_sl_price:.6f}")
                logger.info(f"   PnL: {pnl_pct:+.2f}% margin (${unrealized_pnl:+.2f})")
                
                # Slett gammel SL
                if existing_sl:
                    for order in current_orders:
                        if order['type'] == 'STOP_MARKET' and order.get('closePosition', False):
                            self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                
                # Sett ny SL
                side = 'SELL' if is_long else 'BUY'
                self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='STOP_MARKET',
                    stopPrice=new_sl_price,
                    closePosition=True,
                    workingType='MARK_PRICE'
                )
                logger.info(f"   ✅ SL @ ${new_sl_price:.6f}")
            
            # Oppdater multiple partial TPs
            if should_update_tps:
                # Slett alle gamle TP orders først med comprehensive cleanup
                for order in current_orders:
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT', 'LIMIT']:
                        try:
                            self.client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                            logger.info(f"   🗑️  Cancelled {order['type']} order {order['orderId']}")
                        except Exception as cancel_e:
                            logger.warning(f"   Could not cancel {order['type']}: {cancel_e}")
                
                # Plasser nye partial TP orders
                side = 'SELL' if is_long else 'BUY'
                step_size = self._get_step_size(symbol)
                
                logger.info(f"   💰 Setting {len(partial_tp_levels)} partial TPs:")
                for tp_price, qty_pct, reason in partial_tp_levels:
                    qty = abs(amt) * qty_pct
                    qty = self._round_quantity(qty, step_size)
                    
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='TAKE_PROFIT_MARKET',
                        quantity=qty,
                        stopPrice=tp_price,
                        workingType='MARK_PRICE'
                    )
                    logger.info(f"      • {reason}: {qty_pct*100:.0f}% @ ${tp_price:.6f} (qty: {qty})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Failed to adjust SL - {e}")
            return False
    
    async def _set_tpsl_for_position(self, position: Dict) -> bool:
        """Set TP/SL orders for a single position"""
        symbol = position['symbol']
        amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        
        if amt == 0:
            return False
        
        logger.info(f"🛡️ Setting TP/SL for {symbol}: amt={amt}, entry=${entry_price}")
        
        # Get precision
        price_precision = self._get_price_precision(symbol)
        step_size = self._get_quantity_precision(symbol)
        
        # Get leverage for this position
        leverage = float(position.get('leverage', 30))
        
        # 🎯 LEVERAGE-ADJUSTED TP/SL FOR 30x
        # Goal: 2% SL = 2% margin loss (not 60% loss!)
        # With 30x leverage: 2% margin loss = 0.067% price move (2% / 30)
        price_tp_pct = self.tp_pct / leverage  # 3% / 30 = 0.1% price
        price_sl_pct = self.sl_pct / leverage  # 2% / 30 = 0.067% price
        
        logger.info(f"   🎯 Leverage {leverage}x: TP {self.tp_pct*100:.1f}% margin = {price_tp_pct*100:.2f}% price, SL {self.sl_pct*100:.1f}% margin = {price_sl_pct*100:.2f}% price")
        
        # Calculate TP/SL prices with leverage adjustment
        if amt > 0:  # LONG position
            tp_price = round(entry_price * (1 + price_tp_pct), price_precision)
            sl_price = round(entry_price * (1 - price_sl_pct), price_precision)
            side = 'SELL'
        else:  # SHORT position
            tp_price = round(entry_price * (1 - price_tp_pct), price_precision)
            sl_price = round(entry_price * (1 + price_sl_pct), price_precision)
            side = 'BUY'
        
        # Calculate quantities
        partial_qty = self._round_quantity(abs(amt) * self.partial_tp, step_size)
        remaining_qty = self._round_quantity(abs(amt) - partial_qty, step_size)
        
        logger.info(f"   TP: ${tp_price:.8f}, SL: ${sl_price:.8f}")
        logger.info(f"   Partial: {partial_qty}, Trailing: {remaining_qty}")
        
        # Cancel existing orders first - use comprehensive cleanup
        try:
            cancelled = self._cancel_all_orders_for_symbol(symbol)
            if cancelled > 0:
                logger.info(f"   Cleaned up {cancelled} existing orders before setting new TP/SL")
        except Exception as e:
            logger.warning(f"   Could not cancel orders: {e}")
        
        try:
            # 1. Partial TP order
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=tp_price,
                quantity=partial_qty,
                workingType='MARK_PRICE',
                reduceOnly=True
            )
            logger.info(f"   ✅ TP: {partial_qty} @ ${tp_price:.8f}")
            
            # 2. Trailing stop for remaining
            trail_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='TRAILING_STOP_MARKET',
                quantity=remaining_qty,
                callbackRate=self.trail_pct * 100,
                workingType='MARK_PRICE',
                reduceOnly=True
            )
            logger.info(f"   ✅ Trailing: {remaining_qty} @ {self.trail_pct*100:.1f}%")
            
            # 3. Stop loss (DUAL PROTECTION) - Both STOP_MARKET and STOP_LOSS_LIMIT for redundancy
            # First: STOP_MARKET with closePosition=True (primary protection)
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                stopPrice=sl_price,
                closePosition=True,
                workingType='MARK_PRICE'
            )
            logger.info(f"   ✅ SL (STOP_MARKET): Full position @ ${sl_price:.8f}")
            
            # Second: STOP_LOSS_LIMIT as backup (triggers if STOP_MARKET fails)
            # Set limit price slightly worse to ensure execution
            limit_sl_price = round(sl_price * (0.998 if side == 'SELL' else 1.002), price_precision)
            try:
                sl_limit_order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='STOP',  # STOP order type (limit order triggered by stop price)
                    stopPrice=sl_price,
                    price=limit_sl_price,
                    quantity=abs(amt),
                    timeInForce='GTC',
                    workingType='MARK_PRICE',
                    reduceOnly=True
                )
                logger.info(f"   ✅ SL BACKUP (STOP_LIMIT): Full position @ stop ${sl_price:.8f}, limit ${limit_sl_price:.8f}")
            except Exception as backup_e:
                logger.warning(f"   ⚠️  Could not set backup SL: {backup_e}")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to set TP/SL: {e}")
            return False
    
    async def check_all_positions(self) -> Dict:
        """Check all positions and set TP/SL if missing
        
        🚨 FIX: Re-evaluates AI sentiment, warns if changed, and enforces emergency SL if breached.
        """
        try:
            # Get all positions
            positions = self.client.futures_position_information()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            
            if not open_positions:
                return {"checked": 0, "newly_protected": 0, "adjusted": 0}
            
            # 🚨 EMERGENCY CHECK: Close positions where SL should have triggered but didn't
            emergency_closed = 0
            for position in open_positions:
                symbol = position['symbol']
                amt = float(position['positionAmt'])
                entry_price = float(position['entryPrice'])
                mark_price = float(position['markPrice'])
                unrealized_pnl = float(position['unRealizedProfit'])
                leverage = float(position.get('leverage', 30))
                
                # Calculate PnL percentage of margin
                margin = abs(amt * entry_price) / leverage
                pnl_pct = (unrealized_pnl / margin) * 100 if margin > 0 else 0
                
                # 🚨 EMERGENCY: If losing more than 3% margin, force close immediately
                if pnl_pct < -3.0:
                    logger.error(f"🚨 EMERGENCY: {symbol} losing {pnl_pct:.2f}% - SL FAILED! Force closing...")
                    try:
                        side = 'SELL' if amt > 0 else 'BUY'
                        # Cancel all orders first
                        self._cancel_all_orders_for_symbol(symbol)
                        # Force close with market order
                        self.client.futures_create_order(
                            symbol=symbol,
                            side=side,
                            type='MARKET',
                            quantity=abs(amt),
                            reduceOnly=True
                        )
                        logger.error(f"✅ EMERGENCY CLOSE: {symbol} closed at ${mark_price:.6f} (loss: {pnl_pct:.2f}%)")
                        emergency_closed += 1
                        continue  # Skip to next position
                    except Exception as e:
                        logger.error(f"❌ EMERGENCY CLOSE FAILED for {symbol}: {e}")
            
            if emergency_closed > 0:
                logger.error(f"🚨 Emergency closed {emergency_closed} positions with failed SL")
                return {"status": "ok", "positions": 0, "protected": 0, "unprotected": 0}
            
            # 🚨 FIX #3: Re-evaluate AI sentiment for open positions
            if hasattr(self, 'ai_engine') and self.ai_engine:
                symbols = [p['symbol'] for p in open_positions]
                try:
                    current_positions_map = {p['symbol']: float(p['positionAmt']) for p in open_positions}
                    ai_signals = await self.ai_engine.get_trading_signals(symbols, current_positions_map)  # 🎯 RENAMED from 'signals'
                    
                    for signal in ai_signals:
                        symbol = signal['symbol']
                        ai_action = signal['action']
                        ai_confidence = signal['confidence']
                        
                        # Find matching position
                        pos = next((p for p in open_positions if p['symbol'] == symbol), None)
                        if pos:
                            amt = float(pos['positionAmt'])
                            current_direction = 'BUY' if amt > 0 else 'SELL'
                            
                            # Check if AI disagrees or is weak
                            if ai_action == 'HOLD' and ai_confidence < 0.5:
                                logger.warning(
                                    f"⚠️ {symbol}: AI sentiment weak (HOLD {ai_confidence:.0%}) - consider closing"
                                )
                            elif ai_action != current_direction and ai_action != 'HOLD':
                                logger.warning(
                                    f"🚨 {symbol}: AI changed from {current_direction} to {ai_action} "
                                    f"({ai_confidence:.0%}) - consider closing!"
                                )
                except Exception as e:
                    logger.debug(f"Could not re-evaluate AI sentiment: {e}")
            
            # 🎯 STEG 1: DYNAMISK JUSTERING for alle posisjoner (hver 10 sek med check_interval=10)
            adjusted_count = 0
            adjusted_symbols = set()  # 🎯 Track which symbols we adjusted
            for position in open_positions:
                try:
                    if await self._adjust_tpsl_dynamically(position, ai_signals):
                        adjusted_count += 1
                        adjusted_symbols.add(position['symbol'])  # 🎯 Remember this symbol
                except Exception as e:
                    logger.debug(f"Could not adjust {position['symbol']}: {type(e).__name__}: {e}")
            
            if adjusted_count > 0:
                logger.info(f"🔄 Dynamically adjusted TP/SL for {adjusted_count} positions")
            
            # Get all open orders
            all_orders = self.client.futures_get_open_orders()
            
            # Group orders by symbol
            orders_by_symbol = {}
            for order in all_orders:
                symbol = order['symbol']
                if symbol not in orders_by_symbol:
                    orders_by_symbol[symbol] = []
                orders_by_symbol[symbol].append(order)
            
            # 🎯 STEG 2: BASIC BESKYTTELSE - Check each position for TP/SL existence
            protected = 0
            unprotected = 0
            newly_protected = 0
            
            for position in open_positions:
                symbol = position['symbol']
                
                # 🎯 SKIP hvis vi nettopp justerte denne symbolet dynamisk!
                if symbol in adjusted_symbols:
                    protected += 1
                    logger.debug(f"✅ {symbol} dynamically adjusted (skipping re-check)")
                    continue
                
                orders = orders_by_symbol.get(symbol, [])
                
                # Check if position has TP/SL
                has_tp = any(o['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] for o in orders)
                # Check for STOP_MARKET with closePosition=True (the real safety net)
                has_sl = any(o['type'] == 'STOP_MARKET' and o.get('closePosition', False) for o in orders)
                
                if has_tp and has_sl:
                    protected += 1
                    logger.debug(f"✅ {symbol} already protected")
                else:
                    unprotected += 1
                    logger.warning(f"⚠️ {symbol} UNPROTECTED - setting TP/SL now...")
                    
                    # Set TP/SL asynchronously
                    success = await self._set_tpsl_for_position(position)
                    if success:
                        newly_protected += 1
            
            logger.info(
                f"📊 Position check: {len(open_positions)} total, "
                f"{protected} protected, {unprotected} unprotected, "
                f"{newly_protected} newly protected"
            )
            
            # 🔥 ORPHANED ORDER CLEANUP: Cancel orders for symbols with no position
            try:
                all_open_orders = self.client.futures_get_open_orders()
                if all_open_orders:
                    open_symbols = {p['symbol'] for p in open_positions}
                    orphaned_symbols = set()
                    
                    for order in all_open_orders:
                        symbol = order['symbol']
                        if symbol not in open_symbols:
                            orphaned_symbols.add(symbol)
                    
                    if orphaned_symbols:
                        logger.warning(f"🗑️  Found orphaned orders for {len(orphaned_symbols)} symbols with no position: {orphaned_symbols}")
                        total_cancelled = 0
                        for symbol in orphaned_symbols:
                            cancelled = self._cancel_all_orders_for_symbol(symbol)
                            total_cancelled += cancelled
                        logger.info(f"✅ Cleaned up {total_cancelled} orphaned orders")
            except Exception as cleanup_exc:
                logger.warning(f"Could not clean orphaned orders: {cleanup_exc}")
            
            return {
                "status": "ok",
                "positions": len(open_positions),
                "protected": protected,
                "unprotected": unprotected,
                "newly_protected": newly_protected
            }
            
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
            return {"status": "error", "error": str(e)}
    
    async def monitor_loop(self) -> None:
        """Main monitoring loop"""
        logger.info(f"🔍 Starting Position Monitor (interval: {self.check_interval}s)")
        
        while True:
            try:
                result = await self.check_all_positions()
                if result.get('newly_protected', 0) > 0:
                    logger.info(f"✅ Protected {result['newly_protected']} positions")
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    def run(self) -> None:
        """Run the position monitor"""
        asyncio.run(self.monitor_loop())


def main():
    """Entry point for standalone execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = PositionMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
