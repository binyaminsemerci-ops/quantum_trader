#!/usr/bin/env python3
"""
Robust Exit Engine - Continuous Position Monitoring + Exit Emission

Monitors open positions and emits reduceOnly plans based on:
- Initial SL: k1 * ATR from entry (e.g. 1.8 * ATR)
- Partial TP: k2 * ATR (e.g. 1.2 * ATR) when trend confirms
- Trailing SL: k3 * ATR, tightens as profit grows
- Time-based exits: No progress after N bars → close
- Regime-aware: CHOP → quick exits, TREND → let runners go

All exits are reduceOnly=true and emit to quantum:stream:apply.plan

Redis Input Keys:
- quantum:positions:open (HASH per symbol: entry_price, qty, side, entry_ts, sl, tp)
- quantum:market_state:{symbol} (HASH: price, atr, regime, trend_conf)

Redis Output Stream:
- quantum:stream:apply.plan (reduceOnly=true plans)
"""

import os
import sys
import time
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class ExitEngineConfig:
    """Exit engine configuration"""
    
    def __init__(self):
        # ATR multipliers for exits
        self.ATR_MULT_INITIAL_SL = float(os.getenv('EXIT_ATR_MULT_INITIAL_SL', '1.8'))
        self.ATR_MULT_TP1 = float(os.getenv('EXIT_ATR_MULT_TP1', '1.2'))
        self.ATR_MULT_TRAIL = float(os.getenv('EXIT_ATR_MULT_TRAIL', '1.0'))
        
        # Partial TP fractions
        self.TP1_FRACTION = float(os.getenv('EXIT_TP1_FRACTION', '0.33'))  # Take 33% at TP1
        
        # Time-based exit (bars without progress)
        self.MAX_BARS_NO_PROGRESS_15M = int(os.getenv('EXIT_MAX_BARS_NO_PROGRESS_15M', '12'))  # 3 hours
        self.MAX_BARS_NO_PROGRESS_1H = int(os.getenv('EXIT_MAX_BARS_NO_PROGRESS_1H', '24'))  # 24 hours
        
        # Regime multipliers (adjust exits based on regime)
        self.REGIME_CHOP_SL_MULT = float(os.getenv('EXIT_REGIME_CHOP_SL_MULT', '0.8'))  # Tighter SL in chop
        self.REGIME_TREND_SL_MULT = float(os.getenv('EXIT_REGIME_TREND_SL_MULT', '1.2'))  # Wider SL in trend
        
        # Trend confirmation threshold for TP1
        self.MIN_TREND_CONF_FOR_TP = float(os.getenv('EXIT_MIN_TREND_CONF_FOR_TP', '0.6'))
        
        # Loop interval
        self.CHECK_INTERVAL_SEC = int(os.getenv('EXIT_CHECK_INTERVAL_SEC', '15'))
        
        # Redis
        self.REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        
        # Streams
        self.STREAM_APPLY_PLAN = os.getenv('STREAM_APPLY_PLAN', 'quantum:stream:apply.plan')
        self.STREAM_HARVEST_EVENTS = os.getenv('STREAM_HARVEST_EVENTS', 'quantum:stream:harvest.events')
        
        logger.info(f"ExitEngine Config:")
        logger.info(f"  Initial SL: {self.ATR_MULT_INITIAL_SL} * ATR")
        logger.info(f"  TP1: {self.ATR_MULT_TP1} * ATR ({self.TP1_FRACTION*100:.0f}% close)")
        logger.info(f"  Trail: {self.ATR_MULT_TRAIL} * ATR")
        logger.info(f"  Check interval: {self.CHECK_INTERVAL_SEC}s")


class Position:
    """Open position tracker"""
    
    def __init__(self, symbol: str, side: str, qty: float, entry_price: float, entry_ts: float, 
                 initial_sl: float, initial_tp: float, atr: float):
        self.symbol = symbol
        self.side = side  # LONG or SHORT
        self.qty = qty
        self.entry_price = entry_price
        self.entry_ts = entry_ts
        self.current_sl = initial_sl
        self.current_tp = initial_tp
        self.atr = atr
        
        # Tracking
        self.highest_price = entry_price if side == 'LONG' else entry_price
        self.lowest_price = entry_price if side == 'SHORT' else entry_price
        self.tp1_taken = False
        self.bars_no_progress = 0
        self.last_update_ts = time.time()
    
    def update_price(self, price: float):
        """Update current price and tracking"""
        if self.side == 'LONG':
            if price > self.highest_price:
                self.highest_price = price
                self.bars_no_progress = 0
            else:
                self.bars_no_progress += 1
        else:  # SHORT
            if price < self.lowest_price:
                self.lowest_price = price
                self.bars_no_progress = 0
            else:
                self.bars_no_progress += 1
        
        self.last_update_ts = time.time()
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized PnL %"""
        if self.side == 'LONG':
            return (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            return (self.entry_price - current_price) / self.entry_price
    
    def age_minutes(self) -> float:
        """Position age in minutes"""
        return (time.time() - self.entry_ts) / 60


class RobustExitEngine:
    """
    Continuous exit engine that emits reduceOnly plans
    """
    
    def __init__(self, config: ExitEngineConfig):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Local position cache
        self.positions: Dict[str, Position] = {}
        
        # Dedup: track emitted plan IDs
        self.emitted_plans: set = set()
        
        logger.info("RobustExitEngine initialized")
    
    def _get_market_state(self, symbol: str) -> Dict:
        """Get current market state for symbol"""
        market_key = f"quantum:market_state:{symbol}"
        data = self.redis.hgetall(market_key)
        
        if not data:
            return {}
        
        return {
            'price': float(data.get('price', '0')),
            'atr': float(data.get('atr', '0')),
            'regime': data.get('regime', 'BASE'),
            'trend_conf': float(data.get('trend_conf', '0.5')),
            'last_update_ts': float(data.get('last_update_ts', '0'))
        }
    
    def _load_open_positions(self):
        """Load open positions from Redis"""
        # Check for position keys
        position_keys = self.redis.keys("quantum:position:*")
        
        for key in position_keys:
            symbol = key.split(':')[-1]
            data = self.redis.hgetall(key)
            
            if not data:
                continue
            
            qty = float(data.get('qty', '0'))
            if abs(qty) < 1e-8:
                continue  # Empty position
            
            # Create Position object if not exists
            if symbol not in self.positions:
                side = 'LONG' if qty > 0 else 'SHORT'
                entry_price = float(data.get('entry_price', '0'))
                entry_ts = float(data.get('entry_ts', time.time()))
                atr = float(data.get('atr', '0'))
                
                # Calculate initial SL/TP
                regime_mult = 1.0
                initial_sl = entry_price - (atr * self.config.ATR_MULT_INITIAL_SL * regime_mult) if side == 'LONG' else entry_price + (atr * self.config.ATR_MULT_INITIAL_SL * regime_mult)
                initial_tp = entry_price + (atr * self.config.ATR_MULT_TP1) if side == 'LONG' else entry_price - (atr * self.config.ATR_MULT_TP1)
                
                pos = Position(
                    symbol=symbol,
                    side=side,
                    qty=abs(qty),
                    entry_price=entry_price,
                    entry_ts=entry_ts,
                    initial_sl=initial_sl,
                    initial_tp=initial_tp,
                    atr=atr
                )
                
                self.positions[symbol] = pos
                logger.info(f"Loaded position: {symbol} {side} qty={abs(qty):.4f} entry=${entry_price:.2f}")
    
    def _emit_exit_plan(self, symbol: str, action: str, qty: float, reason: str, metadata: Dict = None):
        """Emit reduceOnly exit plan to apply.plan stream"""
        # Generate deterministic plan_id
        plan_data = f"{symbol}:{action}:{qty}:{reason}:{time.time()}"
        plan_id = hashlib.sha256(plan_data.encode()).hexdigest()[:16]
        
        # Dedup check
        if plan_id in self.emitted_plans:
            logger.debug(f"{symbol}: Plan {plan_id[:8]} already emitted (dedup)")
            return
        
        # Create exit plan
        plan = {
            'plan_id': plan_id,
            'symbol': symbol,
            'action': action,
            'side': 'CLOSE',
            'qty': str(qty),
            'reduceOnly': 'true',
            'decision': 'EXECUTE',
            'kill_score': '0',  # Exits bypass kill_score
            'reason': reason,
            'source': 'robust_exit_engine',
            'timestamp': str(time.time())
        }
        
        # Add metadata
        if metadata:
            for k, v in metadata.items():
                plan[k] = str(v)
        
        # Emit to stream
        self.redis.xadd(self.config.STREAM_APPLY_PLAN, plan)
        
        # Track emission
        self.emitted_plans.add(plan_id)
        
        # Publish harvest event
        self.redis.xadd(self.config.STREAM_HARVEST_EVENTS, {
            'event': 'EXIT_PLAN_EMITTED',
            'symbol': symbol,
            'action': action,
            'plan_id': plan_id,
            'reason': reason,
            'timestamp': str(time.time())
        })
        
        logger.info(f"HARVEST_DECISION symbol={symbol} action={action} qty={qty:.4f} reason={reason} plan_id={plan_id[:8]}")
    
    def _evaluate_position_exits(self, symbol: str, pos: Position):
        """Evaluate position for exit opportunities"""
        # Get market state
        market = self._get_market_state(symbol)
        if not market or market.get('price', 0) <= 0:
            logger.warning(f"{symbol}: No market data - skipping exit evaluation")
            return
        
        price = market['price']
        atr = market['atr']
        regime = market['regime']
        trend_conf = market['trend_conf']
        
        # Update position price tracking
        pos.update_price(price)
        
        # Calculate PnL
        pnl_pct = pos.unrealized_pnl_pct(price)
        age_min = pos.age_minutes()
        
        # Exit Rule 1: Stop Loss Hit
        if pos.side == 'LONG' and price <= pos.current_sl:
            self._emit_exit_plan(
                symbol, 'FULL_CLOSE_PROPOSED', pos.qty,
                f'stop_loss_hit pnl={pnl_pct*100:.1f}% sl=${pos.current_sl:.2f}',
                {'pnl_pct': pnl_pct, 'age_min': age_min, 'exit_type': 'STOP'}
            )
            return
        
        if pos.side == 'SHORT' and price >= pos.current_sl:
            self._emit_exit_plan(
                symbol, 'FULL_CLOSE_PROPOSED', pos.qty,
                f'stop_loss_hit pnl={pnl_pct*100:.1f}% sl=${pos.current_sl:.2f}',
                {'pnl_pct': pnl_pct, 'age_min': age_min, 'exit_type': 'STOP'}
            )
            return
        
        # Exit Rule 2: Partial TP (TP1) - only if trend confirms
        if not pos.tp1_taken and trend_conf >= self.config.MIN_TREND_CONF_FOR_TP:
            tp_hit = False
            if pos.side == 'LONG' and price >= pos.current_tp:
                tp_hit = True
            elif pos.side == 'SHORT' and price <= pos.current_tp:
                tp_hit = True
            
            if tp_hit:
                partial_qty = pos.qty * self.config.TP1_FRACTION
                self._emit_exit_plan(
                    symbol, 'PARTIAL_33', partial_qty,
                    f'tp1_hit pnl={pnl_pct*100:.1f}% tp=${pos.current_tp:.2f} trend_conf={trend_conf:.2f}',
                    {'pnl_pct': pnl_pct, 'age_min': age_min, 'exit_type': 'TP1', 'trend_conf': trend_conf}
                )
                pos.tp1_taken = True
                pos.qty -= partial_qty  # Update remaining qty
                
                # Move SL to breakeven after TP1
                pos.current_sl = pos.entry_price
                logger.info(f"{symbol}: TP1 taken, SL moved to breakeven ${pos.entry_price:.2f}")
                return
        
        # Exit Rule 3: Trailing Stop (after breakeven)
        if pnl_pct > 0 and atr > 0:
            trail_distance = atr * self.config.ATR_MULT_TRAIL
            
            if pos.side == 'LONG':
                new_trail_sl = price - trail_distance
                # Only move SL up (reduce risk), never down
                if new_trail_sl > pos.current_sl:
                    pos.current_sl = new_trail_sl
                    logger.info(f"{symbol}: Trailing SL updated to ${new_trail_sl:.2f} (trail_dist=${trail_distance:.2f})")
            else:  # SHORT
                new_trail_sl = price + trail_distance
                # Only move SL down (reduce risk), never up
                if new_trail_sl < pos.current_sl:
                    pos.current_sl = new_trail_sl
                    logger.info(f"{symbol}: Trailing SL updated to ${new_trail_sl:.2f} (trail_dist=${trail_distance:.2f})")
        
        # Exit Rule 4: Time-based exit (no progress)
        max_bars = self.config.MAX_BARS_NO_PROGRESS_15M if age_min < 180 else self.config.MAX_BARS_NO_PROGRESS_1H
        
        if pos.bars_no_progress >= max_bars:
            self._emit_exit_plan(
                symbol, 'FULL_CLOSE_PROPOSED', pos.qty,
                f'time_exit no_progress={pos.bars_no_progress} bars age={age_min:.0f}min pnl={pnl_pct*100:.1f}%',
                {'pnl_pct': pnl_pct, 'age_min': age_min, 'exit_type': 'TIME_EXIT', 'bars_no_progress': pos.bars_no_progress}
            )
            return
        
        # Exit Rule 5: Regime-aware quick exit in CHOP
        if regime == 'CHOP' and pnl_pct < -0.01:  # Losing > 1% in chop
            self._emit_exit_plan(
                symbol, 'FULL_CLOSE_PROPOSED', pos.qty,
                f'chop_quick_exit regime={regime} pnl={pnl_pct*100:.1f}%',
                {'pnl_pct': pnl_pct, 'age_min': age_min, 'exit_type': 'CHOP_EXIT', 'regime': regime}
            )
            return
        
        # Log status (no exit triggered)
        logger.debug(f"{symbol}: {pos.side} pnl={pnl_pct*100:.1f}% age={age_min:.0f}min sl=${pos.current_sl:.2f} bars_stale={pos.bars_no_progress}")
    
    def run_once(self):
        """Run one evaluation cycle"""
        # Load open positions
        self._load_open_positions()
        
        if not self.positions:
            logger.debug("No open positions to evaluate")
            return
        
        logger.info(f"Evaluating {len(self.positions)} positions for exits")
        
        # Evaluate each position
        for symbol, pos in list(self.positions.items()):
            try:
                self._evaluate_position_exits(symbol, pos)
            except Exception as e:
                logger.error(f"{symbol}: Error evaluating exits: {e}", exc_info=True)
        
        # Cleanup closed positions (check if still in Redis)
        for symbol in list(self.positions.keys()):
            pos_key = f"quantum:position:{symbol}"
            if not self.redis.exists(pos_key):
                logger.info(f"{symbol}: Position closed - removing from tracker")
                del self.positions[symbol]
    
    def run_loop(self):
        """Main loop"""
        logger.info("Starting exit engine loop")
        
        while True:
            try:
                self.run_once()
                time.sleep(self.config.CHECK_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("Shutdown signal received")
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(self.config.CHECK_INTERVAL_SEC)


def main():
    """Main entry point"""
    config = ExitEngineConfig()
    engine = RobustExitEngine(config)
    engine.run_loop()


if __name__ == '__main__':
    main()
