#!/usr/bin/env python3
"""
Trade History Logger - Completed Trade Tracking
================================================

Watches apply.result stream for completed trades (position closures)
and updates ledger with trade history for performance tracking.

Updates:
- quantum:ledger:{symbol} with trade_history JSON array
- Increments total_trades, winning_trades, losing_trades
- Tracks total_pnl_usdt, total_fees_usdt, total_volume_usdt

Used by: Performance Tracker for win_rate, avg_pnl_pct, sharpe_ratio

Author: Quantum Trader Team  
Date: 2026-02-04
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Optional
from datetime import datetime

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

STREAM_APPLY_RESULT = "quantum:stream:apply.result"
CONSUMER_GROUP = "trade_history_logger"
CONSUMER_NAME = f"trade_logger_{os.getpid()}"

LOG_LEVEL = os.getenv("TRADE_LOGGER_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [TRADE-LOGGER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class TradeHistoryLogger:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        self._ensure_consumer_group()
        
    def _ensure_consumer_group(self):
        """Create consumer group if not exists"""
        try:
            self.redis.xgroup_create(
                STREAM_APPLY_RESULT,
                CONSUMER_GROUP,
                id='0',
                mkstream=True
            )
            logger.info(f"Created consumer group: {CONSUMER_GROUP}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group exists: {CONSUMER_GROUP}")
            else:
                raise
    
    def is_trade_closure(self, result: Dict) -> bool:
        """
        Determine if this result represents a completed trade (position closure)
        
        Trade closure indicators:
        - reduceOnly=True (partial or full close)
        - Position went to FLAT (full close detected in snapshot)
        - executed=True
        """
        try:
            executed = result.get(b'executed', b'false').decode().lower() == 'true'
            if not executed:
                return False
            
            # Check if reduceOnly
            reduce_only_str = result.get(b'reduceOnly', result.get(b'reduce_only', b'false')).decode()
            reduce_only = reduce_only_str.lower() in ('true', '1', 'yes')
            
            return reduce_only
            
        except Exception as e:
            logger.debug(f"Error checking trade closure: {e}")
            return False
    
    def extract_trade_data(self, result: Dict) -> Optional[Dict]:
        """Extract trade data from apply.result"""
        try:
            symbol = result.get(b'symbol', b'').decode()
            plan_id = result.get(b'plan_id', b'').decode()
            
            # Parse steps_results
            steps_str = result.get(b'steps_results', b'[]').decode()
            steps = json.loads(steps_str) if steps_str else []
            
            if not steps:
                return None
            
            last_step = steps[-1]
            
            # Extract order details
            order_id = last_step.get('order_id', 0)
            executed_qty = float(last_step.get('executed_qty', 0))
            avg_price = float(last_step.get('avg_price', 0))
            side = last_step.get('side', 'UNKNOWN')
            
            # Get realized PnL if available (from position snapshot after close)
            realized_pnl = float(result.get(b'realized_pnl', b'0').decode())
            
            trade_data = {
                'symbol': symbol,
                'plan_id': plan_id,
                'order_id': order_id,
                'executed_qty': executed_qty,
                'avg_price': avg_price,
                'side': side,
                'realized_pnl': realized_pnl,
                'timestamp': int(time.time())
            }
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Failed to extract trade data: {e}")
            return None
    
    def get_position_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get current position snapshot to calculate P&L"""
        try:
            snapshot_key = f"quantum:position:snapshot:{symbol}"
            data = self.redis.hgetall(snapshot_key)
            
            if not data:
                return None
            
            return {
                'position_amt': float(data.get(b'position_amt', b'0').decode()),
                'entry_price': float(data.get(b'entry_price', b'0').decode()),
                'unrealized_pnl': float(data.get(b'unrealized_pnl', b'0').decode()),
                'side': data.get(b'side', b'FLAT').decode()
            }
            
        except Exception as e:
            logger.debug(f"Failed to get snapshot for {symbol}: {e}")
            return None
    
    def update_ledger_history(self, trade: Dict):
        """Update ledger with completed trade"""
        try:
            symbol = trade['symbol']
            ledger_key = f"quantum:ledger:{symbol}"
            
            # Get current ledger
            ledger_data = self.redis.hgetall(ledger_key)
            
            # Parse existing trade history
            history_str = ledger_data.get(b'trade_history', b'[]').decode()
            trade_history = json.loads(history_str) if history_str else []
            
            # Calculate P&L for this trade
            executed_qty = trade['executed_qty']
            avg_price = trade['avg_price']
            
            # Get snapshot to determine if position is now flat (full close)
            snapshot = self.get_position_snapshot(symbol)
            is_full_close = (snapshot and abs(snapshot['position_amt']) < 1e-6)
            
            # Use realized_pnl if available, otherwise estimate
            if trade['realized_pnl'] != 0:
                pnl_usdt = trade['realized_pnl']
            else:
                # Estimate P&L (will be improved when we have entry tracking)
                entry_price = float(ledger_data.get(b'entry_price', b'0').decode()) if ledger_data else 0
                if entry_price > 0:
                    if trade['side'] == 'SELL':  # Closing LONG
                        pnl_usdt = executed_qty * (avg_price - entry_price)
                    else:  # Closing SHORT
                        pnl_usdt = executed_qty * (entry_price - avg_price)
                else:
                    pnl_usdt = 0
            
            # Calculate P&L%
            volume_usdt = executed_qty * avg_price
            pnl_pct = (pnl_usdt / volume_usdt * 100) if volume_usdt > 0 else 0
            
            # Add to history
            trade_record = {
                'timestamp': trade['timestamp'],
                'order_id': trade['order_id'],
                'qty': executed_qty,
                'price': avg_price,
                'side': trade['side'],
                'pnl_usdt': round(pnl_usdt, 2),
                'pnl_pct': round(pnl_pct, 4),
                'volume_usdt': round(volume_usdt, 2),
                'is_full_close': is_full_close
            }
            
            trade_history.append(trade_record)
            
            # Keep last 100 trades
            if len(trade_history) > 100:
                trade_history = trade_history[-100:]
            
            # Update aggregates
            total_trades = int(ledger_data.get(b'total_trades', b'0').decode()) + 1 if ledger_data else 1
            winning_trades = int(ledger_data.get(b'winning_trades', b'0').decode()) if ledger_data else 0
            losing_trades = int(ledger_data.get(b'losing_trades', b'0').decode()) if ledger_data else 0
            
            if pnl_usdt > 0:
                winning_trades += 1
            elif pnl_usdt < 0:
                losing_trades += 1
            
            total_pnl = float(ledger_data.get(b'total_pnl_usdt', b'0').decode()) if ledger_data else 0
            total_pnl += pnl_usdt
            
            total_volume = float(ledger_data.get(b'total_volume_usdt', b'0').decode()) if ledger_data else 0
            total_volume += volume_usdt
            
            # Write to ledger
            ledger_update = {
                b'trade_history': json.dumps(trade_history).encode(),
                b'total_trades': str(total_trades).encode(),
                b'winning_trades': str(winning_trades).encode(),
                b'losing_trades': str(losing_trades).encode(),
                b'total_pnl_usdt': str(round(total_pnl, 2)).encode(),
                b'total_volume_usdt': str(round(total_volume, 2)).encode(),
                b'last_trade_ts': str(trade['timestamp']).encode()
            }
            
            self.redis.hset(ledger_key, mapping=ledger_update)
            
            logger.info(
                f"✅ {symbol}: Trade logged | "
                f"P&L=${pnl_usdt:.2f} ({pnl_pct:+.2f}%) | "
                f"Total: {total_trades} trades, {winning_trades}W/{losing_trades}L, "
                f"${total_pnl:.2f} cumulative"
            )
            
        except Exception as e:
            logger.error(f"Failed to update ledger history for {trade.get('symbol', 'UNKNOWN')}: {e}")
    
    def process_result(self, msg_id: bytes, result: Dict):
        """Process one apply.result message"""
        try:
            # Check if this is a trade closure
            if not self.is_trade_closure(result):
                return
            
            # Extract trade data
            trade = self.extract_trade_data(result)
            if not trade:
                return
            
            # Update ledger history
            self.update_ledger_history(trade)
            
        except Exception as e:
            logger.error(f"Failed to process result {msg_id.decode()}: {e}")
    
    def run_forever(self):
        """Main loop: consume apply.result stream"""
        logger.info(f"Trade History Logger started")
        logger.info(f"Consuming: {STREAM_APPLY_RESULT}")
        logger.info(f"Consumer group: {CONSUMER_GROUP} ({CONSUMER_NAME})")
        
        while True:
            try:
                # Read from stream
                messages = self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {STREAM_APPLY_RESULT: '>'},
                    count=10,
                    block=5000  # 5 second block
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for msg_id, result in stream_messages:
                        self.process_result(msg_id, result)
                        
                        # ACK message
                        self.redis.xack(STREAM_APPLY_RESULT, CONSUMER_GROUP, msg_id)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)


def main():
    logger.info("=" * 70)
    logger.info("TRADE HISTORY LOGGER - Completed Trade Tracking")
    logger.info("=" * 70)
    
    logger_service = TradeHistoryLogger()
    logger_service.run_forever()


if __name__ == "__main__":
    main()
