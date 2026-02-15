"""
Emergency Exit Worker - LAST LINE OF DEFENSE

This service exists solely to close ALL open positions during 
catastrophic or uncertain system states.

Principles:
- Market orders only
- Reduce-only
- No strategy, no intelligence
- Survival > cost

Triggered by: system.panic_close

DO NOT ADD:
- Retry logic
- Optimization
- Conditions
- Intelligence
"""

import os
import sys
import time
import json
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis.asyncio as aioredis
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [EEW] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - DO NOT MODIFY WITHOUT AUTHORIZATION
# =============================================================================

# Redis streams (matches REDIS_STREAMS_SCHEMA.md)
PANIC_CLOSE_STREAM = "system:panic_close"
PANIC_COMPLETE_STREAM = "system:panic_close:completed"
TRADING_HALT_KEY = "system:state:trading"

# Processed events (idempotency)
PROCESSED_EVENTS_KEY = "system:panic_close:processed"

# Authorized trigger sources (ONLY these can trigger panic_close)
# Matches issued_by enum in schema
AUTHORIZED_SOURCES = frozenset(["risk_kernel", "exit_brain", "ops", "watchdog"])

# Timing
MAX_TRIGGER_AGE_SECONDS = 60  # Reject triggers older than this (prevent replay)
RATE_LIMIT_PAUSE_MS = 100     # Brief pause between orders if rate limited

# Consumer group
CONSUMER_GROUP = "emergency_exit_worker"
CONSUMER_NAME = f"eew_{os.getpid()}"


@dataclass
class Position:
    """Open position data"""
    symbol: str
    amount: float  # Positive = long, negative = short
    entry_price: float
    unrealized_pnl: float
    notional: float


@dataclass
class PanicCloseResult:
    """Result of panic close execution - matches system:panic_close:completed schema"""
    event_id: str = ""  # Original event_id from panic_close
    trigger_source: str = ""
    trigger_reason: str = ""
    ts_started: int = 0  # Epoch ms
    ts_completed: int = 0  # Epoch ms
    positions_total: int = 0  # Positions found open
    positions_closed: int = 0
    positions_failed: int = 0
    failed_symbols: List[str] = field(default_factory=list)
    total_notional_usd: float = 0.0
    execution_time_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "positions_total": self.positions_total,
            "positions_closed": self.positions_closed,
            "positions_failed": self.positions_failed,
            "failed_symbols": self.failed_symbols,
            "ts_started": self.ts_started,
            "ts_completed": self.ts_completed,
            "execution_time_ms": self.execution_time_ms
        }


class EmergencyExitWorker:
    """
    Emergency Exit Worker - Unconditional position closer
    
    When triggered:
    1. Fetches ALL open positions
    2. Sends MARKET close orders for each
    3. Publishes completion event
    4. Halts all trading
    
    NO intelligence. NO optimization. SURVIVAL ONLY.
    """
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.binance: Optional[BinanceClient] = None
        self._running = False
        
        # Load credentials from environment
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    async def start(self):
        """Start the Emergency Exit Worker"""
        logger.info("=" * 60)
        logger.info("EMERGENCY EXIT WORKER STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode: {'TESTNET' if self.testnet else 'PRODUCTION'}")
        
        # Connect to Redis
        self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        
        # Connect to Binance
        if self.testnet:
            self.binance = BinanceClient(
                self.api_key, 
                self.api_secret,
                testnet=True
            )
        else:
            self.binance = BinanceClient(self.api_key, self.api_secret)
        
        # Create consumer group if not exists
        try:
            await self.redis.xgroup_create(
                PANIC_CLOSE_STREAM, 
                CONSUMER_GROUP, 
                id="0",
                mkstream=True
            )
            logger.info(f"Created consumer group: {CONSUMER_GROUP}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Consumer group setup: {e}")
        
        self._running = True
        logger.info(f"✅ Emergency Exit Worker READY")
        logger.info(f"   Listening on: {PANIC_CLOSE_STREAM}")
        logger.info(f"   Consumer: {CONSUMER_NAME}")
        logger.info("=" * 60)
        
        # Main loop
        await self._listen_loop()
    
    async def stop(self):
        """Stop the worker"""
        self._running = False
        if self.redis:
            await self.redis.close()
        logger.info("Emergency Exit Worker stopped")
    
    async def _listen_loop(self):
        """Main event loop - listen for panic_close events"""
        while self._running:
            try:
                # Read from stream with blocking
                messages = await self.redis.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {PANIC_CLOSE_STREAM: ">"},
                    count=1,
                    block=1000  # 1 second timeout
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        await self._handle_panic_close(msg_id, msg_data)
                        
                        # Acknowledge message
                        await self.redis.xack(PANIC_CLOSE_STREAM, CONSUMER_GROUP, msg_id)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(1)
    
    async def _handle_panic_close(self, msg_id: str, event: Dict[str, str]):
        """
        Handle panic_close event
        
        CRITICAL: This function MUST execute regardless of external state.
        """
        logger.warning("=" * 60)
        logger.warning("🚨 PANIC CLOSE EVENT RECEIVED 🚨")
        logger.warning("=" * 60)
        
        # Parse event data per schema
        try:
            event_id = event.get("event_id", str(uuid.uuid4()))  # Fallback to new UUID
            source = event.get("issued_by", event.get("source", "unknown"))  # Support both
            reason = event.get("reason", "No reason provided")
            ts = int(event.get("ts", int(time.time() * 1000)))  # Epoch ms
            timestamp = ts / 1000.0  # Convert to seconds for validation
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")
            event_id = str(uuid.uuid4())
            source = "unknown"
            reason = f"Parse error: {e}"
            timestamp = time.time()
        
        logger.info(f"Event ID: {event_id}")
        logger.info(f"Source: {source}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Redis Msg ID: {msg_id}")
        
        # Validate trigger
        if not self._validate_trigger(source, timestamp):
            logger.error("❌ TRIGGER VALIDATION FAILED - IGNORING")
            return
        
        # Execute panic close
        result = await self._execute_panic_close(event_id, source, reason)
        
        # Publish completion
        await self._publish_completion(result)
        
        # Halt all trading
        await self._halt_trading(result)
        
        logger.warning("=" * 60)
        logger.warning("🔒 PANIC CLOSE COMPLETE - SYSTEM HALTED 🔒")
        logger.warning("=" * 60)
    
    def _validate_trigger(self, source: str, timestamp: float) -> bool:
        """
        Validate panic_close trigger
        
        Returns True if trigger is valid and should be executed.
        """
        # Check source authorization
        if source not in AUTHORIZED_SOURCES:
            logger.error(f"UNAUTHORIZED SOURCE: {source}")
            logger.error(f"Authorized sources: {AUTHORIZED_SOURCES}")
            return False
        
        # Check timestamp (prevent replay attacks)
        age = time.time() - timestamp
        if age > MAX_TRIGGER_AGE_SECONDS:
            logger.error(f"STALE TRIGGER: {age:.1f}s old (max {MAX_TRIGGER_AGE_SECONDS}s)")
            return False
        
        if age < -5:  # Allow 5s clock skew
            logger.error(f"FUTURE TRIGGER: {-age:.1f}s in future")
            return False
        
        logger.info(f"✅ Trigger validated (source={source}, age={age:.1f}s)")
        return True
    
    async def _execute_panic_close(self, event_id: str, source: str, reason: str) -> PanicCloseResult:
        """
        Execute panic close - close ALL positions
        
        NO retries. NO optimization. Fire and forget.
        """
        ts_start = int(time.time() * 1000)  # Epoch ms
        
        result = PanicCloseResult(
            event_id=event_id,
            trigger_source=source,
            trigger_reason=reason,
            ts_started=ts_start
        )
        
        # Fetch all open positions
        positions = self._fetch_open_positions()
        result.positions_total = len(positions)
        result.total_notional_usd = sum(p.notional for p in positions)
        
        if not positions:
            logger.info("No open positions found - nothing to close")
            result.ts_completed = int(time.time() * 1000)
            result.execution_time_ms = result.ts_completed - result.ts_started
            return result
        
        logger.info(f"Found {len(positions)} open positions")
        logger.info(f"Total notional: ${result.total_notional_usd:,.2f}")
        
        # Close each position
        for pos in positions:
            success = await self._close_position(pos)
            if success:
                result.positions_closed += 1
            else:
                result.positions_failed += 1
                result.failed_symbols.append(pos.symbol)
        
        result.ts_completed = int(time.time() * 1000)  # Epoch ms
        result.execution_time_ms = result.ts_completed - result.ts_started
        
        logger.info(f"Closed: {result.positions_closed}/{result.positions_total}")
        if result.failed_symbols:
            logger.error(f"Failed symbols: {result.failed_symbols}")
        logger.info(f"Execution time: {result.execution_time_ms}ms")
        
        return result
    
    def _fetch_open_positions(self) -> List[Position]:
        """Fetch all open positions from exchange"""
        try:
            raw_positions = self.binance.futures_position_information()
            positions = []
            
            for p in raw_positions:
                amount = float(p.get("positionAmt", 0))
                if amount == 0:
                    continue
                
                positions.append(Position(
                    symbol=p["symbol"],
                    amount=amount,
                    entry_price=float(p.get("entryPrice", 0)),
                    unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                    notional=abs(float(p.get("notional", 0)))
                ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []
    
    async def _close_position(self, pos: Position) -> bool:
        """
        Close a single position with MARKET order
        
        NO RETRIES. Fire and forget.
        """
        try:
            # Determine close side (opposite of position)
            close_side = "SELL" if pos.amount > 0 else "BUY"
            close_qty = abs(pos.amount)
            
            logger.info(f"Closing {pos.symbol}: {close_side} {close_qty} (reduce-only)")
            
            # Send market order
            order = self.binance.futures_create_order(
                symbol=pos.symbol,
                side=close_side,
                type="MARKET",
                quantity=close_qty,
                reduceOnly=True
            )
            
            order_id = order.get("orderId", "unknown")
            logger.info(f"✅ {pos.symbol} closed - Order ID: {order_id}")
            return True
            
        except BinanceAPIException as e:
            logger.error(f"❌ {pos.symbol} FAILED: {e.message}")
            return False
            
        except Exception as e:
            logger.error(f"❌ {pos.symbol} FAILED: {e}")
            return False
    
    async def _publish_completion(self, result: PanicCloseResult):
        """Publish to system:panic_close:completed per schema"""
        try:
            await self.redis.xadd(
                PANIC_COMPLETE_STREAM,
                {
                    "event_id": result.event_id,
                    "positions_total": str(result.positions_total),
                    "positions_closed": str(result.positions_closed),
                    "positions_failed": str(result.positions_failed),
                    "failed_symbols": json.dumps(result.failed_symbols),
                    "ts_started": str(result.ts_started),
                    "ts_completed": str(result.ts_completed),
                    "execution_time_ms": str(result.execution_time_ms)
                }
            )
            logger.info(f"Published to {PANIC_COMPLETE_STREAM}")
        except Exception as e:
            logger.error(f"Failed to publish completion: {e}")
    
    async def _halt_trading(self, result: PanicCloseResult):
        """Set system to halted state"""
        try:
            state = {
                "halted": "true",
                "reason": result.trigger_reason,
                "source": result.trigger_source,
                "event_id": result.event_id,
                "ts": str(int(time.time() * 1000)),  # Epoch ms
                "positions_closed": str(result.positions_closed),
                "requires_manual_reset": "true"
            }
            
            await self.redis.hset(TRADING_HALT_KEY, mapping=state)
            logger.info(f"System halted - state set in {TRADING_HALT_KEY}")
            
        except Exception as e:
            logger.error(f"Failed to set halt state: {e}")


async def main():
    """Main entry point"""
    worker = EmergencyExitWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
