"""
Backlog Drain Module
====================

Safely drain historical trade.intent backlog with:
- Throttling (max 5 events/sec)
- Age filtering (drop > 30 min old)
- Symbol allowlist
- Confidence filtering
- Audit logging to separate stream
- NO actual trading (SAFE_DRAIN mode)

Usage:
    python -m backend.services.execution.backlog_drain \
        --mode dry-run \
        --allowlist "BTCUSDT,ETHUSDT,SOLUSDT" \
        --min-conf 0.60 \
        --max-age-min 30 \
        --throttle 5 \
        --max-events 500
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BacklogDrainMetrics:
    """Track drain metrics"""
    
    def __init__(self):
        self.total_read = 0
        self.dropped_old = 0
        self.dropped_not_allowlisted = 0
        self.dropped_low_conf = 0
        self.drained_ok = 0
        self.errors = 0
        self.start_time = time.time()
    
    def summary(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            "total_read": self.total_read,
            "dropped_old": self.dropped_old,
            "dropped_not_allowlisted": self.dropped_not_allowlisted,
            "dropped_low_conf": self.dropped_low_conf,
            "drained_ok": self.drained_ok,
            "errors": self.errors,
            "elapsed_sec": round(elapsed, 2),
            "events_per_sec": round(self.total_read / elapsed, 2) if elapsed > 0 else 0
        }


class BacklogDrainer:
    """Controlled backlog drainer with throttling and filtering"""
    
    def __init__(
        self,
        redis_url: str,
        stream_key: str,
        consumer_group: str,
        consumer_name: str,
        audit_stream: str,
        allowlist: Set[str],
        min_confidence: float,
        max_age_minutes: int,
        throttle_events_per_sec: int,
        dry_run: bool = False
    ):
        self.redis_url = redis_url
        self.stream_key = stream_key
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self.audit_stream = audit_stream
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self.max_age_minutes = max_age_minutes
        self.throttle_delay = 1.0 / throttle_events_per_sec
        self.dry_run = dry_run
        
        self.metrics = BacklogDrainMetrics()
        self.redis_client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info(f"✅ Connected to Redis: {self.redis_url}")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Disconnected from Redis")
    
    def parse_stream_id_timestamp(self, stream_id: str) -> datetime:
        """Extract timestamp from Redis stream ID (format: TIMESTAMP-SEQUENCE)"""
        timestamp_ms = int(stream_id.split('-')[0])
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    
    def get_event_age_seconds(self, event_data: Dict, stream_id: str) -> float:
        """Calculate event age in seconds"""
        # Try to get timestamp from event payload
        timestamp_str = None
        
        # Check for timestamp in payload
        if 'timestamp' in event_data:
            timestamp_str = event_data['timestamp']
        elif 'payload' in event_data:
            try:
                payload = json.loads(event_data['payload'])
                timestamp_str = payload.get('timestamp')
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Parse timestamp or use stream ID
        if timestamp_str:
            try:
                # Handle various timestamp formats
                if 'T' in timestamp_str:
                    if '+' in timestamp_str or 'Z' in timestamp_str:
                        event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        event_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
                else:
                    event_time = datetime.fromtimestamp(float(timestamp_str), tz=timezone.utc)
            except (ValueError, AttributeError):
                event_time = self.parse_stream_id_timestamp(stream_id)
        else:
            event_time = self.parse_stream_id_timestamp(stream_id)
        
        age_seconds = (datetime.now(timezone.utc) - event_time).total_seconds()
        return age_seconds
    
    def extract_symbol_and_confidence(self, event_data: Dict) -> tuple[Optional[str], Optional[float]]:
        """Extract symbol and confidence from event data"""
        symbol = None
        confidence = None
        
        # Direct fields (from new consumer format)
        if 'symbol' in event_data:
            symbol = event_data['symbol']
        if 'confidence' in event_data:
            try:
                confidence = float(event_data['confidence'])
            except (ValueError, TypeError):
                pass
        
        # Try payload if direct fields not found
        if (symbol is None or confidence is None) and 'payload' in event_data:
            try:
                payload = json.loads(event_data['payload'])
                if symbol is None:
                    symbol = payload.get('symbol')
                if confidence is None:
                    confidence = payload.get('confidence')
                    if confidence is not None:
                        confidence = float(confidence)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        
        return symbol, confidence
    
    async def write_audit(
        self,
        original_id: str,
        symbol: Optional[str],
        age_sec: float,
        confidence: Optional[float],
        action: str
    ):
        """Write audit event to separate stream"""
        if self.dry_run:
            logger.info(f"  [DRY-RUN] Would write audit: {action} for {symbol}")
            return
        
        audit_data = {
            'original_id': original_id,
            'symbol': symbol or 'UNKNOWN',
            'age_sec': str(round(age_sec, 2)),
            'confidence': str(confidence) if confidence is not None else 'null',
            'action': action,
            'drained_at_ts': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            await self.redis_client.xadd(self.audit_stream, audit_data)
        except Exception as e:
            logger.error(f"  ❌ Failed to write audit: {e}")
    
    async def process_event(
        self,
        stream_id: str,
        event_data: Dict
    ) -> str:
        """
        Process single event and return action taken
        
        Returns:
            Action string: dropped_old, dropped_not_allowlisted, dropped_low_conf, drained_ok
        """
        self.metrics.total_read += 1
        
        # Calculate age
        age_sec = self.get_event_age_seconds(event_data, stream_id)
        age_min = age_sec / 60
        
        # Extract symbol and confidence
        symbol, confidence = self.extract_symbol_and_confidence(event_data)
        
        # Filter 1: Age check
        if age_min > self.max_age_minutes:
            action = 'dropped_old'
            self.metrics.dropped_old += 1
            logger.info(
                f"  ⏰ DROP (age): {stream_id} | {symbol} | "
                f"age={age_min:.1f}min (>{self.max_age_minutes}min)"
            )
            await self.write_audit(stream_id, symbol, age_sec, confidence, action)
            return action
        
        # Filter 2: Allowlist check
        if symbol and symbol not in self.allowlist:
            action = 'dropped_not_allowlisted'
            self.metrics.dropped_not_allowlisted += 1
            logger.info(
                f"  🚫 DROP (allowlist): {stream_id} | {symbol} | "
                f"not in allowlist"
            )
            await self.write_audit(stream_id, symbol, age_sec, confidence, action)
            return action
        
        # Filter 3: Confidence check
        if confidence is not None and confidence < self.min_confidence:
            action = 'dropped_low_conf'
            self.metrics.dropped_low_conf += 1
            logger.info(
                f"  📉 DROP (confidence): {stream_id} | {symbol} | "
                f"conf={confidence:.2f} (<{self.min_confidence})"
            )
            await self.write_audit(stream_id, symbol, age_sec, confidence, action)
            return action
        
        # Passed all filters - drain OK (no actual trading in SAFE_DRAIN)
        action = 'drained_ok'
        self.metrics.drained_ok += 1
        logger.info(
            f"  ✅ DRAINED: {stream_id} | {symbol} | "
            f"conf={confidence:.2f if confidence else 'N/A'} | "
            f"age={age_min:.1f}min"
        )
        await self.write_audit(stream_id, symbol, age_sec, confidence, action)
        return action
    
    async def drain_batch(self, max_events: int):
        """
        Drain batch of events from backlog
        
        Reads from consumer group starting at '0' to get historical backlog
        """
        logger.info("=" * 80)
        logger.info(f"🔄 DRAIN BATCH START")
        logger.info(f"  Mode: {'DRY-RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"  Stream: {self.stream_key}")
        logger.info(f"  Consumer Group: {self.consumer_group}")
        logger.info(f"  Consumer Name: {self.consumer_name}")
        logger.info(f"  Allowlist: {sorted(self.allowlist)}")
        logger.info(f"  Min Confidence: {self.min_confidence}")
        logger.info(f"  Max Age: {self.max_age_minutes} minutes")
        logger.info(f"  Throttle: {1/self.throttle_delay:.1f} events/sec ({self.throttle_delay:.3f}s per event)")
        logger.info(f"  Max Events: {max_events}")
        logger.info("=" * 80)
        
        events_processed = 0
        
        try:
            while events_processed < max_events:
                # Read from consumer group
                # Using '>' reads new messages
                # Using '0' reads pending messages (backlog)
                # We want backlog first, so we'll try pending first
                
                # First, check for pending messages
                pending = await self.redis_client.xpending_range(
                    self.stream_key,
                    self.consumer_group,
                    min='-',
                    max='+',
                    count=1
                )
                
                if pending:
                    # We have pending messages, claim them
                    pending_id = pending[0]['message_id']
                    messages = await self.redis_client.xclaim(
                        self.stream_key,
                        self.consumer_group,
                        self.consumer_name,
                        min_idle_time=0,  # Claim regardless of idle time
                        message_ids=[pending_id]
                    )
                else:
                    # No pending, read new messages using XREADGROUP with '>'
                    # '>' reads undelivered messages (backlog that group hasn't seen yet)
                    result = await self.redis_client.xreadgroup(
                        groupname=self.consumer_group,
                        consumername=self.consumer_name,
                        streams={self.stream_key: '>'},
                        count=1,
                        block=0  # Non-blocking
                    )
                    
                    if not result:
                        logger.info("📭 No more undelivered messages in backlog")
                        break
                    
                    messages = result[0][1]  # [(stream_name, [(id, data), ...])]
                
                if not messages:
                    logger.info("📭 No messages available")
                    break
                
                # Process message
                stream_id, event_data = messages[0] if isinstance(messages[0], tuple) else (messages[0]['message_id'], messages[0]['data'])
                
                logger.info(f"\n📨 Event {events_processed + 1}/{max_events}: {stream_id}")
                
                try:
                    action = await self.process_event(stream_id, event_data)
                    
                    # ACK the message (remove from pending)
                    if not self.dry_run:
                        await self.redis_client.xack(
                            self.stream_key,
                            self.consumer_group,
                            stream_id
                        )
                        logger.debug(f"  ✓ ACKed: {stream_id}")
                    else:
                        logger.debug(f"  [DRY-RUN] Would ACK: {stream_id}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing {stream_id}: {e}", exc_info=True)
                    self.metrics.errors += 1
                    # Don't ACK on error
                
                events_processed += 1
                
                # Throttle
                if events_processed < max_events:
                    await asyncio.sleep(self.throttle_delay)
        
        except Exception as e:
            logger.error(f"❌ Fatal error in drain_batch: {e}", exc_info=True)
            self.metrics.errors += 1
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DRAIN BATCH COMPLETE")
        logger.info("=" * 80)
        summary = self.metrics.summary()
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 80)
        
        return summary


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Drain trade.intent backlog with throttling and filtering'
    )
    
    parser.add_argument(
        '--mode',
        choices=['dry-run', 'live'],
        default='dry-run',
        help='Dry-run (no ACK) or live mode'
    )
    parser.add_argument(
        '--allowlist',
        type=str,
        default='BTCUSDT,ETHUSDT,SOLUSDT',
        help='Comma-separated list of allowed symbols'
    )
    parser.add_argument(
        '--min-conf',
        type=float,
        default=0.60,
        help='Minimum confidence threshold'
    )
    parser.add_argument(
        '--max-age-min',
        type=int,
        default=30,
        help='Maximum age in minutes (older events dropped)'
    )
    parser.add_argument(
        '--throttle',
        type=int,
        default=5,
        help='Max events per second'
    )
    parser.add_argument(
        '--max-events',
        type=int,
        default=500,
        help='Maximum events to process in this batch'
    )
    parser.add_argument(
        '--redis-url',
        type=str,
        default='redis://quantum_redis:6379/0',
        help='Redis connection URL'
    )
    parser.add_argument(
        '--stream',
        type=str,
        default='quantum:stream:trade.intent',
        help='Redis stream key'
    )
    parser.add_argument(
        '--consumer-group',
        type=str,
        default='quantum:group:execution:trade.intent',
        help='Consumer group name'
    )
    parser.add_argument(
        '--consumer-name',
        type=str,
        default='backlog_drain_1',
        help='Consumer name'
    )
    parser.add_argument(
        '--audit-stream',
        type=str,
        default='quantum:stream:trade.intent.drain_audit',
        help='Audit stream key'
    )
    
    args = parser.parse_args()
    
    # Parse allowlist
    allowlist = set(s.strip() for s in args.allowlist.split(',') if s.strip())
    
    # Create drainer
    drainer = BacklogDrainer(
        redis_url=args.redis_url,
        stream_key=args.stream,
        consumer_group=args.consumer_group,
        consumer_name=args.consumer_name,
        audit_stream=args.audit_stream,
        allowlist=allowlist,
        min_confidence=args.min_conf,
        max_age_minutes=args.max_age_min,
        throttle_events_per_sec=args.throttle,
        dry_run=(args.mode == 'dry-run')
    )
    
    try:
        await drainer.connect()
        summary = await drainer.drain_batch(max_events=args.max_events)
        
        # Print final report
        print("\n" + "=" * 80)
        print("📋 DRAIN REPORT")
        print("=" * 80)
        print(f"Mode: {args.mode.upper()}")
        print(f"Total Read: {summary['total_read']}")
        print(f"Dropped (old): {summary['dropped_old']}")
        print(f"Dropped (not allowlisted): {summary['dropped_not_allowlisted']}")
        print(f"Dropped (low confidence): {summary['dropped_low_conf']}")
        print(f"Drained OK: {summary['drained_ok']}")
        print(f"Errors: {summary['errors']}")
        print(f"Elapsed: {summary['elapsed_sec']}s")
        print(f"Throughput: {summary['events_per_sec']} events/sec")
        print("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await drainer.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
