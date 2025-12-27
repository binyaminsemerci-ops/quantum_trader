"""
Backlog Drain Service
Safely processes historical trade.intent events with throttling and filtering.

Usage:
    python -m backend.services.execution.backlog_drain_service --mode=dry-run
    python -m backend.services.execution.backlog_drain_service --mode=live --throttle=5
"""
import asyncio
import logging
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class DrainConfig:
    """Configuration for backlog drain"""
    # Throttling
    events_per_second: int = 2  # Max events to process per second
    batch_size: int = 10  # Process in small batches
    
    # Time filter
    max_age_hours: int = 24  # Drop events older than this
    
    # Symbol filter (allowlist)
    allowed_symbols: Optional[List[str]] = None  # None = all allowed
    
    # Confidence filter
    min_confidence: float = 0.6  # Drop events with confidence < this
    
    # Safety
    dry_run: bool = True  # If True, only log what would be done
    max_events: Optional[int] = None  # Max events to process (None = all)


class BacklogDrainService:
    """
    Safely drains historical trade.intent events from Redis stream.
    
    Features:
    - Throttled processing (1-5 events/sec)
    - Age-based filtering (drop old events)
    - Symbol allowlist filtering
    - Confidence threshold filtering
    - Dry-run mode for safety
    - Detailed metrics and reporting
    """
    
    def __init__(self, config: DrainConfig):
        self.config = config
        self.stats = {
            "total_processed": 0,
            "filtered_age": 0,
            "filtered_symbol": 0,
            "filtered_confidence": 0,
            "executed": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }
        
        # Initialize Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis = None
        self.redis_url = f"redis://{redis_host}:{redis_port}"
        
        logger.info("[BACKLOG_DRAIN] Initialized with config:")
        logger.info(f"  Throttle: {config.events_per_second} events/sec")
        logger.info(f"  Max age: {config.max_age_hours} hours")
        logger.info(f"  Min confidence: {config.min_confidence}")
        logger.info(f"  Dry run: {config.dry_run}")
        if config.allowed_symbols:
            logger.info(f"  Symbol allowlist: {config.allowed_symbols}")
    
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info(f"[BACKLOG_DRAIN] Connected to Redis: {self.redis_url}")
    
    async def get_stream_info(self) -> Dict[str, Any]:
        """Get stream metadata"""
        info = await self.redis.execute_command(
            "XINFO", "STREAM", "quantum:stream:trade.intent"
        )
        # Parse Redis response
        return dict(zip(info[::2], info[1::2]))
    
    async def should_process_event(self, event_data: Dict[str, Any], event_id: str) -> tuple[bool, Optional[str]]:
        """
        Determine if event should be processed based on filters.
        
        Returns:
            (should_process, reason_if_filtered)
        """
        # Extract timestamp from event ID (format: timestamp-sequence)
        event_timestamp_ms = int(event_id.split("-")[0])
        event_time = datetime.fromtimestamp(event_timestamp_ms / 1000)
        age_hours = (datetime.now() - event_time).total_seconds() / 3600
        
        # Filter 1: Age
        if age_hours > self.config.max_age_hours:
            return False, f"too_old ({age_hours:.1f}h)"
        
        # Filter 2: Symbol allowlist
        if self.config.allowed_symbols:
            symbol = event_data.get("symbol", "")
            if symbol not in self.config.allowed_symbols:
                return False, f"symbol_not_allowed ({symbol})"
        
        # Filter 3: Confidence threshold
        confidence = float(event_data.get("confidence", 0.0))
        if confidence < self.config.min_confidence:
            return False, f"low_confidence ({confidence:.2f})"
        
        return True, None
    
    async def process_event(self, event_id: str, event_data: Dict[str, Any]) -> bool:
        """
        Process a single event.
        
        Returns:
            True if successfully processed, False otherwise
        """
        try:
            symbol = event_data.get("symbol", "UNKNOWN")
            side = event_data.get("side", "UNKNOWN")
            source = event_data.get("source", "UNKNOWN")
            confidence = event_data.get("confidence", 0.0)
            
            if self.config.dry_run:
                logger.info(
                    f"[BACKLOG_DRAIN] [DRY-RUN] Would process: "
                    f"{symbol} {side} (conf={confidence}, source={source})"
                )
            else:
                logger.info(
                    f"[BACKLOG_DRAIN] [LIVE] Processing: "
                    f"{symbol} {side} (conf={confidence}, source={source})"
                )
                # In live mode, we would:
                # 1. Validate event data
                # 2. Check current market conditions
                # 3. Execute trade if still valid
                # For now, just mark as processed
                pass
            
            self.stats["executed"] += 1
            return True
            
        except Exception as e:
            logger.error(f"[BACKLOG_DRAIN] Error processing event {event_id}: {e}", exc_info=True)
            self.stats["errors"] += 1
            return False
    
    async def drain_backlog(self):
        """
        Main drain loop: Read historical events with throttling and filtering.
        """
        self.stats["start_time"] = time.time()
        
        try:
            # Get stream info
            stream_info = await self.get_stream_info()
            total_length = int(stream_info.get("length", 0))
            
            logger.info(f"[BACKLOG_DRAIN] Stream length: {total_length} events")
            logger.info(f"[BACKLOG_DRAIN] Starting drain process...")
            
            # Read historical events using XRANGE (not XREADGROUP to avoid affecting live consumer)
            last_id = "0-0"  # Start from beginning
            processed_count = 0
            
            while True:
                # Check if we've hit max_events limit
                if self.config.max_events and processed_count >= self.config.max_events:
                    logger.info(f"[BACKLOG_DRAIN] Reached max_events limit ({self.config.max_events})")
                    break
                
                # Read batch
                batch = await self.redis.xrange(
                    "quantum:stream:trade.intent",
                    min=f"({last_id}",  # Exclusive start
                    max="+",
                    count=self.config.batch_size
                )
                
                if not batch:
                    logger.info("[BACKLOG_DRAIN] No more events to process")
                    break
                
                # Process batch with throttling
                batch_start = time.time()
                
                for event_id, event_data in batch:
                    self.stats["total_processed"] += 1
                    processed_count += 1
                    
                    # Apply filters
                    should_process, filter_reason = await self.should_process_event(event_data, event_id)
                    
                    if not should_process:
                        if "too_old" in filter_reason:
                            self.stats["filtered_age"] += 1
                        elif "symbol_not_allowed" in filter_reason:
                            self.stats["filtered_symbol"] += 1
                        elif "low_confidence" in filter_reason:
                            self.stats["filtered_confidence"] += 1
                        
                        logger.debug(f"[BACKLOG_DRAIN] Filtered {event_id}: {filter_reason}")
                        continue
                    
                    # Process event
                    await self.process_event(event_id, event_data)
                    
                    # Throttle: ensure we don't exceed events_per_second
                    await asyncio.sleep(1.0 / self.config.events_per_second)
                    
                    last_id = event_id
                
                # Log batch progress
                batch_duration = time.time() - batch_start
                logger.info(
                    f"[BACKLOG_DRAIN] Batch complete: {len(batch)} events in {batch_duration:.2f}s "
                    f"(total: {processed_count}/{total_length})"
                )
                
                # Print stats every 100 events
                if processed_count % 100 == 0:
                    self._print_stats()
        
        finally:
            self.stats["end_time"] = time.time()
            self._print_final_report()
    
    def _print_stats(self):
        """Print current statistics"""
        logger.info("[BACKLOG_DRAIN] === PROGRESS STATS ===")
        logger.info(f"  Total processed: {self.stats['total_processed']}")
        logger.info(f"  Filtered (age): {self.stats['filtered_age']}")
        logger.info(f"  Filtered (symbol): {self.stats['filtered_symbol']}")
        logger.info(f"  Filtered (confidence): {self.stats['filtered_confidence']}")
        logger.info(f"  Executed: {self.stats['executed']}")
        logger.info(f"  Errors: {self.stats['errors']}")
    
    def _print_final_report(self):
        """Print final drain report"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        events_per_sec = self.stats["total_processed"] / duration if duration > 0 else 0
        
        logger.info("")
        logger.info("[BACKLOG_DRAIN] ═══════════════════════════════════════")
        logger.info("[BACKLOG_DRAIN] 📊 FINAL DRAIN REPORT")
        logger.info("[BACKLOG_DRAIN] ═══════════════════════════════════════")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Total processed: {self.stats['total_processed']}")
        logger.info(f"  Throughput: {events_per_sec:.2f} events/sec")
        logger.info("")
        logger.info(f"  ✅ Executed: {self.stats['executed']}")
        logger.info(f"  🔇 Filtered (age): {self.stats['filtered_age']}")
        logger.info(f"  🔇 Filtered (symbol): {self.stats['filtered_symbol']}")
        logger.info(f"  🔇 Filtered (confidence): {self.stats['filtered_confidence']}")
        logger.info(f"  ❌ Errors: {self.stats['errors']}")
        logger.info("")
        
        total_filtered = (
            self.stats['filtered_age'] + 
            self.stats['filtered_symbol'] + 
            self.stats['filtered_confidence']
        )
        filter_rate = (total_filtered / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0
        logger.info(f"  Filter rate: {filter_rate:.1f}%")
        logger.info("[BACKLOG_DRAIN] ═══════════════════════════════════════")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drain trade.intent backlog safely")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run", help="Execution mode")
    parser.add_argument("--throttle", type=int, default=2, help="Events per second (1-10)")
    parser.add_argument("--max-age-hours", type=int, default=24, help="Max event age in hours")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Min confidence threshold")
    parser.add_argument("--max-events", type=int, default=None, help="Max events to process")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol allowlist")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Parse symbols
    allowed_symbols = None
    if args.symbols:
        allowed_symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Create config
    config = DrainConfig(
        events_per_second=min(max(args.throttle, 1), 10),  # Clamp 1-10
        max_age_hours=args.max_age_hours,
        min_confidence=args.min_confidence,
        dry_run=(args.mode == "dry-run"),
        max_events=args.max_events,
        allowed_symbols=allowed_symbols,
    )
    
    # Run drain
    service = BacklogDrainService(config)
    try:
        await service.initialize()
        await service.drain_backlog()
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
