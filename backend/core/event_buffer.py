"""
PATCH-P0-03: EventBus Disk Buffer for Redis Failover
======================================================

Prevents event loss during Redis outages by writing events to disk
and replaying them when Redis recovers.

Features:
- .jsonl format for easy append/recovery
- Automatic rotation at max size
- Replay on reconnect with deduplication
- Minimal performance overhead
"""

import asyncio
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventBuffer:
    """
    Disk-backed event buffer for Redis failover.
    
    When Redis publish fails, events are written to a .jsonl file.
    On Redis reconnect, events are replayed to subscribers.
    """
    
    MAX_FILE_SIZE_MB = 100  # Rotate at 100MB
    MAX_REPLAY_EVENTS = 10000  # Max events to replay
    BUFFER_DIR = Path("data/event_buffer")
    
    def __init__(self, buffer_dir: Optional[Path] = None):
        """
        Initialize event buffer.
        
        Args:
            buffer_dir: Directory for buffer files (default: data/event_buffer)
        """
        self.buffer_dir = buffer_dir or self.BUFFER_DIR
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_file = self._get_current_buffer_file()
        self.file_handle: Optional[Any] = None
        self.events_written = 0
        self.events_replayed = 0
        
        # In-memory deque for quick access (last 1000 events)
        self.recent_events: deque = deque(maxlen=1000)
        
        logger.info(
            f"EventBuffer initialized: buffer_dir={self.buffer_dir}, "
            f"current_file={self.current_file.name}"
        )
    
    def _get_current_buffer_file(self) -> Path:
        """Get current buffer file path (dated)."""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.buffer_dir / f"events_{date_str}.jsonl"
    
    def _rotate_if_needed(self) -> None:
        """Rotate buffer file if max size exceeded."""
        if not self.current_file.exists():
            return
        
        size_mb = self.current_file.stat().st_size / (1024 * 1024)
        
        if size_mb >= self.MAX_FILE_SIZE_MB:
            logger.warning(
                f"EventBuffer rotating: {self.current_file.name} "
                f"({size_mb:.1f}MB >= {self.MAX_FILE_SIZE_MB}MB)"
            )
            
            # Close current file
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            
            # Rename to .old
            old_file = self.current_file.with_suffix(".jsonl.old")
            self.current_file.rename(old_file)
            
            # Create new file
            self.current_file = self._get_current_buffer_file()
            logger.info(f"EventBuffer new file: {self.current_file.name}")
    
    async def write_event(
        self,
        event_type: str,
        event_data: dict,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Write event to disk buffer (async).
        
        Args:
            event_type: Event type/topic
            event_data: Event payload
            event_id: Optional unique event ID for deduplication
        """
        try:
            # Check rotation
            self._rotate_if_needed()
            
            # Prepare event record
            record = {
                "id": event_id or f"{datetime.utcnow().timestamp()}-{self.events_written}",
                "type": event_type,
                "data": event_data,
                "timestamp": datetime.utcnow().isoformat(),
                "buffered": True,
            }
            
            # Write to file (sync I/O in thread to avoid blocking)
            await asyncio.to_thread(self._write_record_sync, record)
            
            # Add to recent events (for quick replay)
            self.recent_events.append(record)
            
            self.events_written += 1
            
            if self.events_written % 100 == 0:
                logger.info(f"EventBuffer: {self.events_written} events written")
        
        except Exception as e:
            logger.error(f"EventBuffer write failed: {e}")
    
    def _write_record_sync(self, record: dict) -> None:
        """Write record to file (sync, called in thread)."""
        # Open file in append mode
        if not self.file_handle or self.file_handle.closed:
            self.file_handle = open(self.current_file, "a", encoding="utf-8")
        
        # Write JSON line
        self.file_handle.write(json.dumps(record) + "\n")
        self.file_handle.flush()
    
    async def replay_events(
        self,
        replay_callback: Any,
        max_events: Optional[int] = None,
    ) -> int:
        """
        Replay buffered events through callback.
        
        Args:
            replay_callback: Async function(event_type, event_data) to call
            max_events: Max events to replay (default: MAX_REPLAY_EVENTS)
        
        Returns:
            Number of events replayed
        """
        max_events = max_events or self.MAX_REPLAY_EVENTS
        
        logger.warning(f"EventBuffer replay started: max_events={max_events}")
        
        try:
            # Close write handle
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            
            replayed = 0
            seen_ids = set()
            
            # Read and replay events
            if self.current_file.exists():
                with open(self.current_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if replayed >= max_events:
                            logger.warning(f"EventBuffer replay limit reached: {max_events}")
                            break
                        
                        try:
                            record = json.loads(line.strip())
                            
                            # Deduplication
                            event_id = record.get("id")
                            if event_id in seen_ids:
                                continue
                            seen_ids.add(event_id)
                            
                            # Replay event
                            await replay_callback(
                                record["type"],
                                record["data"]
                            )
                            
                            replayed += 1
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"EventBuffer corrupt line: {e}")
                            continue
            
            self.events_replayed = replayed
            
            logger.warning(
                f"EventBuffer replay complete: {replayed} events replayed"
            )
            
            # Clear buffer file after successful replay
            if replayed > 0:
                await self.clear_buffer()
            
            return replayed
        
        except Exception as e:
            logger.error(f"EventBuffer replay failed: {e}")
            return 0
    
    async def clear_buffer(self) -> None:
        """Clear buffer file (after successful replay)."""
        try:
            if self.current_file.exists():
                self.current_file.unlink()
                logger.info("EventBuffer cleared after replay")
        except Exception as e:
            logger.error(f"EventBuffer clear failed: {e}")
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        file_size_mb = 0
        if self.current_file.exists():
            file_size_mb = self.current_file.stat().st_size / (1024 * 1024)
        
        return {
            "events_written": self.events_written,
            "events_replayed": self.events_replayed,
            "file_size_mb": round(file_size_mb, 2),
            "current_file": self.current_file.name,
            "recent_events_count": len(self.recent_events),
        }
    
    def close(self) -> None:
        """Close buffer file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


# Global singleton (initialized by EventBus)
_event_buffer: Optional[EventBuffer] = None


def get_event_buffer() -> EventBuffer:
    """Get global EventBuffer instance."""
    global _event_buffer
    if _event_buffer is None:
        _event_buffer = EventBuffer()
    return _event_buffer
