"""
SPRINT 1 - D2: Disk Buffer for EventBus Fallback

Provides at-least-once delivery during Redis outages by buffering
events to local disk in JSONL format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DiskBuffer:
    """
    Local disk buffer for event persistence during Redis outages.
    
    Features:
    - JSONL format (one event per line)
    - Append-only writes
    - Ordered replay by timestamp
    - Atomic file operations
    
    File structure:
        runtime/eventbus_buffer/YYYY-MM-DD_HH-MM-SS.jsonl
    """
    
    def __init__(self, buffer_dir: str = "runtime/eventbus_buffer"):
        """
        Initialize disk buffer.
        
        Args:
            buffer_dir: Directory for buffer files
        """
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        
        # Current buffer file (rotates daily or after replay)
        self._current_file: Optional[Path] = None
        self._ensure_buffer_file()
        
        logger.info(f"DiskBuffer initialized: {self.buffer_dir}")
    
    def _ensure_buffer_file(self) -> Path:
        """Ensure current buffer file exists."""
        if self._current_file and self._current_file.exists():
            return self._current_file
        
        # Create new buffer file with timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self._current_file = self.buffer_dir / f"buffer_{timestamp}.jsonl"
        self._current_file.touch(exist_ok=True)
        
        logger.info(f"New buffer file: {self._current_file}")
        return self._current_file
    
    def write(self, event_type: str, message: dict) -> bool:
        """
        Write event to disk buffer.
        
        Args:
            event_type: Event type identifier
            message: Event message dict (must be JSON-serializable)
        
        Returns:
            True if write successful, False otherwise
        """
        try:
            buffer_entry = {
                "event_type": event_type,
                "message": message,
                "buffered_at": datetime.utcnow().isoformat()
            }
            
            # Append to buffer file (atomic write)
            buffer_file = self._ensure_buffer_file()
            with open(buffer_file, "a") as f:
                f.write(json.dumps(buffer_entry) + "\n")
                f.flush()  # Ensure write to disk
            
            logger.warning(
                f"📁 Event buffered to disk: {event_type} → {buffer_file.name}"
            )
            return True
        
        except Exception as e:
            logger.critical(
                f"❌ CRITICAL: Failed to buffer event to disk: {e}. "
                f"Event LOST: {event_type}"
            )
            return False
    
    def read_all(self) -> list[dict]:
        """
        Read all buffered events from all buffer files.
        
        Returns:
            List of buffered event dicts, sorted by buffered_at timestamp
        """
        buffered_events = []
        
        # Read all .jsonl files in buffer directory
        for buffer_file in sorted(self.buffer_dir.glob("*.jsonl")):
            try:
                with open(buffer_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            entry = json.loads(line.strip())
                            buffered_events.append(entry)
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse line {line_num} in {buffer_file.name}: {e}"
                            )
            except Exception as e:
                logger.error(f"Failed to read buffer file {buffer_file.name}: {e}")
        
        # Sort by timestamp to preserve order
        buffered_events.sort(key=lambda e: e.get("buffered_at", ""))
        
        logger.info(f"📖 Read {len(buffered_events)} buffered events from disk")
        return buffered_events
    
    def clear(self) -> int:
        """
        Clear all buffer files after successful replay.
        
        Returns:
            Number of files deleted
        """
        deleted = 0
        
        for buffer_file in self.buffer_dir.glob("*.jsonl"):
            try:
                buffer_file.unlink()
                deleted += 1
                logger.info(f"🗑️  Deleted buffer file: {buffer_file.name}")
            except Exception as e:
                logger.error(f"Failed to delete buffer file {buffer_file.name}: {e}")
        
        # Reset current file
        self._current_file = None
        
        logger.info(f"🧹 Cleared {deleted} buffer files")
        return deleted
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dict with buffer stats (file_count, total_events, oldest_event)
        """
        files = list(self.buffer_dir.glob("*.jsonl"))
        
        total_events = 0
        oldest_event = None
        
        for buffer_file in files:
            try:
                with open(buffer_file, "r") as f:
                    for line in f:
                        total_events += 1
                        try:
                            entry = json.loads(line.strip())
                            timestamp = entry.get("buffered_at")
                            if oldest_event is None or timestamp < oldest_event:
                                oldest_event = timestamp
                        except:
                            pass
            except:
                pass
        
        return {
            "file_count": len(files),
            "total_events": total_events,
            "oldest_event": oldest_event,
            "buffer_dir": str(self.buffer_dir),
        }
