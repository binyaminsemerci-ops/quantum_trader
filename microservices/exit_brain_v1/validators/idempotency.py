"""
Idempotency — In-memory duplicate intent detection.

Tracks recently seen idempotency keys within a configurable time window.
Used by: exit_intent_gateway_validator.py

No Redis. No IO. Pure in-process state.
"""

from __future__ import annotations

import time
from collections import OrderedDict

# Default window: 5 minutes
DEFAULT_WINDOW_SEC = 300.0

# Maximum entries to prevent unbounded memory growth
MAX_ENTRIES = 10_000


class IdempotencyTracker:
    """
    In-memory tracker of recently seen idempotency keys.

    Keys older than window_sec are evicted on every check.
    """

    def __init__(self, window_sec: float = DEFAULT_WINDOW_SEC) -> None:
        self._window_sec = window_sec
        self._seen: OrderedDict[str, float] = OrderedDict()

    def is_duplicate(self, key: str) -> bool:
        """
        Check if this key was seen within the window.

        Returns True if duplicate, False if new.
        Side effect: registers the key if new.
        """
        self._evict_expired()

        if key in self._seen:
            return True

        self._seen[key] = time.time()

        # Bound memory
        while len(self._seen) > MAX_ENTRIES:
            self._seen.popitem(last=False)

        return False

    def _evict_expired(self) -> None:
        """Remove keys older than window_sec."""
        cutoff = time.time() - self._window_sec
        while self._seen:
            oldest_key, oldest_ts = next(iter(self._seen.items()))
            if oldest_ts < cutoff:
                self._seen.popitem(last=False)
            else:
                break

    def clear(self) -> None:
        """Clear all tracked keys."""
        self._seen.clear()

    @property
    def size(self) -> int:
        return len(self._seen)
