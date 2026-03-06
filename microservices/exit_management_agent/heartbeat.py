"""heartbeat: maintain quantum:exit_agent:heartbeat TTL key.

Writes a Unix-epoch timestamp string via RedisClient.set_with_ttl().
That method enforces the quantum:exit_agent: namespace restriction, so
no other key can accidentally be written here.

The heartbeat key is intentionally NOT a stream entry (XADD) — it is a
plain string key with an EX TTL so that downstream monitors can detect
agent absence by checking key existence rather than stream lag.
"""
from __future__ import annotations

import logging
import time

from .redis_io import RedisClient

_log = logging.getLogger("exit_management_agent.heartbeat")


class HeartbeatWriter:
    """
    Write / refresh the heartbeat key each tick.

    The key is:   AgentConfig.heartbeat_key  (default: quantum:exit_agent:heartbeat)
    The value is: Unix epoch as integer string  (e.g. "1700000000")
    The TTL is:   AgentConfig.heartbeat_ttl_sec (default: 60 seconds)

    A missing key means the agent has not run for > TTL seconds.
    """

    def __init__(
        self,
        redis: RedisClient,
        key: str,
        ttl_sec: int,
    ) -> None:
        self._redis = redis
        self._key = key
        self._ttl_sec = ttl_sec
        self._last_beat: float = 0.0

    async def beat(self) -> None:
        """
        Refresh the heartbeat key.
        Swallows Redis errors (agent should keep running even if heartbeat fails).
        """
        now = time.time()
        value = str(int(now))
        try:
            await self._redis.set_with_ttl(self._key, value, self._ttl_sec)
            self._last_beat = now
            _log.debug("Heartbeat refreshed: key=%s TTL=%ds", self._key, self._ttl_sec)
        except Exception as exc:
            _log.warning("Heartbeat write failed (key=%s): %s", self._key, exc)

    @property
    def last_beat(self) -> float:
        """Unix timestamp of the most recent successful heartbeat. 0.0 = never."""
        return self._last_beat
