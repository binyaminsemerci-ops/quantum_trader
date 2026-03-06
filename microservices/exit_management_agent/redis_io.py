"""redis_io: async Redis client with strict write-guard.

Write permission model
----------------------
XADD  — only to streams in _ALLOWED_WRITE_STREAMS
SET    — only to keys with prefix _ALLOWED_SET_KEYS_PREFIX

Any attempt to write outside these boundaries raises RuntimeError immediately,
before any network call is made.  This is a defence-in-depth measure; the agent
itself should never call these methods with forbidden targets.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

_log = logging.getLogger("exit_management_agent.redis_io")

# ── WRITE ALLOWLIST ────────────────────────────────────────────────────────────
# Exhaustive; must be reviewed and explicitly extended when new streams are added.
_ALLOWED_WRITE_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:exit.audit",
        "quantum:stream:exit.metrics",
    }
)

# Only keys with this prefix may be SET.
_ALLOWED_SET_KEYS_PREFIX: str = "quantum:exit_agent:"

# Explicitly listed forbidden streams for defence-in-depth (belt-and-suspenders).
_FORBIDDEN_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:apply.plan",
        "quantum:stream:trade.intent",
        "quantum:stream:harvest.intent",
        "quantum:stream:exit.intent",
        "quantum:stream:harvest.suggestions",
    }
)


class RedisClient:
    """
    Async Redis client for exit_management_agent.

    READ operations are unrestricted.
    WRITE operations are guarded — only the audit/metrics streams and the
    heartbeat key prefix are allowed.  All other writes raise RuntimeError
    without touching the network.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._client = aioredis.Redis(
            host=self._host,
            port=self._port,
            decode_responses=True,
        )
        await self._client.ping()
        _log.info("Redis connected: %s:%d", self._host, self._port)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── READ OPERATIONS ────────────────────────────────────────────────────────

    async def scan_position_keys(
        self,
        match: str = "quantum:position:*",
        batch: int = 200,
    ) -> list:
        """
        SCAN for position hash keys.
        Excludes :snapshot: and :ledger: sub-keys (used by other services).
        Returns at most `batch` keys.
        """
        keys: list = []
        cursor = 0
        while True:
            cursor, batch_keys = await self._client.scan(
                cursor=cursor, match=match, count=batch
            )
            keys.extend(
                k
                for k in batch_keys
                if ":snapshot:" not in k and ":ledger:" not in k
            )
            if cursor == 0 or len(keys) >= batch:
                break
        return keys

    async def hgetall_position(self, key: str) -> dict:
        """Read all fields from a position hash. Returns empty dict if missing."""
        return await self._client.hgetall(key)

    async def get_mark_price_from_ticker(self, symbol: str) -> Optional[float]:
        """
        Try quantum:ticker:{symbol} hash for a fresh mark price.
        Checks both 'markPrice' and 'mark_price' field names.
        Returns None if the key is absent or the price cannot be parsed.
        """
        data = await self._client.hgetall(f"quantum:ticker:{symbol}")
        if not data:
            return None
        raw = data.get("markPrice") or data.get("mark_price")
        if raw:
            try:
                price = float(raw)
                return price if price > 0.0 else None
            except (ValueError, TypeError):
                pass
        return None

    # ── WRITE OPERATIONS (guarded) ─────────────────────────────────────────────

    async def xadd(self, stream: str, fields: dict) -> None:
        """
        Append an entry to a Redis stream.

        Raises RuntimeError immediately (no network call) if:
          - stream is in _FORBIDDEN_STREAMS, OR
          - stream is not in _ALLOWED_WRITE_STREAMS
        """
        if stream in _FORBIDDEN_STREAMS:
            raise RuntimeError(
                f"[WRITE_GUARD] Attempt to write to categorically forbidden stream: {stream}"
            )
        if stream not in _ALLOWED_WRITE_STREAMS:
            raise RuntimeError(
                f"[WRITE_GUARD] Stream not on allowlist: {stream!r}. "
                f"Allowed streams: {sorted(_ALLOWED_WRITE_STREAMS)}"
            )
        await self._client.xadd(stream, fields)

    async def set_with_ttl(self, key: str, value: str, ttl_sec: int) -> None:
        """
        SET key with EX ttl.

        Raises RuntimeError immediately if key does not start with
        _ALLOWED_SET_KEYS_PREFIX  ("quantum:exit_agent:").
        """
        if not key.startswith(_ALLOWED_SET_KEYS_PREFIX):
            raise RuntimeError(
                f"[WRITE_GUARD] SET target outside allowed namespace: {key!r}. "
                f"Allowed prefix: {_ALLOWED_SET_KEYS_PREFIX!r}"
            )
        await self._client.set(key, value, ex=ttl_sec)
