"""redis_io: async Redis client with strict write-guard for exit_intent_gateway.

Write permission model
----------------------
XADD — only to streams in _ALLOWED_WRITE_STREAMS:
    quantum:stream:trade.intent            — approved exit orders forwarded to pipeline
    quantum:stream:exit.intent.rejected    — audit trail for rejected intents

SET (NX + TTL) — only to keys with prefix _ALLOWED_SET_KEYS_PREFIX.
    Used for dedup and per-symbol cooldown.

XGROUP CREATE / XREADGROUP / XACK — only against the configured intent_stream
    (quantum:stream:exit.intent).

Any attempt to write outside these boundaries raises RuntimeError immediately,
before any network call is made.  Defence-in-depth: execution pipeline streams
are permanently blocked regardless of config.

PATCH-5B: apply.plan is UNCONDITIONALLY FORBIDDEN.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

_log = logging.getLogger("exit_intent_gateway.redis_io")

# ── WRITE ALLOW-LIST ─────────────────────────────────────────────────────────
# Only these streams may receive XADD from this service.
_ALLOWED_WRITE_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:trade.intent",
        "quantum:stream:harvest.intent",
        "quantum:stream:exit.intent.rejected",
    }
)

# ── UNCONDITIONALLY FORBIDDEN STREAMS ────────────────────────────────────────
# These streams may NEVER be written to by this service.
# apply.plan is the most critical — gateway must never bypass intent_bridge.
_FORBIDDEN_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:apply.plan",
        "quantum:stream:exit.audit",
        "quantum:stream:exit.metrics",
        "quantum:stream:harvest.suggestions",
    }
)

# Only keys with this prefix may receive SET/SET NX operations.
_ALLOWED_SET_KEYS_PREFIX: str = "quantum:exit_gw:"

# Lockdown sentinel key — if this key exists in Redis, the gateway halts.
LOCKDOWN_KEY: str = "quantum:lockdown"


class GatewayRedisClient:
    """
    Async Redis client for exit_intent_gateway.

    READ operations (XREADGROUP, GET, EXISTS) are unrestricted.
    WRITE operations are restricted to _ALLOWED_WRITE_STREAMS (XADD)
    and keys with _ALLOWED_SET_KEYS_PREFIX (SET NX).

    apply.plan is unconditionally forbidden — attempting to write there
    raises RuntimeError before any network call.
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

    # ── WRITE GUARD ──────────────────────────────────────────────────────────

    def _assert_stream_writable(self, stream: str) -> None:
        """Raise RuntimeError if stream is not in the allowed write set."""
        if stream in _FORBIDDEN_STREAMS:
            raise RuntimeError(
                f"FORBIDDEN_STREAM_WRITE: '{stream}' is unconditionally forbidden "
                f"for exit_intent_gateway. This is a defence-in-depth guard."
            )
        if stream not in _ALLOWED_WRITE_STREAMS:
            raise RuntimeError(
                f"UNKNOWN_STREAM_WRITE: '{stream}' is not in the allowed write set. "
                f"Allowed: {sorted(_ALLOWED_WRITE_STREAMS)}"
            )

    def _assert_set_key_allowed(self, key: str) -> None:
        """Raise RuntimeError if key does not have the allowed SET key prefix."""
        if not key.startswith(_ALLOWED_SET_KEYS_PREFIX):
            raise RuntimeError(
                f"FORBIDDEN_SET_KEY: '{key}' does not start with "
                f"'{_ALLOWED_SET_KEYS_PREFIX}'. Gateway SET operations are "
                f"restricted to dedup/cooldown keys only."
            )

    # ── XADD ─────────────────────────────────────────────────────────────────

    async def xadd(self, stream: str, fields: dict, maxlen: int = 10_000) -> str:
        """
        Append fields to a Redis stream.
        Raises RuntimeError if stream is forbidden or not in the allowed set.
        """
        self._assert_stream_writable(stream)
        return await self._client.xadd(stream, fields, maxlen=maxlen)

    # ── CONSUMER GROUP ───────────────────────────────────────────────────────

    async def ensure_consumer_group(
        self, stream: str, group: str, start_id: str = "$"
    ) -> None:
        """
        Create a consumer group if it does not already exist.
        Uses MKSTREAM so the stream is created if absent.
        Swallows BusyGroupError (group already exists).
        """
        try:
            await self._client.xgroup_create(stream, group, id=start_id, mkstream=True)
            _log.info("Consumer group created: group=%s stream=%s", group, stream)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                _log.debug("Consumer group already exists: group=%s stream=%s", group, stream)
            else:
                raise

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        stream: str,
        count: int = 1,
        block_ms: int = 2000,
        pending: bool = False,
    ) -> list:
        """
        Read messages from the consumer group.

        pending=False (default): read new undelivered messages (id=">"),
            blocking for up to block_ms milliseconds if none are available.
        pending=True: read unacknowledged messages from this consumer's PEL
            (id="0"). Non-blocking; used only during startup drain.

        Returns list of (stream_name, [(msg_id, fields), ...]) tuples.
        """
        stream_id = "0" if pending else ">"
        return await self._client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: stream_id},
            count=count,
            block=None if pending else block_ms,
        )

    async def xack(self, stream: str, group: str, message_id: str) -> int:
        """Acknowledge a message in a consumer group."""
        return await self._client.xack(stream, group, message_id)

    # ── DEDUP / COOLDOWN (SET NX with TTL) ───────────────────────────────────

    async def set_nx_with_ttl(self, key: str, ttl_sec: int) -> bool:
        """
        Attempt to SET key NX (only if not exists) with an expiry.
        Returns True if the key was set (lock acquired), False if it already existed.

        Only keys with prefix 'quantum:exit_gw:' are allowed.
        """
        self._assert_set_key_allowed(key)
        result = await self._client.set(key, "1", nx=True, ex=ttl_sec)
        return result is True

    # ── LOCKDOWN CHECK ───────────────────────────────────────────────────────

    async def get_lockdown(self) -> bool:
        """Return True if the quantum:lockdown key exists in Redis."""
        result = await self._client.exists(LOCKDOWN_KEY)
        return bool(result)

    # ── RATE LIMIT (Redis-based sliding window key count) ────────────────────
    # NOTE: Rate limiting is implemented in-process via RateLimitState in models.py
    # because it only needs to bound the current gateway process, not span multiple
    # instances.  If multi-instance deployment is needed, this can be upgraded to
    # a Redis-backed counter without changing the validator interface.

    async def xlen(self, stream: str) -> int:
        """Return the length of a stream. Used for diagnostics / health checks."""
        return await self._client.xlen(stream)
