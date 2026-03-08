"""redis_io: async Redis client with strict write-guard.

Write permission model
----------------------
XADD  — only to streams in the effective allowed set:
          _AUDIT_STREAMS          always (exit.audit, exit.metrics)
          exit.intent             only when live_writes_enabled=True at construction
SET    — only to keys with prefix _ALLOWED_SET_KEYS_PREFIX

Any attempt to write outside these boundaries raises RuntimeError immediately,
before any network call is made.  This is a defence-in-depth measure; the agent
itself should never call these methods with forbidden targets.

PATCH-5A change: quantum:stream:exit.intent is moved OUT of _FORBIDDEN_STREAMS
and into a conditional allowed set.  trade.intent and apply.plan remain
unconditionally forbidden so no misconfiguration can route through the execution
pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

import redis.asyncio as aioredis

_log = logging.getLogger("exit_management_agent.redis_io")

# ── IMMUTABLE AUDIT STREAMS ───────────────────────────────────────────────────
# Always writable regardless of live_writes_enabled.
_AUDIT_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:exit.audit",
        "quantum:stream:exit.metrics",
    }
)

# ── LIVE INTENT STREAM (PATCH-5A) ─────────────────────────────────────────────
# Allowed only when live_writes_enabled=True is passed to RedisClient.__init__.
_LIVE_INTENT_STREAM: str = "quantum:stream:exit.intent"

# Only keys with this prefix may be SET.
_ALLOWED_SET_KEYS_PREFIX: str = "quantum:exit_agent:"

# Backwards-compatibility alias: audit.py imports this name directly.
# It refers to the streams that are unconditionally (statically) allowed for
# audit writes, which is what audit.py needs to validate at construction time.
_ALLOWED_WRITE_STREAMS: frozenset = _AUDIT_STREAMS

# ── UNCONDITIONALLY FORBIDDEN STREAMS ─────────────────────────────────────────
# These streams may NEVER be written to by this agent, regardless of config.
# Defence-in-depth: execution pipeline streams are permanently blocked here.
#
# NOTE: quantum:stream:exit.intent is intentionally absent — it is
# conditionally allowed via _LIVE_INTENT_STREAM above.
_FORBIDDEN_STREAMS: frozenset = frozenset(
    {
        "quantum:stream:apply.plan",
        "quantum:stream:trade.intent",
        "quantum:stream:harvest.intent",
        "quantum:stream:harvest.suggestions",
    }
)


class RedisClient:
    """
    Async Redis client for exit_management_agent.

    READ operations are unrestricted.
    WRITE operations depend on live_writes_enabled:

      live_writes_enabled=False (default, PATCH-1 behaviour):
        Allowed XADD targets: exit.audit, exit.metrics
        Forbidden (all others, including exit.intent)

      live_writes_enabled=True (PATCH-5A):
        Allowed XADD targets: exit.audit, exit.metrics, exit.intent
        Forbidden: apply.plan, trade.intent, harvest.intent, harvest.suggestions

    In both cases, any write to an unrecognised stream raises RuntimeError
    without touching the network.
    """

    def __init__(self, host: str, port: int, live_writes_enabled: bool = False) -> None:
        self._host = host
        self._port = port
        self._client: Optional[aioredis.Redis] = None
        self._live_writes_enabled = live_writes_enabled

        # Build the effective allowed set once at construction time.
        if live_writes_enabled:
            self._allowed_write_streams: frozenset = _AUDIT_STREAMS | frozenset(
                {_LIVE_INTENT_STREAM}
            )
            _log.info(
                "RedisClient: live_writes_enabled=True — "
                "%s added to allowed write streams",
                _LIVE_INTENT_STREAM,
            )
        else:
            # Shadow mode: exit.intent is also effectively forbidden.
            self._allowed_write_streams = _AUDIT_STREAMS

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
          - stream is in _FORBIDDEN_STREAMS (unconditionally blocked), OR
          - stream is exit.intent AND live_writes_enabled=False, OR
          - stream is not in the effective allowed set
        """
        # Unconditional hard block (trade.intent, apply.plan, etc.)
        if stream in _FORBIDDEN_STREAMS:
            raise RuntimeError(
                f"[WRITE_GUARD] Attempt to write to categorically forbidden stream: {stream}"
            )
        # Conditional block for exit.intent when live writes are off.
        if stream == _LIVE_INTENT_STREAM and not self._live_writes_enabled:
            raise RuntimeError(
                f"[WRITE_GUARD] Attempt to write to {stream!r} while "
                "live_writes_enabled=False. Set EXIT_AGENT_LIVE_WRITES_ENABLED=true "
                "and restart the agent to enable PATCH-5A writes."
            )
        # Final allowlist check catches any other unrecognised stream.
        if stream not in self._allowed_write_streams:
            raise RuntimeError(
                f"[WRITE_GUARD] Stream not on allowlist: {stream!r}. "
                f"Allowed streams: {sorted(self._allowed_write_streams)}"
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
