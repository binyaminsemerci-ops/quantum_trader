"""ownership_flag: write quantum:exit_agent:active_flag with TTL each tick.

PATCH-6: activates the exit-ownership kill-switch already deployed in
AutonomousTrader (PATCH-2).  When EMA is healthy and ownership transfer
is enabled, this module writes a short-TTL key each tick.  AutonomousTrader
reads that key and suspends its own _monitor_positions() loop, making EMA
the sole active exit-owner for open positions.

Key:   quantum:exit_agent:active_flag   (AgentConfig.active_flag_key)
Value: "PATCH-6"
TTL:   AgentConfig.active_flag_ttl_sec  (default 30s, ~6× the tick interval)

Fail-safe: if EMA dies or encounters a sustained Redis error, the key
expires within active_flag_ttl_sec seconds and AutonomousTrader resumes
exit evaluation automatically. PATCH-2 is fail-open: it returns False
(not suspended) when the key is absent or Redis is unreachable.

Guards: this module writes ONLY when BOTH conditions hold:
  1. ownership_transfer_enabled=True  (EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=true)
  2. testnet_mode == "true"           (EXIT_AGENT_TESTNET_MODE=true)

The testnet guard prevents accidental exit-ownership seizure on a live
account before the operator explicitly enables it during production review.
"""
from __future__ import annotations

import logging

from .redis_io import RedisClient

_log = logging.getLogger("exit_management_agent.ownership_flag")

# Value written to the flag key; PATCH identifier for auditability.
_FLAG_VALUE: str = "PATCH-6"


class OwnershipFlagWriter:
    """
    Write / refresh the exit-ownership active flag each tick.

    When both ``enabled`` is True *and* ``testnet_mode`` resolves to ``"true"``,
    calls ``redis.set_with_ttl()`` to keep the flag alive.  The flag expires
    automatically when EMA stops running, allowing AutonomousTrader to resume.

    When disabled OR on non-testnet: a DEBUG log is emitted and ``write()``
    returns False without touching Redis.
    """

    def __init__(
        self,
        redis: RedisClient,
        enabled: bool,
        flag_key: str,
        ttl_sec: int,
        testnet_mode: str,
    ) -> None:
        self._redis = redis
        self._enabled = enabled
        self._flag_key = flag_key
        self._ttl_sec = ttl_sec
        self._testnet_mode = testnet_mode.lower().strip()
        # Track whether we've already emitted the one-time WARNING at startup
        # so that routine per-tick renewals are logged at INFO (not WARNING).
        # This prevents ~17,280 WARNING lines/day drowning the journal.
        self._started: bool = False

    async def write(self) -> bool:
        """
        Refresh the exit-ownership flag in Redis.

        Returns True if the flag was written, False if skipped (disabled or
        non-testnet).  Redis errors are swallowed with an ERROR log so that
        a transient Redis blip never aborts the main tick loop.  When Redis
        errors prevent renewal, the existing key expires after active_flag_ttl_sec
        and AutonomousTrader automatically resumes (PATCH-2 fail-open design).
        """
        if not self._enabled:
            _log.debug(
                "OWNERSHIP_FLAG_SKIP reason=disabled key=%s", self._flag_key
            )
            return False

        if self._testnet_mode != "true":
            _log.debug(
                "OWNERSHIP_FLAG_SKIP reason=not_testnet testnet_mode=%s key=%s",
                self._testnet_mode,
                self._flag_key,
            )
            return False

        try:
            await self._redis.set_with_ttl(self._flag_key, _FLAG_VALUE, self._ttl_sec)
            if not self._started:
                # First successful write after startup — emit WARNING once for
                # journalctl auditability without flooding on every subsequent tick.
                _log.warning(
                    "OWNERSHIP_FLAG_ACTIVE key=%s value=%s ttl=%d patch=PATCH-6 "
                    "action=AT_exit_suspended",
                    self._flag_key,
                    _FLAG_VALUE,
                    self._ttl_sec,
                )
                self._started = True
            else:
                _log.info(
                    "OWNERSHIP_FLAG_SET key=%s ttl=%d patch=PATCH-6",
                    self._flag_key,
                    self._ttl_sec,
                )
            return True
        except Exception as exc:
            # Reset _started so that recovery after an error gap emits WARNING.
            self._started = False
            _log.error(
                "OWNERSHIP_FLAG_WRITE_ERROR key=%s error=%s patch=PATCH-6",
                self._flag_key,
                exc,
            )
            return False
