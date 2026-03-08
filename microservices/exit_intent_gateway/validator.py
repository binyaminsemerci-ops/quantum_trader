"""validator: stateful gateway validator for exit_intent_gateway (PATCH-5B).

Validation Rules
----------------
V1  TESTNET_REQUIRED    — config.testnet_mode must be "true"
V2  GATEWAY_ENABLED     — config.enabled must be True
V3  LOCKDOWN            — quantum:lockdown key must not exist in Redis
V4  STALE               — ts_epoch must be within stale_sec of now
V5  DEDUP               — intent_id must not already have a dedup key in Redis
V6  COOLDOWN            — symbol must not be in cooldown
V7  ACTION_WHITELIST    — action must be in GATEWAY_ACTION_WHITELIST
V8  SOURCE_CHECK        — source must be "exit_management_agent"
V9  RATE_LIMIT          — in-process sliding-window rate limit

All checks are async (V3/V5/V6 need Redis; others are synchronous).
Checks run in order; the first failure short-circuits the remainder.
V5 and V6 are side-effecting: they SET dedup/cooldown keys on pass.

Design principle: fail-closed.  Any unexpected exception in validation
is treated as a rejection (VALIDATION_ERROR rule) with error logged.
"""
from __future__ import annotations

import logging
import time

from .config import GatewayConfig
from .models import (
    GATEWAY_ACTION_WHITELIST,
    EXPECTED_SOURCE,
    GatewayValidationResult,
    IntentMessage,
    RateLimitState,
)
from .redis_io import GatewayRedisClient

_log = logging.getLogger("exit_intent_gateway.validator")


class GatewayValidator:
    """
    Async validator that runs 9 sequential checks against an IntentMessage.

    Construct once per gateway instance; share across ticks.
    The RateLimitState is in-process (not Redis-backed) and accumulates
    over the lifetime of the validator.
    """

    def __init__(
        self,
        config: GatewayConfig,
        redis: GatewayRedisClient,
    ) -> None:
        self._cfg = config
        self._redis = redis
        self._rate_state = RateLimitState(window_sec=60.0)

    async def validate(self, msg: IntentMessage) -> GatewayValidationResult:
        """
        Run all validation checks in order.

        Returns GatewayValidationResult with passed=True only if all checks pass.
        V5 (dedup) and V6 (cooldown) SET keys on pass; if a later check fails
        those keys are left set (conservative — prevents retry storms).
        """
        try:
            return await self._run_checks(msg)
        except Exception as exc:  # pragma: no cover
            _log.error(
                "VALIDATION_ERROR intent_id=%s symbol=%s: %s",
                msg.intent_id,
                msg.symbol,
                exc,
                exc_info=True,
            )
            return GatewayValidationResult(
                passed=False,
                rule="VALIDATION_ERROR",
                reason=f"Unexpected validation error: {exc}",
                intent_id=msg.intent_id,
                symbol=msg.symbol,
            )

    async def _run_checks(self, msg: IntentMessage) -> GatewayValidationResult:
        def _fail(rule: str, reason: str) -> GatewayValidationResult:
            return GatewayValidationResult(
                passed=False,
                rule=rule,
                reason=reason,
                intent_id=msg.intent_id,
                symbol=msg.symbol,
            )

        def _pass() -> GatewayValidationResult:
            return GatewayValidationResult(
                passed=True,
                rule="PASS",
                reason="All validation checks passed",
                intent_id=msg.intent_id,
                symbol=msg.symbol,
            )

        # V1 — testnet mode hard check.
        if self._cfg.testnet_mode != "true":
            return _fail("V1_TESTNET_REQUIRED", f"testnet_mode={self._cfg.testnet_mode!r}")

        # V2 — gateway enabled flag.
        if not self._cfg.enabled:
            return _fail("V2_GATEWAY_DISABLED", "EXIT_GATEWAY_ENABLED=false")

        # V3 — lockdown key check (async Redis read).
        if await self._redis.get_lockdown():
            return _fail("V3_LOCKDOWN", "quantum:lockdown key is set — all activity halted")

        # V4 — staleness check.
        age_sec = time.time() - msg.ts_epoch
        if age_sec > self._cfg.stale_sec:
            return _fail(
                "V4_STALE",
                f"intent age {age_sec:.1f}s exceeds stale_sec={self._cfg.stale_sec}",
            )

        # V5 — dedup: SET NX with TTL on intent_id.
        dedup_key = f"quantum:exit_gw:dedup:{msg.intent_id}"
        acquired_dedup = await self._redis.set_nx_with_ttl(dedup_key, self._cfg.dedup_ttl_sec)
        if not acquired_dedup:
            return _fail(
                "V5_DEDUP",
                f"intent_id={msg.intent_id!r} already processed (duplicate)",
            )

        # V6 — per-symbol cooldown: SET NX with TTL on symbol.
        cooldown_key = f"quantum:exit_gw:cooldown:{msg.symbol}"
        acquired_cooldown = await self._redis.set_nx_with_ttl(cooldown_key, self._cfg.cooldown_sec)
        if not acquired_cooldown:
            return _fail(
                "V6_COOLDOWN",
                f"symbol={msg.symbol!r} is in cooldown (cooldown_sec={self._cfg.cooldown_sec})",
            )

        # V7 — action whitelist.
        if msg.action not in GATEWAY_ACTION_WHITELIST:
            return _fail(
                "V7_ACTION",
                f"action={msg.action!r} not in whitelist {sorted(GATEWAY_ACTION_WHITELIST)}",
            )

        # V8 — source check.
        if msg.source != EXPECTED_SOURCE:
            return _fail(
                "V8_SOURCE",
                f"source={msg.source!r} expected={EXPECTED_SOURCE!r}",
            )

        # V9 — in-process rate limit.
        if not self._rate_state.check_and_record(self._cfg.rate_limit):
            return _fail(
                "V9_RATE_LIMIT",
                f"rate limit {self._cfg.rate_limit}/60s exceeded",
            )

        return _pass()
