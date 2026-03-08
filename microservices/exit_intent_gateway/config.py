"""config: read ENV variables into a frozen GatewayConfig dataclass.

Environment Variables
---------------------
REDIS_HOST                         default: 127.0.0.1
REDIS_PORT                         default: 6379
TESTNET_MODE                       REQUIRED — must be "true"
EXIT_GATEWAY_ENABLED               default: false
EXIT_GATEWAY_INTENT_STREAM         default: quantum:stream:exit.intent
EXIT_GATEWAY_TRADE_STREAM          default: quantum:stream:trade.intent
EXIT_GATEWAY_REJECTED_STREAM       default: quantum:stream:exit.intent.rejected
EXIT_GATEWAY_GROUP                 default: exit-intent-gateway
EXIT_GATEWAY_CONSUMER              default: gateway-1
EXIT_GATEWAY_STALE_SEC             default: 60
EXIT_GATEWAY_DEDUP_TTL_SEC         default: 300
EXIT_GATEWAY_COOLDOWN_SEC          default: 90
EXIT_GATEWAY_RATE_LIMIT            default: 10 (per 60s window)
EXIT_GATEWAY_LOG_LEVEL             default: INFO

Security constraints
--------------------
TESTNET_MODE must be "true" — gateway will hard-abort at startup if not set.
EXIT_GATEWAY_ENABLED defaults to false — gateway runs in inert/standby mode
  until explicitly enabled via env override.

PATCH-5B: gateway is intentionally fail-closed. Any configuration error
or missing TESTNET_MODE will abort immediately without processing records.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("exit_intent_gateway.config")

_INTENT_STREAM_DEFAULT = "quantum:stream:exit.intent"
_TRADE_STREAM_DEFAULT = "quantum:stream:trade.intent"
_REJECTED_STREAM_DEFAULT = "quantum:stream:exit.intent.rejected"


@dataclass(frozen=True)
class GatewayConfig:
    redis_host: str
    redis_port: int

    # TESTNET hard-guard: always read from env; gateway aborts if not "true".
    testnet_mode: str  # raw string — comparison is done as lowercase == "true"

    # Feature flag — default false (inert until explicitly enabled).
    enabled: bool

    # Stream names.
    intent_stream: str    # source: quantum:stream:exit.intent
    trade_stream: str     # sink:   quantum:stream:trade.intent
    rejected_stream: str  # audit:  quantum:stream:exit.intent.rejected

    # Consumer group settings.
    group: str
    consumer: str

    # Validation thresholds.
    stale_sec: int          # reject intents older than this many seconds
    dedup_ttl_sec: int      # Redis TTL for dedup keys (quantum:exit_gw:dedup:{id})
    cooldown_sec: int       # Redis TTL for per-symbol cooldown keys
    rate_limit: int         # max intents published per 60-second window

    log_level: str

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        testnet_mode = os.getenv("TESTNET_MODE", "false").lower().strip()
        if testnet_mode != "true":
            _log.critical(
                "GATEWAY_TESTNET_REQUIRED_ABORT — TESTNET_MODE=%r is not 'true'. "
                "Exit intent gateway MUST run in testnet mode. Refusing to start.",
                testnet_mode,
            )
            # Abort is raised here so tests can catch it; main.py also checks.
            raise RuntimeError(
                f"TESTNET_MODE={testnet_mode!r} — gateway requires TESTNET_MODE=true"
            )

        enabled = os.getenv("EXIT_GATEWAY_ENABLED", "false").lower() == "true"
        if enabled:
            _log.warning(
                "EXIT_GATEWAY_ENABLED=true — gateway will process and forward intents "
                "to trade.intent. Ensure PATCH-5A live writes are also enabled."
            )
        else:
            _log.info(
                "EXIT_GATEWAY_ENABLED=false — gateway running in standby/inert mode. "
                "Messages will be consumed and ACK'd but not forwarded."
            )

        return cls(
            redis_host=os.getenv("REDIS_HOST", "127.0.0.1"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            testnet_mode=testnet_mode,
            enabled=enabled,
            intent_stream=os.getenv("EXIT_GATEWAY_INTENT_STREAM", _INTENT_STREAM_DEFAULT),
            trade_stream=os.getenv("EXIT_GATEWAY_TRADE_STREAM", _TRADE_STREAM_DEFAULT),
            rejected_stream=os.getenv("EXIT_GATEWAY_REJECTED_STREAM", _REJECTED_STREAM_DEFAULT),
            group=os.getenv("EXIT_GATEWAY_GROUP", "exit-intent-gateway"),
            consumer=os.getenv("EXIT_GATEWAY_CONSUMER", "gateway-1"),
            stale_sec=int(os.getenv("EXIT_GATEWAY_STALE_SEC", "60")),
            dedup_ttl_sec=int(os.getenv("EXIT_GATEWAY_DEDUP_TTL_SEC", "300")),
            cooldown_sec=int(os.getenv("EXIT_GATEWAY_COOLDOWN_SEC", "90")),
            rate_limit=int(os.getenv("EXIT_GATEWAY_RATE_LIMIT", "10")),
            log_level=os.getenv("EXIT_GATEWAY_LOG_LEVEL", "INFO"),
        )
