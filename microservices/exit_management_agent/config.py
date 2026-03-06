"""config: read ENV variables into a frozen AgentConfig dataclass.

Environment Variables
---------------------
REDIS_HOST                        default: 127.0.0.1
REDIS_PORT                        default: 6379
EXIT_AGENT_ENABLED                default: true
EXIT_AGENT_LOOP_SEC               default: 5.0
EXIT_AGENT_HEARTBEAT_KEY          default: quantum:exit_agent:heartbeat
EXIT_AGENT_HEARTBEAT_TTL_SEC      default: 60
EXIT_AGENT_AUDIT_STREAM           default: quantum:stream:exit.audit
EXIT_AGENT_METRICS_STREAM         default: quantum:stream:exit.metrics
EXIT_AGENT_LOG_LEVEL              default: INFO
EXIT_AGENT_DRY_RUN                default: true  (PATCH-1: always True in code)
EXIT_AGENT_SYMBOL_ALLOWLIST       default: ""    (empty = all symbols)
EXIT_AGENT_MAX_POSITIONS_PER_LOOP default: 50
EXIT_AGENT_MAX_HOLD_SEC           default: 14400 (4 hours)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("exit_management_agent.config")


@dataclass(frozen=True)
class AgentConfig:
    redis_host: str
    redis_port: int
    enabled: bool
    loop_sec: float
    heartbeat_key: str
    heartbeat_ttl_sec: int
    audit_stream: str
    metrics_stream: str
    log_level: str
    # PATCH-1: dry_run is ALWAYS True regardless of env value.
    # The field exists so later patches can unlock it via code change + review.
    dry_run: bool
    symbol_allowlist: frozenset  # frozenset[str]; empty = allow all
    max_positions_per_loop: int
    max_hold_sec: float

    @classmethod
    def from_env(cls) -> "AgentConfig":
        sym_raw = os.getenv("EXIT_AGENT_SYMBOL_ALLOWLIST", "").strip()
        allowlist: frozenset = frozenset(
            s.strip().upper() for s in sym_raw.split(",") if s.strip()
        )

        # Warn if someone tries to disable dry-run — we block it in PATCH-1.
        env_dry_run = os.getenv("EXIT_AGENT_DRY_RUN", "true").lower()
        if env_dry_run != "true":
            _log.warning(
                "EXIT_AGENT_DRY_RUN=%s ignored — PATCH-1 is always shadow-only. "
                "DRY_RUN is hard-coded True until PATCH-5 is reviewed and merged.",
                env_dry_run,
            )

        return cls(
            redis_host=os.getenv("REDIS_HOST", "127.0.0.1"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            enabled=os.getenv("EXIT_AGENT_ENABLED", "true").lower() == "true",
            loop_sec=float(os.getenv("EXIT_AGENT_LOOP_SEC", "5.0")),
            heartbeat_key=os.getenv(
                "EXIT_AGENT_HEARTBEAT_KEY", "quantum:exit_agent:heartbeat"
            ),
            heartbeat_ttl_sec=int(os.getenv("EXIT_AGENT_HEARTBEAT_TTL_SEC", "60")),
            audit_stream=os.getenv(
                "EXIT_AGENT_AUDIT_STREAM", "quantum:stream:exit.audit"
            ),
            metrics_stream=os.getenv(
                "EXIT_AGENT_METRICS_STREAM", "quantum:stream:exit.metrics"
            ),
            log_level=os.getenv("EXIT_AGENT_LOG_LEVEL", "INFO"),
            dry_run=True,  # PATCH-1 hard-coded — never read from env here.
            symbol_allowlist=allowlist,
            max_positions_per_loop=int(
                os.getenv("EXIT_AGENT_MAX_POSITIONS_PER_LOOP", "50")
            ),
            max_hold_sec=float(os.getenv("EXIT_AGENT_MAX_HOLD_SEC", "14400")),
        )

    def is_symbol_allowed(self, symbol: str) -> bool:
        if not self.symbol_allowlist:
            return True
        return symbol.upper() in self.symbol_allowlist
