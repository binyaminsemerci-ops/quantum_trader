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
EXIT_AGENT_INTENT_STREAM          default: quantum:stream:exit.intent  (PATCH-5A)
EXIT_AGENT_LOG_LEVEL              default: INFO
EXIT_AGENT_DRY_RUN                default: true  (PATCH-1: always True in code)
EXIT_AGENT_LIVE_WRITES_ENABLED            default: false (PATCH-5A: gates exit.intent writes)
EXIT_AGENT_SYMBOL_ALLOWLIST               default: ""    (empty = all symbols)
EXIT_AGENT_MAX_POSITIONS_PER_LOOP         default: 50
EXIT_AGENT_MAX_HOLD_SEC                   default: 14400 (4 hours)
EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED     default: false (PATCH-6: gates active_flag writes)
EXIT_AGENT_ACTIVE_FLAG_TTL_SEC            default: 30   (PATCH-6: ~6× tick interval; clamped [10,120])
EXIT_AGENT_TESTNET_MODE                   default: false (PATCH-6: testnet guard)
EXIT_AGENT_SCORING_MODE                   default: shadow (PATCH-7A: shadow|formula|ai)
EXIT_AGENT_QWEN3_ENDPOINT                 default: http://localhost:11434  (PATCH-7B: Ollama/OpenAI-compat)
EXIT_AGENT_QWEN3_TIMEOUT_MS               default: 2000   (PATCH-7B: clamped [200, 10000])
EXIT_AGENT_QWEN3_SHADOW                   default: true   (PATCH-7B: true=audit-only; false=Qwen3 drives live)
EXIT_AGENT_QWEN3_MODEL                    default: qwen3:8b (PATCH-7B: model tag passed to Ollama)
EXIT_AGENT_QWEN3_API_KEY                  default: "" (PATCH-7B-ext: bearer token for external endpoints; never logged)
EXIT_AGENT_DECISION_TTL_SEC               default: 14400 (PATCH-8A: TTL in seconds for quantum:hash:exit.decision:{id})

NOTE: EXIT_AGENT_ACTIVE_FLAG_KEY is intentionally NOT a configurable env var.
      The key "quantum:exit_agent:active_flag" is a binary protocol constant
      shared with the PATCH-2 reader in AutonomousTrader.  Allowing env
      override would enable silent split-brain misconfigurations.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

_log = logging.getLogger("exit_management_agent.config")

_INTENT_STREAM_DEFAULT: str = "quantum:stream:exit.intent"


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
    intent_stream: str
    log_level: str
    # PATCH-1: dry_run is ALWAYS True regardless of env value.
    # The field exists so later patches can unlock it via code change + review.
    dry_run: bool
    # PATCH-5A: live_writes_enabled gates publishing to quantum:stream:exit.intent.
    # When False (default): full PATCH-1 shadow-only behaviour is preserved.
    # When True:  only validated, actionable exits with whitelisted actions are
    #             published to exit.intent.  No write ever reaches trade.intent
    #             or apply.plan — the gateway (PATCH-5B) handles that separately.
    live_writes_enabled: bool
    symbol_allowlist: frozenset  # frozenset[str]; empty = allow all
    max_positions_per_loop: int
    max_hold_sec: float
    # PATCH-6: ownership-transfer fields.
    # ownership_transfer_enabled=True causes EMA to write active_flag each tick,
    # engaging the PATCH-2 kill-switch in AutonomousTrader.
    # Requires testnet_mode="true" as a secondary safety guard.
    ownership_transfer_enabled: bool
    # F1-HARDCODED: active_flag_key is NOT configurable via env.  It is a binary
    # protocol constant shared with autonomous_trader.py PATCH-2 reader (line 320):
    #   await self.redis.get("quantum:exit_agent:active_flag")
    # Allowing env override risks silent split-brain where EMA writes a different
    # key than AT reads.  Always set to the constant below by from_env().
    active_flag_key: str         # always "quantum:exit_agent:active_flag"
    active_flag_ttl_sec: int     # clamped to [10, 120] — see from_env()
    testnet_mode: str            # "true" required to allow ownership transfer
    # PATCH-7A: controls which exit path drives live decisions.
    # "shadow"  — ScoringEngine runs alongside DecisionEngine; only legacy drives live path.
    # "formula" — ScoringEngine drives live path; DecisionEngine audited as comparison.
    # "ai"      — PATCH-7B: formula engine runs first, Qwen3 refines within allowed actions.
    scoring_mode: str            # "shadow" | "formula" | "ai"
    # PATCH-7B: Qwen3 inference config (defaults keep PATCH-7A tests buildable without changes).
    qwen3_endpoint: str = "http://localhost:11434"   # Ollama or OpenAI-compat base URL
    qwen3_timeout_ms: int = 2000                     # clamped [200, 10000]
    qwen3_shadow: bool = True                        # True = audit-only; False = Qwen3 drives live
    qwen3_model: str = "qwen3:8b"                    # model tag
    # PATCH-7B-ext: bearer token for external endpoints (Groq, Together, OpenRouter).
    # Empty string = no Authorization header (Ollama local default).
    # NEVER log this field — treat as a secret at the system boundary.
    qwen3_api_key: str = ""                          # never logged; see EXIT_AGENT_QWEN3_API_KEY
    # PATCH-7B-rl: min seconds between successive Qwen3 API calls (rate-throttle).
    # Default 3.0 ≈ 20 RPM — safely under Groq free-tier ~30 RPM.
    # Set to 0.0 to disable throttle (e.g. local Ollama).
    qwen3_min_interval_sec: float = 3.0              # see EXIT_AGENT_QWEN3_MIN_INTERVAL_SEC
    # PATCH-8A: TTL applied to quantum:hash:exit.decision:{decision_id} snapshots.
    # Default 14400s = 4 h, matching max_hold_sec (positions rarely survive longer).
    decision_ttl_sec: int = 14400                    # see EXIT_AGENT_DECISION_TTL_SEC

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

        # PATCH-5A: read live_writes_enabled from env.
        # Default is false; must be explicitly set to "true" to allow exit.intent writes.
        live_writes_enabled = (
            os.getenv("EXIT_AGENT_LIVE_WRITES_ENABLED", "false").lower() == "true"
        )
        if live_writes_enabled:
            _log.warning(
                "EXIT_AGENT_LIVE_WRITES_ENABLED=true — "
                "exit_management_agent will publish validated exits to exit.intent. "
                "PATCH-5A is active. Gateway (PATCH-5B) is still required before "
                "any order reaches trade.intent or apply.plan."
            )

        # PATCH-6: ownership-transfer guard.
        # Default false; must be explicitly set to "true" AND testnet_mode="true".
        ownership_transfer_enabled = (
            os.getenv("EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED", "false").lower() == "true"
        )
        testnet_mode = os.getenv("EXIT_AGENT_TESTNET_MODE", "false").lower().strip()
        # PATCH-7A: scoring_mode controls which exit engine drives live decisions.
        scoring_mode = os.getenv("EXIT_AGENT_SCORING_MODE", "shadow").lower().strip()
        if scoring_mode not in ("shadow", "formula", "ai"):
            _log.warning(
                "EXIT_AGENT_SCORING_MODE=%r is not a recognised value — "
                "defaulting to 'shadow'. Valid values: shadow | formula | ai",
                scoring_mode,
            )
            scoring_mode = "shadow"
        if scoring_mode == "formula":
            _log.warning(
                "EXIT_AGENT_SCORING_MODE=formula — "
                "ScoringEngine is now the LIVE decision path. "
                "DecisionEngine runs in background for audit comparison. "
                "PATCH-7A formula mode is active."
            )
        elif scoring_mode == "ai":
            _log.warning(
                "EXIT_AGENT_SCORING_MODE=ai — PATCH-7B active. "
                "Formula engine runs first; Qwen3 refines within allowed actions. "
                "qwen3_shadow will be logged at startup."
            )

        if ownership_transfer_enabled:
            _log.warning(
                "EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=true — "
                "exit_management_agent will write quantum:exit_agent:active_flag "
                "each tick, suspending AutonomousTrader exit evaluation. "
                "PATCH-6 is active. testnet_mode=%s",
                testnet_mode,
            )
            if testnet_mode != "true":
                _log.warning(
                    "PATCH-6 ownership transfer is BLOCKED: "
                    "EXIT_AGENT_TESTNET_MODE must be 'true' — "
                    "active_flag will NOT be written."
                )

        # PATCH-6: active_flag_ttl_sec — F2 CLAMP [10, 120].
        # Upper bound: prevents a misconfigured large TTL from making the flag
        #   effectively permanent when rollback Option B (env=false) is used.
        #   At TTL=120s and tick=5s the flag still renews 24× per cycle.
        # Lower bound: ensures ≥2 ticks can fail before AT resumes accidentally.
        _ACTIVE_FLAG_TTL_MIN = 10
        _ACTIVE_FLAG_TTL_MAX = 120
        active_flag_ttl_sec_raw = int(
            os.getenv("EXIT_AGENT_ACTIVE_FLAG_TTL_SEC", "30")
        )
        if active_flag_ttl_sec_raw > _ACTIVE_FLAG_TTL_MAX:
            _log.warning(
                "EXIT_AGENT_ACTIVE_FLAG_TTL_SEC=%d exceeds maximum %d — clamping. "
                "A very large TTL would prevent env=false rollback from working in "
                "time; use 'redis-cli DEL quantum:exit_agent:active_flag' instead.",
                active_flag_ttl_sec_raw,
                _ACTIVE_FLAG_TTL_MAX,
            )
            active_flag_ttl_sec_raw = _ACTIVE_FLAG_TTL_MAX
        elif active_flag_ttl_sec_raw < _ACTIVE_FLAG_TTL_MIN:
            _log.warning(
                "EXIT_AGENT_ACTIVE_FLAG_TTL_SEC=%d below minimum %d — clamping.",
                active_flag_ttl_sec_raw,
                _ACTIVE_FLAG_TTL_MIN,
            )
            active_flag_ttl_sec_raw = _ACTIVE_FLAG_TTL_MIN
        active_flag_ttl_sec = active_flag_ttl_sec_raw

        # PATCH-7B: Qwen3 config.
        _QWEN3_TIMEOUT_MIN = 200
        _QWEN3_TIMEOUT_MAX = 10000
        qwen3_timeout_ms_raw = int(os.getenv("EXIT_AGENT_QWEN3_TIMEOUT_MS", "2000"))
        if qwen3_timeout_ms_raw < _QWEN3_TIMEOUT_MIN:
            _log.warning(
                "EXIT_AGENT_QWEN3_TIMEOUT_MS=%d below minimum %d — clamping.",
                qwen3_timeout_ms_raw, _QWEN3_TIMEOUT_MIN,
            )
            qwen3_timeout_ms_raw = _QWEN3_TIMEOUT_MIN
        elif qwen3_timeout_ms_raw > _QWEN3_TIMEOUT_MAX:
            _log.warning(
                "EXIT_AGENT_QWEN3_TIMEOUT_MS=%d exceeds maximum %d — clamping.",
                qwen3_timeout_ms_raw, _QWEN3_TIMEOUT_MAX,
            )
            qwen3_timeout_ms_raw = _QWEN3_TIMEOUT_MAX

        qwen3_shadow = os.getenv("EXIT_AGENT_QWEN3_SHADOW", "true").lower() != "false"
        qwen3_endpoint = os.getenv("EXIT_AGENT_QWEN3_ENDPOINT", "http://localhost:11434")
        qwen3_model = os.getenv("EXIT_AGENT_QWEN3_MODEL", "qwen3:8b")
        # PATCH-7B-ext: read api key but never log it.
        qwen3_api_key = os.getenv("EXIT_AGENT_QWEN3_API_KEY", "")
        # PATCH-7B-rl: rate-throttle interval (0 = disabled).
        qwen3_min_interval_sec = float(os.getenv("EXIT_AGENT_QWEN3_MIN_INTERVAL_SEC", "3.0"))
        if qwen3_min_interval_sec < 0.0:
            qwen3_min_interval_sec = 0.0
        # PATCH-8A: TTL for decision snapshot hashes.
        decision_ttl_sec = int(os.getenv("EXIT_AGENT_DECISION_TTL_SEC", "14400"))
        if decision_ttl_sec < 1:
            _log.warning(
                "EXIT_AGENT_DECISION_TTL_SEC=%d is below minimum 1 — clamping to 1.",
                decision_ttl_sec,
            )
            decision_ttl_sec = 1

        if scoring_mode == "ai":
            _log.warning(
                "PATCH-7B Qwen3 config: endpoint=%s model=%s timeout_ms=%d shadow=%s",
                qwen3_endpoint, qwen3_model, qwen3_timeout_ms_raw, qwen3_shadow,
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
            intent_stream=os.getenv(
                "EXIT_AGENT_INTENT_STREAM", _INTENT_STREAM_DEFAULT
            ),
            log_level=os.getenv("EXIT_AGENT_LOG_LEVEL", "INFO"),
            dry_run=True,  # PATCH-1 hard-coded — never read from env here.
            live_writes_enabled=live_writes_enabled,
            symbol_allowlist=allowlist,
            max_positions_per_loop=int(
                os.getenv("EXIT_AGENT_MAX_POSITIONS_PER_LOOP", "50")
            ),
            max_hold_sec=float(os.getenv("EXIT_AGENT_MAX_HOLD_SEC", "14400")),
            # PATCH-6 fields.
            ownership_transfer_enabled=ownership_transfer_enabled,
            # F1: key is hardcoded — never read from env. See field comment above.
            active_flag_key="quantum:exit_agent:active_flag",
            active_flag_ttl_sec=active_flag_ttl_sec,  # clamped by F2 guard above
            testnet_mode=testnet_mode,
            scoring_mode=scoring_mode,
            qwen3_endpoint=qwen3_endpoint,
            qwen3_timeout_ms=qwen3_timeout_ms_raw,
            qwen3_shadow=qwen3_shadow,
            qwen3_model=qwen3_model,
            qwen3_api_key=qwen3_api_key,
            qwen3_min_interval_sec=qwen3_min_interval_sec,
            decision_ttl_sec=decision_ttl_sec,
        )

    def is_symbol_allowed(self, symbol: str) -> bool:
        if not self.symbol_allowlist:
            return True
        return symbol.upper() in self.symbol_allowlist
