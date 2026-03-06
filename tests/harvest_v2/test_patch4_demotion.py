"""PATCH-4 demotion tests for harvest_v2.

Chokepoint under test
---------------------
harvest_v2.py — inner loop body:

    if cfg.stream_live:
        # PATCH-4: harvest_v2 live write ownership demoted.
        if os.getenv("HV2_LIVE_WRITES_ENABLED", "").lower() != "true":
            logger.warning("[HV2] HV2_LIVE_WRITE_BLOCKED patch=PATCH-4 ...")
        else:
            live_payload = _build_live_payload(pos, result, decision)
            redis.xadd(cfg.stream_live, live_payload)
            ...

Strategy
--------
* Import harvest_v2 as a fully-qualified module (sys.path insert in the module
  adds microservices/harvest_v2/ so its sub-packages resolve).
* Replace all network-touching classes (RedisClient, ConfigLoader, etc.) with
  recording fakes injected via unittest.mock.patch.
* Run main() for exactly ONE tick: time.sleep side-effect flips _RUNNING False.
* Verify whether redis.xadd was called with cfg.stream_live as the target.

Fakes
-----
_RecordingRedis     – records (stream, fields) for every xadd call; no network.
_LiveConfig         – HarvestV2Config-like with stream_live set to apply.plan.
_ShadowConfig       – stream_live="" (outer guard: never reaches PATCH-4 block).
_ExitEvaluator      – always returns ("EXIT", result) so the emit block is hit.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch, MagicMock
import pytest

import microservices.harvest_v2.harvest_v2 as hv2


# ── Shared stream names ───────────────────────────────────────────────────
_APPLY_PLAN = "quantum:stream:apply.plan"
_SHADOW     = "quantum:stream:harvest.v2.shadow"


# ── Fake: recording Redis ─────────────────────────────────────────────────
class _RecordingRedis:
    """Drop-in for RedisClient — records every xadd without network I/O."""

    def __init__(self):
        self.xadd_calls: list[tuple[str, dict]] = []

    def xadd(self, stream: str, fields: dict, maxlen: int = 50_000):
        self.xadd_calls.append((stream, fields))
        return "0-1"

    def hgetall(self, name: str) -> dict:
        return {}

    def keys(self, pattern: str):
        return []

    def hget(self, *a):
        return None

    def hset(self, *a, **kw):
        pass

    def hincrby(self, *a, **kw):
        pass

    def hincrbyfloat(self, *a, **kw):
        pass

    def expire(self, *a, **kw):
        pass

    @property
    def raw(self):
        return self


# ── Fake: config loaders ──────────────────────────────────────────────────
@dataclass
class _FakeHV2Config:
    stream_shadow: str            = _SHADOW
    stream_live: str              = _APPLY_PLAN   # live-enabled config
    atr_window: int               = 50
    max_position_age_sec: int     = 86400
    scan_interval_sec: float      = 0.001
    config_refresh_interval_sec: int = 60
    metrics_key: str              = "quantum:metrics:harvest_v2"
    r_stop_base: float            = 0.5
    r_target_base: float          = 3.0
    trailing_step: float          = 0.3
    r_emit_step: float            = 0.05
    partial_25_r: float           = 1.0
    partial_50_r: float           = 1.5
    partial_75_r: float           = 2.0
    heat_sensitivity: float       = 0.5
    vol_factor_low: float         = 0.7
    vol_factor_mid: float         = 1.0
    vol_factor_high: float        = 1.4


class _LiveConfigLoader:
    """ConfigLoader stub — returns config with stream_live set."""
    def __init__(self, redis): pass
    def get(self) -> _FakeHV2Config:
        return _FakeHV2Config(stream_live=_APPLY_PLAN)


class _ShadowConfigLoader:
    """ConfigLoader stub — stream_live left empty (never reaches PATCH-4 guard)."""
    def __init__(self, redis): pass
    def get(self) -> _FakeHV2Config:
        return _FakeHV2Config(stream_live="")


# ── Fake: position provider ───────────────────────────────────────────────
@dataclass
class _FakePosition:
    symbol: str           = "BTCUSDT"
    side: str             = "LONG"
    quantity: float       = 1.0
    entry_risk_usdt: float= 100.0
    unrealized_pnl: float = 500.0   # R_net = 5.0 → triggers EXIT
    atr_value: float      = 100.0


class _FakeFetchResult:
    def __init__(self, positions):
        self.positions        = positions
        self.total_keys       = len(positions)
        self.skipped_invalid  = 0
        self.skipped_stale    = 0


class _EmittingPositionProvider:
    """Always returns one position with a high R that forces an EXIT decision."""
    def __init__(self, redis, max_age_sec=86400):
        self.max_age_sec = max_age_sec

    def fetch_positions(self):
        return _FakeFetchResult([_FakePosition()])


class _EmptyPositionProvider:
    """Returns no positions — useful when we only want to verify nothing fires."""
    def __init__(self, redis, max_age_sec=86400):
        self.max_age_sec = max_age_sec

    def fetch_positions(self):
        return _FakeFetchResult([])


# ── Fake: heat + ATR ─────────────────────────────────────────────────────
class _FakeHeatProvider:
    def __init__(self, redis): pass
    def get_heat(self) -> float:
        return 0.0


class _FakeATRProvider:
    def get_atr(self, pos) -> Optional[float]:
        return 10.0   # non-None so position is not skipped


# ── Fake: state manager ───────────────────────────────────────────────────
class _FakeSymbolState:
    partial_stage = 0
    max_R_seen    = 0.0

    def update_max_R(self, v: float):
        self.max_R_seen = max(self.max_R_seen, v)

    def record_emission(self, decision: str, r: float):
        pass


class _FakeStateManager:
    def __init__(self, redis, atr_window=50): pass
    def set_atr_window(self, v): pass
    def get(self, pos): return _FakeSymbolState()
    def save(self, state): pass


# ── Fake: evaluator that always emits EXIT ────────────────────────────────
class _EvalResult:
    def __init__(self):
        self.decision    = "EXIT"
        self.R_net       = 5.0
        self.R_stop      = 0.5
        self.R_target    = 3.0
        self.regime      = "MID_VOL"
        self.vol_factor  = 1.0
        self.emit_reason = "kill_score_close_ok"


class _ExitEvaluator:
    def evaluate(self, **kw):
        return ("EXIT", _EvalResult())


# ── Fake: metrics writer ──────────────────────────────────────────────────
class _FakeMetrics:
    def __init__(self, redis, key): pass
    def record_start(self): pass
    def emission(self, *a, **kw): pass
    def tick(self, **kw): pass


# ── Helper: run exactly one main() tick ──────────────────────────────────
def _one_tick(env: dict, config_loader_cls, pos_provider_cls):
    """
    Run hv2.main() for exactly one loop iteration.

    Returns the _RecordingRedis so callers can inspect xadd_calls.
    Uses time.sleep side-effect to flip _RUNNING=False after first tick.
    """
    recording = _RecordingRedis()

    def _stop_after_first(_duration):
        hv2._RUNNING = False

    patches = [
        patch.object(hv2, "RedisClient",     return_value=recording),
        patch.object(hv2, "ConfigLoader",    config_loader_cls),
        patch.object(hv2, "PositionProvider",pos_provider_cls),
        patch.object(hv2, "HeatProvider",    _FakeHeatProvider),
        patch.object(hv2, "ATRProvider",     _FakeATRProvider),
        patch.object(hv2, "StateManager",    _FakeStateManager),
        patch.object(hv2, "ExitEvaluator",   _ExitEvaluator),
        patch.object(hv2, "MetricsWriter",   _FakeMetrics),
        patch("time.sleep", side_effect=_stop_after_first),
        patch("time.monotonic", return_value=9999.0),
    ]

    with patch.dict(os.environ, env, clear=False):
        for p in patches:
            p.start()
        hv2._RUNNING = True
        try:
            hv2.main()
        finally:
            for p in reversed(patches):
                p.stop()
            hv2._RUNNING = True  # restore for next test

    return recording


# ═══════════════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildLivePayload:
    """Pure function — verify _build_live_payload output shape."""

    def test_full_close_fields_present(self):
        pos    = _FakePosition()
        result = _EvalResult()
        payload = hv2._build_live_payload(pos, result, "EXIT")

        assert payload["source"]   == "harvest_v2"
        assert payload["action"]   == "FULL_CLOSE_PROPOSED"
        assert payload["decision"] == "EXECUTE"
        assert payload["symbol"]   == "BTCUSDT"
        assert payload["reduceOnly"] == "true"
        steps = json.loads(payload["steps"])
        assert steps[0]["type"] == "market_reduce_only"

    def test_partial_25_uses_partial_action(self):
        pos    = _FakePosition()
        result = _EvalResult()
        payload = hv2._build_live_payload(pos, result, "PARTIAL_25")
        assert payload["action"] == "PARTIAL_CLOSE_PROPOSED"

    def test_kill_score_is_zero_string(self):
        pos    = _FakePosition()
        result = _EvalResult()
        payload = hv2._build_live_payload(pos, result, "EXIT")
        assert payload["kill_score"] == "0.0"


class TestLiveWriteBlocked:
    """PATCH-4 core: live write is suppressed unless HV2_LIVE_WRITES_ENABLED=true."""

    def test_blocked_env_absent(self):
        env = {}
        if "HV2_LIVE_WRITES_ENABLED" in os.environ:
            env = {}
        rec = _one_tick(
            env={k: v for k, v in os.environ.items() if k != "HV2_LIVE_WRITES_ENABLED"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], "no xadd to apply.plan when env var absent"

    def test_blocked_env_false(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "false"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], "no xadd to apply.plan when env=false"

    def test_blocked_env_yes(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "yes"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], "no xadd to apply.plan when env=yes"

    def test_blocked_env_1(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "1"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], "no xadd to apply.plan when env=1"

    def test_blocked_env_empty_string(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": ""},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], "no xadd to apply.plan when env empty string"


class TestLiveWriteAllowed:
    """With opt-in env var, live writes go through (rollback-ability check)."""

    def test_allowed_env_true_lowercase(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "true"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert len(live_writes) == 1, "xadd to apply.plan when env=true"

    def test_allowed_env_TRUE_uppercase(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "TRUE"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert len(live_writes) == 1, "guard is case-insensitive"


class TestShadowStreamUnaffected:
    """Shadow writes to harvest.v2.shadow must always proceed regardless of env var."""

    def test_shadow_written_when_guard_blocks_live(self):
        rec = _one_tick(
            env={},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        shadow_writes = [s for s, _ in rec.xadd_calls if s == _SHADOW]
        assert len(shadow_writes) == 1, "shadow stream written even when live is blocked"

    def test_shadow_written_when_live_enabled(self):
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "true"},
            config_loader_cls=_LiveConfigLoader,
            pos_provider_cls=_EmittingPositionProvider,
        )
        shadow_writes = [s for s, _ in rec.xadd_calls if s == _SHADOW]
        assert len(shadow_writes) == 1, "shadow stream written when live is active"

    def test_shadow_never_writes_apply_plan(self):
        """_publish_shadow equivalent: shadow target is never apply.plan."""
        rec = _one_tick(
            env={},
            config_loader_cls=_ShadowConfigLoader,    # stream_live=""
            pos_provider_cls=_EmittingPositionProvider,
        )
        all_streams = [s for s, _ in rec.xadd_calls]
        assert _APPLY_PLAN not in all_streams, (
            "shadow-only config must never write to apply.plan"
        )


class TestOuterStreamLiveGate:
    """If cfg.stream_live is empty the outer guard short-circuits before PATCH-4 block."""

    def test_no_write_when_stream_live_empty_regardless_of_env(self):
        """Even with HV2_LIVE_WRITES_ENABLED=true, empty stream_live → no live write."""
        rec = _one_tick(
            env={"HV2_LIVE_WRITES_ENABLED": "true"},
            config_loader_cls=_ShadowConfigLoader,    # stream_live=""
            pos_provider_cls=_EmittingPositionProvider,
        )
        live_writes = [s for s, _ in rec.xadd_calls if s == _APPLY_PLAN]
        assert live_writes == [], (
            "outer cfg.stream_live gate prevents write even when env var is true"
        )
