#!/usr/bin/env python3
"""PATCH-11: Wire 3-tier AI brain (Qwen3 -> Mistral -> DeepSeek evaluator).

Patches:
  config.py       — add Mistral/DeepSeek fields + env reads
  main.py         — swap Qwen3Layer -> AIBrain, add DeepSeekEvaluator
  replay_writer.py — add evaluator support
  qwen3_layer.py  — expand ALLOWED_ACTIONS + system prompt (add PARTIAL_CLOSE_50)
"""
import sys

AGENT = "/opt/quantum/microservices/exit_management_agent"


def patch(path, old, new, label):
    with open(path) as f:
        content = f.read()
    count = content.count(old)
    if count == 0:
        print(f"FAIL {label}: oldString NOT FOUND in {path}")
        sys.exit(1)
    if count > 1:
        print(f"FAIL {label}: oldString matches {count} times in {path}")
        sys.exit(1)
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print(f"OK   {label}")


# ─────────────────────────────────────────────────────────────────────────────
# config.py — add new dataclass fields
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/config.py",
    '    reward_premature_close_threshold_sec: int = 300    # see EXIT_AGENT_REWARD_PREMATURE_CLOSE_THRESHOLD_SEC',
    '    reward_premature_close_threshold_sec: int = 300    # see EXIT_AGENT_REWARD_PREMATURE_CLOSE_THRESHOLD_SEC\n'
    '    # PATCH-11: Tier-2 fallback (Mistral Small 3.1) + Tier-3 evaluator (DeepSeek-R1).\n'
    '    # Both share qwen3_endpoint and qwen3_api_key — all on Groq, single API key.\n'
    '    mistral_model: str = "mistral-small-3.1-24b-instruct"\n'
    '    mistral_timeout_ms: int = 1500\n'
    '    mistral_min_interval_sec: float = 3.0\n'
    '    deepseek_model: str = "deepseek-r1-distill-llama-70b"\n'
    '    deepseek_timeout_ms: int = 8000\n'
    '    deepseek_enabled: bool = True',
    "config.py: add Mistral/DeepSeek fields to dataclass",
)

# ─────────────────────────────────────────────────────────────────────────────
# config.py — add env var reads in from_env() before the reward threshold check
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/config.py",
    '        if reward_premature_close_threshold_sec < 0:\n            reward_premature_close_threshold_sec = 0',
    '        if reward_premature_close_threshold_sec < 0:\n            reward_premature_close_threshold_sec = 0\n'
    '        # PATCH-11: Tier-2 (Mistral) + Tier-3 (DeepSeek) config — share Groq endpoint/key.\n'
    '        mistral_model = os.getenv("EXIT_AGENT_MISTRAL_MODEL", "mistral-small-3.1-24b-instruct")\n'
    '        mistral_timeout_ms = max(200, min(10000, int(os.getenv("EXIT_AGENT_MISTRAL_TIMEOUT_MS", "1500"))))\n'
    '        mistral_min_interval_sec = max(0.0, float(os.getenv("EXIT_AGENT_MISTRAL_MIN_INTERVAL_SEC", "3.0")))\n'
    '        deepseek_model = os.getenv("EXIT_AGENT_DEEPSEEK_MODEL", "deepseek-r1-distill-llama-70b")\n'
    '        deepseek_timeout_ms = max(200, min(30000, int(os.getenv("EXIT_AGENT_DEEPSEEK_TIMEOUT_MS", "8000"))))\n'
    '        deepseek_enabled = os.getenv("EXIT_AGENT_DEEPSEEK_ENABLED", "true").lower() == "true"',
    "config.py: add Mistral/DeepSeek env reads in from_env()",
)

# ─────────────────────────────────────────────────────────────────────────────
# config.py — add new fields to return cls(...)
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/config.py",
    '            reward_premature_close_threshold_sec=reward_premature_close_threshold_sec,\n        )',
    '            reward_premature_close_threshold_sec=reward_premature_close_threshold_sec,\n'
    '            mistral_model=mistral_model,\n'
    '            mistral_timeout_ms=mistral_timeout_ms,\n'
    '            mistral_min_interval_sec=mistral_min_interval_sec,\n'
    '            deepseek_model=deepseek_model,\n'
    '            deepseek_timeout_ms=deepseek_timeout_ms,\n'
    '            deepseek_enabled=deepseek_enabled,\n'
    '        )',
    "config.py: add Mistral/DeepSeek to return cls()",
)

# ─────────────────────────────────────────────────────────────────────────────
# main.py — add AIBrain + DeepSeekEvaluator import
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/main.py",
    'from .qwen3_layer import Qwen3Layer',
    'from .qwen3_layer import Qwen3Layer\nfrom .ai_brain import AIBrain, DeepSeekEvaluator',
    "main.py: add AIBrain/DeepSeekEvaluator import",
)

# ─────────────────────────────────────────────────────────────────────────────
# main.py — replace single Qwen3Layer init with AIBrain 3-tier init
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/main.py",
    '        # PATCH-7B: Qwen3 constrained decision layer — instantiated always;\n'
    '        # only called when scoring_mode="ai" and formula_action is in _ALLOWED_ACTIONS.\n'
    '        self._qwen3 = Qwen3Layer(\n'
    '            endpoint=config.qwen3_endpoint,\n'
    '            timeout_ms=config.qwen3_timeout_ms,\n'
    '            shadow=config.qwen3_shadow,\n'
    '            model=config.qwen3_model,\n'
    '            api_key=config.qwen3_api_key,\n'
    '            min_interval_sec=config.qwen3_min_interval_sec,\n'
    '        )',
    '        # PATCH-11: 3-tier AI brain.\n'
    '        # Tier 1 (primary):  Qwen3-32b   via Groq — live exit decisions\n'
    '        # Tier 2 (fallback): Mistral 3.1 via Groq — takes over on Tier 1 failure / 429\n'
    '        _ai_primary = Qwen3Layer(\n'
    '            endpoint=config.qwen3_endpoint,\n'
    '            timeout_ms=config.qwen3_timeout_ms,\n'
    '            shadow=config.qwen3_shadow,\n'
    '            model=config.qwen3_model,\n'
    '            api_key=config.qwen3_api_key,\n'
    '            min_interval_sec=config.qwen3_min_interval_sec,\n'
    '        )\n'
    '        _ai_fallback = Qwen3Layer(\n'
    '            endpoint=config.qwen3_endpoint,\n'
    '            timeout_ms=config.mistral_timeout_ms,\n'
    '            shadow=config.qwen3_shadow,\n'
    '            model=config.mistral_model,\n'
    '            api_key=config.qwen3_api_key,\n'
    '            min_interval_sec=config.mistral_min_interval_sec,\n'
    '        )\n'
    '        self._qwen3 = AIBrain(primary=_ai_primary, fallback=_ai_fallback)\n'
    '        # Tier 3 (offline evaluator): DeepSeek-R1-Distill via Groq\n'
    '        self._deepseek_eval = DeepSeekEvaluator(\n'
    '            endpoint=config.qwen3_endpoint,\n'
    '            model=config.deepseek_model,\n'
    '            api_key=config.qwen3_api_key,\n'
    '            timeout_ms=config.deepseek_timeout_ms,\n'
    '            enabled=config.deepseek_enabled,\n'
    '        )',
    "main.py: replace Qwen3Layer with AIBrain 3-tier",
)

# ─────────────────────────────────────────────────────────────────────────────
# main.py — pass DeepSeek evaluator to ReplayWriter
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/main.py",
    '            _replay_writer = ReplayWriter(\n'
    '                redis=self._redis,\n'
    '                replay_stream=config.replay_stream,\n'
    '                enabled=True,\n'
    '            )',
    '            _replay_writer = ReplayWriter(\n'
    '                redis=self._redis,\n'
    '                replay_stream=config.replay_stream,\n'
    '                enabled=True,\n'
    '                evaluator=self._deepseek_eval,\n'
    '            )',
    "main.py: pass DeepSeek evaluator to ReplayWriter",
)

# ─────────────────────────────────────────────────────────────────────────────
# replay_writer.py — add evaluator param to __init__
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/replay_writer.py",
    '    def __init__(\n'
    '        self,\n'
    '        redis,\n'
    '        replay_stream: str = _REPLAY_STREAM_DEFAULT,\n'
    '        enabled: bool = True,\n'
    '    ) -> None:\n'
    '        self._redis = redis\n'
    '        self._replay_stream = replay_stream\n'
    '        self._enabled = enabled',
    '    def __init__(\n'
    '        self,\n'
    '        redis,\n'
    '        replay_stream: str = _REPLAY_STREAM_DEFAULT,\n'
    '        enabled: bool = True,\n'
    '        evaluator=None,\n'
    '    ) -> None:\n'
    '        self._redis = redis\n'
    '        self._replay_stream = replay_stream\n'
    '        self._enabled = enabled\n'
    '        self._evaluator = evaluator',
    "replay_writer.py: add evaluator param to __init__",
)

# ─────────────────────────────────────────────────────────────────────────────
# replay_writer.py — call evaluator in write() before xadd
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/replay_writer.py",
    '        record = self._build_record(decision_id, symbol, snapshot, outcome, result)\n'
    '        try:\n'
    '            await self._redis.xadd(self._replay_stream, record)',
    '        record = self._build_record(decision_id, symbol, snapshot, outcome, result)\n'
    '        # PATCH-11: DeepSeek-R1 offline evaluation (post-trade verdict).\n'
    '        if self._evaluator is not None:\n'
    '            try:\n'
    '                eval_fields = await self._evaluator.evaluate_replay(record)\n'
    '                record.update(eval_fields)\n'
    '            except Exception as _eval_exc:\n'
    '                _log.warning("PATCH-11: DeepSeek eval error for %s: %s", symbol, _eval_exc)\n'
    '        try:\n'
    '            await self._redis.xadd(self._replay_stream, record)',
    "replay_writer.py: call DeepSeek evaluator before xadd",
)

# ─────────────────────────────────────────────────────────────────────────────
# qwen3_layer.py — expand ALLOWED_ACTIONS (add PARTIAL_CLOSE_50)
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/qwen3_layer.py",
    '{"HOLD", "PARTIAL_CLOSE_25", "FULL_CLOSE", "TIME_STOP_EXIT"}',
    '{"HOLD", "PARTIAL_CLOSE_25", "PARTIAL_CLOSE_50", "FULL_CLOSE", "TIME_STOP_EXIT"}',
    "qwen3_layer.py: add PARTIAL_CLOSE_50 to ALLOWED_ACTIONS",
)

# ─────────────────────────────────────────────────────────────────────────────
# qwen3_layer.py — system prompt: add PARTIAL_CLOSE_50 action line
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/qwen3_layer.py",
    '    "  PARTIAL_CLOSE_25\\n"\n    "  FULL_CLOSE\\n"',
    '    "  PARTIAL_CLOSE_25\\n"\n    "  PARTIAL_CLOSE_50\\n"\n    "  FULL_CLOSE\\n"',
    "qwen3_layer.py: add PARTIAL_CLOSE_50 to system prompt",
)

# ─────────────────────────────────────────────────────────────────────────────
# qwen3_layer.py — system prompt: update action count 4 -> 5
# ─────────────────────────────────────────────────────────────────────────────
patch(
    f"{AGENT}/qwen3_layer.py",
    '"<one of the 4 actions>"',
    '"<one of the 5 actions>"',
    "qwen3_layer.py: update prompt action count 4->5",
)

print("\nPATCH-11 complete — all 10 patches applied")
