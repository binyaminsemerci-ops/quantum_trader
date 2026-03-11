"""
patch11_correct_path.py — Apply PATCH-11 to the CORRECT service directory.

WorkingDirectory=/home/qt/quantum_trader  (actual service source)
Previous patches went to /opt/quantum/    (wrong — service never read those).

Steps executed:
  0. Copy groq_client.py / judge_validator.py / ai_judge.py to correct dir
  1. config.py  — add 8 PATCH-11 fields
  2. models.py  — insert JudgeResult after Qwen3LayerResult
  3. scoring_engine.py — extend FORMULA_QTY_MAP with 6 PATCH-11 actions
  4. main.py    — wire AIJudge + OfflineEvaluator (replace Qwen3Layer)
  5. Clear __pycache__
  6. Update systemd env vars (mistral/deepseek -> fallback/evaluator)
  7. systemctl daemon-reload
"""
import ast
import os
import shutil
import subprocess
import sys

SVC_BASE = "/home/qt/quantum_trader/microservices/exit_management_agent"
OPT_BASE = "/opt/quantum/microservices/exit_management_agent"
SERVICE_FILE = "/etc/systemd/system/quantum-exit-management-agent.service"

_errors = []


def backup(path):
    bak = path + ".bak11"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)


def rd(path):
    with open(path) as f:
        return f.read()


def wr(path, content):
    with open(path, "w") as f:
        f.write(content)


def ok_syntax(label, src):
    try:
        ast.parse(src)
        print(f"  [SYNTAX OK] {label}")
        return True
    except SyntaxError as exc:
        print(f"  [SYNTAX FAIL] {label}: {exc}")
        return False


def sub1(label, src, old, new):
    if old not in src:
        msg = f"  [MISS] Not found — {label}"
        print(msg)
        _errors.append(msg)
        return src
    n = src.count(old)
    if n > 1:
        print(f"  [WARN] {n} occurrences found; replacing first — {label}")
    return src.replace(old, new, 1)


# ══════════════════════════════════════════════════════════════
# 0. Copy new module files
# ══════════════════════════════════════════════════════════════
print("\n── 0. Copy new modules to correct service dir ──")
for fname in ("groq_client.py", "judge_validator.py", "ai_judge.py"):
    src = os.path.join(OPT_BASE, fname)
    dst = os.path.join(SVC_BASE, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  [COPIED] {fname}")
    else:
        msg = f"  [MISSING] {fname} not at {src}"
        print(msg)
        _errors.append(msg)


# ══════════════════════════════════════════════════════════════
# 1. config.py
# ══════════════════════════════════════════════════════════════
print("\n── 1. config.py ──")
cfg_path = os.path.join(SVC_BASE, "config.py")
backup(cfg_path)
cfg = rd(cfg_path)

if "fallback_model:" in cfg:
    print("  [SKIP] already patched")
else:
    # 1a. Insert 8 new dataclass fields before the @classmethod boundary
    cfg = sub1(
        "config.py dataclass fields",
        cfg,
        "\n    @classmethod\n    def from_env(cls",
        """
    # PATCH-11 (proper): 3-tier AIJudge — shared Groq endpoint/api_key.
    fallback_model: str = "openai/gpt-oss-20b"
    fallback_timeout_ms: int = 3000
    fallback_min_interval_sec: float = 3.0
    evaluator_model: str = "llama-3.3-70b-versatile"
    evaluator_timeout_ms: int = 8000
    evaluator_enabled: bool = True
    judge_confidence_threshold: float = 0.55
    judge_strict_validation: bool = False

    @classmethod
    def from_env(cls""",
    )

    # 1b. Insert 8 from_env locals before the "if scoring_mode == 'ai':" log warning
    cfg = sub1(
        "config.py from_env locals",
        cfg,
        '        if scoring_mode == "ai":\n            _log.warning(\n                "PATCH-7B Qwen3 config:',
        '''        # PATCH-11 (proper): Tier-2 fallback + offline evaluator.
        fallback_model = os.getenv("EXIT_AGENT_FALLBACK_MODEL", "openai/gpt-oss-20b")
        fallback_timeout_ms = max(200, min(10000, int(os.getenv("EXIT_AGENT_FALLBACK_TIMEOUT_MS", "3000"))))
        fallback_min_interval_sec = max(0.0, float(os.getenv("EXIT_AGENT_FALLBACK_MIN_INTERVAL_SEC", "3.0")))
        evaluator_model = os.getenv("EXIT_AGENT_EVALUATOR_MODEL", "llama-3.3-70b-versatile")
        evaluator_timeout_ms = max(200, min(30000, int(os.getenv("EXIT_AGENT_EVALUATOR_TIMEOUT_MS", "8000"))))
        evaluator_enabled = os.getenv("EXIT_AGENT_EVALUATOR_ENABLED", "true").lower() == "true"
        judge_confidence_threshold = float(os.getenv("EXIT_AGENT_JUDGE_CONFIDENCE_THRESHOLD", "0.55"))
        judge_strict_validation = os.getenv("EXIT_AGENT_JUDGE_STRICT_VALIDATION", "false").lower() == "true"

        if scoring_mode == "ai":
            _log.warning(
                "PATCH-7B Qwen3 config:''',
    )

    # 1c. Insert 8 kwargs in return cls() — target the last existing kwarg+closing paren
    cfg = sub1(
        "config.py return cls kwargs",
        cfg,
        "            reward_premature_close_threshold_sec=reward_premature_close_threshold_sec,\n        )",
        """            reward_premature_close_threshold_sec=reward_premature_close_threshold_sec,
            fallback_model=fallback_model,
            fallback_timeout_ms=fallback_timeout_ms,
            fallback_min_interval_sec=fallback_min_interval_sec,
            evaluator_model=evaluator_model,
            evaluator_timeout_ms=evaluator_timeout_ms,
            evaluator_enabled=evaluator_enabled,
            judge_confidence_threshold=judge_confidence_threshold,
            judge_strict_validation=judge_strict_validation,
        )""",
    )
    print("  [PATCHED]")

if ok_syntax("config.py", cfg):
    wr(cfg_path, cfg)
    print("  [WRITTEN] config.py")
else:
    _errors.append("config.py syntax failure")


# ══════════════════════════════════════════════════════════════
# 2. models.py — insert JudgeResult after Qwen3LayerResult
# ══════════════════════════════════════════════════════════════
print("\n── 2. models.py ──")
mdl_path = os.path.join(SVC_BASE, "models.py")
backup(mdl_path)
mdl = rd(mdl_path)

JUDGE_RESULT = '''

@dataclass(frozen=True)
class JudgeResult:
    """
    [PATCH-11] Output of AIJudge.evaluate() for one position.
    tier: "t1"=primary, "t2"=fallback, "t3"=disagreement-resolved, "t0"=formula
    """
    action: str
    confidence: float
    reason_codes: tuple
    risk_note: str
    tier: str
    fallback: bool
    shadow_mode: str
    primary_raw: str
    fallback_raw: str
    primary_validation: str
    fallback_validation: str
    latency_ms: float
    formula_action: str

    def as_qwen3_layer_result(self) -> "Qwen3LayerResult":
        """Backward-compat shim for ExitDecision.qwen3_result."""
        reason = "{tier}:{codes}".format(
            tier=self.tier,
            codes=",".join(str(c) for c in self.reason_codes) or self.risk_note or "ok",
        )
        return Qwen3LayerResult(
            action=self.action,
            confidence=self.confidence,
            reason=reason[:200],
            fallback=self.fallback,
            latency_ms=self.latency_ms,
            raw=self.primary_raw[:500],
        )

    @property
    def reason(self) -> str:
        return ",".join(str(c) for c in self.reason_codes) or self.risk_note or "ai_judge"

'''

if "class JudgeResult" in mdl:
    print("  [SKIP] JudgeResult already present")
else:
    mdl = sub1(
        "models.py JudgeResult insert after Qwen3LayerResult",
        mdl,
        "\n@dataclass(frozen=True)\nclass DecisionSnapshot:",
        JUDGE_RESULT + "\n@dataclass(frozen=True)\nclass DecisionSnapshot:",
    )
    print("  [PATCHED]")

if ok_syntax("models.py", mdl):
    wr(mdl_path, mdl)
    print("  [WRITTEN] models.py")
else:
    _errors.append("models.py syntax failure")


# ══════════════════════════════════════════════════════════════
# 3. scoring_engine.py — extend FORMULA_QTY_MAP
# ══════════════════════════════════════════════════════════════
print("\n── 3. scoring_engine.py ──")
se_path = os.path.join(SVC_BASE, "scoring_engine.py")
backup(se_path)
se = rd(se_path)

if "REDUCE_25" in se:
    print("  [SKIP] PATCH-11 actions already present")
else:
    se = sub1(
        "scoring_engine.py FORMULA_QTY_MAP",
        se,
        "FORMULA_QTY_MAP: dict = {\n    FULL_CLOSE:        1.0,\n    TIME_STOP_EXIT:    1.0,\n    PARTIAL_CLOSE_25:  0.25,\n    TIGHTEN_TRAIL:     None,\n    MOVE_TO_BREAKEVEN: None,\n    HOLD:              None,\n}",
        "FORMULA_QTY_MAP: dict = {\n    FULL_CLOSE:        1.0,\n    TIME_STOP_EXIT:    1.0,\n    PARTIAL_CLOSE_25:  0.25,\n    TIGHTEN_TRAIL:     None,\n    MOVE_TO_BREAKEVEN: None,\n    HOLD:              None,\n    # PATCH-11 AIJudge actions\n    \"REDUCE_25\":           0.25,\n    \"REDUCE_50\":           0.50,\n    \"HARVEST_70_KEEP_30\":  0.70,\n    \"DEFENSIVE_HOLD\":      None,\n    \"TOXICITY_UNWIND\":     1.0,\n    \"QUARANTINE\":          None,\n}",
    )
    print("  [PATCHED]")

if ok_syntax("scoring_engine.py", se):
    wr(se_path, se)
    print("  [WRITTEN] scoring_engine.py")
else:
    _errors.append("scoring_engine.py syntax failure")


# ══════════════════════════════════════════════════════════════
# 4. main.py — replace Qwen3Layer with AIJudge
# ══════════════════════════════════════════════════════════════
print("\n── 4. main.py ──")
main_path = os.path.join(SVC_BASE, "main.py")
backup(main_path)
main = rd(main_path)

if "from .ai_judge import AIJudge" in main:
    print("  [SKIP] already wired to AIJudge")
else:
    # 4a. Swap import
    main = sub1(
        "main.py import",
        main,
        "from .qwen3_layer import Qwen3Layer",
        "from .groq_client import GroqModelClient\nfrom .ai_judge import AIJudge, OfflineEvaluator",
    )

    # 4b. Swap __init__ construction
    OLD_INIT = (
        "        self._qwen3 = Qwen3Layer(\n"
        "            endpoint=config.qwen3_endpoint,\n"
        "            timeout_ms=config.qwen3_timeout_ms,\n"
        "            shadow=config.qwen3_shadow,\n"
        "            model=config.qwen3_model,\n"
        "            api_key=config.qwen3_api_key,\n"
        "            min_interval_sec=config.qwen3_min_interval_sec,\n"
        "        )"
    )
    NEW_INIT = (
        "        # PATCH-11 (proper): 3-tier AIJudge.\n"
        "        _primary_client = GroqModelClient(\n"
        "            endpoint=config.qwen3_endpoint,\n"
        "            model=config.qwen3_model,\n"
        "            api_key=config.qwen3_api_key,\n"
        "            timeout_ms=config.qwen3_timeout_ms,\n"
        "            min_interval_sec=config.qwen3_min_interval_sec,\n"
        "        )\n"
        "        _fallback_client = GroqModelClient(\n"
        "            endpoint=config.qwen3_endpoint,\n"
        "            model=config.fallback_model,\n"
        "            api_key=config.qwen3_api_key,\n"
        "            timeout_ms=config.fallback_timeout_ms,\n"
        "            min_interval_sec=config.fallback_min_interval_sec,\n"
        "        )\n"
        "        _eval_client = GroqModelClient(\n"
        "            endpoint=config.qwen3_endpoint,\n"
        "            model=config.evaluator_model,\n"
        "            api_key=config.qwen3_api_key,\n"
        "            timeout_ms=config.evaluator_timeout_ms,\n"
        "            min_interval_sec=3.0,\n"
        "        )\n"
        "        _judge_shadow_mode = \"shadow\" if config.qwen3_shadow else \"live\"\n"
        "        self._qwen3 = AIJudge(\n"
        "            primary=_primary_client,\n"
        "            fallback=_fallback_client,\n"
        "            shadow_mode=_judge_shadow_mode,\n"
        "            confidence_threshold=config.judge_confidence_threshold,\n"
        "            strict_validation=config.judge_strict_validation,\n"
        "        )\n"
        "        self._deepseek_eval = OfflineEvaluator(\n"
        "            client=_eval_client,\n"
        "            enabled=config.evaluator_enabled,\n"
        "        )"
    )
    main = sub1("main.py __init__ Qwen3Layer → AIJudge", main, OLD_INIT, NEW_INIT)

    # 4c. Fix _tick shadow check: self._cfg.qwen3_shadow → qr.shadow_mode
    main = sub1(
        "main.py _tick shadow check",
        main,
        "                            if self._cfg.qwen3_shadow or qr.fallback:",
        "                            # PATCH-11: shadow_mode baked into JudgeResult.\n                            if qr.shadow_mode == \"shadow\" or qr.fallback:",
    )

    # 4d. Fix qwen3_result=qr → qr.as_qwen3_layer_result()
    main = sub1(
        "main.py _tick qwen3_result",
        main,
        "                                qwen3_result=qr,\n                            )",
        "                                qwen3_result=qr.as_qwen3_layer_result(),\n                            )",
    )
    print("  [PATCHED]")

if ok_syntax("main.py", main):
    wr(main_path, main)
    print("  [WRITTEN] main.py")
else:
    _errors.append("main.py syntax failure")


# ══════════════════════════════════════════════════════════════
# 5. Clear __pycache__
# ══════════════════════════════════════════════════════════════
print("\n── 5. Clear __pycache__ ──")
pycache = os.path.join(SVC_BASE, "__pycache__")
if os.path.isdir(pycache):
    shutil.rmtree(pycache)
    print("  [CLEARED]")
else:
    print("  [none to clear]")


# ══════════════════════════════════════════════════════════════
# 6. Update systemd env vars
# ══════════════════════════════════════════════════════════════
print("\n── 6. systemd env vars ──")
svc = rd(SERVICE_FILE)
if not os.path.exists(SERVICE_FILE + ".bak11"):
    shutil.copy2(SERVICE_FILE, SERVICE_FILE + ".bak11")

for old, new in [
    ("Environment=EXIT_AGENT_MISTRAL_MODEL=mistral-small-3.1-24b-instruct",
     "Environment=EXIT_AGENT_FALLBACK_MODEL=openai/gpt-oss-20b"),
    ("Environment=EXIT_AGENT_MISTRAL_TIMEOUT_MS=2000",
     "Environment=EXIT_AGENT_FALLBACK_TIMEOUT_MS=3000"),
    ("Environment=EXIT_AGENT_DEEPSEEK_MODEL=deepseek-r1-distill-llama-70b",
     "Environment=EXIT_AGENT_EVALUATOR_MODEL=llama-3.3-70b-versatile"),
    ("Environment=EXIT_AGENT_DEEPSEEK_TIMEOUT_MS=8000",
     "Environment=EXIT_AGENT_EVALUATOR_TIMEOUT_MS=8000"),
    ("Environment=EXIT_AGENT_DEEPSEEK_ENABLED=true",
     "Environment=EXIT_AGENT_EVALUATOR_ENABLED=true"),
    ("Environment=EXIT_AGENT_QWEN3_SHADOW=false",
     "Environment=EXIT_AGENT_QWEN3_SHADOW=true"),
]:
    if old in svc:
        svc = svc.replace(old, new, 1)
        print(f"  [REPLACED] ...{old[-40:]}")
    elif new in svc:
        print(f"  [SKIP] already correct: {new}")
    else:
        print(f"  [MISS] not found: {old}")

for var in (
    "Environment=EXIT_AGENT_JUDGE_CONFIDENCE_THRESHOLD=0.55",
    "Environment=EXIT_AGENT_JUDGE_STRICT_VALIDATION=false",
    "Environment=EXIT_AGENT_FALLBACK_MIN_INTERVAL_SEC=3.0",
):
    key = var.split("=")[1]
    if key not in svc:
        anchor = "Environment=EXIT_AGENT_EVALUATOR_ENABLED=true"
        if anchor in svc:
            svc = svc.replace(anchor, f"{anchor}\n{var}", 1)
            print(f"  [ADDED] {var}")
        else:
            print(f"  [WARN] anchor missing; skipped {var}")

wr(SERVICE_FILE, svc)
print("  [WRITTEN] service file")

r = subprocess.run(["systemctl", "daemon-reload"], capture_output=True, text=True)
print("  [OK] daemon-reload" if r.returncode == 0 else f"  [ERR] daemon-reload: {r.stderr.strip()}")


# ══════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
if _errors:
    print(f"ERRORS ({len(_errors)}):")
    for e in _errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("PATCH-11 correct-path deploy COMPLETE.")
    print("Next: systemctl restart quantum-exit-management-agent")
