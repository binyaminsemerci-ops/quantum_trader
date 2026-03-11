"""
patch11_proper.py — PATCH-11 (proper): wire AIJudge + correct Groq models.

Applies in-place patches to:
  1. config.py       — replace mistral/deepseek fields → fallback/evaluator
  2. models.py       — add JudgeResult dataclass
  3. scoring_engine.py — extend FORMULA_QTY_MAP with PATCH-11 actions
  4. main.py         — wire AIJudge + OfflineEvaluator; drop AIBrain + Qwen3Layer

Creates .bak files before modifying.  Validates Python syntax after each file.
Prints PASS or FAIL for each step.

Run on VPS:
  python3 /tmp/patch11_proper.py
"""
import ast
import os
import shutil
import sys

BASE = "/opt/quantum/microservices/exit_management_agent"

OK = True  # global pass/fail tracker


def backup(path: str) -> str:
    bak = path + ".bak11"
    shutil.copy2(path, bak)
    return bak


def read(path: str) -> str:
    with open(path) as f:
        return f.read()


def write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def check_syntax(label: str, src: str) -> bool:
    try:
        ast.parse(src)
        print(f"  [SYNTAX OK] {label}")
        return True
    except SyntaxError as exc:
        print(f"  [SYNTAX FAIL] {label}: {exc}")
        return False


def replace_once(label: str, src: str, old: str, new: str) -> str:
    if old not in src:
        print(f"  [WARN] Target not found in {label}: {repr(old[:80])}")
        return src
    if src.count(old) > 1:
        print(f"  [WARN] Target appears {src.count(old)}× in {label} — using first")
    return src.replace(old, new, 1)


# ═══════════════════════════════════════════════════════════
# 1. config.py — replace mistral/deepseek PATCH-11 fields
# ═══════════════════════════════════════════════════════════
print("\n── 1. config.py ──")
cfg_path = os.path.join(BASE, "config.py")
backup(cfg_path)
cfg = read(cfg_path)

# 1a. Dataclass field block
cfg = replace_once(
    "config.py dataclass fields",
    cfg,
    """    # PATCH-11: Tier-2 fallback (Mistral Small 3.1) + Tier-3 evaluator (DeepSeek-R1).
    # Both share qwen3_endpoint and qwen3_api_key — all on Groq, single API key.
    mistral_model: str = "mistral-small-3.1-24b-instruct"
    mistral_timeout_ms: int = 1500
    mistral_min_interval_sec: float = 3.0
    deepseek_model: str = "deepseek-r1-distill-llama-70b"
    deepseek_timeout_ms: int = 8000
    deepseek_enabled: bool = True""",
    """    # PATCH-11 (proper): 3-tier AI judge.
    # Tier 1 (primary):  qwen/qwen3-32b           via Groq — live exit decisions
    # Tier 2 (fallback): openai/gpt-oss-20b        via Groq — takes over on hard/soft fail
    # Offline evaluator: llama-3.3-70b-versatile   via Groq — replay/forensic only
    # All tiers share qwen3_endpoint and qwen3_api_key (single Groq API key).
    fallback_model: str = "openai/gpt-oss-20b"
    fallback_timeout_ms: int = 3000
    fallback_min_interval_sec: float = 3.0
    evaluator_model: str = "llama-3.3-70b-versatile"
    evaluator_timeout_ms: int = 8000
    evaluator_enabled: bool = True
    # Judge thresholds and validation flags.
    judge_confidence_threshold: float = 0.55
    judge_strict_validation: bool = False""",
)

# 1b. from_env() local vars block
cfg = replace_once(
    "config.py from_env PATCH-11 locals",
    cfg,
    """        # PATCH-11: Tier-2 (Mistral) + Tier-3 (DeepSeek) config — share Groq endpoint/key.
        mistral_model = os.getenv("EXIT_AGENT_MISTRAL_MODEL", "mistral-small-3.1-24b-instruct")
        mistral_timeout_ms = max(200, min(10000, int(os.getenv("EXIT_AGENT_MISTRAL_TIMEOUT_MS", "1500"))))
        mistral_min_interval_sec = max(0.0, float(os.getenv("EXIT_AGENT_MISTRAL_MIN_INTERVAL_SEC", "3.0")))
        deepseek_model = os.getenv("EXIT_AGENT_DEEPSEEK_MODEL", "deepseek-r1-distill-llama-70b")
        deepseek_timeout_ms = max(200, min(30000, int(os.getenv("EXIT_AGENT_DEEPSEEK_TIMEOUT_MS", "8000"))))
        deepseek_enabled = os.getenv("EXIT_AGENT_DEEPSEEK_ENABLED", "true").lower() == "true\"""",
    """        # PATCH-11 (proper): Tier-2 fallback (GPT-OSS-20b) + offline evaluator (llama).
        fallback_model = os.getenv("EXIT_AGENT_FALLBACK_MODEL", "openai/gpt-oss-20b")
        fallback_timeout_ms = max(200, min(10000, int(os.getenv("EXIT_AGENT_FALLBACK_TIMEOUT_MS", "3000"))))
        fallback_min_interval_sec = max(0.0, float(os.getenv("EXIT_AGENT_FALLBACK_MIN_INTERVAL_SEC", "3.0")))
        evaluator_model = os.getenv("EXIT_AGENT_EVALUATOR_MODEL", "llama-3.3-70b-versatile")
        evaluator_timeout_ms = max(200, min(30000, int(os.getenv("EXIT_AGENT_EVALUATOR_TIMEOUT_MS", "8000"))))
        evaluator_enabled = os.getenv("EXIT_AGENT_EVALUATOR_ENABLED", "true").lower() == "true"
        judge_confidence_threshold = float(os.getenv("EXIT_AGENT_JUDGE_CONFIDENCE_THRESHOLD", "0.55"))
        judge_strict_validation = os.getenv("EXIT_AGENT_JUDGE_STRICT_VALIDATION", "false").lower() == "true\"""",
)

# 1c. return cls() kwargs
cfg = replace_once(
    "config.py return cls kwargs",
    cfg,
    """            mistral_model=mistral_model,
            mistral_timeout_ms=mistral_timeout_ms,
            mistral_min_interval_sec=mistral_min_interval_sec,
            deepseek_model=deepseek_model,
            deepseek_timeout_ms=deepseek_timeout_ms,
            deepseek_enabled=deepseek_enabled,""",
    """            fallback_model=fallback_model,
            fallback_timeout_ms=fallback_timeout_ms,
            fallback_min_interval_sec=fallback_min_interval_sec,
            evaluator_model=evaluator_model,
            evaluator_timeout_ms=evaluator_timeout_ms,
            evaluator_enabled=evaluator_enabled,
            judge_confidence_threshold=judge_confidence_threshold,
            judge_strict_validation=judge_strict_validation,""",
)

if check_syntax("config.py", cfg):
    write(cfg_path, cfg)
    print("  [WRITTEN] config.py")
else:
    OK = False
    print("  [ABORTED] config.py not written")


# ═══════════════════════════════════════════════════════════
# 2. models.py — add JudgeResult after Qwen3LayerResult
# ═══════════════════════════════════════════════════════════
print("\n── 2. models.py ──")
mdl_path = os.path.join(BASE, "models.py")
backup(mdl_path)
mdl = read(mdl_path)

JUDGE_RESULT_CLASS = '''
@dataclass(frozen=True)
class JudgeResult:
    """
    [PATCH-11] Output of AIJudge.evaluate() for one position.

    Supersedes Qwen3LayerResult for the 3-tier judge path.

    tier: "t1"=primary, "t2"=fallback, "t3"=disagreement-resolved, "t0"=formula
    shadow_mode: "shadow" | "hybrid" | "live"
    fallback: True when AI tiers both failed and formula was used (tier="t0")
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
        """Returns Qwen3LayerResult for backward compat with ExitDecision.qwen3_result."""
        reason = f"{self.tier}:{','.join(self.reason_codes) or self.risk_note or 'ok'}"
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
        """Convenience alias for main.py live_reason assignment."""
        return ",".join(self.reason_codes) or self.risk_note or "ai_judge"

'''

INSERT_AFTER = """    raw: str


@dataclass(frozen=True)
class DecisionSnapshot:"""

if INSERT_AFTER in mdl:
    mdl = mdl.replace(
        INSERT_AFTER,
        f"""    raw: str
{JUDGE_RESULT_CLASS}
@dataclass(frozen=True)
class DecisionSnapshot:""",
        1,
    )
    print("  [PATCHED] JudgeResult inserted after Qwen3LayerResult")
elif "class JudgeResult" in mdl:
    print("  [SKIP] JudgeResult already present in models.py")
else:
    print("  [WARN] Could not find insert target — skipping models.py patch")

if check_syntax("models.py", mdl):
    write(mdl_path, mdl)
    print("  [WRITTEN] models.py")
else:
    OK = False
    print("  [ABORTED] models.py not written")


# ═══════════════════════════════════════════════════════════
# 3. scoring_engine.py — extend FORMULA_QTY_MAP
# ═══════════════════════════════════════════════════════════
print("\n── 3. scoring_engine.py ──")
se_path = os.path.join(BASE, "scoring_engine.py")
backup(se_path)
se = read(se_path)

se = replace_once(
    "scoring_engine.py FORMULA_QTY_MAP",
    se,
    """FORMULA_QTY_MAP: dict = {
    FULL_CLOSE:        1.0,
    TIME_STOP_EXIT:    1.0,
    PARTIAL_CLOSE_25:  0.25,
    PARTIAL_CLOSE_50:  0.50,
    PARTIAL_CLOSE_75:  0.75,
    TIGHTEN_TRAIL:     None,
    MOVE_TO_BREAKEVEN: None,
    HOLD:              None,
}""",
    """FORMULA_QTY_MAP: dict = {
    FULL_CLOSE:        1.0,
    TIME_STOP_EXIT:    1.0,
    PARTIAL_CLOSE_25:  0.25,
    PARTIAL_CLOSE_50:  0.50,
    PARTIAL_CLOSE_75:  0.75,
    TIGHTEN_TRAIL:     None,
    MOVE_TO_BREAKEVEN: None,
    HOLD:              None,
    # ── PATCH-11 actions (AIJudge enum) ──────────────────────────────────────
    "REDUCE_25":           0.25,   # exit 25 %
    "REDUCE_50":           0.50,   # exit 50 %
    "HARVEST_70_KEEP_30":  0.70,   # exit 70 %, keep 30 %
    "DEFENSIVE_HOLD":      None,   # hold with heightened monitoring
    "TOXICITY_UNWIND":     1.0,    # full close due to signal toxicity
    "QUARANTINE":          None,   # lock action, await manual review
}""",
)

if check_syntax("scoring_engine.py", se):
    write(se_path, se)
    print("  [WRITTEN] scoring_engine.py")
else:
    OK = False
    print("  [ABORTED] scoring_engine.py not written")


# ═══════════════════════════════════════════════════════════
# 4. main.py — wire AIJudge + OfflineEvaluator
# ═══════════════════════════════════════════════════════════
print("\n── 4. main.py ──")
main_path = os.path.join(BASE, "main.py")
backup(main_path)
main = read(main_path)

# 4a. Replace imports
main = replace_once(
    "main.py imports",
    main,
    """from .qwen3_layer import Qwen3Layer
from .ai_brain import AIBrain, DeepSeekEvaluator""",
    """from .groq_client import GroqModelClient
from .ai_judge import AIJudge, OfflineEvaluator""",
)

# 4b. Replace __init__ AI construction block
main = replace_once(
    "main.py __init__ AI construction",
    main,
    """        # PATCH-11: 3-tier AI brain.
        # Tier 1 (primary):  Qwen3-32b   via Groq — live exit decisions
        # Tier 2 (fallback): Mistral 3.1 via Groq — takes over on Tier 1 failure / 429
        _ai_primary = Qwen3Layer(
            endpoint=config.qwen3_endpoint,
            timeout_ms=config.qwen3_timeout_ms,
            shadow=config.qwen3_shadow,
            model=config.qwen3_model,
            api_key=config.qwen3_api_key,
            min_interval_sec=config.qwen3_min_interval_sec,
        )
        _ai_fallback = Qwen3Layer(
            endpoint=config.qwen3_endpoint,
            timeout_ms=config.mistral_timeout_ms,
            shadow=config.qwen3_shadow,
            model=config.mistral_model,
            api_key=config.qwen3_api_key,
            min_interval_sec=config.mistral_min_interval_sec,
        )
        self._qwen3 = AIBrain(primary=_ai_primary, fallback=_ai_fallback)
        # Tier 3 (offline evaluator): DeepSeek-R1-Distill via Groq
        self._deepseek_eval = DeepSeekEvaluator(
            endpoint=config.qwen3_endpoint,
            model=config.deepseek_model,
            api_key=config.qwen3_api_key,
            timeout_ms=config.deepseek_timeout_ms,
            enabled=config.deepseek_enabled,
        )""",
    """        # PATCH-11 (proper): 3-tier AIJudge.
        # Tier 1 (primary):  qwen/qwen3-32b          via Groq — live exit decisions
        # Tier 2 (fallback): openai/gpt-oss-20b       via Groq — hard/soft fail escalation
        # Offline evaluator: llama-3.3-70b-versatile  via Groq — replay forensics only
        _primary_client = GroqModelClient(
            endpoint=config.qwen3_endpoint,
            model=config.qwen3_model,
            api_key=config.qwen3_api_key,
            timeout_ms=config.qwen3_timeout_ms,
            min_interval_sec=config.qwen3_min_interval_sec,
        )
        _fallback_client = GroqModelClient(
            endpoint=config.qwen3_endpoint,
            model=config.fallback_model,
            api_key=config.qwen3_api_key,
            timeout_ms=config.fallback_timeout_ms,
            min_interval_sec=config.fallback_min_interval_sec,
        )
        _judge_shadow_mode = "shadow" if config.qwen3_shadow else "live"
        self._qwen3 = AIJudge(
            primary=_primary_client,
            fallback=_fallback_client,
            shadow_mode=_judge_shadow_mode,
            confidence_threshold=config.judge_confidence_threshold,
            strict_validation=config.judge_strict_validation,
        )
        # Offline evaluator — NEVER called from the tick path; only from ReplayWriter.
        _eval_client = GroqModelClient(
            endpoint=config.qwen3_endpoint,
            model=config.evaluator_model,
            api_key=config.qwen3_api_key,
            timeout_ms=config.evaluator_timeout_ms,
            min_interval_sec=3.0,
        )
        self._deepseek_eval = OfflineEvaluator(
            client=_eval_client,
            enabled=config.evaluator_enabled,
        )""",
)

# 4c. Fix _tick shadow check + reason access
main = replace_once(
    "main.py _tick shadow check",
    main,
    """                            if self._cfg.qwen3_shadow or qr.fallback:
                                live_action = score_state.formula_action
                                live_confidence = score_state.formula_confidence
                                live_reason = score_state.formula_reason
                                live_urgency = score_state.formula_urgency
                            else:
                                live_action = qr.action
                                live_confidence = qr.confidence
                                live_reason = qr.reason
                                # urgency stays formula-derived (Qwen3 does not score urgency)
                                live_urgency = score_state.formula_urgency
                            dec = ExitDecision(
                                snapshot=snap,
                                action=live_action,
                                reason=live_reason,
                                urgency=live_urgency,
                                R_net=p.R_net,
                                confidence=live_confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=FORMULA_QTY_MAP.get(live_action),
                                dry_run=self._cfg.dry_run,
                                score_state=score_state,
                                qwen3_result=qr,
                            )""",
    """                            # PATCH-11 (proper): shadow_mode is baked into JudgeResult.
                            # "shadow" or tier="t0" (formula fallback) → formula drives live.
                            if qr.shadow_mode == "shadow" or qr.fallback:
                                live_action = score_state.formula_action
                                live_confidence = score_state.formula_confidence
                                live_reason = score_state.formula_reason
                                live_urgency = score_state.formula_urgency
                            else:
                                live_action = qr.action
                                live_confidence = qr.confidence
                                live_reason = qr.reason  # JudgeResult.reason property
                                # urgency stays formula-derived (judge does not score urgency)
                                live_urgency = score_state.formula_urgency
                            dec = ExitDecision(
                                snapshot=snap,
                                action=live_action,
                                reason=live_reason,
                                urgency=live_urgency,
                                R_net=p.R_net,
                                confidence=live_confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=FORMULA_QTY_MAP.get(live_action),
                                dry_run=self._cfg.dry_run,
                                score_state=score_state,
                                qwen3_result=qr.as_qwen3_layer_result(),
                            )""",
)

if check_syntax("main.py", main):
    write(main_path, main)
    print("  [WRITTEN] main.py")
else:
    OK = False
    print("  [ABORTED] main.py not written")


# ═══════════════════════════════════════════════════════════
# 5. Final summary
# ═══════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════════")
if OK:
    print("PATCH-11 (proper) applied SUCCESSFULLY.")
    print("Next: restart quantum-exit-management-agent")
else:
    print("PATCH-11 encountered ERRORS — review above output.")
    print("Backups are at <file>.bak11")
