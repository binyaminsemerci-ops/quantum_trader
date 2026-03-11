"""
PATCH-10B: Wire kill chain + council into ScoringEngine (scoring_mode=ai path).

ScoringEngine.score() now reads kill_chain_level, thesis_score, conviction_budget,
council_action from PerceptionResult (added in PATCH-10A) and uses them to
override the formula action when the kill chain fires.
"""
import os

BASE = "/opt/quantum/microservices/exit_management_agent"

def read(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return f.read()

def write(name, content):
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        f.write(content)

def patch(name, old, new):
    content = read(name)
    if old not in content:
        raise AssertionError(f"{name}: target not found:\n{repr(old[:120])}")
    count = content.count(old)
    if count > 1:
        raise AssertionError(f"{name}: ambiguous — {count} matches")
    write(name, content.replace(old, new, 1))
    print(f"  [OK] {name}")

# ── Target: add kill chain override between _apply_decision_map call and return ──

OLD_SE = (
    "        action, urgency, reason = _apply_decision_map(\n"
    "            exit_score=exit_score,\n"
    "            d1=d1, d2=d2, d3=d3, d4=d4,\n"
    "            R_net=p.R_net,\n"
    "            r_effective_t1=p.r_effective_t1,\n"
    "        )\n"
    "\n"
    "        return ExitScoreState("
)

NEW_SE = (
    "        action, urgency, reason = _apply_decision_map(\n"
    "            exit_score=exit_score,\n"
    "            d1=d1, d2=d2, d3=d3, d4=d4,\n"
    "            R_net=p.R_net,\n"
    "            r_effective_t1=p.r_effective_t1,\n"
    "        )\n"
    "\n"
    "        # ── PATCH-10: Kill chain + council override ─────────────────────\n"
    "        # Reads the pre-computed signals from PerceptionResult (added in PATCH-10A).\n"
    "        # getattr with defaults keeps this backward-compatible with older PerceptionResults.\n"
    "        _kc       = getattr(p, 'kill_chain_level',    0)\n"
    "        _kc_rsn   = getattr(p, 'kill_chain_reason',   '')\n"
    "        _cncl_act = getattr(p, 'council_action',      HOLD)\n"
    "        _cncl_cf  = getattr(p, 'council_confidence',  0.0)\n"
    "        _cncl_cs  = getattr(p, 'council_consensus',   1.0)\n"
    "        _cncl_rsn = getattr(p, 'council_reasoning',   '')\n"
    "\n"
    "        if _kc >= 4:\n"
    "            action     = FULL_CLOSE\n"
    "            urgency    = URGENCY_HIGH\n"
    "            reason     = f'KillChain-L{_kc}: {_kc_rsn}'\n"
    "            exit_score = max(exit_score, 0.85)\n"
    "        elif _kc >= 3 and p.R_net >= p.r_effective_lock:\n"
    "            action     = PARTIAL_CLOSE_50\n"
    "            urgency    = URGENCY_MEDIUM\n"
    "            reason     = f'KillChain-L3 unwind: {_kc_rsn}'\n"
    "            exit_score = max(exit_score, 0.65)\n"
    "        elif _kc >= 2 and p.R_net >= p.r_effective_lock:\n"
    "            if action in (HOLD, TIGHTEN_TRAIL, MOVE_TO_BREAKEVEN):\n"
    "                action     = PARTIAL_CLOSE_25\n"
    "                urgency    = URGENCY_MEDIUM\n"
    "                reason     = f'KillChain-L2 de-risk: {_kc_rsn}'\n"
    "                exit_score = max(exit_score, 0.35)\n"
    "        elif (\n"
    "            _cncl_act not in (HOLD, TIGHTEN_TRAIL, MOVE_TO_BREAKEVEN)\n"
    "            and _cncl_cf  > 0.65\n"
    "            and _cncl_cs  > 0.55\n"
    "            and _kc       >= 1\n"
    "            and p.R_net   >= p.r_effective_lock\n"
    "            and action    == HOLD\n"
    "        ):\n"
    "            action     = _cncl_act\n"
    "            urgency    = URGENCY_MEDIUM\n"
    "            reason     = f'Council({_cncl_cs:.0%}): {_cncl_rsn}'\n"
    "            exit_score = max(exit_score, 0.20)\n"
    "\n"
    "        return ExitScoreState("
)

print("\n=== PATCH-10B: ScoringEngine kill chain wire ===\n")
patch("scoring_engine.py", OLD_SE, NEW_SE)
print("\nDone. Restart service.\n")
