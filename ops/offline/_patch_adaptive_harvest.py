"""
Patch: Adaptive Harvest Engine (PATCH-9A)
=========================================
Replaces the rigid T1/T2/T3 (2R/4R/6R) fixed thresholds with a
regime-aware, giveback-triggered continuous harvest algorithm.

Changes:
  1. NEW: adaptive_harvest_engine.py
  2. models.py       — add market_regime: str = "NORMAL"
  3. redis_io.py     — add get_market_regime() async method
  4. perception.py   — add regime lookup, pass to PerceptionResult
  5. decision_engine.py — replace Rules 4/4b/4c with adaptive harvest
  6. scoring_engine.py  — D3 giveback row -> PARTIAL_CLOSE (not just TIGHTEN_TRAIL)
"""
import pathlib, sys

BASE = pathlib.Path("/opt/quantum/microservices/exit_management_agent")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CREATE adaptive_harvest_engine.py
# ─────────────────────────────────────────────────────────────────────────────
AHE_SRC = '''\
"""adaptive_harvest_engine: regime-aware, giveback-triggered adaptive harvest.

Philosophy
----------
Instead of waiting for fixed R multiples (2R / 4R / 6R), this engine:
  1. Selects a TP profile based on the current market regime.
  2. Scales each TP threshold by 1/sqrt(leverage) — same as the existing R targets.
  3. Applies a giveback discount: if profit is already being given back,
     lower the required R threshold (up to DISCOUNT_MAX at DISCOUNT_FULL_AT giveback).
  4. Returns the highest triggered TP level as a PARTIAL_CLOSE action.
  5. Enforces r_lock as a minimum floor (never harvest below break-even buffer).

Regime profiles (base R at 1x leverage, same unit as r_effective_t1):
  CHOP     — 0.8R / 1.5R / 2.5R   — extremely quick exits (avoid whipsaws)
  RANGE    — 1.0R / 2.0R / 3.5R   — quick scalp exits
  VOLATILE — 1.2R / 2.5R / 4.5R   — moderate exits with wider tolerance
  TREND    — 2.0R / 4.0R / 6.0R   — let profits run (matches legacy thresholds)
  NORMAL   — 1.5R / 3.0R / 5.0R   — default balanced (slightly more aggressive)

Human analogy
-------------
  - Position in RANGE regime at R=1.1 -> take 25% off. Don't wait for 2R.
  - Position in NORMAL at R=1.8 giving back 25% of peak ->
    effective TP2 threshold = 3.0 * (1-0.25) = 2.25R, not reached.
    effective TP1 threshold = 1.5 * (1-0.25) = 1.125R < 1.8R -> HARVEST 25%.
  - Position in CHOP at any R giving back 20% ->
    effective TP1 = 0.8 * 0.8 = 0.64R -> harvest as soon as above lock.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

# ── Regime TP profiles ───────────────────────────────────────────────────────
# Each: (base_r_unscaled, close_fraction, label)
# base_r_unscaled: R multiple at 1x leverage (will be divided by sqrt(lev))
# close_fraction: raw fraction -> mapped to PARTIAL_CLOSE_25/50/75
REGIME_PROFILES: dict = {
    "CHOP":     [(0.8, 0.25, "TP1"), (1.5, 0.50, "TP2"), (2.5, 0.75, "TP3")],
    "RANGE":    [(1.0, 0.25, "TP1"), (2.0, 0.50, "TP2"), (3.5, 0.75, "TP3")],
    "VOLATILE": [(1.2, 0.25, "TP1"), (2.5, 0.50, "TP2"), (4.5, 0.75, "TP3")],
    "TREND":    [(2.0, 0.25, "TP1"), (4.0, 0.50, "TP2"), (6.0, 0.75, "TP3")],
    "NORMAL":   [(1.5, 0.25, "TP1"), (3.0, 0.50, "TP2"), (5.0, 0.75, "TP3")],
}

# ── Giveback discount ────────────────────────────────────────────────────────
# At or above FULL_DISCOUNT_AT giveback: apply MAX_DISCOUNT to the R threshold.
# Linear interpolation between MIN_GIVEBACK_ACTIVATE and FULL_DISCOUNT_AT.
# Example: 30% giveback -> threshold * (1 - 0.30) -> fires 30% earlier.
_MIN_GIVEBACK_ACTIVATE: float = 0.10  # ignore noise below 10% giveback
_FULL_DISCOUNT_AT: float      = 0.35  # 35%+ giveback = maximum discount
_MAX_DISCOUNT: float           = 0.35  # up to 35% threshold reduction

# ── Action constants (must match decision_engine.py, scoring_engine.py) ──────
PARTIAL_CLOSE_25 = "PARTIAL_CLOSE_25"
PARTIAL_CLOSE_50 = "PARTIAL_CLOSE_50"
PARTIAL_CLOSE_75 = "PARTIAL_CLOSE_75"


def get_adaptive_harvest(
    R_net: float,
    giveback_pct: float,
    r_lock: float,
    leverage: float,
    regime: str = "NORMAL",
) -> Optional[Tuple[str, float, float, str]]:
    """
    Compute regime-aware adaptive harvest decision.

    Args:
        R_net:        Current R-multiple (signed; negative = loss).
        giveback_pct: Fraction [0..1] of peak profit already given back.
        r_lock:       Leverage-scaled break-even lock R (minimum to harvest).
        leverage:     Position leverage (>= 1.0).
        regime:       Market regime (CHOP/RANGE/VOLATILE/TREND/NORMAL).
                      Case-insensitive. Unknown values fall back to NORMAL.

    Returns:
        (action, qty_fraction, confidence, reason)  if a harvest fires.
        None  if position is below lock or no TP threshold reached.

    Notes:
        - R_net must be >= r_lock for any harvest.
        - Giveback in [_MIN_GIVEBACK_ACTIVATE, _FULL_DISCOUNT_AT] gives proportional discount.
        - Giveback >= _FULL_DISCOUNT_AT applies full _MAX_DISCOUNT.
        - Effective threshold is always >= r_lock (floors at break-even).
        - Returns the HIGHEST triggered TP level (take the most applicable action).
    """
    if R_net < r_lock or r_lock <= 0.0:
        return None

    scale = math.sqrt(max(float(leverage), 1.0))
    regime_key = regime.upper() if regime else "NORMAL"
    profile_raw = REGIME_PROFILES.get(regime_key, REGIME_PROFILES["NORMAL"])

    # Scale R thresholds for leverage
    scaled = [(base_r / scale, frac, label) for base_r, frac, label in profile_raw]

    # Giveback discount
    if giveback_pct >= _MIN_GIVEBACK_ACTIVATE:
        raw = (giveback_pct - _MIN_GIVEBACK_ACTIVATE) / max(
            _FULL_DISCOUNT_AT - _MIN_GIVEBACK_ACTIVATE, 0.001
        )
        discount = min(raw, 1.0) * _MAX_DISCOUNT
    else:
        discount = 0.0

    # Check TP levels from highest to lowest; return the highest triggered
    best: Optional[Tuple[str, float, float, str]] = None
    for r_base, frac, label in reversed(scaled):
        effective = max(r_base * (1.0 - discount), r_lock)
        if R_net >= effective:
            action, qty_frac = _map_fraction_to_action(frac)
            conf = min(0.70 + giveback_pct * 0.30, 0.92)
            gb_str = f"{giveback_pct:.0%}" if giveback_pct >= 0.01 else "none"
            disc_str = f", giveback-discount \u2212{discount:.0%}" if discount > 0.01 else ""
            reason = (
                f"Adaptive harvest {label}: R_net={R_net:.2f} \u2265 "
                f"{effective:.2f}R [{regime_key} regime{disc_str}] "
                f"giveback={gb_str}"
            )
            best = (action, qty_frac, conf, reason)
            break  # highest triggered (reversed), stop here

    return best


def _map_fraction_to_action(close_fraction: float) -> Tuple[str, float]:
    """Map raw close fraction to the nearest allowed PARTIAL_CLOSE action."""
    if close_fraction >= 0.60:
        return PARTIAL_CLOSE_75, 0.75
    elif close_fraction >= 0.35:
        return PARTIAL_CLOSE_50, 0.50
    else:
        return PARTIAL_CLOSE_25, 0.25
'''

ahe_path = BASE / "adaptive_harvest_engine.py"
ahe_path.write_text(AHE_SRC)
print("  [CREATED] adaptive_harvest_engine.py")

# ─────────────────────────────────────────────────────────────────────────────
# 2. models.py — add market_regime field to PerceptionResult
# ─────────────────────────────────────────────────────────────────────────────
f = BASE / "models.py"
src = f.read_text()
old_models = "    r_effective_t1: float\n    r_effective_t2: float      # 4R/sqrt(lev) -> PARTIAL_CLOSE_50\n    r_effective_t3: float      # 6R/sqrt(lev) -> PARTIAL_CLOSE_75\n    r_effective_lock: float"
new_models = (
    "    r_effective_t1: float\n"
    "    r_effective_t2: float      # 4R/sqrt(lev) -> PARTIAL_CLOSE_50\n"
    "    r_effective_t3: float      # 6R/sqrt(lev) -> PARTIAL_CLOSE_75\n"
    "    r_effective_lock: float\n"
    "    market_regime: str = \"NORMAL\"   # CHOP|RANGE|VOLATILE|TREND|NORMAL (PATCH-9A)"
)
assert old_models in src, "models.py: PerceptionResult R targets not found"
f.write_text(src.replace(old_models, new_models, 1))
print("  [OK] models.py")

# ─────────────────────────────────────────────────────────────────────────────
# 3. redis_io.py — add get_market_regime() after get_mark_price_from_ticker()
# ─────────────────────────────────────────────────────────────────────────────
f = BASE / "redis_io.py"
src = f.read_text()

# Find insertion point: after the closing of get_mark_price_from_ticker()
insertion_marker = "    async def get_mark_price_from_ticker(self, symbol: str) -> Optional[float]:"
# We'll insert the new method before the WRITE OPERATIONS section
write_ops_marker = "    # ── WRITE OPERATIONS (guarded)"
old_rio = write_ops_marker
new_rio = (
    "    async def get_market_regime(self, symbol: str) -> str:\n"
    "        \"\"\"\n"
    "        Read market regime for a symbol from Redis.\n"
    "        Tries multiple key patterns (meta_regime, market state).\n"
    "        Returns 'NORMAL' if no regime data is available.\n"
    "        \"\"\"\n"
    "        candidates = [\n"
    "            (f\"quantum:meta_regime:{symbol}\", \"state\"),\n"
    "            (f\"quantum:meta_regime:{symbol}\", \"regime\"),\n"
    "            (f\"quantum:market:regime:{symbol}\", \"regime\"),\n"
    "            (f\"quantum:market:{symbol}\", \"regime\"),\n"
    "            (f\"quantum:regime:{symbol}\", \"state\"),\n"
    "        ]\n"
    "        for key, field in candidates:\n"
    "            try:\n"
    "                val = await self._client.hget(key, field)\n"
    "                if val and isinstance(val, str) and val.upper() in (\n"
    "                    \"CHOP\", \"RANGE\", \"VOLATILE\", \"TREND\", \"NORMAL\",\n"
    "                    \"TRENDING\", \"RANGING\", \"CHOPPY\",\n"
    "                ):\n"
    "                    # Normalise aliases\n"
    "                    mapped = {\"TRENDING\": \"TREND\", \"RANGING\": \"RANGE\", \"CHOPPY\": \"CHOP\"}\n"
    "                    return mapped.get(val.upper(), val.upper())\n"
    "            except Exception:\n"
    "                pass\n"
    "        return \"NORMAL\"\n"
    "\n"
    "    # ── WRITE OPERATIONS (guarded)"
)
assert old_rio in src, "redis_io.py: WRITE OPERATIONS marker not found"
f.write_text(src.replace(old_rio, new_rio, 1))
print("  [OK] redis_io.py")

# ─────────────────────────────────────────────────────────────────────────────
# 4. perception.py — add regime reading, pass to PerceptionResult
# ─────────────────────────────────────────────────────────────────────────────
f = BASE / "perception.py"
src = f.read_text()

# 4a: add regime lookup in compute() after r_targets
old_perc_r = (
    "        r_t1, r_t2, r_t3, r_lock = _get_r_targets(snapshot.leverage)\n"
    "\n"
    "        return PerceptionResult("
)
new_perc_r = (
    "        r_t1, r_t2, r_t3, r_lock = _get_r_targets(snapshot.leverage)\n"
    "\n"
    "        # ── 7. Market regime (PATCH-9A) ────────────────────────────────────\n"
    "        regime = \"NORMAL\"\n"
    "        if self._redis is not None:\n"
    "            try:\n"
    "                regime = await self._redis.get_market_regime(snapshot.symbol)\n"
    "            except Exception as exc:\n"
    "                _log.debug(\"%s: regime lookup failed: %s\", snapshot.symbol, exc)\n"
    "\n"
    "        return PerceptionResult("
)
assert old_perc_r in src, "perception.py: r_targets + PerceptionResult return not found"
src = src.replace(old_perc_r, new_perc_r, 1)

# 4b: pass market_regime to PerceptionResult constructor
old_perc_result = (
    "            r_effective_t1=r_t1,\n"
    "            r_effective_t2=r_t2,\n"
    "            r_effective_t3=r_t3,\n"
    "            r_effective_lock=r_lock,\n"
    "        )"
)
new_perc_result = (
    "            r_effective_t1=r_t1,\n"
    "            r_effective_t2=r_t2,\n"
    "            r_effective_t3=r_t3,\n"
    "            r_effective_lock=r_lock,\n"
    "            market_regime=regime,\n"
    "        )"
)
assert old_perc_result in src, "perception.py: PerceptionResult constructor tail not found"
f.write_text(src.replace(old_perc_result, new_perc_result, 1))
print("  [OK] perception.py")

# ─────────────────────────────────────────────────────────────────────────────
# 5. decision_engine.py — replace fixed T1/T2/T3 rules with adaptive harvest
# ─────────────────────────────────────────────────────────────────────────────
f = BASE / "decision_engine.py"
src = f.read_text()

# 5a: add import at the top (after existing imports)
old_import = "from .models import ExitDecision, PerceptionResult, PositionSnapshot"
new_import = (
    "from .models import ExitDecision, PerceptionResult, PositionSnapshot\n"
    "from .adaptive_harvest_engine import get_adaptive_harvest"
)
assert old_import in src, "decision_engine.py: models import not found"
src = src.replace(old_import, new_import, 1)

# 5b: replace Rules 4c / 4b / 4 (the three T1/T2/T3 rules) with single adaptive rule
old_rules_4 = (
    "        # ── Rule 4c: Partial harvest at T3 (6R) ──────────────────────────\n"
    "        if p.R_net >= p.r_effective_t3:\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=PARTIAL_CLOSE_75,\n"
    "                reason=(\n"
    "                    f\"Harvest T3: R_net={p.R_net:.2f} >= T3={p.r_effective_t3:.2f}R \"\n"
    "                    f\"(leverage={snap.leverage:.0f}x scaled) -- take 75%\"\n"
    "                ),\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=0.80,\n"
    "                suggested_qty_fraction=0.75,\n"
    "                dry_run=dry_run,\n"
    "            )\n"
    "\n"
    "        # ── Rule 4b: Partial harvest at T2 (4R) ──────────────────────────\n"
    "        if p.R_net >= p.r_effective_t2:\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=PARTIAL_CLOSE_50,\n"
    "                reason=(\n"
    "                    f\"Harvest T2: R_net={p.R_net:.2f} >= T2={p.r_effective_t2:.2f}R \"\n"
    "                    f\"(leverage={snap.leverage:.0f}x scaled) -- take 50%\"\n"
    "                ),\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=0.77,\n"
    "                suggested_qty_fraction=0.50,\n"
    "                dry_run=dry_run,\n"
    "            )\n"
    "\n"
    "        # ── Rule 4: Partial harvest at T1 (2R) ────────────────────────────\n"
    "        if p.R_net >= p.r_effective_t1:\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=PARTIAL_CLOSE_25,\n"
    "                reason=(\n"
    "                    f\"Harvest T1: R_net={p.R_net:.2f} >= T1={p.r_effective_t1:.2f}R \"\n"
    "                    f\"(leverage={snap.leverage:.0f}x scaled)\"\n"
    "                ),\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=0.75,\n"
    "                suggested_qty_fraction=0.25,\n"
    "                dry_run=dry_run,\n"
    "            )"
)
new_rules_4 = (
    "        # ── Rule 4: Adaptive harvest (PATCH-9A: regime-aware + giveback-triggered) ───\n"
    "        # Replaces fixed T1/T2/T3 (2R/4R/6R) with regime-scaled dynamic thresholds.\n"
    "        # CHOP: harvests at 0.8R | RANGE: 1.0R | VOLATILE: 1.2R | TREND: 2.0R | NORMAL: 1.5R\n"
    "        # Giveback discount: up to -35% on threshold when position is giving back profit.\n"
    "        # Never fires below r_effective_lock (break-even floor always respected).\n"
    "        _harvest = get_adaptive_harvest(\n"
    "            R_net=p.R_net,\n"
    "            giveback_pct=p.giveback_pct,\n"
    "            r_lock=p.r_effective_lock,\n"
    "            leverage=snap.leverage,\n"
    "            regime=getattr(p, \"market_regime\", \"NORMAL\"),\n"
    "        )\n"
    "        if _harvest is not None:\n"
    "            _h_action, _h_frac, _h_conf, _h_reason = _harvest\n"
    "            return _decision(\n"
    "                snap=snap,\n"
    "                action=_h_action,\n"
    "                reason=_h_reason,\n"
    "                urgency=URGENCY_MEDIUM,\n"
    "                r_net=p.R_net,\n"
    "                confidence=_h_conf,\n"
    "                suggested_qty_fraction=_h_frac,\n"
    "                dry_run=dry_run,\n"
    "            )"
)
assert old_rules_4 in src, "decision_engine.py: T1/T2/T3 rule block not found"
f.write_text(src.replace(old_rules_4, new_rules_4, 1))
print("  [OK] decision_engine.py")

# ─────────────────────────────────────────────────────────────────────────────
# 6. scoring_engine.py — update D3 giveback row to PARTIAL_CLOSE (not just trail)
# ─────────────────────────────────────────────────────────────────────────────
f = BASE / "scoring_engine.py"
src = f.read_text()

old_d3_row = (
    "    # Row 4: giveback dominant\n"
    "    if exit_score >= _THRESH_GIVEBACK and d3 >= _THRESH_GIVEBACK_D3:\n"
    "        return (\n"
    "            TIGHTEN_TRAIL,\n"
    "            URGENCY_MEDIUM,\n"
    "            f\"Score={exit_score:.3f} D3={d3:.3f} — giveback at profit\",\n"
    "        )"
)
new_d3_row = (
    "    # Row 4: giveback dominant (PATCH-9A: partial close when in profit, trail below)\n"
    "    if exit_score >= _THRESH_GIVEBACK and d3 >= _THRESH_GIVEBACK_D3:\n"
    "        # If already past T1 profit zone with significant giveback -> partial close\n"
    "        if R_net >= r_effective_t1:\n"
    "            return (\n"
    "                PARTIAL_CLOSE_50,\n"
    "                URGENCY_MEDIUM,\n"
    "                f\"Score={exit_score:.3f} D3={d3:.3f} — giveback at T1+ profit, lock 50%\",\n"
    "            )\n"
    "        elif R_net > 0:\n"
    "            return (\n"
    "                TIGHTEN_TRAIL,\n"
    "                URGENCY_MEDIUM,\n"
    "                f\"Score={exit_score:.3f} D3={d3:.3f} — giveback below T1, tighten trail\",\n"
    "            )\n"
    "        return (\n"
    "            TIGHTEN_TRAIL,\n"
    "            URGENCY_MEDIUM,\n"
    "            f\"Score={exit_score:.3f} D3={d3:.3f} — giveback at profit\",\n"
    "        )"
)
assert old_d3_row in src, "scoring_engine.py: D3 giveback row not found"

# Also need to add R_net and r_effective_t1 to _apply_decision_map signature
old_sig = (
    "def _apply_decision_map(\n"
    "    exit_score: float,\n"
    "    d1: float,\n"
    "    d2: float,\n"
    "    d3: float,\n"
    "    d4: float,\n"
    "    R_net: float,\n"
    "    r_effective_t1: float,\n"
    ") -> tuple:"
)
assert old_sig in src, "scoring_engine.py: _apply_decision_map signature not found"
# The signature already has R_net and r_effective_t1, good.

src = src.replace(old_d3_row, new_d3_row, 1)

# Also update the call to _apply_decision_map to pass r_effective_t1 - it already has it.
f.write_text(src)
print("  [OK] scoring_engine.py")

print("\nAll PATCH-9A changes applied.")
print("Next: restart quantum-exit-management-agent")
