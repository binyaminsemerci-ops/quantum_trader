#!/usr/bin/env python3
"""
_inject_replay_scenarios.py — Testnet scenario injection for EMA replay diversity.

Writes controlled quantum:position:{symbol} hashes to Redis, targeting every
formula decision zone that passes through Qwen3. The live EMA service processes
each position each tick, builds decision snapshots, and records them to the
pending-decision set. Deleting a position triggers OutcomeTracker → RewardEngine
→ ReplayWriter → quantum:stream:exit.replay.

Design constraints
------------------
- Only uses symbols without live positions (safe injection).
- Never writes R_net near the hard-guard threshold (-1.5) — hard guards bypass
  Qwen3 entirely, which defeats the divergence goal.
- Avoids the MoveToBreakeven / TightenTrail formula zones (R_net 0.48-0.61 at
  L10) which also bypass Qwen3.
- Uses leverage=10 so formula thresholds are achievable at realistic R_net levels.

Formula zones targeted (leverage=10, r_lock=0.474, r_t1=0.632):
  HOLD zone        : R_net in [-0.90, 0.0]    — formula=HOLD, Qwen3 called
  HOLD zone above  : R_net in [0.0, 0.48]     — formula=HOLD, Qwen3 called
  PARTIAL_CLOSE_25 : R_net >= 0.613           — formula=PARTIAL_CLOSE_25, Qwen3 called
  * Avoid R_net [0.48, 0.61] → MOVE_TO_BREAKEVEN (Qwen3 bypassed)
  * Avoid R_net <= -1.36 + ages → FULL_CLOSE formula via D1 score
  * Never R_net <= -1.5 → hard guard FULL_CLOSE, Qwen3 bypassed

Usage (run on VPS):
    /home/qt/quantum_trader_venv/bin/python _inject_replay_scenarios.py [--dry-run]

Requires:
    EXIT_AGENT_LOOP_SEC must be set to >=15 so Groq 429 rate is manageable.
    redis package: available in /home/qt/quantum_trader_venv
"""
from __future__ import annotations

import sys
import time
import math
import logging
from datetime import datetime, timezone
from typing import Optional

import redis

# ── Config ────────────────────────────────────────────────────────────────────
_REDIS_HOST = "127.0.0.1"
_REDIS_PORT = 6379

_ENTRY_RISK_USDT = 20.0   # fixed risk per scenario in USDT
_LEVERAGE = 10            # L10: r_t1≈0.632, r_lock≈0.474 (vs default L2 r_t1=1.414)

_HOLD_TICKS = 10          # ticks to hold each position; should match loops ≤ EXIT_AGENT_LOOP_SEC

# The sleep between inject and close must be >= EXIT_AGENT_LOOP_SEC * HOLD_TICKS.
# Script reads EXIT_AGENT_LOOP_SEC from the environment. Default 15 here.
import os
_LOOP_SEC = float(os.getenv("EXIT_AGENT_LOOP_SEC", "15"))
_HOLD_SEC = _HOLD_TICKS * _LOOP_SEC                 # seconds to hold each position
_POST_CLOSE_PAUSE = _LOOP_SEC * 2                   # let OutcomeTracker fire

# Real position symbols to NEVER overwrite during injection
_DO_NOT_TOUCH = frozenset({"BTCUSDT", "ETHUSDT"})

# ── Scenarios ─────────────────────────────────────────────────────────────────
# Each tuple: (symbol, side, target_R_net, description)
# Arranged to cover all Qwen3-reachable formula zones.
_SCENARIOS: list[tuple[str, str, float, str]] = [
    # -- HOLD zone: slight loss (D1 building, formula=HOLD, Qwen3 called) --
    ("SOLUSDT",   "LONG",  -0.15, "hold_slight_loss_sol_long"),
    ("SOLUSDT",   "SHORT", -0.25, "hold_moderate_loss_sol_short"),
    ("BNBUSDT",   "LONG",  -0.35, "hold_moderate_loss_bnb_long"),
    ("BNBUSDT",   "SHORT", -0.50, "hold_deeper_loss_bnb_short"),
    ("AVAXUSDT",  "LONG",  -0.20, "hold_slight_loss_avax_long"),
    ("AVAXUSDT",  "SHORT", -0.40, "hold_moderate_loss_avax_short"),
    ("NEARUSDT",  "LONG",  -0.30, "hold_moderate_loss_near_long"),
    ("NEARUSDT",  "SHORT", -0.60, "hold_deeper_loss_near_short"),
    ("DOGEUSDT",  "LONG",  -0.45, "hold_moderate_loss_doge_long"),
    ("DOGEUSDT",  "SHORT", -0.75, "hold_deep_loss_doge_short"),

    # -- HOLD zone: below-BE profit but far below MOVE_TO_BE threshold (R_net < 0.48) --
    ("SOLUSDT",   "LONG",   0.10, "hold_slight_profit_sol_long"),
    ("BNBUSDT",   "SHORT",  0.25, "hold_moderate_profit_bnb_short"),
    ("AVAXUSDT",  "LONG",   0.38, "hold_stronger_profit_avax_long"),
    ("NEARUSDT",  "SHORT",  0.45, "hold_near_MBE_threshold_near_short"),

    # -- PARTIAL_CLOSE_25 zone: R_net >= 0.613, formula triggers harvest --
    # Qwen3 may disagree (prefer HOLD to let profits run)
    ("SOLUSDT",   "SHORT",  0.65, "harvest_zone_sol_short"),
    ("BNBUSDT",   "LONG",   0.70, "harvest_zone_bnb_long"),
    ("AVAXUSDT",  "SHORT",  0.75, "harvest_zone_avax_short"),
    ("NEARUSDT",  "LONG",   0.80, "harvest_zone_near_long"),
    ("DOGEUSDT",  "SHORT",  0.85, "harvest_zone_doge_short"),
    ("SOLUSDT",   "LONG",   0.90, "harvest_strong_sol_long"),
    ("BNBUSDT",   "SHORT",  1.00, "harvest_strong_bnb_short"),
    ("AVAXUSDT",  "LONG",   1.10, "harvest_strong_avax_long"),

    # -- Second pass: repeat HOLD zone with different symbols for breadth --
    ("DOGEUSDT",  "LONG",  -0.55, "hold_deeper_loss_doge_long_v2"),
    ("SOLUSDT",   "SHORT", -0.70, "hold_deep_loss_sol_short_v2"),
    ("BNBUSDT",   "LONG",  -0.80, "hold_deepest_loss_bnb_long_v2"),
    ("AVAXUSDT",  "SHORT", -0.90, "hold_deepest_loss_avax_short_v2"),

    # -- Second pass: PARTIAL_CLOSE_25 with mixed sides --
    ("NEARUSDT",  "SHORT",  0.65, "harvest_v2_near_short"),
    ("DOGEUSDT",  "LONG",   0.72, "harvest_v2_doge_long"),
    ("SOLUSDT",   "SHORT",  0.78, "harvest_v2_sol_short"),

    # -- Edge cases: borderline R_net near formula decision boundaries --
    ("BNBUSDT",   "LONG",  -0.05, "near_zero_bnb_long"),     # almost breakeven, HOLD
    ("AVAXUSDT",  "SHORT",  0.02, "tiny_profit_avax_short"),  # tiny profit, HOLD
    ("NEARUSDT",  "LONG",   0.62, "at_harvest_boundary_near"),  # just at PARTIAL_CLOSE_25
    ("DOGEUSDT",  "SHORT",  0.63, "just_above_harvest_doge"),  # definite P25
    ("SOLUSDT",   "LONG",  -0.85, "near_D1_boundary_sol"),   # high D1, still HOLD
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
_log = logging.getLogger("scenario_injector")

# ── Redis helpers ─────────────────────────────────────────────────────────────

def get_mark_price(r: redis.Redis, symbol: str) -> float:
    """Read current mark price from quantum:ticker:{symbol}."""
    for field in (b"markPrice", b"mark_price", b"price"):
        raw = r.hget(f"quantum:ticker:{symbol}", field)
        if raw:
            return float(raw)
    raise ValueError(f"No mark price found for {symbol}")


def compute_entry_price(mark: float, side: str, R_net: float, qty: float, risk: float) -> float:
    """
    Compute entry_price that produces the target R_net at the current mark price.

    For LONG : R_net = (mark - entry) * qty / risk  →  entry = mark - R_net*risk/qty
    For SHORT: R_net = (entry - mark) * qty / risk  →  entry = mark + R_net*risk/qty
    """
    if side.upper() == "LONG":
        return mark - (R_net * risk / qty)
    else:
        return mark + (R_net * risk / qty)


def inject_position(
    r: redis.Redis,
    symbol: str,
    side: str,
    target_R_net: float,
    dry_run: bool = False,
) -> dict:
    """
    Write or preview a quantum:position:{symbol} hash.

    Returns dict with injection metadata.
    Raises ValueError if symbol is in _DO_NOT_TOUCH or no ticker available.
    """
    if symbol in _DO_NOT_TOUCH:
        raise ValueError(f"{symbol} is a live position — skipping")

    mark = get_mark_price(r, symbol)
    qty = (_ENTRY_RISK_USDT * _LEVERAGE) / mark   # standard sizing
    entry = compute_entry_price(mark, side, target_R_net, qty, _ENTRY_RISK_USDT)

    if entry <= 0:
        raise ValueError(f"Computed entry_price={entry:.4f} is non-positive for {symbol} {side} R={target_R_net}")

    if side.upper() == "LONG":
        unreal_pnl = (mark - entry) * qty
    else:
        unreal_pnl = (entry - mark) * qty

    now = int(time.time())
    pos_hash = {
        "symbol": symbol,
        "side": side.upper(),
        "quantity": f"{qty:.8f}",
        "entry_price": f"{entry:.8f}",
        "leverage": str(_LEVERAGE),
        "opened_at": str(now),
        "source": "scenario_injection",
        "risk_missing": "0",
        "sync_timestamp": str(now),
        "unrealized_pnl": f"{unreal_pnl:.8f}",
        "entry_risk_usdt": f"{_ENTRY_RISK_USDT:.8f}",
        "stop_loss": "0",
        "take_profit": "0",
    }

    if not dry_run:
        r.hset(f"quantum:position:{symbol}", mapping=pos_hash)

    return {
        "symbol": symbol,
        "side": side,
        "mark": mark,
        "entry": entry,
        "R_net": target_R_net,
        "qty": qty,
        "unreal_pnl": unreal_pnl,
    }


def close_position(r: redis.Redis, symbol: str, dry_run: bool = False) -> None:
    """Remove the position hash from Redis, triggering OutcomeTracker."""
    if not dry_run:
        r.delete(f"quantum:position:{symbol}")


# ── Score prediction helper (for pre-flight validation) ───────────────────────

def predict_formula_action(R_net: float, age_sec: float = 10.0, max_hold: float = 14400.0, leverage: float = 10.0) -> str:
    """
    Predict which formula action would fire for a fresh position.
    Mirrors scoring_engine.py logic — for planning purposes only.
    """
    scale = math.sqrt(max(leverage, 1.0))
    r_t1 = 2.0 / scale
    r_lock = 1.5 / scale

    # Dimensions
    d1 = max(0.0, min(1.0, -R_net / 1.5))
    if r_t1 != r_lock:
        d2 = max(0.0, min(1.0, (R_net - r_lock) / (r_t1 - r_lock)))
    else:
        d2 = 0.0
    d3 = 0.0  # no giveback on fresh inject
    age_frac = max(0.0, min(1.0, age_sec / max_hold))
    d4 = age_frac ** 1.5
    d5 = 0.0  # no SL set

    score = 0.30 * d1 + 0.25 * d2 + 0.20 * d3 + 0.15 * d4 + 0.10 * d5

    if score >= 0.27 and d1 >= 0.40:
        return "FULL_CLOSE"
    if score >= 0.22 and d2 >= 0.80:
        return "PARTIAL_CLOSE_25"
    if score >= 0.12 and d4 >= 0.85:
        return "TIME_STOP_EXIT"
    if score >= 0.08 and d3 >= 0.60:
        return "TIGHTEN_TRAIL"
    if score >= 0.015 and d2 >= 0.05:
        return "MOVE_TO_BREAKEVEN"
    return "HOLD"


def qwen3_will_be_called(formula_action: str) -> bool:
    """True if the formula action doesn't bypass Qwen3."""
    return formula_action not in ("TIGHTEN_TRAIL", "MOVE_TO_BREAKEVEN", "FULL_CLOSE")
    # Note: FULL_CLOSE from hard guard bypasses Qwen3, but FULL_CLOSE from scoring
    # still gets Qwen3. Conservatively treating formula-FULL_CLOSE as Qwen3-bypassed
    # here for planning; in practice scoring FULL_CLOSE DOES call Qwen3.


# ── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    r = redis.Redis(host=_REDIS_HOST, port=_REDIS_PORT)
    r.ping()

    replay_start = r.xlen("quantum:stream:exit.replay")
    audit_start = r.xlen("quantum:stream:exit.audit")

    _log.info(
        "=== Scenario Injector Start | dry_run=%s | loop_sec=%.0f | hold_ticks=%d | hold_sec=%.0f ===",
        dry_run, _LOOP_SEC, _HOLD_TICKS, _HOLD_SEC,
    )
    _log.info(
        "Replay stream at start: %d | Audit stream at start: %d",
        replay_start, audit_start,
    )
    _log.info(
        "Scenarios planned: %d | Expected new records: ~%d",
        len(_SCENARIOS), len(_SCENARIOS) * _HOLD_TICKS,
    )
    _log.info("")

    # Pre-flight: print prediction for each scenario
    _log.info("--- Pre-flight scenario plan ---")
    qwen3_eligible = 0
    bypass_count = 0
    for sym, side, R_net, label in _SCENARIOS:
        pred = predict_formula_action(R_net, age_sec=10.0)
        called = qwen3_will_be_called(pred)
        symbol_safe = sym not in _DO_NOT_TOUCH
        status = "OK" if (called and symbol_safe) else ("BYPASS_QWEN3" if not called else "SKIP_LIVE_POS")
        if called and symbol_safe:
            qwen3_eligible += 1
        else:
            bypass_count += 1
        _log.info("  [%s] %-10s %-6s R=%+.2f  formula=%-18s qwen3=%s",
                  status, sym, side, R_net, pred, "YES" if called else "NO")
    _log.info("")
    _log.info("Qwen3-eligible scenarios: %d | Bypassed: %d", qwen3_eligible, bypass_count)
    _log.info("")

    if dry_run:
        _log.info("[DRY-RUN] Exiting without injecting positions.")
        return

    scenarios_run = 0
    scenarios_skipped = 0
    records_at_scenario_start = replay_start

    for i, (sym, side, R_net, label) in enumerate(_SCENARIOS, 1):
        _log.info("[%d/%d] === Scenario: %s ===", i, len(_SCENARIOS), label)

        # Safety check
        if sym in _DO_NOT_TOUCH:
            _log.warning("  SKIP — %s is in _DO_NOT_TOUCH", sym)
            scenarios_skipped += 1
            continue

        # Pre-flight validation
        pred = predict_formula_action(R_net, age_sec=10.0)
        if not qwen3_will_be_called(pred):
            _log.warning(
                "  SKIP — formula=%s bypasses Qwen3 for %s R=%+.2f (would not generate divergence data)",
                pred, sym, R_net,
            )
            scenarios_skipped += 1
            continue

        # Inject
        try:
            info = inject_position(r, sym, side, R_net)
            _log.info(
                "  INJECT %s %s | mark=%.4f entry=%.4f R_net=%+.2f qty=%.6f | predicted=%s",
                sym, side, info["mark"], info["entry"], R_net, info["qty"], pred,
            )
        except Exception as exc:
            _log.error("  INJECT_FAIL: %s", exc)
            scenarios_skipped += 1
            continue

        # Hold: agent processes HOLD_TICKS ticks
        _log.info("  Holding %ds (%d ticks × %.0fs)...", int(_HOLD_SEC), _HOLD_TICKS, _LOOP_SEC)
        time.sleep(_HOLD_SEC)

        # Live mark price (may have drifted)
        try:
            live_mark = get_mark_price(r, sym)
            _log.info("  Mark drifted: %.4f → %.4f (%.3f%%)",
                      info["mark"], live_mark, 100 * (live_mark - info["mark"]) / info["mark"])
        except Exception:
            pass

        # Close (delete) — triggers OutcomeTracker
        close_position(r, sym)
        _log.info("  CLOSE %s → OutcomeTracker firing", sym)

        # Brief pause so OutcomeTracker can process the close
        time.sleep(_POST_CLOSE_PAUSE)

        replay_now = r.xlen("quantum:stream:exit.replay")
        new_records = replay_now - records_at_scenario_start
        records_at_scenario_start = replay_now
        scenarios_run += 1

        _log.info(
            "  --- After scenario %d/%d | +%d replay records | total=%d ---",
            i, len(_SCENARIOS), new_records, replay_now,
        )
        _log.info("")

    # Final
    replay_final = r.xlen("quantum:stream:exit.replay")
    audit_final = r.xlen("quantum:stream:exit.audit")
    new_replay = replay_final - replay_start
    _log.info("=== DONE ===")
    _log.info("Scenarios: run=%d skipped=%d", scenarios_run, scenarios_skipped)
    _log.info("Replay records: %d → %d (+%d)", replay_start, replay_final, new_replay)
    _log.info("Audit records : %d → %d (+%d)", audit_start, audit_final, audit_final - audit_start)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
