#!/usr/bin/env python3
"""
DAG 8 — FREEZE Exit Analyzer & Phased Recovery Plan
======================================================
Evaluates 5 criteria to determine when/how to safely exit FREEZE mode.
Outputs a recommendation and phase plan.

Phases:
  PHASE 0 — FREEZE (current): no new positions
  PHASE 1 — SHADOW:  shadow signals only, 0 live exposure
  PHASE 2 — PAPER:   paper trading on 5 symbols, validates execution path
  PHASE 3 — LIVE_XS: live on 5 symbols, 25% normal size
  PHASE 4 — LIVE_SM: live on 10 symbols, 50% normal size
  PHASE 5 — LIVE_NORMAL: full resume

Gate criteria (all must be GREEN for PHASE 1 transition):
  C1: Drawdown < DD_TARGET_PCT from peak
  C2: Layer 2 gate OPEN (accuracy >= 55%, n_trades >= 30)
  C3: Fear & Greed > FEAR_GREED_MIN (not extreme fear)
  C4: System health GREEN for HEALTH_STREAK_NEEDED consecutive checks
  C5: Operator approval (SET quantum:dag8:exit_approved APPROVED)

Operator commands:
  redis-cli SET quantum:dag8:exit_approved APPROVED     -- approve transition
  redis-cli SET quantum:dag8:exit_approved DENIED       -- deny transition
  redis-cli SET quantum:dag8:force_phase 2              -- force a specific phase (emergency use)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s dag8 %(message)s",
)
log = logging.getLogger("dag8")

# ── Config ───────────────────────────────────────────────────────────────
REDIS_HOST   = os.getenv("REDIS_HOST", "localhost")
REDIS_DB     = 0

DD_TARGET_PCT       = float(os.getenv("DD_TARGET_PCT",   "22.0"))   # C1
FEAR_GREED_MIN      = int(os.getenv("FEAR_GREED_MIN",    "20"))     # C3
HEALTH_STREAK_NEEDED = int(os.getenv("HEALTH_STREAK_NEEDED", "3"))  # C4
CHECK_INTERVAL      = int(os.getenv("CHECK_INTERVAL",    "60"))     # seconds

# Redis keys (live db=0, read-only)
KEY_EQUITY          = "quantum:equity:current"
KEY_MODE            = "quantum:mode"
KEY_HEALTH          = "quantum:health:truth:latest"
KEY_GATE            = "quantum:sandbox:gate:latest"
KEY_ACCURACY        = "quantum:sandbox:accuracy:latest"
KEY_FNG             = "quantum:sentiment:fear_greed"
KEY_DAG5            = "quantum:dag5:lockdown_guard:latest"
KEY_APPROVE         = "quantum:dag8:exit_approved"
KEY_FORCE_PHASE     = "quantum:dag8:force_phase"
KEY_STATUS          = "quantum:dag8:freeze_exit:status"
KEY_HISTORY         = "quantum:dag8:freeze_exit:history"
KEY_PHASE           = "quantum:dag8:current_phase"

# Phase descriptions
PHASES = {
    0: "FREEZE",
    1: "SHADOW_ONLY",
    2: "PAPER_TRADE",
    3: "LIVE_XS_5SYM_25PCT",
    4: "LIVE_SM_10SYM_50PCT",
    5: "LIVE_NORMAL",
}

PHASE_CRITERIA = {
    # phase_target: {min_criteria_green, additional_criteria}
    1: {"min_green": 5, "note": "All 5 criteria must be GREEN"},
    2: {"min_green": 5, "note": "Phase 1 must be stable ≥24h"},
    3: {"min_green": 5, "note": "Phase 2 paper_pnl must be positive"},
    4: {"min_green": 5, "note": "Phase 3 stable ≥48h, no SL hits"},
    5: {"min_green": 5, "note": "Phase 4 full validation complete"},
}


# ── Criterion Evaluators ──────────────────────────────────────────────────
async def check_c1_drawdown(r: aioredis.Redis) -> Tuple[bool, str]:
    """C1: Drawdown below target."""
    try:
        equity_data = await r.hgetall(KEY_EQUITY)
        if not equity_data:
            return False, "C1 NO_DATA (quantum:equity:current not found)"
        eq   = float(equity_data.get("equity",  equity_data.get("balance", 0)))
        peak = float(equity_data.get("peak",    equity_data.get("peak_equity", eq)))

        if peak <= 0:
            return False, "C1 peak=0"
        dd = (peak - eq) / peak * 100
        ok = dd < DD_TARGET_PCT
        return ok, f"C1 dd={dd:.2f}% target=<{DD_TARGET_PCT}% → {'GREEN' if ok else 'RED'}"
    except Exception as e:
        return False, f"C1 ERROR {e}"


async def check_c2_gate(r: aioredis.Redis) -> Tuple[bool, str]:
    """C2: Layer 2 research sandbox gate is OPEN."""
    try:
        gate_data = await r.hgetall(KEY_GATE)
        if not gate_data:
            return False, "C2 NO_GATE_DATA"
        gate_status = gate_data.get("gate", "CLOSED").upper()
        acc_data = await r.hgetall(KEY_ACCURACY)
        accuracy = float(acc_data.get("accuracy_pct", 0.0)) if acc_data else 0.0
        n_signals = int(acc_data.get("n_exit_signals", 0)) if acc_data else 0
        pf = float(acc_data.get("profit_factor", 0.0)) if acc_data else 0.0
        ok = gate_status == "OPEN"
        return ok, (f"C2 gate={gate_status} acc={accuracy:.1f}% "
                    f"n={n_signals} pf={pf:.2f} → {'GREEN' if ok else 'RED'}")
    except Exception as e:
        return False, f"C2 ERROR {e}"


async def check_c3_fear_greed(r: aioredis.Redis) -> Tuple[bool, str]:
    """C3: Fear & Greed above minimum threshold."""
    try:
        fng = await r.hgetall(KEY_FNG)
        if not fng:
            return False, "C3 NO_FNG_DATA"
        value  = int(fng.get("value", 0))
        regime = fng.get("regime", "unknown")
        ok = value > FEAR_GREED_MIN
        return ok, (f"C3 fng={value} regime={regime} "
                    f"target=>{FEAR_GREED_MIN} → {'GREEN' if ok else 'RED'}")
    except Exception as e:
        return False, f"C3 ERROR {e}"


async def check_c4_health_streak(r: aioredis.Redis, state: dict) -> Tuple[bool, str]:
    """C4: System health GREEN for N consecutive checks."""
    try:
        health = await r.hgetall(KEY_HEALTH)
        status = health.get("overall_health", "UNKNOWN").upper() if health else "UNKNOWN"
        is_green = status == "GREEN"

        if is_green:
            state["health_streak"] = state.get("health_streak", 0) + 1
        else:
            state["health_streak"] = 0

        streak = state["health_streak"]
        ok = streak >= HEALTH_STREAK_NEEDED
        return ok, (f"C4 health={status} streak={streak}/{HEALTH_STREAK_NEEDED} "
                    f"→ {'GREEN' if ok else 'RED'}")
    except Exception as e:
        return False, f"C4 ERROR {e}"


async def check_c5_operator(r: aioredis.Redis) -> Tuple[bool, str]:
    """C5: Operator has manually approved the transition."""
    try:
        val = await r.get(KEY_APPROVE)
        if val is None:
            return False, "C5 NOT_SET (SET quantum:dag8:exit_approved APPROVED to approve)"
        ok = val.upper() == "APPROVED"
        return ok, f"C5 operator={val} → {'GREEN' if ok else 'RED'}"
    except Exception as e:
        return False, f"C5 ERROR {e}"


# ── Phase Transition Logic ────────────────────────────────────────────────
def compute_recommendation(criteria_results: list, n_green: int,
                            current_phase: int, force_phase: Optional[int]) -> dict:
    """Map criteria results to a recommendation."""
    if force_phase is not None:
        return {
            "recommendation": "FORCE_PHASE",
            "target_phase":   force_phase,
            "phase_name":     PHASES.get(force_phase, "UNKNOWN"),
            "reason":         f"Operator force override: phase {force_phase}",
        }

    if n_green == 5:
        if current_phase == 0:
            return {
                "recommendation": "TRANSITION",
                "target_phase":   1,
                "phase_name":     PHASES[1],
                "reason":         "All 5 criteria GREEN — safe to enter SHADOW mode",
                "action":         "redis-cli SET quantum:mode SHADOW",
            }
        elif current_phase >= 1:
            return {
                "recommendation": "MONITOR",
                "target_phase":   current_phase,
                "phase_name":     PHASES.get(current_phase, "UNKNOWN"),
                "reason":         f"Phase {current_phase} stable — criteria still all GREEN",
            }
    elif n_green >= 3:
        return {
            "recommendation": "APPROACHING",
            "target_phase":   current_phase,
            "phase_name":     PHASES.get(current_phase, "UNKNOWN"),
            "reason":         f"{n_green}/5 criteria GREEN — getting closer but not ready",
            "missing":        [d for ok, d in criteria_results if not ok],
        }
    else:
        return {
            "recommendation": "HOLD_FREEZE",
            "target_phase":   0,
            "phase_name":     "FREEZE",
            "reason":         f"Only {n_green}/5 criteria GREEN — maintain FREEZE",
            "missing":        [d for ok, d in criteria_results if not ok],
        }


# ── Main Evaluation Loop ──────────────────────────────────────────────────
async def evaluate(r: aioredis.Redis, state: dict) -> dict:
    ts = int(time.time())
    ts_human = datetime.now(timezone.utc).isoformat()

    # Read current phase
    phase_str = await r.get(KEY_PHASE)
    current_phase = int(phase_str) if phase_str and phase_str.isdigit() else 0

    # Check force override
    force_str = await r.get(KEY_FORCE_PHASE)
    force_phase = int(force_str) if force_str and force_str.isdigit() else None

    # Evaluate all 5 criteria
    c1 = await check_c1_drawdown(r)
    c2 = await check_c2_gate(r)
    c3 = await check_c3_fear_greed(r)
    c4 = await check_c4_health_streak(r, state)
    c5 = await check_c5_operator(r)

    criteria = [c1, c2, c3, c4, c5]
    n_green  = sum(1 for ok, _ in criteria if ok)

    recommendation = compute_recommendation(criteria, n_green, current_phase, force_phase)

    result = {
        "ts":             ts,
        "ts_human":       ts_human,
        "current_phase":  current_phase,
        "phase_name":     PHASES.get(current_phase, "UNKNOWN"),
        "n_green":        n_green,
        "criteria_green": f"{n_green}/5",
        "c1_drawdown":    c1[1],
        "c2_gate":        c2[1],
        "c3_fng":         c3[1],
        "c4_health":      c4[1],
        "c5_operator":    c5[1],
        "recommendation": recommendation["recommendation"],
        "target_phase":   recommendation.get("target_phase", current_phase),
        "reason":         recommendation.get("reason", ""),
    }

    # Log summary
    status_icon = "🔴" if n_green < 3 else ("🟡" if n_green < 5 else "🟢")
    log.info(
        f"[DAG8] {status_icon} {n_green}/5 GREEN | phase={PHASES.get(current_phase)} "
        f"→ {recommendation['recommendation']}"
    )
    for ok, desc in criteria:
        log.info(f"  {'✅' if ok else '❌'} {desc}")

    # Apply transition if recommended
    if recommendation["recommendation"] in ("TRANSITION", "FORCE_PHASE"):
        target = recommendation["target_phase"]
        new_phase = PHASES.get(target, "SHADOW_ONLY")
        await r.set(KEY_PHASE, str(target))
        log.info(f"[DAG8] ⚡ TRANSITION → Phase {target} ({new_phase})")
        await r.set("quantum:dag8:transition_ts", ts)
        # Reset operator approval so next transition requires explicit re-approval
        if recommendation["recommendation"] == "TRANSITION":
            await r.delete(KEY_APPROVE)

    return result


async def status_publisher(r: aioredis.Redis, state: dict):
    while True:
        try:
            if "last_result" in state:
                flat = {k: str(v) for k, v in state["last_result"].items()}
                await r.hset(KEY_STATUS, mapping=flat)
        except Exception as e:
            log.warning(f"status publish failed: {e}")
        await asyncio.sleep(CHECK_INTERVAL)


async def main():
    log.info("[DAG8] Freeze Exit Analyzer starting")
    log.info(f"[DAG8] Criteria: dd<{DD_TARGET_PCT}% | gate_open | fng>{FEAR_GREED_MIN} "
             f"| health_streak={HEALTH_STREAK_NEEDED} | operator_approved")

    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
    await r.ping()
    log.info("[DAG8] Redis OK")
    log.info("[DAG8] Operator commands:")
    log.info("  SET quantum:dag8:exit_approved APPROVED    # approve FREEZE exit")
    log.info("  SET quantum:dag8:force_phase 2             # force phase (emergency)")
    log.info("  SET quantum:dag8:exit_approved DENIED      # block transition")

    state = {"health_streak": 0}
    asyncio.create_task(status_publisher(r, state))

    check_count = 0
    while True:
        try:
            result = await evaluate(r, state)
            state["last_result"] = result
            check_count += 1

            # Push to history (capped at 100)
            await r.lpush(KEY_HISTORY, json.dumps(result))
            await r.ltrim(KEY_HISTORY, 0, 99)

            # Write latest status
            flat = {k: str(v) for k, v in result.items()}
            await r.hset(KEY_STATUS, mapping=flat)

        except Exception as e:
            log.error(f"Evaluation error: {e}", exc_info=True)

        await asyncio.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
