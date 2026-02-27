#!/usr/bin/env python3
"""
dag5_lockdown_guard.py — Autonomous Lockdown Mode Guardian

Monitors system health and escalates to LOCKDOWN when critical conditions
are detected. LOCKDOWN persists until manually cleared by an operator.

Trigger logic (evaluated every CHECK_INTERVAL_SEC):
  T1 — CRITICAL (any one → LOCKDOWN immediately):
    • T1 service dead: exit_monitor, governor, intent_executor,
                       execution, emergency-exit-worker
    • apply_plan_lag > PLAN_LAG_LOCKDOWN_THRESHOLD (5000)
    • position_mismatch == true
    • Drawdown from equity peak > DD_PCT_LOCKDOWN (35%)
    • orphan_position_count > ORPHAN_LOCKDOWN_THRESHOLD (2)

  T2 — WARNING (logged, published as YELLOW alarm, no mode change):
    • apply_plan_lag > PLAN_LAG_WARN_THRESHOLD (1000)
    • apply_result_lag > RESULT_LAG_WARN_THRESHOLD (5000)
    • Drawdown from equity peak > DD_PCT_WARN (28%)
    • T2/T3 service dead (harvest-brain, risk-brake, reconcile, ai-engine)

Mode escalation rules:
  NORMAL → LOCKDOWN  (on T1 trigger)
  FREEZE → LOCKDOWN  (on T1 trigger)
  LOCKDOWN → <stays until operator clears>

Manual clear:
  redis-cli SET quantum:system:mode FREEZE   # back to normal freeze
  redis-cli DEL quantum:dag5:lockdown_active  # clear lock flag

What LOCKDOWN does beyond mode flag:
  • Publishes quantum:stream:apply.plan with EMERGENCY_LOCKDOWN_HALT signal
  • Sets quantum:governor:halt=1  (governor stops approving new plans)
  • Sets quantum:learning:frozen=1 (belt + suspenders)

State published every tick:
  quantum:dag5:lockdown_guard:latest  (hash, TTL=120s)
"""

import os
import sys
import time
import logging
import signal
import json
import redis as redis_lib

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s dag5 %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dag5")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
CHECK_INTERVAL_SEC = int(os.getenv("DAG5_INTERVAL_SEC", "15"))

# Drawdown thresholds (% drop from equity peak)
DD_PCT_WARN     = float(os.getenv("DAG5_DD_WARN_PCT", "28.0"))
DD_PCT_LOCKDOWN = float(os.getenv("DAG5_DD_LOCKDOWN_PCT", "35.0"))

# Stream lag thresholds
PLAN_LAG_WARN      = int(os.getenv("DAG5_PLAN_LAG_WARN", "1000"))
PLAN_LAG_LOCKDOWN  = int(os.getenv("DAG5_PLAN_LAG_LOCKDOWN", "5000"))
RESULT_LAG_WARN    = int(os.getenv("DAG5_RESULT_LAG_WARN", "5000"))

# Orphan threshold
ORPHAN_LOCKDOWN = int(os.getenv("DAG5_ORPHAN_LOCKDOWN", "2"))

# T1 services — death of any → immediate LOCKDOWN
T1_SERVICES = [
    "quantum-exit-monitor",
    "quantum-governor",
    "quantum-intent-executor",
    "quantum-execution",
    "quantum-emergency-exit-worker",
]

# T2 services — death = WARNING alarm only
T2_SERVICES = [
    "quantum-harvest-brain",
    "quantum-harvest-v2",
    "quantum-risk-brake",
    "quantum-reconcile-engine",
    "quantum-ai-engine",
]

# Redis keys
TRUTH_KEY        = "quantum:health:truth:latest"
EQUITY_KEY       = "quantum:equity:current"
MODE_KEY         = "quantum:system:mode"
GOVERNOR_HALT    = "quantum:governor:halt"
LEARNING_FROZEN  = "quantum:learning:frozen"
LOCKDOWN_ACTIVE  = "quantum:dag5:lockdown_active"
POSITION_PREFIX  = "quantum:position:"
STATE_KEY        = "quantum:dag5:lockdown_guard:latest"
STATE_TTL        = 120

# ── Graceful shutdown ─────────────────────────────────────────────────────
_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    logger.info("Signal %s — shutting down", sig)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def check_service_active(service: str) -> bool:
    """Fast systemd is-active check via /run/systemd/units."""
    import subprocess
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", service],
            capture_output=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def compute_drawdown_pct(r: redis_lib.Redis) -> tuple[float, float, float]:
    """Returns (equity, peak, drawdown_pct)."""
    raw = r.hgetall(EQUITY_KEY)
    if not raw:
        return 0.0, 0.0, 0.0
    equity = _safe_float(raw.get(b"equity", raw.get("equity", 0)))
    peak   = _safe_float(raw.get(b"peak",   raw.get("peak",   0)))
    if peak <= 0:
        return equity, peak, 0.0
    dd_pct = max(0.0, (peak - equity) / peak * 100.0)
    return equity, peak, dd_pct


def get_truth(r: redis_lib.Redis) -> dict:
    """Read Runtime Truth Engine snapshot (decoded)."""
    raw = r.hgetall(TRUTH_KEY)
    return {_decode(k): _decode(v) for k, v in raw.items()}


def get_unrealized_pnl_total(r: redis_lib.Redis) -> float:
    """Sum unrealized_pnl across all open positions."""
    total = 0.0
    for key in r.keys(f"{POSITION_PREFIX}*"):
        raw = r.hgetall(key)
        if raw:
            total += _safe_float(raw.get(b"unrealized_pnl", raw.get("unrealized_pnl", 0)))
    return total


def activate_lockdown(r: redis_lib.Redis, reason: str):
    """
    Set LOCKDOWN mode with full side-effects:
      1. quantum:system:mode = LOCKDOWN
      2. quantum:governor:halt = 1
      3. quantum:learning:frozen = 1
      4. quantum:dag5:lockdown_active = <reason + timestamp>
      5. Push emergency signal to apply.plan stream (rate-limited: max once per 60s)
    """
    current_mode = _decode(r.get(MODE_KEY))
    if current_mode == "LOCKDOWN":
        return  # idempotent

    logger.critical("[DAG5] ⚠  LOCKDOWN ACTIVATED  ⚠  reason=%s", reason)

    pipe = r.pipeline()
    pipe.set(MODE_KEY, "LOCKDOWN")
    pipe.set(GOVERNOR_HALT, "1")
    pipe.set(LEARNING_FROZEN, "1")
    pipe.set(LOCKDOWN_ACTIVE, json.dumps({
        "reason":     reason,
        "activated":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ts":         int(time.time()),
    }))
    pipe.execute()

    # Emergency signal to apply.plan (intent_executor will halt)
    # Rate-limit: only if not sent in last 60s
    last_signal_key = "quantum:dag5:last_lockdown_signal"
    if not r.exists(last_signal_key):
        r.xadd("quantum:stream:apply.plan", {
            "plan_id":      "dag5_lockdown_halt",
            "symbol":       "ALL",
            "action":       "LOCKDOWN_HALT",
            "decision":     "HALT",
            "reason_codes": reason,
            "source":       "dag5_lockdown_guard",
            "timestamp":    str(int(time.time())),
        })
        r.setex(last_signal_key, 60, "1")
        logger.info("[DAG5] Emergency LOCKDOWN_HALT signal sent to apply.plan")


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_triggers(r: redis_lib.Redis) -> tuple[list, list, list]:
    """
    Returns (lockdown_triggers, warn_triggers, info_lines).
    """
    lockdown = []
    warnings = []
    info     = []

    truth = get_truth(r)
    equity, peak, dd_pct = compute_drawdown_pct(r)

    # ── Truth-engine-derived checks ──────────────────────────────────────
    services_dead_str = truth.get("services_dead", "")
    dead_services = [s.strip() for s in services_dead_str.split(",") if s.strip()]

    for svc in dead_services:
        if svc in T1_SERVICES:
            lockdown.append(f"T1_DEAD:{svc}")
        elif svc in T2_SERVICES:
            warnings.append(f"T2_DEAD:{svc}")

    # Stream lags (from truth)
    plan_lag   = int(truth.get("apply_plan_lag", 0))
    result_lag = int(truth.get("apply_result_lag", 0))

    if plan_lag >= PLAN_LAG_LOCKDOWN:
        lockdown.append(f"PLAN_LAG:{plan_lag}>={PLAN_LAG_LOCKDOWN}")
    elif plan_lag >= PLAN_LAG_WARN:
        warnings.append(f"PLAN_LAG_WARN:{plan_lag}>={PLAN_LAG_WARN}")

    if result_lag >= RESULT_LAG_WARN:
        warnings.append(f"RESULT_LAG_WARN:{result_lag}>={RESULT_LAG_WARN}")

    # Position mismatch
    if truth.get("position_mismatch", "false").lower() == "true":
        lockdown.append("POSITION_MISMATCH:true")

    # Orphan positions
    orphan_count = int(truth.get("orphan_position_count", 0))
    if orphan_count > ORPHAN_LOCKDOWN:
        lockdown.append(f"ORPHANS:{orphan_count}>{ORPHAN_LOCKDOWN}")

    # Exit monitor dead (extra-critical, not in services_dead always)
    if truth.get("exit_monitor_alive", "true").lower() == "false":
        lockdown.append("EXIT_MONITOR_DEAD")

    # ── Drawdown ─────────────────────────────────────────────────────────
    if equity > 0 and peak > 0:
        info.append(f"equity={equity:.2f} peak={peak:.2f} dd={dd_pct:.2f}%")
        if dd_pct >= DD_PCT_LOCKDOWN:
            lockdown.append(f"DRAWDOWN:{dd_pct:.1f}%>={DD_PCT_LOCKDOWN}%")
        elif dd_pct >= DD_PCT_WARN:
            warnings.append(f"DRAWDOWN_WARN:{dd_pct:.1f}%>={DD_PCT_WARN}%")

    # ── Unrealized PnL summary ────────────────────────────────────────────
    upnl = get_unrealized_pnl_total(r)
    info.append(f"total_unrealized_pnl={upnl:.2f}")

    return lockdown, warnings, info


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    logger.info("[DAG5] Lockdown Guard starting — interval=%ds", CHECK_INTERVAL_SEC)
    logger.info("[DAG5] Thresholds: DD_warn=%.1f%% DD_lock=%.1f%% plan_lag_lock=%d orphan_lock=%d",
                DD_PCT_WARN, DD_PCT_LOCKDOWN, PLAN_LAG_LOCKDOWN, ORPHAN_LOCKDOWN)

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
        logger.info("[DAG5] Redis OK")
    except redis_lib.ConnectionError as e:
        logger.error("[DAG5] Redis FAILED: %s", e)
        sys.exit(1)

    # Show current state at startup
    current_mode = _decode(r.get(MODE_KEY))
    equity, peak, dd_pct = compute_drawdown_pct(r)
    logger.info("[DAG5] Startup state: mode=%s equity=%.2f peak=%.2f dd=%.2f%%",
                current_mode, equity, peak, dd_pct)

    lockdown_active = r.get(LOCKDOWN_ACTIVE)
    if lockdown_active:
        logger.warning("[DAG5] Pre-existing LOCKDOWN_ACTIVE flag: %s",
                       _decode(lockdown_active))

    consecutive_errors = 0

    while _RUNNING:
        tick_start = time.monotonic()

        try:
            current_mode = _decode(r.get(MODE_KEY))
            lockdown_triggers, warn_triggers, info_lines = evaluate_triggers(r)

            # ── Fire lockdown ─────────────────────────────────────────────
            if lockdown_triggers and current_mode != "LOCKDOWN":
                reasons = " | ".join(lockdown_triggers)
                activate_lockdown(r, reasons)
                current_mode = "LOCKDOWN"

            # ── Log ───────────────────────────────────────────────────────
            if lockdown_triggers:
                logger.critical("[DAG5] LOCKDOWN triggers: %s", " | ".join(lockdown_triggers))
            if warn_triggers:
                logger.warning("[DAG5] WARNINGS: %s", " | ".join(warn_triggers))

            for line in info_lines:
                logger.debug("[DAG5] %s", line)

            # ── Publish state ─────────────────────────────────────────────
            equity, peak, dd_pct = compute_drawdown_pct(r)
            upnl = float(info_lines[-1].split("=")[-1]) if info_lines else 0.0

            state = {
                "mode":             current_mode,
                "lockdown_active":  "1" if r.exists(LOCKDOWN_ACTIVE) else "0",
                "lockdown_reasons": " | ".join(lockdown_triggers) if lockdown_triggers else "",
                "warnings":         " | ".join(warn_triggers) if warn_triggers else "",
                "equity":           f"{equity:.2f}",
                "peak":             f"{peak:.2f}",
                "drawdown_pct":     f"{dd_pct:.2f}",
                "unrealized_pnl":   f"{upnl:.2f}",
                "dd_warn_pct":      str(DD_PCT_WARN),
                "dd_lock_pct":      str(DD_PCT_LOCKDOWN),
                "ts":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            r.hset(STATE_KEY, mapping=state)
            r.expire(STATE_KEY, STATE_TTL)

            log_level = "CRITICAL" if lockdown_triggers else ("WARNING" if warn_triggers else "INFO")
            log_fn = getattr(logger, log_level.lower())
            log_fn("[DAG5_TICK] mode=%s dd=%.2f%% upnl=%.2f lockdown=%s warn=%s",
                   current_mode, dd_pct, upnl,
                   len(lockdown_triggers), len(warn_triggers))

            consecutive_errors = 0

        except Exception as e:
            consecutive_errors += 1
            logger.error("[DAG5] Exception (consecutive=%d): %s", consecutive_errors, e)
            if consecutive_errors >= 10:
                logger.critical("[DAG5] 10 consecutive errors — sleeping 60s")
                time.sleep(60)
                consecutive_errors = 0

        elapsed   = time.monotonic() - tick_start
        sleep_for = max(0.0, CHECK_INTERVAL_SEC - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    logger.info("[DAG5] Lockdown Guard stopped")


if __name__ == "__main__":
    main()
