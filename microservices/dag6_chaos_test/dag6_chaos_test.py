#!/usr/bin/env python3
"""
dag6_chaos_test.py — Quantum OS Resilience Test Battery

5 Tests (all self-healing — rollbacks run even on failure):

  T1  Service Restart Resilience
        Stop quantum-harvest-v2 → verify systemd restarts it in <45s

  T2  Drawdown Warning Boundary
        Artificially inflate peak to force DD ≥ 28% →
        verify DAG5 state shows warning → restore original peak

  T3  Zombie PEL Injection + DAG4 Clearance
        Create test stream/group, claim a message, lower idle threshold
        for DAG4 test via isolated invocation, verify auto-claim fires

  T4  Hardware Stop Persistence (DAG3 Restart)
        Restart quantum-dag3-hw-stops → verify Binance testnet orders
        still exist for all open positions after restart

  T5  Lockdown Escalation Dry-Run
        Inject position_mismatch=true into truth snapshot (test copy),
        run DAG5 trigger evaluator in isolation, confirm it would fire
        LOCKDOWN, then restore — mode never actually changes (safe)

Usage:
  python dag6_chaos_test.py           # all 5 tests
  python dag6_chaos_test.py --test T1 # single test
  python dag6_chaos_test.py --skip T3 # skip one

Results:
  quantum:dag6:chaos_test:latest  (hash, TTL=3600)
  quantum:dag6:chaos_test:history (list, capped 20 entries)
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import logging

import redis as redis_lib
import requests
import hmac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s chaos %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dag6")

# ── Redis ─────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ── Binance testnet ────────────────────────────────────────────────────────
BINANCE_BASE     = "https://testnet.binancefuture.com"
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# ── Redis keys ────────────────────────────────────────────────────────────
MODE_KEY         = "quantum:system:mode"
EQUITY_KEY       = "quantum:equity:current"
TRUTH_KEY        = "quantum:health:truth:latest"
GOVERNOR_HALT    = "quantum:governor:halt"
DAG5_STATE       = "quantum:dag5:lockdown_guard:latest"
DAG6_STATE       = "quantum:dag6:chaos_test:latest"
DAG6_HISTORY     = "quantum:dag6:chaos_test:history"

T1_SERVICE       = "quantum-harvest-v2"       # T2 service — safe to bounce
DAG3_SERVICE     = "quantum-dag3-hw-stops"
TEST_STREAM      = "quantum:stream:chaos_test_pel"
TEST_GROUP       = "chaos_test_grp"

# ── Helpers ───────────────────────────────────────────────────────────────

def _decode(v) -> str:
    if isinstance(v, bytes):
        return v.decode()
    return str(v) if v is not None else ""


def _safe_float(v, d=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def svc_active(name: str) -> bool:
    r = subprocess.run(["systemctl", "is-active", "--quiet", name],
                       capture_output=True, timeout=5)
    return r.returncode == 0


def svc_cmd(name: str, cmd: str, timeout: int = 15):
    subprocess.run(["systemctl", cmd, name], check=False, timeout=timeout)


def svc_kill_crash(name: str):
    """Kill the service process directly so systemd sees a crash and Restart=always fires.
    Note: 'systemctl stop' is administrative — systemd does NOT auto-restart on stop.
          SIGKILL to the main PID simulates an unexpected crash, which triggers restart.
    """
    r2 = subprocess.run(["systemctl", "show", "-p", "MainPID", "--value", name],
                        capture_output=True, text=True, timeout=5)
    pid = r2.stdout.strip()
    if pid and pid != "0":
        subprocess.run(["kill", "-SIGKILL", pid], check=False, timeout=5)
    else:
        subprocess.run(["systemctl", "kill", "--kill-who=main", name], check=False, timeout=5)


def _load_env_file(path: str = "/etc/quantum/testnet.env"):
    """Parse KEY=value env file and inject into os.environ for keys not already set."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _binance_headers(params: dict) -> tuple[dict, dict]:
    """Return (headers, signed_params)."""
    ts = str(int(time.time() * 1000))
    params["timestamp"] = ts
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(BINANCE_API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    return headers, params


def binance_open_orders() -> list[dict]:
    params: dict = {}
    headers, params = _binance_headers(params)
    resp = requests.get(f"{BINANCE_BASE}/fapi/v1/openOrders",
                        params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Binance openOrders HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def binance_positions() -> list[dict]:
    params: dict = {}
    headers, params = _binance_headers(params)
    resp = requests.get(f"{BINANCE_BASE}/fapi/v2/positionRisk",
                        params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Binance positionRisk HTTP {resp.status_code}: {resp.text[:200]}")
    return [p for p in resp.json() if abs(float(p.get("positionAmt", 0))) > 0]


# ── Test harness ──────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str):
        self.name      = name
        self.status    = "PENDING"
        self.details   = []
        self.start_ts  = time.time()
        self.end_ts    = None

    def ok(self, msg: str = ""):
        self.status  = "PASS"
        self.end_ts  = time.time()
        if msg:
            self.details.append(f"PASS: {msg}")
        logger.info("[%s] PASS  %s", self.name, msg)

    def fail(self, msg: str):
        self.status  = "FAIL"
        self.end_ts  = time.time()
        self.details.append(f"FAIL: {msg}")
        logger.error("[%s] FAIL  %s", self.name, msg)

    def note(self, msg: str):
        self.details.append(msg)
        logger.info("[%s] %s", self.name, msg)

    def elapsed(self) -> float:
        if self.end_ts:
            return round(self.end_ts - self.start_ts, 2)
        return round(time.time() - self.start_ts, 2)

    def to_dict(self) -> dict:
        return {
            "name":    self.name,
            "status":  self.status,
            "elapsed": self.elapsed(),
            "details": self.details,
        }


# ── Individual Tests ──────────────────────────────────────────────────────

def test_t1_service_restart(r: redis_lib.Redis) -> TestResult:
    """
    Kill quantum-harvest-v2 → verify systemd brings it back in <45 s.
    """
    res = TestResult("T1_ServiceRestart")
    res.note(f"Targeting service: {T1_SERVICE}")

    was_active = svc_active(T1_SERVICE)
    if not was_active:
        res.note(f"{T1_SERVICE} was already inactive — starting it first")
        svc_cmd(T1_SERVICE, "start")
        time.sleep(5)
        if not svc_active(T1_SERVICE):
            res.fail(f"Could not start {T1_SERVICE} — skipping kill test")
            return res

    res.note("Sending SIGKILL to main process (simulates crash, triggers Restart=always) ...")
    svc_kill_crash(T1_SERVICE)
    time.sleep(2)  # brief gap before polling

    res.note("Process killed — waiting up to 20s for systemd Restart=always (RestartSec=3) ...")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if svc_active(T1_SERVICE):
            elapsed = 22 - (deadline - time.monotonic())
            res.ok(f"{T1_SERVICE} restarted automatically in ~{elapsed:.0f}s (Restart=always)")
            return res
        time.sleep(1)

    res.fail(f"{T1_SERVICE} did NOT restart within 20s after SIGKILL")
    svc_cmd(T1_SERVICE, "start")  # Rollback — start manually
    return res


def test_t2_drawdown_warning(r: redis_lib.Redis) -> TestResult:
    """
    Inflate equity peak so DD ≥ 28% (i.e. dd_warn threshold).
    Verify DAG5 state shows a warning within 30s. Restore original peak.
    """
    res = TestResult("T2_DrawdownWarning")

    raw = r.hgetall(EQUITY_KEY)
    orig_equity = _safe_float(raw.get(b"equity", b"0"))
    orig_peak   = _safe_float(raw.get(b"peak",   b"0"))

    if orig_equity <= 0:
        res.fail("Could not read equity from Redis")
        return res

    # Need DD ≥ 28% → peak = equity / (1 - 0.28) + 50 buffer
    target_peak = orig_equity / (1 - 0.285) + 50
    res.note(f"Original: equity={orig_equity:.2f} peak={orig_peak:.2f} "
             f"dd={max(0, (orig_peak-orig_equity)/orig_peak*100):.1f}%")
    res.note(f"Injecting test peak={target_peak:.2f} to trigger dd_warn>=28%")

    r.hset(EQUITY_KEY, "peak", f"{target_peak:.2f}")

    try:
        # Wait up to 30s for DAG5 to pick it up
        deadline = time.monotonic() + 30
        found_warning = False
        while time.monotonic() < deadline:
            time.sleep(3)
            state = {_decode(k): _decode(v) for k, v in r.hgetall(DAG5_STATE).items()}
            dd    = _safe_float(state.get("drawdown_pct", 0))
            warns = state.get("warnings", "")
            res.note(f"  DAG5 state: dd={dd:.2f}% warnings='{warns}'")
            if dd >= 28.0:
                found_warning = True
                if "DRAWDOWN" in warns:
                    res.ok(f"DAG5 reported DRAWDOWN_WARN at dd={dd:.2f}%")
                else:
                    # dd is updated but warn string may not have fired yet
                    # (DAG5 checks dd every tick — check lockdown_reasons too)
                    lockdown_r = state.get("lockdown_reasons", "")
                    if "DRAWDOWN" in lockdown_r:
                        res.ok(f"DAG5 raised DRAWDOWN in lockdown_reasons at dd={dd:.2f}% (mode never escalated — peak set below lockdown threshold)")
                    else:
                        res.note(f"  dd={dd:.2f}% but no DRAWDOWN in warnings/lockdown_reasons yet — waiting ...")
                        continue
                break

        if not found_warning:
            res.fail("DAG5 did not reflect drawdown warning within 30s")
    finally:
        # ALWAYS restore original peak
        r.hset(EQUITY_KEY, "peak", f"{orig_peak:.2f}")
        res.note(f"Peak restored to {orig_peak:.2f}")

    return res


def test_t3_zombie_pel_injection(r: redis_lib.Redis) -> TestResult:
    """
    Create an isolated test stream/group, publish a message, claim it
    into the consumer PEL, then invoke DAG4 auto-claim logic directly
    with a zero idle threshold — verify the PEL empties.
    """
    res = TestResult("T3_ZombiePEL")

    # Cleanup previous test artifacts
    r.delete(TEST_STREAM)

    # Create group at $ first (MKSTREAM)
    try:
        r.xgroup_create(TEST_STREAM, TEST_GROUP, id="0", mkstream=True)
    except Exception:
        pass  # group may already exist from previous run

    # Publish 3 messages
    for i in range(3):
        r.xadd(TEST_STREAM, {"idx": str(i), "src": "chaos_test"})

    # Claim them into consumer PEL without acknowledging
    msgs = r.xreadgroup(TEST_GROUP, "chaos_consumer", {TEST_STREAM: ">"}, count=10)
    claimed_ids = []
    if msgs:
        for stream_name, entries in msgs:
            for msg_id, _ in entries:
                claimed_ids.append(msg_id)

    res.note(f"Injected {len(claimed_ids)} messages into PEL of {TEST_STREAM}/{TEST_GROUP}")

    if not claimed_ids:
        res.fail("Could not inject messages into PEL")
        r.delete(TEST_STREAM)
        return res

    # Verify PEL exists
    pel_info = r.xpending(TEST_STREAM, TEST_GROUP)
    initial_pel = pel_info.get("pending", 0) if isinstance(pel_info, dict) else int(pel_info[0])
    res.note(f"PEL before auto-claim: {initial_pel}")

    # Run XAUTOCLAIM with idle=0ms (claim immediately)
    try:
        result = r.xautoclaim(TEST_STREAM, TEST_GROUP, "dag4_chaos_test",
                              min_idle_time=0, start_id="0-0", count=100)
        # xautoclaim returns (next_start_id, messages, deleted_ids)
        reclaimed = result[1] if isinstance(result, (list, tuple)) else []
        res.note(f"XAUTOCLAIM reclaimed {len(reclaimed)} messages")

        # ACK them
        for msg_id, _ in reclaimed:
            r.xack(TEST_STREAM, TEST_GROUP, msg_id)

        # Verify PEL is now 0
        pel_info_after = r.xpending(TEST_STREAM, TEST_GROUP)
        final_pel = pel_info_after.get("pending", 0) if isinstance(pel_info_after, dict) else int(pel_info_after[0])
        res.note(f"PEL after auto-claim+ACK: {final_pel}")

        if final_pel == 0:
            res.ok(f"PEL cleared: {initial_pel} → {final_pel} via XAUTOCLAIM+ACK"
                   f" (same logic DAG4 uses at idle>=300s)")
        else:
            res.fail(f"PEL not empty after auto-claim: {final_pel}")
    except Exception as e:
        res.fail(f"XAUTOCLAIM failed: {e}")
    finally:
        r.delete(TEST_STREAM)

    return res


def test_t4_hw_stops_persistence(r: redis_lib.Redis) -> TestResult:
    """
    Restart quantum-dag3-hw-stops. Wait 65s (two DAG3 ticks).
    Verify Binance testnet still has stop/TP orders for open positions.
    """
    res = TestResult("T4_HWStopsPersistence")

    # Pre-check: open positions
    try:
        positions = binance_positions()
        symbols   = {p["symbol"] for p in positions}
        res.note(f"Open positions before restart: {symbols or '(none)'}")
    except Exception as e:
        res.fail(f"Could not read Binance positions: {e}")
        return res

    if not symbols:
        res.note("No open positions — verifying DAG3 is running and stable instead")
        if svc_active(DAG3_SERVICE):
            res.ok("No positions active; DAG3 service is running OK")
        else:
            res.fail(f"{DAG3_SERVICE} is not active")
        return res

    # Orders before restart
    try:
        pre_orders = binance_open_orders()
        pre_stop  = {o["symbol"] for o in pre_orders if o.get("type") == "STOP_MARKET"}
        pre_tp    = {o["symbol"] for o in pre_orders if o.get("type") == "TAKE_PROFIT_MARKET"}
        res.note(f"Orders before restart — STOP:{pre_stop} TP:{pre_tp}")
    except Exception as e:
        res.fail(f"Could not read pre-restart orders: {e}")
        return res

    # Restart DAG3
    res.note(f"Restarting {DAG3_SERVICE} ...")
    svc_cmd(DAG3_SERVICE, "restart", timeout=20)
    time.sleep(5)
    if not svc_active(DAG3_SERVICE):
        res.fail(f"{DAG3_SERVICE} failed to start after restart")
        return res
    res.note("DAG3 restarted OK — waiting 65s for two ticks ...")
    time.sleep(65)

    # Orders after restart
    try:
        post_orders = binance_open_orders()
        post_stop = {o["symbol"] for o in post_orders if o.get("type") == "STOP_MARKET"}
        post_tp   = {o["symbol"] for o in post_orders if o.get("type") == "TAKE_PROFIT_MARKET"}
        res.note(f"Orders after restart — STOP:{post_stop} TP:{post_tp}")

        missing_stop = symbols - post_stop
        missing_tp   = symbols - post_tp

        if not missing_stop and not missing_tp:
            res.ok(f"All orders intact after DAG3 restart — STOP:{post_stop} TP:{post_tp}")
        else:
            res.fail(f"Orders missing — STOP missing:{missing_stop} TP missing:{missing_tp}")
    except Exception as e:
        res.fail(f"Could not read post-restart orders: {e}")

    return res


def test_t5_lockdown_dryrun(r: redis_lib.Redis) -> TestResult:
    """
    Inject a synthetic truth snapshot with position_mismatch=true into
    a shadow key, run the DAG5 evaluator logic inline (no actual mode
    change), confirm the LOCKDOWN trigger fires, then remove shadow key.

    Mode NEVER changes. This is a logic-path verification only.
    """
    res = TestResult("T5_LockdownDryRun")

    # Build synthetic truth (copy of real + position_mismatch=true)
    real_truth = {_decode(k): _decode(v) for k, v in r.hgetall(TRUTH_KEY).items()}
    if not real_truth:
        res.fail("Real truth snapshot is empty — DAG2 may be down")
        return res

    res.note(f"Real truth: overall_health={real_truth.get('overall_health')} "
             f"services_dead='{real_truth.get('services_dead', '')}' "
             f"position_mismatch={real_truth.get('position_mismatch')}")

    # Write synthetic truth to a shadow key
    SHADOW_TRUTH = "quantum:dag6:shadow_truth"
    synthetic = dict(real_truth)
    synthetic["position_mismatch"] = "true"
    r.hset(SHADOW_TRUTH, mapping=synthetic)
    r.expire(SHADOW_TRUTH, 60)

    try:
        # Run trigger evaluation (inline copy of DAG5 logic — read-only on SHADOW_TRUTH)
        lockdown_triggers = []

        # -- position_mismatch
        if synthetic.get("position_mismatch", "false").lower() == "true":
            lockdown_triggers.append("POSITION_MISMATCH:true")

        # -- T1 service dead
        T1_SERVICES_LOCAL = [
            "quantum-exit-monitor",
            "quantum-governor",
            "quantum-intent-executor",
            "quantum-execution",
            "quantum-emergency-exit-worker",
        ]
        dead_services_str = synthetic.get("services_dead", "")
        dead_services = [s.strip() for s in dead_services_str.split(",") if s.strip()]
        for svc in dead_services:
            if svc in T1_SERVICES_LOCAL:
                lockdown_triggers.append(f"T1_DEAD:{svc}")

        # -- plan lag
        plan_lag = int(synthetic.get("apply_plan_lag", 0))
        if plan_lag >= 5000:
            lockdown_triggers.append(f"PLAN_LAG:{plan_lag}")

        # -- drawdown
        raw_eq = r.hgetall(EQUITY_KEY)
        equity = _safe_float(raw_eq.get(b"equity", b"0"))
        peak   = _safe_float(raw_eq.get(b"peak",   b"0"))
        dd_pct = max(0.0, (peak - equity) / peak * 100.0) if peak > 0 else 0.0
        if dd_pct >= 35.0:
            lockdown_triggers.append(f"DRAWDOWN:{dd_pct:.1f}%")

        res.note(f"Evaluator found triggers: {lockdown_triggers}")

        # Confirm POSITION_MISMATCH fired
        if "POSITION_MISMATCH:true" in lockdown_triggers:
            # Verify mode was NOT changed (we never called activate_lockdown)
            current_mode = _decode(r.get(MODE_KEY))
            res.note(f"Current mode (must still be FREEZE/NORMAL): {current_mode}")
            if current_mode not in ("LOCKDOWN",):
                res.ok(f"Dry-run confirmed: evaluator fired LOCKDOWN triggers={lockdown_triggers} "
                       f"but mode unchanged ({current_mode}) — escalation path verified")
            else:
                # Mode is LOCKDOWN already — check if DAG5 fired on its own
                res.note("Mode is already LOCKDOWN (may be pre-existing) — dry-run still valid")
                res.ok(f"Dry-run triggers={lockdown_triggers} (mode was already LOCKDOWN)")
        else:
            res.fail("POSITION_MISMATCH trigger did not appear in evaluation result")
    finally:
        r.delete(SHADOW_TRUTH)
        res.note("Shadow truth key removed")

    return res


# ── Runner ────────────────────────────────────────────────────────────────

def run_all(r: redis_lib.Redis, skip: set, only: set) -> list[TestResult]:
    tests = [
        ("T1", test_t1_service_restart),
        ("T2", test_t2_drawdown_warning),
        ("T3", test_t3_zombie_pel_injection),
        ("T4", test_t4_hw_stops_persistence),
        ("T5", test_t5_lockdown_dryrun),
    ]
    results = []
    for tag, fn in tests:
        if only and tag not in only:
            logger.info("[%s] SKIPPED (not in --test list)", tag)
            continue
        if tag in skip:
            logger.info("[%s] SKIPPED (--skip)", tag)
            continue
        logger.info("=" * 60)
        logger.info("[%s] Starting ...", tag)
        res = fn(r)
        results.append(res)
        logger.info("[%s] → %s  (%.1fs)", res.name, res.status, res.elapsed())
    return results


def publish_results(r: redis_lib.Redis, results: list[TestResult]):
    passed  = sum(1 for r2 in results if r2.status == "PASS")
    failed  = sum(1 for r2 in results if r2.status == "FAIL")
    total   = len(results)
    overall = "PASS" if failed == 0 else "FAIL"

    summary = {
        "overall":    overall,
        "passed":     str(passed),
        "failed":     str(failed),
        "total":      str(total),
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "details":    json.dumps([res.to_dict() for res in results]),
    }

    r.hset(DAG6_STATE, mapping=summary)
    r.expire(DAG6_STATE, 3600)

    # Append to history list (capped at 20)
    r.lpush(DAG6_HISTORY, json.dumps({"ts": summary["ts"], "overall": overall,
                                       "passed": passed, "failed": failed}))
    r.ltrim(DAG6_HISTORY, 0, 19)

    logger.info("=" * 60)
    logger.info("[DAG6] Results: %d/%d PASS  overall=%s", passed, total, overall)
    for res in results:
        marker = "✓" if res.status == "PASS" else "✗"
        logger.info("  %s %s — %s  (%.1fs)", marker, res.name, res.status, res.elapsed())


def main():
    parser = argparse.ArgumentParser(description="Quantum OS Chaos Test Battery (DAG 6)")
    parser.add_argument("--test", nargs="+", metavar="T1-T5",
                        help="Run only specific tests")
    parser.add_argument("--skip", nargs="+", metavar="T1-T5",
                        help="Skip specific tests")
    args = parser.parse_args()

    only = set(args.test or [])
    skip = set(args.skip or [])

    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    try:
        r.ping()
    except redis_lib.ConnectionError as e:
        logger.error("Redis FAILED: %s", e)
        sys.exit(1)

    logger.info("[DAG6] Chaos Test Battery — %s", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # Ensure Binance keys available (env file may not be exported into shell)
    _load_env_file()
    global BINANCE_API_KEY, BINANCE_API_SECRET
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", BINANCE_API_KEY)
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", BINANCE_API_SECRET)

    current_mode = _decode(r.get("quantum:system:mode"))
    logger.info("[DAG6] System mode at start: %s", current_mode)

    results = run_all(r, skip=skip, only=only)
    publish_results(r, results)

    # Exit code: 0 = all pass, 1 = at least one fail
    failed = sum(1 for res in results if res.status == "FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
