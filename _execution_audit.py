#!/usr/bin/env python3
"""
EXECUTION AUDIT — Full pipeline verification
Checks: ENTRY flow, SL integrity, ATR risk tracking, execution gap, RL multiplier.
Run on VPS: python3 /tmp/_execution_audit.py
"""
import redis, json, subprocess, time
from datetime import datetime, timezone

r = redis.Redis(decode_responses=True)

SEP = "=" * 70

def ts_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def get_deploy_ms() -> int:
    try:
        result = subprocess.run(
            ["systemctl", "show", "quantum-intent-bridge", "--property=ActiveEnterTimestamp"],
            capture_output=True, text=True
        )
        ts_str = result.stdout.strip().replace("ActiveEnterTimestamp=", "")
        # e.g. "Tue 2026-02-25 00:32:55 UTC"
        for fmt in ["%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S"]:
            try:
                dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except:
                pass
    except:
        pass
    return 0

# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("EXECUTION AUDIT — Quantum Trader Production")
print(SEP)

deploy_ms = get_deploy_ms()
deploy_label = ts_to_utc(deploy_ms) if deploy_ms else "UNKNOWN"
print(f"\nDeploy timestamp: {deploy_label}")
print(f"Audit time:       {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[1] ENTRY FLOW — apply.plan → apply.result")
print(SEP)

# Scan apply.plan for all recent ENTRY_PROPOSED (last 5000 messages)
apply_plan_msgs = r.xrevrange("quantum:stream:apply.plan", count=5000)
entry_msgs = [m for m in apply_plan_msgs if m[1].get("action") == "ENTRY_PROPOSED"]
total_entries = len(entry_msgs)

post_deploy = [m for m in entry_msgs if int(m[0].split("-")[0]) >= deploy_ms] if deploy_ms else []
pre_deploy  = [m for m in entry_msgs if int(m[0].split("-")[0]) < deploy_ms]  if deploy_ms else entry_msgs

print(f"  ENTRY_PROPOSED in last 5000 apply.plan msgs : {total_entries}")
print(f"  Pre-deploy                                   : {len(pre_deploy)}")
print(f"  Post-deploy (since {deploy_label}): {len(post_deploy)}")

# For each post-deploy ENTRY, check entry_price and atr_value
entry_price_ok = sum(1 for m in post_deploy if float(m[1].get("entry_price", "0") or "0") > 0)
atr_ok         = sum(1 for m in post_deploy if float(m[1].get("atr_value", "0") or "0") > 0)
risk_ok        = sum(1 for m in post_deploy if m[1].get("entry_risk_usdt") and float(m[1].get("entry_risk_usdt","0")) > 0)
risk_missing   = sum(1 for m in post_deploy if m[1].get("risk_missing") == "1")

if post_deploy:
    print(f"\n  Post-deploy field quality ({len(post_deploy)} entries):")
    print(f"    entry_price > 0       : {entry_price_ok}/{len(post_deploy)}")
    print(f"    atr_value > 0         : {atr_ok}/{len(post_deploy)}")
    print(f"    entry_risk_usdt > 0   : {risk_ok}/{len(post_deploy)}")
    print(f"    risk_missing = 1      : {risk_missing}/{len(post_deploy)}  ← should be 0")

# Build plan_id index from apply.result
print(f"\n  Building apply.result plan_id index (last 5000)...")
apply_result_msgs = r.xrevrange("quantum:stream:apply.result", count=5000)
result_by_plan = {}
for mid, fields in apply_result_msgs:
    pid = fields.get("plan_id")
    if pid:
        result_by_plan[pid] = fields

# Match ENTRY plans to their results
matched = 0
executed = 0
failed = 0
missing_result = 0
entry_risk_post = []

for mid, fields in post_deploy:
    pid = fields.get("plan_id")
    if not pid:
        missing_result += 1
        continue
    result = result_by_plan.get(pid)
    if result is None:
        missing_result += 1
        continue
    matched += 1
    was_executed = result.get("executed", "False").lower() in ("true", "1")
    if was_executed:
        executed += 1
        # Extract entry_risk_usdt from steps_results if available
        steps_json = result.get("steps_results", "[]")
        try:
            steps = json.loads(steps_json) if steps_json else []
            for step in (steps if isinstance(steps, list) else []):
                risk = float(step.get("entry_risk_usdt", 0) or 0)
                if risk > 0:
                    entry_risk_post.append(risk)
        except:
            pass
    else:
        failed += 1

print(f"\n  apply.result match for post-deploy ENTRYs:")
print(f"    Matched to apply.result  : {matched}/{len(post_deploy)}")
print(f"    Executed (executed=True) : {executed}")
print(f"    Not executed             : {failed}")
print(f"    No result found          : {missing_result}")
if entry_risk_post:
    avg_risk = sum(entry_risk_post) / len(entry_risk_post)
    print(f"    entry_risk_usdt from steps: {len(entry_risk_post)} entries, avg={avg_risk:.4f} USDT")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[2] EXECUTION.RESULT STREAM GAP")
print(SEP)

exec_result_len = r.xlen("quantum:stream:execution.result")
print(f"  Stream length: {exec_result_len}")

if exec_result_len > 0:
    newest_msgs = r.xrevrange("quantum:stream:execution.result", count=3)
    oldest_msgs = r.xrange("quantum:stream:execution.result", count=1)

    newest_ts = int(newest_msgs[0][0].split("-")[0]) if newest_msgs else 0
    oldest_ts = int(oldest_msgs[0][0].split("-")[0]) if oldest_msgs else 0
    apply_result_newest = int(apply_result_msgs[0][0].split("-")[0]) if apply_result_msgs else 0

    print(f"  Oldest entry  : {ts_to_utc(oldest_ts)}")
    print(f"  Newest entry  : {ts_to_utc(newest_ts)}")
    print(f"  apply.result newest: {ts_to_utc(apply_result_newest)}")

    gap_hours = (apply_result_newest - newest_ts) / 3_600_000
    print(f"  GAP between execution.result → apply.result: {gap_hours:.1f} hours")
    if gap_hours > 24:
        print(f"  ⚠️  execution.result has been SILENT for {gap_hours:.0f} hours!")
        print(f"     → Either the execution service is paper-trading (not writing fills)")
        print(f"       or execution.result events were discontinued")
    else:
        print(f"  ✅ execution.result is recent")

    # Show last 3 execution.result payloads
    print(f"\n  Last 3 execution.result entries:")
    for mid, fields in newest_msgs:
        ts = int(mid.split("-")[0])
        try:
            payload = json.loads(fields.get("payload", "{}"))
        except:
            payload = {}
        print(f"  [{ts_to_utc(ts)}] event_type={fields.get('event_type','?')}")
        for k, v in list(payload.items())[:8]:
            print(f"    {k}: {str(v)[:80]}")

# Also check apply.result executed=True count
executed_true = sum(1 for _, f in apply_result_msgs if f.get("executed", "").lower() in ("true", "1"))
executed_false = sum(1 for _, f in apply_result_msgs if f.get("executed", "").lower() in ("false", "0"))
print(f"\n  apply.result last 5000: executed=True: {executed_true}, executed=False: {executed_false}")
if executed_false == len(apply_result_msgs):
    print(f"  ⚠️  ALL apply.result entries have executed=False → paper mode or blocked!")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[3] OPEN POSITIONS — SL integrity & risk tracking")
print(SEP)

all_pos_keys = r.keys("quantum:position:*")
real_pos = [k for k in all_pos_keys if not any(x in k for x in ["snapshot", "ledger", "claim", "cooldown"])]
print(f"  Total position keys  : {len(all_pos_keys)}")
print(f"  Real open positions  : {len(real_pos)}")

sl_ok = 0; sl_wrong = 0; sl_missing = 0
risk_ok_pos = 0; risk_zero_pos = 0; risk_missing_pos = 0
rl_mult_vals = []
pos_details = []

for k in real_pos:
    symbol = k.replace("quantum:position:", "")
    t = r.type(k)
    pos = {}
    if t == "hash":
        pos = r.hgetall(k)
    elif t == "string":
        raw = r.get(k)
        try:
            pos = json.loads(raw) if raw else {}
        except:
            pos = {}
    if not pos:
        continue

    side = pos.get("side", pos.get("direction", "")).upper()
    try:
        ep = float(pos.get("entry_price", 0) or 0)
        sl = float(pos.get("sl_price", pos.get("stop_loss", 0)) or 0)
    except:
        ep, sl = 0.0, 0.0

    # SL direction check
    if ep > 0 and sl > 0 and side in ("LONG", "BUY", "SHORT", "SELL"):
        is_long = side in ("LONG", "BUY")
        if is_long and sl < ep:
            sl_ok += 1
            detail = "SL✅"
        elif not is_long and sl > ep:
            sl_ok += 1
            detail = "SL✅"
        else:
            sl_wrong += 1
            detail = f"SL❌ ({side} entry={ep:.4f} sl={sl:.4f})"
    else:
        sl_missing += 1
        detail = f"SL? ep={ep} sl={sl}"

    # Risk tracking
    try:
        risk = float(pos.get("entry_risk_usdt", 0) or 0)
        rm   = int(pos.get("risk_missing", 0) or 0)
    except:
        risk, rm = 0.0, 1

    if risk > 0 and rm == 0:
        risk_ok_pos += 1
        risk_detail = f"risk={risk:.2f}✅"
    elif risk == 0 and rm == 1:
        risk_zero_pos += 1
        risk_detail = "risk=0❌"
    else:
        risk_missing_pos += 1
        risk_detail = f"risk={risk}? rm={rm}"

    # RL multiplier
    try:
        mult = float(pos.get("rl_multiplier", pos.get("size_multiplier", 1.0)) or 1.0)
        rl_mult_vals.append(mult)
        mult_detail = f"rl_mult={mult:.3f}"
    except:
        mult_detail = "rl_mult=?"

    pos_details.append(f"  {symbol:25s} {side:5s} {detail:35s} {risk_detail:20s} {mult_detail}")

print(f"\n  SL integrity:")
print(f"    Correct direction    : {sl_ok}/{len(real_pos)}")
print(f"    WRONG direction ❌   : {sl_wrong}")
print(f"    Missing/zero         : {sl_missing}")

print(f"\n  ATR risk tracking:")
print(f"    entry_risk_usdt > 0  : {risk_ok_pos}/{len(real_pos)}")
print(f"    entry_risk_usdt = 0  : {risk_zero_pos}  ← pre-fix positions")
print(f"    risk field absent    : {risk_missing_pos}")

if rl_mult_vals:
    non_default = [m for m in rl_mult_vals if abs(m - 1.0) > 0.01]
    print(f"\n  RL multiplier:")
    print(f"    Positions with rl_multiplier field : {len(rl_mult_vals)}")
    print(f"    Non-default (≠1.0)                 : {len(non_default)}  ← RL is sizing!")
    if rl_mult_vals:
        print(f"    Range: {min(rl_mult_vals):.3f} – {max(rl_mult_vals):.3f}")

if pos_details:
    print(f"\n  Per-position detail:")
    for d in pos_details[:15]:
        print(d)
    if len(pos_details) > 15:
        print(f"  ... ({len(pos_details)-15} more)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[4] RL SIZING STATUS")
print(SEP)

rl_exp_raw = r.llen("rl:experience")
rl_policy_updates = r.get("rl:policy_updates") or "0"
rl_staging = r.exists("rl:model:staging:ready")

print(f"  rl:experience buffer   : {rl_exp_raw}")
print(f"  rl:policy_updates      : {rl_policy_updates}")
print(f"  rl:model:staging:ready : {'YES' if rl_staging else 'no'}")

# Check recent RL logs
try:
    result = subprocess.run(
        ["journalctl", "-u", "quantum-rl-agent", "-n", "50", "--no-pager"],
        capture_output=True, text=True
    )
    lines = result.stdout.splitlines()
    stats_lines = [l for l in lines if "RL Agent Stats" in l or "Policy updated" in l or "Scheduled retrain" in l]
    if stats_lines:
        print(f"\n  Recent RL logs:")
        for l in stats_lines[-4:]:
            ts = l[:19] if len(l) > 19 else l
            msg = l[l.rfind("]")+2:] if "]" in l else l
            print(f"    {msg.strip()[:100]}")
except:
    print("  (could not read RL logs)")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[5] MANUAL LANE + TESTNET MODE")
print(SEP)

MANUAL_LANE_KEY = "quantum:manual_lane:enabled"
lane_val = r.get(MANUAL_LANE_KEY)
lane_ttl = r.ttl(MANUAL_LANE_KEY)

if lane_ttl > 0:
    print(f"  🔓 MANUAL_LANE_ACTIVE  ttl={lane_ttl//60}m {lane_ttl%60}s remaining")
else:
    print(f"  🔒 MANUAL_LANE_OFF  ({MANUAL_LANE_KEY} absent/expired)")
    print(f"     → No new ENTRY orders will be sent to Binance!")
    print(f"     → To enable: redis-cli SET {MANUAL_LANE_KEY} 1 EX <TTL_SECONDS>")
    print(f"       1h = EX 3600 | 4h = EX 14400 | 8h = EX 28800")

# Check testnet vs live
testnet_mode = False
try:
    result_env = subprocess.run(
        ["grep", "BINANCE_USE_TESTNET", "/etc/quantum/intent-executor.env"],
        capture_output=True, text=True
    )
    if "true" in result_env.stdout.lower():
        testnet_mode = True
except:
    pass
try:
    result_env2 = subprocess.run(
        ["grep", "BINANCE_USE_TESTNET", "/etc/quantum/execution.env"],
        capture_output=True, text=True
    )
    if "true" in result_env2.stdout.lower():
        testnet_mode = True
except:
    pass

if testnet_mode:
    print(f"\n  ⚠️  TESTNET MODE — executor pointing to testnet.binancefuture.com")
    print(f"     → Orders go to Binance TESTNET, not real account!")
else:
    print(f"\n  ✅ LIVE MODE — executor pointing to real Binance")

# P3.5 guard status
try:
    result_guard = subprocess.run(
        ["journalctl", "-u", "quantum-intent-executor", "-n", "50", "--no-pager"],
        capture_output=True, text=True
    )
    guard_lines = [l for l in result_guard.stdout.splitlines() if "P3.5_GUARD" in l]
    if guard_lines:
        last = guard_lines[-1]
        reason_part = last.split("reason=")[-1].strip()[:80] if "reason=" in last else "?"
        print(f"\n  P3.5_GUARD (last block): {reason_part}")
except:
    pass

# Drawdown
dd = r.hgetall("quantum:dag5:lockdown_guard:latest")
if dd:
    equity = float(dd.get("equity", 0))
    peak = float(dd.get("peak", 0))
    pct = float(dd.get("drawdown_pct", 0))
    lock = dd.get("lockdown_active", "?")
    print(f"\n  Drawdown: {pct:.2f}%  (equity=${equity:.0f}  peak=${peak:.0f})")
    if pct > 20:
        print(f"  ⚠️  Significant drawdown — {pct:.1f}% from peak")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("[6] SERVICE HEALTH")
print(SEP)

services = [
    "quantum-intent-bridge",
    "quantum-apply-layer",
    "quantum-intent-executor",
    "quantum-execution",
    "quantum-rl-agent",
    "quantum-rl-sizer",
    "quantum-rl-feedback-v2",
    "quantum-autonomous-trader",
]

for svc in services:
    try:
        r2 = subprocess.run(
            ["systemctl", "is-active", svc],
            capture_output=True, text=True
        )
        status = r2.stdout.strip()
        icon = "✅" if status == "active" else "❌"
        print(f"  {svc:40s} {status:10s} {icon}")
    except:
        print(f"  {svc:40s} ERROR")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

issues = []
if lane_ttl <= 0:
    issues.append(f"🔒 MANUAL_LANE_OFF — set `redis-cli SET quantum:manual_lane:enabled 1 EX 14400` to open")
if testnet_mode:
    issues.append(f"⚠️  TESTNET MODE — executor connected to testnet.binancefuture.com, not real Binance!")
if executed == 0 and post_deploy:
    issues.append(f"⚠️  {len(post_deploy)} post-deploy ENTRYs — 0 confirmed executed in apply.result!")
if executed_false == len(apply_result_msgs) and len(apply_result_msgs) > 0:
    issues.append(f"🚨 ALL {len(apply_result_msgs)} apply.result entries: executed=False")
if sl_wrong > 0:
    issues.append(f"🚨 {sl_wrong} positions have SL in WRONG direction!")
if risk_zero_pos > 0:
    issues.append(f"⚠️  {risk_zero_pos} positions still have entry_risk_usdt=0 (pre-fix)")

ok_items = []
if entry_price_ok == len(post_deploy) and post_deploy:
    ok_items.append(f"  ✅ entry_price present in 100% of post-deploy ENTRYs ({len(post_deploy)})")
if atr_ok == len(post_deploy) and post_deploy:
    ok_items.append(f"  ✅ atr_value present in 100% of post-deploy ENTRYs ({len(post_deploy)})")
if sl_ok > 0 and sl_wrong == 0:
    ok_items.append(f"  ✅ All {sl_ok} open positions: SL in correct direction")
if rl_mult_vals and non_default:
    ok_items.append(f"  ✅ RL sizing active: {len(non_default)} positions with non-default multiplier")

for ok in ok_items:
    print(ok)
if issues:
    print()
    for iss in issues:
        print(f"  {iss}")
if not issues:
    print(f"  ✅ No critical issues found — pipeline is healthy")
print()
