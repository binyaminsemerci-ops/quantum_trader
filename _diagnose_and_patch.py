#!/usr/bin/env python3
"""
Three-part diagnostic + fix:
A) Check real Binance FAPI positions vs Redis
B) Diagnose execution.result silence (quantum-execution service logs)
C) ATR-patch the 23 old positions with entry_risk_usdt=0
"""
import redis, json, time, subprocess, os, hmac, hashlib, urllib.parse
import urllib.request

r = redis.Redis(decode_responses=True)

# ── Load API keys from env ─────────────────────────────────────────────────
def load_env_file(path):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except:
        pass
    return env

env = {}
for ef in ["/etc/quantum/intent-executor.env", "/etc/quantum/execution.env",
           "/etc/quantum/autonomous-trader.env", "/etc/quantum/apply-layer.env"]:
    env.update(load_env_file(ef))

API_KEY    = env.get("BINANCE_API_KEY", "")
API_SECRET = env.get("BINANCE_API_SECRET", "")
BASE_URL   = env.get("BINANCE_BASE_URL", "https://fapi.binance.com")

print("=" * 70)
print("A) REAL BINANCE POSITIONS vs REDIS")
print("=" * 70)
print(f"  BASE_URL: {BASE_URL}")
print(f"  API_KEY:  {API_KEY[:8]}..." if API_KEY else "  API_KEY: MISSING")
print()

def binance_signed_get(path, params=None):
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    query = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{path}?{query}&signature={sig}"
    req = urllib.request.Request(url, headers={"X-MBX-APIKEY": API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

# Get real positions from Binance
if API_KEY and API_SECRET:
    result = binance_signed_get("/fapi/v2/positionRisk")
    if isinstance(result, list):
        binance_positions = {p["symbol"]: p for p in result
                             if float(p.get("positionAmt", "0")) != 0}
        print(f"  Real Binance open positions: {len(binance_positions)}")
        for sym, p in sorted(binance_positions.items()):
            amt = float(p["positionAmt"])
            epx = float(p["entryPrice"])
            upnl = float(p["unRealizedProfit"])
            mark = float(p["markPrice"])
            side = "LONG" if amt > 0 else "SHORT"
            print(f"    {sym:20} {side:5} amt={amt:>12.3f}  ep={epx:.6f}  mark={mark:.6f}  pnl={upnl:+.2f}")

        # Compare against Redis
        print(f"\n  Redis open positions:")
        redis_positions = {}
        pos_keys = r.keys("quantum:position:*")
        for k in pos_keys:
            if k.count(":") == 2:
                sym = k.split(":")[-1]
                data = r.hgetall(k)
                if data.get("quantity") and float(data.get("quantity", 0)) > 0:
                    redis_positions[sym] = data
        print(f"    Total Redis positions with qty>0: {len(redis_positions)}")

        # Reconciliation
        only_redis = set(redis_positions.keys()) - set(binance_positions.keys())
        only_binance = set(binance_positions.keys()) - set(redis_positions.keys())
        both = set(redis_positions.keys()) & set(binance_positions.keys())
        print(f"\n  Reconciliation:")
        print(f"    In both Redis + Binance: {len(both)}")
        print(f"    Redis only (phantom)   : {len(only_redis)} {sorted(only_redis)[:10]}")
        print(f"    Binance only (missing) : {len(only_binance)} {sorted(only_binance)[:10]}")
    elif "error" in result:
        print(f"  ❌ Binance API error: {result['error']}")
    else:
        print(f"  ❌ Unexpected response: {str(result)[:200]}")
else:
    print("  ❌ Missing API_KEY or API_SECRET — cannot query Binance")

print()
print("=" * 70)
print("B) EXECUTION.RESULT SILENCE — quantum-execution service diagnostic")
print("=" * 70)

# Check service status + recent logs
def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

status = run_cmd(["systemctl", "is-active", "quantum-execution"])
print(f"  quantum-execution status: {status}")

# Last 50 lines filtered for key events
logs = subprocess.run(
    ["journalctl", "-u", "quantum-execution", "-n", "100", "--no-pager"],
    capture_output=True, text=True
).stdout
print(f"\n  Recent quantum-execution logs (filtered):")
relevant = []
for line in logs.splitlines():
    if any(k in line for k in [
        "ERROR", "error", "except", "Traceback", "REJECT", "order", "ORDER",
        "fapi", "testnet", "connected", "start", "Started", "execution.result",
        "write", "execute", "position", "fill", "FILLED", "submit",
        "BASE_URL", "base_url", "BINANCE", "binance"
    ]):
        relevant.append(line.strip()[-200:])
for r_line in relevant[-30:]:
    print(f"  {r_line}")

if not relevant:
    print("  (No relevant log hits — showing last 15 lines)")
    for line in logs.splitlines()[-15:]:
        print(f"  {line.strip()[-200:]}")

# Check quantum-execution env file
print("\n  quantum-execution env config:")
exec_env = load_env_file("/etc/quantum/execution.env")
for k, v in exec_env.items():
    if "SECRET" in k.upper():
        print(f"    {k}=***")
    elif "KEY" in k.upper():
        print(f"    {k}={v[:8]}...")
    else:
        print(f"    {k}={v}")

# Check if quantum-execution has its own API key config
print("\n  Checking all env files for BINANCE keys used by execution service...")
for ef in ["/etc/quantum/execution.env", "/etc/quantum/intent-executor.env",
           "/etc/quantum/autonomous-trader.env"]:
    ev = load_env_file(ef)
    keys_found = {k: v for k, v in ev.items() if "BINANCE" in k.upper()}
    if keys_found:
        print(f"    {ef}:")
        for k, v in keys_found.items():
            if "SECRET" in k.upper():
                print(f"      {k}=***")
            elif "KEY" in k.upper():
                print(f"      {k}={v[:8]}...")
            else:
                print(f"      {k}={v}")

# Check execution.result stream recency
exec_stream_len = r.xlen("quantum:stream:execution.result") if r.exists("quantum:stream:execution.result") else 0
apply_stream_len = r.xlen("quantum:stream:apply.result") if r.exists("quantum:stream:apply.result") else 0
print(f"\n  quantum:stream:execution.result length: {exec_stream_len}")
print(f"  quantum:stream:apply.result     length: {apply_stream_len}")

# Check if quantum-execution reads from apply.result
print("\n  Checking quantum-execution source code path...")
exec_main = run_cmd(["find", "/home/qt/quantum_trader/microservices",
                     "-name", "main.py", "-path", "*execution*"])
print(f"  main.py paths: {exec_main}")

print()
print("=" * 70)
print("C) PATCH 23 PRE-FIX POSITIONS — add entry_risk_usdt from ATR")
print("=" * 70)

# Get all positions missing entry_risk_usdt or entry_risk_usdt=0
to_patch = []
for pkey in r.keys("quantum:position:*"):
    if pkey.count(":") != 2:
        continue
    sym = pkey.split(":")[-1]
    d = r.hgetall(pkey)
    if not d:
        continue
    risk = float(d.get("entry_risk_usdt", 0) or 0)
    qty = float(d.get("quantity", 0) or 0)
    ep = float(d.get("entry_price", 0) or 0)
    sl = float(d.get("stop_loss", 0) or 0)
    leverage = float(d.get("leverage", 20) or 20)
    if risk == 0 and qty > 0 and ep > 0:
        to_patch.append({
            "key": pkey, "symbol": sym,
            "qty": qty, "entry_price": ep, "stop_loss": sl,
            "leverage": leverage, "side": d.get("side", "SHORT")
        })

print(f"  Positions needing entry_risk_usdt patch: {len(to_patch)}")

patched = 0
for pos in to_patch:
    sym = pos["symbol"]
    ep = pos["entry_price"]
    sl = pos["stop_loss"]
    qty = pos["qty"]
    side = pos["side"]

    # Try to get ATR from Redis market data
    atr_val = 0.0
    for atr_key in [f"quantum:atr:{sym}", f"quantum:market:{sym}:atr",
                    f"quantum:market:{sym}", f"quantum:market:{sym}:binance_main"]:
        atr_raw = r.hget(atr_key, "atr") or r.hget(atr_key, "atr_14") or r.get(atr_key)
        if atr_raw:
            try:
                atr_val = float(atr_raw)
                break
            except:
                pass

    if atr_val <= 0:
        # Fall back to: risk = 1.5 * ATR * qty where ATR ≈ 2% of entry
        atr_val = ep * 0.02

    # For SHORT: SL should be ABOVE entry; risk per unit = |sl - ep| or 1.5*ATR if SL missing
    if sl > 0 and sl != ep:
        risk_per_unit = abs(sl - ep)
    else:
        risk_per_unit = 1.5 * atr_val

    entry_risk_usdt = round(risk_per_unit * qty, 4)
    r.hset(pos["key"], "entry_risk_usdt", entry_risk_usdt)
    r.hset(pos["key"], "risk_patched_by", "c3_activation_retrospective")
    r.hset(pos["key"], "risk_patched_at", int(time.time()))
    patched += 1
    print(f"  ✅ {sym:20} ep={ep:.5f}  risk_per_unit={risk_per_unit:.6f}  "
          f"qty={qty:.1f}  entry_risk_usdt={entry_risk_usdt:.4f}  "
          f"(atr={'from_redis' if atr_val != ep*0.02 else 'estimated@2%'})")

print(f"\n  Patched {patched}/{len(to_patch)} positions with entry_risk_usdt")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
