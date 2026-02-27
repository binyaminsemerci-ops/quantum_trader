#!/usr/bin/env python3
import redis, subprocess, re, time, sys

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
out = []

def p(s):
    out.append(s)
    print(s, flush=True)

p("=== FIX START ===")

# 1. Ghost slots
ledger = r.smembers("quantum:ledger:positions") or set()
p(f"LEDGER: {sorted(ledger)}")
pos_keys = r.keys("quantum:position:*")
cleared = 0
active = []
for k in pos_keys:
    d = r.hgetall(k)
    qty = float(d.get('quantity','0') or '0')
    st = d.get('status','')
    sym = k.split(':')[-1]
    if qty > 0 and st not in ('CLOSED','CLOSED_GHOST'):
        if sym not in ledger:
            r.hset(k, mapping={'quantity':'0','status':'CLOSED_GHOST'})
            cleared += 1
        else:
            active.append(sym)
p(f"GHOST_CLEARED:{cleared}")
p(f"REAL_ACTIVE:{active}")
p(f"FREE_SLOTS:{10-len(active)}")

# 2. Reduce exit_manager timeout 60→15
em = "microservices/autonomous_trader/exit_manager.py"
try:
    code = open(em).read()
    n60 = code.count('timeout=60.0')
    code2 = code.replace('timeout=60.0','timeout=15.0')
    open(em,'w').write(code2)
    p(f"TIMEOUT_FIX:{n60}_occurrences_60->15")
except Exception as e:
    p(f"TIMEOUT_ERR:{e}")

# 3. Restart
for svc in ['quantum-autonomous-trader']:
    r2 = subprocess.run(['systemctl','restart',svc], capture_output=True)
    time.sleep(3)
    r3 = subprocess.run(['systemctl','is-active',svc], capture_output=True, text=True)
    p(f"RESTART:{svc}:{r3.stdout.strip()}")

p("=== FIX DONE ===")
