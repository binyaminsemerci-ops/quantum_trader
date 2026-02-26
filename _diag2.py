import redis, time, json

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

print("=== HARVEST BRAIN INTENTS (last 5) ===")
entries = r.xrevrange("quantum:stream:harvest.intent", count=5)
for eid, data in entries:
    age = int(time.time()) - int(eid.split("-")[0])//1000
    sym = data.get("symbol","?")
    action = data.get("action","?")
    r_level = data.get("r_level","?")
    fraction = data.get("fraction","?")
    print(f"  {sym} action={action} R={r_level} fraction={fraction}  {age}s ago")
if not entries:
    print("  none")

print()
print("=== HARVEST BRAIN LOGS (last 30 lines) ===")
import subprocess
res = subprocess.run(["journalctl","-u","quantum-harvest-brain","-n","30","--no-pager"], capture_output=True, text=True)
for line in res.stdout.strip().split("\n")[-30:]:
    print(" ", line)

print()
print("=== AI ENGINE INTENTS (payload parsed, last 3) ===")
entries = r.xrevrange("quantum:stream:trade.intent", count=3)
for eid, data in entries:
    age = int(time.time()) - int(eid.split("-")[0])//1000
    try:
        p = json.loads(data.get("payload","{}"))
        print(f"  {p.get('symbol','?')} side={p.get('side','?')} confidence={p.get('confidence','?'):.2f}  {age}s ago")
    except:
        print(f"  parse error {age}s ago")

print()
print("=== POSITION STATE BRAIN LOGS (last 20) ===")
res = subprocess.run(["journalctl","-u","quantum-position-state-brain","-n","20","--no-pager"], capture_output=True, text=True)
for line in res.stdout.strip().split("\n")[-20:]:
    print(" ", line)
