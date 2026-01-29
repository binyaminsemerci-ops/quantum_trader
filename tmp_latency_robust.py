import subprocess, re
from statistics import median

def redis_cmd(*args):
    p = subprocess.run(["redis-cli","--raw",*args], capture_output=True, text=True, check=True)
    return p.stdout.splitlines()

def parse_stream(raw_lines):
    # returns list of (id, fields_dict)
    out=[]
    i=0
    while i < len(raw_lines):
        sid = raw_lines[i].strip()
        if re.match(r"^[0-9]+-[0-9]+$", sid):
            fields={}
            i+=1
            while i+1 < len(raw_lines) and not re.match(r"^[0-9]+-[0-9]+$", raw_lines[i].strip()):
                k = raw_lines[i].strip()
                v = raw_lines[i+1].strip()
                fields[k]=v
                i+=2
            out.append((sid, fields))
        else:
            i+=1
    return out

def ts_ms(stream_id: str) -> int:
    return int(stream_id.split("-")[0])

# Tune windows if needed
OBS_COUNT = 2000
APPLY_COUNT = 8000

obs_raw = redis_cmd("XREVRANGE","quantum:stream:apply.heat.observed","+","-","COUNT",str(OBS_COUNT))
apply_raw = redis_cmd("XREVRANGE","quantum:stream:apply.plan","+","-","COUNT",str(APPLY_COUNT))

obs_events = parse_stream(obs_raw)
apply_events = parse_stream(apply_raw)

# Build map: plan_id -> list of (ts_ms, stream_id) sorted oldest->newest
apply_ts_by_plan={}
for sid, f in apply_events:
    pid = f.get("plan_id")
    if pid:
        if pid not in apply_ts_by_plan:
            apply_ts_by_plan[pid] = []
        apply_ts_by_plan[pid].append((ts_ms(sid), sid))

# Sort each plan's timestamps (oldest first for binary search)
for pid in apply_ts_by_plan:
    apply_ts_by_plan[pid].sort()

# Collect latencies for publish_plan_post with robust matching
lat=[]
miss_apply=0
negative_count=0
for sid, f in obs_events:
    if f.get("obs_point") != "publish_plan_post":
        continue
    pid = f.get("plan_id")
    if not pid:
        continue
    
    obs_ts = ts_ms(sid)
    apply_ts_list = apply_ts_by_plan.get(pid)
    
    if not apply_ts_list:
        miss_apply += 1
        continue
    
    # Find closest apply_ts <= obs_ts (most recent before observation)
    best_ats = None
    for ats, _ in apply_ts_list:
        if ats <= obs_ts:
            best_ats = ats
        else:
            break  # timestamps are sorted, no need to check further
    
    if best_ats is None:
        # All apply events are AFTER observation (timing anomaly)
        negative_count += 1
        continue
    
    latency = obs_ts - best_ats
    lat.append(latency)

if not lat:
    print("No publish_plan_post samples with valid matching in window.")
    print(f"obs_window={OBS_COUNT} apply_window={APPLY_COUNT} miss_apply={miss_apply} negative_outliers={negative_count}")
    raise SystemExit(0)

lat.sort()
n=len(lat)
def pct(p):
    idx = int(n*p)
    if idx>=n: idx=n-1
    return lat[idx]

print(f"samples={n} miss_apply={miss_apply} negative_outliers={negative_count} obs_window={OBS_COUNT} apply_window={APPLY_COUNT}")
print(f"p50={pct(0.50)}ms p90={pct(0.90)}ms p95={pct(0.95)}ms p99={pct(0.99)}ms max={lat[-1]}ms")
print(f"min={lat[0]}ms")

# Truth summary for tuning (always calculate headroom against configured MAX_WAIT)
MAX_WAIT_MS = 2000  # Current config
p99_val = pct(0.99)
max_val = lat[-1]
headroom = MAX_WAIT_MS / p99_val if p99_val > 0 else float('inf')
print(f"\n[TRUTH] p99={p99_val}ms max={max_val}ms samples={n} negative_outliers={negative_count} headroom={headroom:.1f}x (max_wait={MAX_WAIT_MS}ms)")
