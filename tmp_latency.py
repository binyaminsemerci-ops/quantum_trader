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

# Map plan_id -> apply_ts_ms (first match is newest; ok for recent window)
apply_ts_by_plan={}
for sid, f in apply_events:
    pid = f.get("plan_id")
    if pid and pid not in apply_ts_by_plan:
        apply_ts_by_plan[pid]=ts_ms(sid)

# Collect latencies for publish_plan_post
lat=[]
miss_apply=0
for sid, f in obs_events:
    if f.get("obs_point") != "publish_plan_post":
        continue
    pid = f.get("plan_id")
    if not pid:
        continue
    ats = apply_ts_by_plan.get(pid)
    if ats is None:
        miss_apply += 1
        continue
    lat.append(ts_ms(sid) - ats)

if not lat:
    print("No publish_plan_post samples in window.")
    print(f"obs_window={OBS_COUNT} apply_window={APPLY_COUNT} miss_apply_match={miss_apply}")
    raise SystemExit(0)

lat.sort()
n=len(lat)
def pct(p):
    idx = int(n*p)
    if idx>=n: idx=n-1
    return lat[idx]

print(f"samples={n} miss_apply_match={miss_apply} obs_window={OBS_COUNT} apply_window={APPLY_COUNT}")
print(f"p50={pct(0.50)}ms p90={pct(0.90)}ms p95={pct(0.95)}ms p99={pct(0.99)}ms max={lat[-1]}ms")
print(f"min={lat[0]}ms")
