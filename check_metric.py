import json,sys
d=json.load(sys.stdin)
r=d.get("data",{}).get("result",[])
print("result_count:", len(r))
for s in r:
    print("labels:", s["metric"])
    print("value:", s["value"])
