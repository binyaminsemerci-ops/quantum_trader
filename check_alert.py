import json,sys
d=json.load(sys.stdin)
alerts=[a for a in d["data"]["alerts"] if "SafeMode" in a["labels"].get("alertname","")]
print(f"SafeMode alerts: {len(alerts)}")
for a in alerts:
    print(f"  State: {a['state']}")
    print(f"  Active since: {a['activeAt']}")
    print(f"  Summary: {a['annotations'].get('summary','N/A')}")
