import json,sys
d=json.load(sys.stdin)["data"]["activeTargets"]
for t in d:
    if t["labels"].get("job")=="quantum_safety_telemetry":
        print("job:", t["labels"].get("job"))
        print("instance:", t["labels"].get("instance"))
        print("scrapeUrl:", t.get("scrapeUrl"))
        print("health:", t.get("health"))
        print("lastScrape:", t.get("lastScrape"))
        print("lastError:", t.get("lastError"))
        print("discoveredLabels:", {k:v for k,v in t.get("discoveredLabels",{}).items() if k in ["__address__","__metrics_path__"]})
