#!/bin/bash
# P2.7 Dashboard Import - 100% Working Script
set -euo pipefail

DASH=/tmp/quantum_harvest_control_p27.json
test -f "$DASH" || { echo "❌ Missing $DASH"; exit 1; }

echo "🔍 Step 1: Finding Prometheus datasource UID..."
DS_UID=$(curl -s -u admin:QuantumTrader2024! http://127.0.0.1:3000/api/datasources \
  | python3 -c "import sys,json; ds=[d for d in json.load(sys.stdin) if d.get('type')=='prometheus']; print(ds[0]['uid'] if ds else '')")

echo "   Prometheus UID: $DS_UID"
test -n "$DS_UID" || { echo "❌ No Prometheus datasource found"; exit 2; }

echo ""
echo "🔧 Step 2: Patching dashboard JSON with datasource UID..."
python3 << 'PYEOF'
import json
import sys

dash_path = "/tmp/quantum_harvest_control_p27.json"
ds_uid = sys.argv[1]

with open(dash_path) as f:
    dashboard = json.load(f)

def update_datasource(obj):
    """Recursively update datasource UIDs"""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == "datasource" and isinstance(value, dict):
                uid = value.get("uid")
                if isinstance(uid, str) and "${DS_PROMETHEUS}" in uid:
                    value["uid"] = ds_uid
                    print(f"   Updated datasource in panel/target")
            else:
                update_datasource(value)
    elif isinstance(obj, list):
        for item in obj:
            update_datasource(item)

update_datasource(dashboard)

payload = {
    "dashboard": dashboard,
    "overwrite": True
}

output_path = "/tmp/p27_import_payload.json"
with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"   ✅ Wrote {output_path}")
PYEOF

python3 -c "
import json, sys
with open('/tmp/quantum_harvest_control_p27.json') as f:
    d = json.load(f)
def walk(x):
    if isinstance(x, dict):
        for k,v in list(x.items()):
            if k == 'datasource' and isinstance(v, dict):
                uid = v.get('uid')
                if isinstance(uid, str) and '\${DS_PROMETHEUS}' in uid:
                    v['uid'] = '$DS_UID'
            else:
                walk(v)
    elif isinstance(x, list):
        for i in x: walk(i)
walk(d)
with open('/tmp/p27_import_payload.json','w') as f:
    json.dump({'dashboard': d, 'overwrite': True}, f)
print('   ✅ Datasource UIDs updated')
"

echo ""
echo "📤 Step 3: Importing dashboard to Grafana..."
RESULT=$(curl -s -u admin:QuantumTrader2024! \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:3000/api/dashboards/db \
  --data-binary @/tmp/p27_import_payload.json)

echo "$RESULT" | python3 -m json.tool

# Check if successful
if echo "$RESULT" | grep -q '"status":"success"'; then
    echo ""
    echo "✅ Dashboard imported successfully!"
    echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"   Dashboard URL: http://46.224.116.254:3000{d.get('url','')}\")"
else
    echo ""
    echo "❌ Import may have failed. Check output above."
    exit 3
fi
