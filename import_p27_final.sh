#!/bin/bash
# P2.7 Dashboard Import - Simplified Working Version
set -euo pipefail

echo "🔍 Step 1: Finding Prometheus datasource UID..."
DS_UID=$(curl -s -u admin:QuantumTrader2024! http://127.0.0.1:3000/api/datasources \
  | python3 -c "import sys,json; ds=[d for d in json.load(sys.stdin) if d.get('type')=='prometheus']; print(ds[0]['uid'] if ds else '')")

echo "   Prometheus UID: $DS_UID"
test -n "$DS_UID" || { echo "❌ No Prometheus datasource found"; exit 2; }

echo ""
echo "🔧 Step 2: Patching dashboard JSON with datasource UID ($DS_UID)..."
python3 << PYEOF
import json

# Load dashboard
with open('/tmp/quantum_harvest_control_p27.json') as f:
    dashboard = json.load(f)

ds_uid = "$DS_UID"

# Recursively update datasource UIDs
def update_datasource(obj):
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == "datasource" and isinstance(value, dict):
                uid = value.get("uid")
                if isinstance(uid, str) and "\${DS_PROMETHEUS}" in uid:
                    value["uid"] = ds_uid
            else:
                update_datasource(value)
    elif isinstance(obj, list):
        for item in obj:
            update_datasource(item)

update_datasource(dashboard)

# Ensure dashboard has required fields
if not dashboard.get("title"):
    dashboard["title"] = "Quantum Harvest Control (P2.7)"
if dashboard.get("id"):
    del dashboard["id"]  # Remove id for new import

# Create import payload
payload = {
    "dashboard": dashboard,
    "overwrite": True
}

with open('/tmp/p27_import_payload.json', 'w') as f:
    json.dump(payload, f, indent=2)

print("   ✅ Dashboard patched and ready for import")
PYEOF

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
    DASH_URL=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('url',''))")
    echo "   Dashboard URL: http://46.224.116.254:3000$DASH_URL"
    echo ""
    echo "   Login: admin / QuantumTrader2024!"
else
    echo ""
    echo "❌ Import may have failed. Check output above."
    exit 3
fi
