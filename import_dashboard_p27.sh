#!/bin/bash
# P2.7 Dashboard Import Script

# Get Prometheus datasource UID
curl -s -u admin:QuantumTrader2024! http://localhost:3001/api/datasources > /tmp/datasources.json
PROM_UID=$(python3 -c "import json; ds=[d for d in json.load(open('/tmp/datasources.json')) if d.get('type')=='prometheus']; print(ds[0]['uid'] if ds else '')")

echo "Prometheus UID: $PROM_UID"

if [ -n "$PROM_UID" ]; then
  # Update dashboard JSON with Prometheus UID
  python3 << 'PYEOF'
import json

with open('/tmp/quantum_harvest_control_p27.json') as f:
    dashboard = json.load(f)

with open('/tmp/datasources.json') as f:
    datasources = json.load(f)
    prom_uid = [d['uid'] for d in datasources if d.get('type') == 'prometheus'][0]

# Recursively update datasource UIDs
def update_datasource(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'datasource':
                if isinstance(v, dict):
                    if '${DS_PROMETHEUS}' in str(v.get('uid', '')):
                        v['uid'] = prom_uid
                    if v.get('type') == 'prometheus' and not v.get('uid'):
                        v['uid'] = prom_uid
            else:
                update_datasource(v)
    elif isinstance(obj, list):
        for item in obj:
            update_datasource(item)

update_datasource(dashboard)

# Wrap in dashboard object for API
dashboard_payload = {
    'dashboard': dashboard,
    'overwrite': True
}

with open('/tmp/dashboard_updated.json', 'w') as f:
    json.dump(dashboard_payload, f, indent=2)

print(f'Dashboard updated with Prometheus UID: {prom_uid}')
PYEOF

  # Import dashboard via Grafana API
  RESULT=$(curl -s -X POST -u admin:QuantumTrader2024! \
    -H 'Content-Type: application/json' \
    -d @/tmp/dashboard_updated.json \
    http://localhost:3001/api/dashboards/db)
  
  echo "Import result:"
  echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"
else
  echo "ERROR: Could not find Prometheus datasource"
  exit 1
fi
