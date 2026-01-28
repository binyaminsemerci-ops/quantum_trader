#!/usr/bin/env python3
"""P2.7 Grafana Dashboard Import Script"""

import json
import urllib.request
import base64

# Grafana credentials (Base64: admin:QuantumTrader2024!)
AUTH_HEADER = 'Basic YWRtaW46UXVhbnR1bVRyYWRlcjIwMjQh'
GRAFANA_URL = 'http://localhost:3001'
DASHBOARD_FILE = '/tmp/quantum_harvest_control_p27.json'

def main():
    # Step 1: Get Prometheus datasource UID
    print("Fetching Prometheus datasource UID...")
    req = urllib.request.Request(f'{GRAFANA_URL}/api/datasources')
    req.add_header('Authorization', AUTH_HEADER)
    
    with urllib.request.urlopen(req) as response:
        datasources = json.loads(response.read())
    
    prom_datasources = [d for d in datasources if d.get('type') == 'prometheus']
    if not prom_datasources:
        print("ERROR: No Prometheus datasource found!")
        return 1
    
    prom_uid = prom_datasources[0]['uid']
    print(f"Prometheus UID: {prom_uid}")
    
    # Step 2: Load dashboard JSON
    print(f"Loading dashboard from {DASHBOARD_FILE}...")
    with open(DASHBOARD_FILE) as f:
        dashboard = json.load(f)
    
    # Step 3: Update datasource UIDs recursively
    print("Updating datasource references...")
    def update_datasource(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == 'datasource' and isinstance(value, dict):
                    # Replace placeholder with actual UID
                    if '${DS_PROMETHEUS}' in str(value.get('uid', '')):
                        value['uid'] = prom_uid
                        print(f"  Updated datasource UID in {key}")
                else:
                    update_datasource(value)
        elif isinstance(obj, list):
            for item in obj:
                update_datasource(item)
    
    update_datasource(dashboard)
    
    # Step 4: Wrap dashboard for API
    dashboard_payload = {
        'dashboard': dashboard,
        'overwrite': True
    }
    
    # Step 5: Import dashboard
    print("Importing dashboard to Grafana...")
    payload_bytes = json.dumps(dashboard_payload).encode('utf-8')
    req = urllib.request.Request(
        f'{GRAFANA_URL}/api/dashboards/db',
        data=payload_bytes,
        method='POST'
    )
    req.add_header('Authorization', AUTH_HEADER)
    req.add_header('Content-Type', 'application/json')
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read())
        print("✅ Dashboard imported successfully!")
        print(json.dumps(result, indent=2))
        return 0
    except urllib.error.HTTPError as e:
        print(f"❌ Import failed: {e}")
        print(e.read().decode())
        return 1

if __name__ == '__main__':
    exit(main())
