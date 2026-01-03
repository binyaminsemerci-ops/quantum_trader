#!/usr/bin/env python3
"""Add Prometheus datasource to all Grafana dashboard panels"""

import json
import sys
from pathlib import Path

def add_datasources_to_dashboard(dashboard_path, datasource_uid="PBFA97CFB590B2093"):
    """Add Prometheus datasource to all panel targets in dashboard"""
    
    with open(dashboard_path, 'r') as f:
        dashboard = json.load(f)
    
    panels_updated = 0
    targets_updated = 0
    
    # Add datasource to all panel targets
    for panel in dashboard.get('panels', []):
        panel_had_update = False
        for target in panel.get('targets', []):
            if 'datasource' not in target:
                target['datasource'] = {
                    'type': 'prometheus',
                    'uid': datasource_uid
                }
                targets_updated += 1
                panel_had_update = True
        
        if panel_had_update:
            panels_updated += 1
    
    # Write back
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"✅ {dashboard_path.name}: Updated {panels_updated} panels, {targets_updated} targets")
    return targets_updated > 0

if __name__ == "__main__":
    dashboards_dir = Path("observability/grafana/dashboards")
    
    if not dashboards_dir.exists():
        print(f"❌ Directory not found: {dashboards_dir}")
        sys.exit(1)
    
    updated_count = 0
    for dashboard_file in dashboards_dir.glob("*.json"):
        if add_datasources_to_dashboard(dashboard_file):
            updated_count += 1
    
    print(f"\n🎉 Updated {updated_count} dashboards total")
