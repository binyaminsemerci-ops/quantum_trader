#!/usr/bin/env python3
"""Remove dead Prometheus scrape targets from config"""
import yaml
import sys

config_path = '/etc/prometheus/prometheus.yml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Find scrape_configs
scrape_configs = config.get('scrape_configs', [])

# Remove dead targets
dead_jobs = ['quantum_trader', 'cadvisor']
original_count = len(scrape_configs)

scrape_configs = [
    job for job in scrape_configs 
    if job.get('job_name') not in dead_jobs
]

removed = original_count - len(scrape_configs)
config['scrape_configs'] = scrape_configs

# Write back
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"✅ Removed {removed} dead targets: {', '.join(dead_jobs)}")
print(f"✅ Active jobs: {len(scrape_configs)}")
