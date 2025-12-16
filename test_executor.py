#!/usr/bin/env python3
"""
Test manual trade via Event-Driven Executor API
"""
import requests
import json

# Test event-driven executor directly
url = "http://localhost:8000/health"
print(f"Testing {url}...")
r = requests.get(url)
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))

# Try to force a rebalance check
print("\n" + "="*80)
print("Checking if Event-Driven Executor is actually running...")
print("="*80)

health = r.json()
print(f"event_driven_active: {health.get('event_driven_active')}")
print(f"allowed_symbols: {len(health.get('risk', {}).get('config', {}).get('allowed_symbols', []))}")
