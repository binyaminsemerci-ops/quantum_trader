#!/usr/bin/env python3
"""
Fix AI Engine asyncio saturation:
1. Disable/reduce external API calls that timeout (funding/volatility/orderbook)
2. Increase TOP_N_BUFFER_INTERVAL to reduce processing frequency
3. Test if HTTP responds
"""
import subprocess, time, urllib.request, json, re

ENV = "/etc/quantum/ai-engine.env"
print("=== AI ENGINE ASYNCIO FIX ===")

with open(ENV) as f:
    content = f.read()

print("[CURRENT CONFIG]")
for line in content.splitlines():
    if any(k in line.upper() for k in ['FUNDING', 'VOLATIL', 'ORDERBOOK', 'TIMEOUT', 'INTERVAL', 'QT_MAX', 'TOP_N']):
        print(f"  {line}")

# Apply fixes
changes = []

# 1. Increase TOP_N_BUFFER_INTERVAL: 2.0 → 30.0 (process every 30s not every 2s)
old = 'TOP_N_BUFFER_INTERVAL_SEC=2.0'
new = 'TOP_N_BUFFER_INTERVAL_SEC=30.0'
if old in content:
    content = content.replace(old, new)
    changes.append(f"TOP_N_BUFFER_INTERVAL_SEC: 2.0 → 30.0")

# 2. Also set in Environment override (service file has TOP_N_BUFFER_INTERVAL_SEC=3.0)
# Just add to env file which overrides service file
if 'TOP_N_BUFFER_INTERVAL_SEC=30.0' not in content and 'TOP_N_BUFFER_INTERVAL_SEC' not in content:
    content += '\nTOP_N_BUFFER_INTERVAL_SEC=30.0\n'
    changes.append("Added TOP_N_BUFFER_INTERVAL_SEC=30.0")

# 3. Disable funding rate enrichment if possible
if 'FUNDING_ENRICHMENT_ENABLED' not in content:
    content += '\nFUNDING_ENRICHMENT_ENABLED=false\n'
    changes.append("Added FUNDING_ENRICHMENT_ENABLED=false")
elif 'FUNDING_ENRICHMENT_ENABLED=true' in content:
    content = content.replace('FUNDING_ENRICHMENT_ENABLED=true', 'FUNDING_ENRICHMENT_ENABLED=false')
    changes.append("FUNDING_ENRICHMENT_ENABLED: true → false")

# 4. Disable volatility enrichment
if 'VOLATILITY_ENRICHMENT_ENABLED' not in content:
    content += '\nVOLATILITY_ENRICHMENT_ENABLED=false\n'
    changes.append("Added VOLATILITY_ENRICHMENT_ENABLED=false")

# 5. Disable orderbook enrichment
if 'ORDERBOOK_ENRICHMENT_ENABLED' not in content:
    content += '\nORDERBOOK_ENRICHMENT_ENABLED=false\n'
    changes.append("Added ORDERBOOK_ENRICHMENT_ENABLED=false")

# 6. Also try USE_PHASE1=false etc
if 'PHASE1_ENABLED' not in content:
    content += '\nPHASE1_ENABLED=false\n'
    changes.append("Added PHASE1_ENABLED=false (disable funding phase)")

with open(ENV, 'w') as f:
    f.write(content)

print(f"\n[CHANGES APPLIED: {len(changes)}]")
for c in changes:
    print(f"  {c}")

# Restart AI Engine
print("\n[RESTARTING AI ENGINE]")
subprocess.run(['systemctl', 'restart', 'quantum-ai-engine'])
print("  Waiting 35s for startup...")
time.sleep(35)

# Test HTTP
print("\n[HTTP TEST]")
for attempt in range(3):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=8) as resp:
            data = resp.read()
            print(f"  ✅ HTTP OK: {data[:100]}")
            break
    except Exception as e:
        print(f"  Attempt {attempt+1}/3: {type(e).__name__}: {str(e)[:60]}")
        if attempt < 2:
            time.sleep(5)

print("\n=== DONE ===")
