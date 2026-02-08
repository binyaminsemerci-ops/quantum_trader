#!/usr/bin/env python3
"""
Cadence monitoring script - checks if system is ready for calibration
"""
import time
import json
import subprocess
import redis
from datetime import datetime

print("=== CADENCE MONITORING - STARTING ===\n")
print("Vil overvåke hvert 30. sekund i 6 minutter:")
print("  1. Learning Cadence status (READY?)")
print("  2. CLM data availability")
print("  3. Trade activity")
print("  4. Equity freshness\n")

r = redis.Redis(decode_responses=True)
calibration_ready = False

for i in range(1, 13):
    print("━" * 50)
    print(f"CHECK #{i} - {datetime.now().strftime('%H:%M:%S')}")
    print("━" * 50)
    print()
    
    # 1. Cadence status
    print("📊 1. LEARNING CADENCE STATUS:")
    cadence_ready = False
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:8007/health"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            status = data.get("status", "unknown")
            print(f"  Status: {status}")
            if "ready" in data:
                ready = data["ready"]
                print(f"  Ready: {ready}")
                cadence_ready = (ready == True)
        else:
            print("  ❌ API not responding")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 2. CLM data
    print("\n📈 2. CLM DATA:")
    clm_has_data = False
    try:
        count = r.xlen("quantum:stream:trade.closed")
        print(f"  Stream length: {count} entries")
        clm_has_data = (count > 0)
    except Exception as e:
        print(f"  Error: {e}")
    
    # 3. Equity status
    print("\n💰 3. EQUITY STATUS:")
    equity_fresh = False
    try:
        d = r.hgetall("quantum:equity:current")
        equity = float(d.get("equity", 0))
        age = time.time() - float(d.get("last_update_ts", 0))
        fresh = "✅" if age < 300 else "❌"
        print(f"  Equity: ${equity:.2f}, Age: {age:.1f}s, Fresh: {fresh}")
        equity_fresh = (age < 300)
    except Exception as e:
        print(f"  Error: {e}")
    
    # 4. Recent activity
    print("\n🎯 4. RECENT POSITION ACTIVITY (last 5 min):")
    has_activity = False
    try:
        result = subprocess.run(
            "journalctl --since '5 minutes ago' 2>/dev/null | grep -i 'position.*opened\\|trade.*executed\\|order.*filled' | wc -l",
            shell=True,
            capture_output=True,
            text=True
        )
        count = int(result.stdout.strip())
        print(f"  Activity events: {count}")
        has_activity = (count > 0)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Check if all conditions met
    print()
    if cadence_ready and equity_fresh:
        print("🎉" * 25)
        print("✅ CALIBRATION KAN KJØRES!")
        print("🎉" * 25)
        print()
        print("Betingelser oppfylt:")
        print(f"  ✅ Cadence: READY")
        print(f"  ✅ Equity: FRESH (<300s)")
        if clm_has_data:
            print(f"  ✅ CLM: {r.xlen('quantum:stream:trade.closed')} entries")
        if has_activity:
            print(f"  ✅ Trading: Active")
        print()
        calibration_ready = True
        break
    
    print()
    if i < 12:
        print("⏳ Waiting 30 seconds...")
        time.sleep(30)
        print()

print("\n=== MONITORING COMPLETE ===")
if not calibration_ready:
    print("\n⚠️  Calibration prerequisites not yet met")
    print("Continue monitoring or check system status")
