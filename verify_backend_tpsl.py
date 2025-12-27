"""
Verify Backend TP/SL System
============================
Checks if the backend is correctly placing TP/SL orders on new AI trades.

This script:
1. Checks if EventDrivenExecutor is running
2. Verifies STAGING_MODE setting
3. Checks Binance credentials
4. Scans logs for TP/SL placement
5. Tests Trading Profile API endpoint
6. Provides actionable recommendations

Author: Quantum Trader Team
Date: 2025-11-26
"""

import os
import sys
import requests
import subprocess
import json
from datetime import datetime

def check_env_variable(var_name: str, required: bool = True) -> tuple[bool, str]:
    """Check if environment variable is set."""
    value = os.getenv(var_name)
    if value:
        # Mask secrets
        if 'SECRET' in var_name or 'KEY' in var_name:
            masked = value[:4] + '...' + value[-4:] if len(value) > 8 else '***'
            return True, masked
        return True, value
    return False, 'NOT SET'

def check_backend_health() -> tuple[bool, str]:
    """Check if backend is running."""
    try:
        resp = requests.get('http://localhost:8000/health', timeout=5)
        if resp.status_code == 200:
            return True, 'ONLINE'
        return False, f'HTTP {resp.status_code}'
    except Exception as e:
        return False, str(e)

def check_trading_profile() -> tuple[bool, dict]:
    """Check Trading Profile configuration."""
    try:
        resp = requests.get('http://localhost:8000/trading-profile/config', timeout=10)
        if resp.status_code == 200:
            config = resp.json()
            return True, config
        return False, {}
    except Exception as e:
        return False, {}

def check_docker_logs(container: str = 'quantum_backend', search_terms: list = None) -> dict:
    """Search Docker logs for specific terms."""
    if search_terms is None:
        search_terms = ['TP order placed', 'SL order placed', 'Event-driven trading mode']
    
    results = {}
    for term in search_terms:
        try:
            cmd = f'docker logs {container} 2>&1 | Select-String "{term}"'
            result = subprocess.run(
                ['powershell', '-Command', cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split('\n') if result.stdout else []
            results[term] = {
                'found': len(lines) > 0 and lines[0] != '',
                'count': len([l for l in lines if l.strip()]),
                'last_occurrence': lines[-1] if lines and lines[0] else None
            }
        except Exception as e:
            results[term] = {'found': False, 'count': 0, 'error': str(e)}
    
    return results

def main():
    print("=" * 80)
    print("🔍 BACKEND TP/SL SYSTEM VERIFICATION")
    print("=" * 80)
    print()
    
    # Check 1: Backend Health
    print("📊 [1/6] Backend Status")
    print("-" * 80)
    is_online, status = check_backend_health()
    if is_online:
        print(f"✅ Backend: {status}")
    else:
        print(f"❌ Backend: {status}")
        print("   ACTION: Start backend with: docker-compose --profile dev up -d")
    print()
    
    # Check 2: Environment Variables
    print("🔑 [2/6] Environment Variables")
    print("-" * 80)
    
    staging_set, staging_value = check_env_variable('STAGING_MODE', required=False)
    if staging_value == 'false' or staging_value == 'NOT SET':
        print(f"✅ STAGING_MODE: {staging_value} (live orders enabled)")
    else:
        print(f"⚠️  STAGING_MODE: {staging_value} (orders will be SIMULATED!)")
    
    binance_key_set, binance_key = check_env_variable('BINANCE_API_KEY')
    if binance_key_set:
        print(f"✅ BINANCE_API_KEY: {binance_key}")
    else:
        print(f"❌ BINANCE_API_KEY: {binance_key}")
    
    binance_secret_set, binance_secret = check_env_variable('BINANCE_API_SECRET')
    if binance_secret_set:
        print(f"✅ BINANCE_API_SECRET: {binance_secret}")
    else:
        print(f"❌ BINANCE_API_SECRET: {binance_secret}")
    
    if not (binance_key_set and binance_secret_set):
        print("   ❌ CRITICAL: Missing Binance credentials!")
        print("   ACTION: Set BINANCE_API_KEY and BINANCE_API_SECRET in .env")
    print()
    
    # Check 3: Trading Profile
    print("⚙️  [3/6] Trading Profile Configuration")
    print("-" * 80)
    if is_online:
        tp_ok, config = check_trading_profile()
        if tp_ok:
            enabled = config.get('enabled', False)
            if enabled:
                print(f"✅ Trading Profile: ENABLED")
                
                risk = config.get('risk', {})
                tpsl = config.get('tpsl', {})
                
                print(f"   Leverage: {risk.get('default_leverage', 0)}x")
                print(f"   Max Positions: {risk.get('max_positions', 0)}")
                print(f"   ATR Period: {tpsl.get('atr_period', 0)} on {tpsl.get('atr_timeframe', 'N/A')}")
                print(f"   TP1: {tpsl.get('atr_mult_tp1', 0)}R (50% close)")
                print(f"   TP2: {tpsl.get('atr_mult_tp2', 0)}R (30% close)")
                print(f"   SL: {tpsl.get('atr_mult_sl', 0)}R")
            else:
                print(f"⚠️  Trading Profile: DISABLED")
                print("   ACTION: Enable in config or API")
        else:
            print(f"❌ Could not fetch Trading Profile config")
    else:
        print("⚠️  Skipped (backend offline)")
    print()
    
    # Check 4: Event-Driven Mode
    print("🤖 [4/6] Event-Driven Executor")
    print("-" * 80)
    log_results = check_docker_logs()
    
    event_driven = log_results.get('Event-driven trading mode', {})
    if event_driven.get('found'):
        print(f"✅ Event-driven mode: ACTIVE ({event_driven.get('count')} occurrences)")
        if event_driven.get('last_occurrence'):
            print(f"   Last: {event_driven.get('last_occurrence')[:100]}...")
    else:
        print(f"❌ Event-driven mode: NOT DETECTED")
        print("   ACTION: Check QT_EVENT_DRIVEN_MODE=true in .env")
    print()
    
    # Check 5: TP/SL Order Placement
    print("🎯 [5/6] TP/SL Order Placement Logs")
    print("-" * 80)
    
    tp_logs = log_results.get('TP order placed', {})
    sl_logs = log_results.get('SL order placed', {})
    
    if tp_logs.get('found'):
        print(f"✅ TP orders: {tp_logs.get('count')} placed")
        if tp_logs.get('last_occurrence'):
            print(f"   Last: {tp_logs.get('last_occurrence')[:100]}...")
    else:
        print(f"⚠️  TP orders: No logs found")
        print("   This could mean:")
        print("   - No AI trades executed yet")
        print("   - TP/SL placement is failing")
        print("   - Logs rotated/cleared")
    
    if sl_logs.get('found'):
        print(f"✅ SL orders: {sl_logs.get('count')} placed")
        if sl_logs.get('last_occurrence'):
            print(f"   Last: {sl_logs.get('last_occurrence')[:100]}...")
    else:
        print(f"⚠️  SL orders: No logs found")
    print()
    
    # Check 6: Summary & Recommendations
    print("=" * 80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    all_good = (
        is_online and
        binance_key_set and
        binance_secret_set and
        (staging_value == 'false' or staging_value == 'NOT SET') and
        event_driven.get('found', False)
    )
    
    if all_good:
        print("✅ SYSTEM STATUS: READY")
        print()
        print("Backend TP/SL system appears to be configured correctly!")
        print()
        if not tp_logs.get('found') and not sl_logs.get('found'):
            print("⚠️  No TP/SL logs detected yet. This is normal if:")
            print("   - System just started")
            print("   - No AI trades triggered yet")
            print("   - Waiting for strong signals (confidence >= threshold)")
            print()
            print("💡 To verify TP/SL works:")
            print("   1. Wait for AI to open a position")
            print("   2. Check logs: docker logs quantum_backend | Select-String 'TP order placed'")
            print("   3. Verify on Binance: Futures → Orders → Check TP/SL exist")
        else:
            print("✅ TP/SL orders ARE being placed automatically!")
            print(f"   TP orders: {tp_logs.get('count', 0)}")
            print(f"   SL orders: {sl_logs.get('count', 0)}")
    else:
        print("⚠️  SYSTEM STATUS: NEEDS ATTENTION")
        print()
        print("Issues detected:")
        if not is_online:
            print("❌ Backend offline")
        if not binance_key_set or not binance_secret_set:
            print("❌ Missing Binance credentials")
        if staging_value not in ['false', 'NOT SET']:
            print(f"❌ STAGING_MODE={staging_value} (should be false)")
        if not event_driven.get('found'):
            print("❌ Event-driven mode not detected")
        print()
        print("ACTION REQUIRED:")
        print("1. Fix issues listed above")
        print("2. Restart backend: docker-compose --profile dev restart")
        print("3. Re-run verification: python verify_backend_tpsl.py")
    
    print()
    print("=" * 80)
    print("🛡️  PROTECTION COVERAGE")
    print("=" * 80)
    print()
    print("AUTOMATIC (Backend TP/SL):")
    if all_good:
        print("✅ Covers: AI-generated trades")
        print("✅ Triggers: When EventDrivenExecutor opens position")
        print("✅ Method: ATR-based multi-target (1.5R/2.5R TP, 1R SL)")
    else:
        print("⚠️  Not functioning - see issues above")
    print()
    print("MANUAL POSITIONS:")
    print("❌ NOT covered by backend system")
    print("✅ SOLUTION: Use Position Protection Service")
    print("   Run: python position_protection_service.py")
    print("   Covers: ALL positions (manual + automated)")
    print("   Check: Every 60 seconds")
    print()
    print("=" * 80)
    
    # Exit code
    sys.exit(0 if all_good else 1)

if __name__ == "__main__":
    main()
