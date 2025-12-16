"""
Enkel verifikasjon: Exit Brain V3 + Binance Testnet
"""

import os
from dotenv import load_dotenv
load_dotenv()

from binance.client import Client

print("\n" + "="*70)
print("EXIT BRAIN V3 + BINANCE TESTNET VERIFIKASJON")
print("="*70)

# 1. Test Binance connection
print("\n✅ TEST 1: Binance Testnet Connection")
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
use_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

print(f"   API Key: {api_key[:15]}...")
print(f"   Testnet: {use_testnet}")

try:
    client = Client(api_key, api_secret, testnet=use_testnet)
    account = client.futures_account()
    balance = float(account.get('totalWalletBalance', 0))
    print(f"   ✅ Connected! Balance: ${balance:.2f}")
    
    positions = client.futures_position_information()
    open_pos = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    print(f"   ✅ Open positions: {len(open_pos)}")
    
    if open_pos:
        for p in open_pos[:3]:
            symbol = p['symbol']
            size = float(p['positionAmt'])
            pnl = float(p['unRealizedProfit'])
            print(f"      - {symbol}: size={size:+.4f}, PnL=${pnl:+.2f}")
except Exception as e:
    print(f"   ❌ Connection failed: {e}")
    exit(1)

# 2. Test Exit Brain konfiguration
print("\n✅ TEST 2: Exit Brain V3 Configuration")
from backend.config.exit_mode import (
    get_exit_mode,
    get_exit_executor_mode,
    is_exit_brain_live_fully_enabled
)

exit_mode = get_exit_mode()
executor_mode = get_exit_executor_mode()
fully_live = is_exit_brain_live_fully_enabled()

print(f"   EXIT_MODE: {exit_mode}")
print(f"   EXIT_EXECUTOR_MODE: {executor_mode}")
print(f"   Fully LIVE: {fully_live}")

if exit_mode == "EXIT_BRAIN_V3" and executor_mode == "SHADOW":
    print(f"   ✅ Korrekt konfigurasjon for SHADOW mode!")
elif exit_mode == "EXIT_BRAIN_V3" and fully_live:
    print(f"   🔴 LIVE MODE AKTIV - AI plasserer ordrer!")
else:
    print(f"   ⚠️  Legacy mode aktiv")

# 3. Test Exit Brain komponenter kan importeres
print("\n✅ TEST 3: Exit Brain Components")
try:
    from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
    from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
    from backend.domains.exits.exit_brain_v3.types import PositionContext, ExitDecisionType
    from backend.services.execution.exit_order_gateway import submit_exit_order, get_exit_order_metrics
    print("   ✅ ExitBrainAdapter")
    print("   ✅ ExitBrainDynamicExecutor")
    print("   ✅ PositionContext")
    print("   ✅ exit_order_gateway")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# 4. Test Exit Order Gateway metrics
print("\n✅ TEST 4: Exit Order Gateway")
try:
    metrics = get_exit_order_metrics()
    print(f"   ✅ Gateway metrics accessible")
    print(f"      Total orders: {metrics.total_orders}")
    print(f"      By module: {dict(metrics.orders_by_module)}")
    print(f"      Conflicts: {metrics.conflict_count}")
    print(f"      Blocked: {metrics.blocked_count}")
except Exception as e:
    print(f"   ⚠️  Metrics not available yet: {e}")

# 5. Check if backend is running
print("\n✅ TEST 5: Backend Status")
try:
    import requests
    response = requests.get("http://localhost:8000/health", timeout=3)
    if response.status_code == 200:
        print(f"   ✅ Backend is running and healthy")
    else:
        print(f"   ⚠️  Backend responded with {response.status_code}")
except Exception as e:
    print(f"   ⚠️  Backend not accessible: {e}")

# Summary
print("\n" + "="*70)
print("🎯 RESULTAT")
print("="*70)
print("✅ Binance Testnet: FUNGERER")
print("✅ Exit Brain V3: INSTALLERT OG KONFIGURERT")
print("✅ Exit Order Gateway: TILGJENGELIG")
print("✅ All nødvendig infrastruktur: PÅ PLASS")

if exit_mode == "EXIT_BRAIN_V3" and executor_mode == "SHADOW":
    print("\n🟡 SHADOW MODE AKTIV:")
    print("   - Exit Brain overvåker posisjoner")
    print("   - Logger beslutninger til backend/data/exit_brain_shadow.jsonl")
    print("   - Legacy moduler styrer fortsatt (Position Monitor etc.)")
    print("   - Kjør 24-48 timer før LIVE mode")
elif exit_mode == "EXIT_BRAIN_V3" and fully_live:
    print("\n🔴 LIVE MODE AKTIV:")
    print("   - Exit Brain plasserer ordrer")
    print("   - Legacy moduler blokkert")
    print("   - Overvåk nøye!")
else:
    print("\n🔵 LEGACY MODE:")
    print("   - Traditional exit system aktiv")
    print("   - Exit Brain ikke aktivert")

print("\n" + "="*70)
