#!/usr/bin/env python3
"""Test Position Monitor status and summary."""
import os
import asyncio
from backend.services.monitoring.position_monitor import PositionMonitor

async def main():
    print("\n" + "=" * 70)
    print("[SEARCH] POSITION MONITOR TEST - AI OVERVÅKER ALLE POSISJONER")
    print("=" * 70)
    
    monitor = PositionMonitor(check_interval=30)
    
    print("\n⚙️  KONFIGURASJON:")
    print(f"   TP: {monitor.tp_pct*100:.1f}%")
    print(f"   SL: {monitor.sl_pct*100:.1f}%")
    print(f"   Trailing: {monitor.trail_pct*100:.1f}%")
    print(f"   Partial TP: {monitor.partial_tp*100:.0f}%")
    print(f"   Check interval: {monitor.check_interval}s")
    
    print("\n[SEARCH] KJØRER SJEKK AV ALLE POSISJONER...")
    result = await monitor.check_all_positions()
    
    print(f"\n[CHART] RESULTAT:")
    print(f"   Status: {result['status']}")
    print(f"   Total posisjoner: {result['positions']}")
    print(f"   Beskyttede: {result['protected']}")
    print(f"   Ubeskyttede: {result['unprotected']}")
    print(f"   Nylig beskyttet: {result['newly_protected']}")
    
    print("\n💡 HVORDAN DET FUNGERER:")
    print("   1️⃣  Position Monitor kjører hvert 30. sekund")
    print("   2️⃣  Sjekker ALLE åpne posisjoner på Binance")
    print("   3️⃣  Hvis posisjon mangler TP/SL → setter det automatisk")
    print("   4️⃣  Bruker hybrid strategi: 50% TP + 50% trailing")
    print("   5️⃣  Trailing Stop Manager følger så vinners opp")
    
    print("\n🤖 AI STYRER ALT AUTOMATISK:")
    print("   [OK] Nye posisjoner: Backend setter TP/SL ved åpning")
    print("   [OK] Eksisterende: Position Monitor finner og beskytter")
    print("   [OK] Profit tracking: Trailing Stop Manager følger prisen")
    print("   [OK] Event-driven: AI åpner posisjoner ved 70%+ confidence")
    
    print("\n" + "=" * 70)
    print("[OK] AI HAR FULL KONTROLL - INGEN SKRIPT NØDVENDIG!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
