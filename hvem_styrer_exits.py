"""
Check who is currently controlling exits
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config.exit_mode import (
    get_exit_mode,
    get_exit_executor_mode,
    is_exit_brain_live_fully_enabled,
)

def main():
    print("\n" + "="*70)
    print("HVEM STYRER EXITS AKKURAT NÅ?")
    print("="*70)
    
    exit_mode = get_exit_mode()
    executor_mode = get_exit_executor_mode()
    fully_live = is_exit_brain_live_fully_enabled()
    
    print(f"\n📊 Konfigurasjon:")
    print(f"   EXIT_MODE: {exit_mode}")
    print(f"   EXIT_EXECUTOR_MODE: {executor_mode}")
    print(f"   Fully LIVE: {'JA' if fully_live else 'NEI'}")
    
    print(f"\n👥 HVEM STYRER:")
    print("="*70)
    
    if exit_mode == "LEGACY":
        print("""
🔵 LEGACY MODE - Tradisjonelt system

AKTIVE MODULER (BRAIN + MUSCLE):
  1. ✅ Position Monitor (position_monitor)
     - Overvåker alle posisjoner hvert 10. sekund
     - Setter automatisk TP/SL hvis de mangler
     - Justerer SL/TP basert på AI sentiment
     - PLASSERER ORDRER DIREKTE til Binance
  
  2. ✅ Hybrid TP/SL System (hybrid_tpsl)
     - Dynamiske TP/SL nivåer per posisjon
     - Reagerer på markedsendringer
     - PLASSERER ORDRER DIREKTE
  
  3. ✅ Trailing Stop Manager (trailing_stop_manager)
     - Flytter SL opp når profit øker
     - PLASSERER ORDRER DIREKTE

Exit Brain V3:
  ❌ IKKE AKTIV (EXIT_MODE=LEGACY)
  
PROBLEM: "Too many cooks" - 3+ moduler som alle er BRAIN+MUSCLE samtidig!
        """)
    
    elif exit_mode == "EXIT_BRAIN_V3":
        if fully_live:
            print("""
🔴 EXIT BRAIN V3 LIVE MODE - AI HAR FULL KONTROLL!

AKTIVE MODULER:

Exit Brain Dynamic Executor (exit_executor):
  ✅ SINGLE MUSCLE - AI plasserer alle exit ordrer
  ✅ Overvåker posisjoner kontinuerlig
  ✅ 5 beslutningstyper:
     - NO_CHANGE: Hold current TP/SL
     - FULL_EXIT_NOW: Emergency market close
     - PARTIAL_CLOSE: Take partial profit
     - MOVE_SL: Adjust stop loss
     - UPDATE_TP_LIMITS: Adjust take profit
  ✅ PLASSERER ORDRER via exit_order_gateway

Legacy Moduler:
  🛑 BLOKKERT av exit_order_gateway
  🛑 Position Monitor: Kjører men ordrer AVVIST
  🛑 Hybrid TP/SL: Kjører men ordrer AVVIST
  🛑 Trailing Stop: Kjører men ordrer AVVIST
  
Gateway blokkerer automatisk alle ordrer fra legacy moduler.
Exit Brain er nå SINGLE MUSCLE for exits!
            """)
        else:
            print("""
🟡 EXIT BRAIN V3 SHADOW MODE - OBSERVASJON

AKTIVE MODULER:

Exit Brain Dynamic Executor (exit_executor):
  🔍 SHADOW MODE - Observerer og logger
  ✅ Overvåker posisjoner kontinuerlig
  ✅ Bestemmer hva den VILLE gjort
  📝 Logger beslutninger til:
     - Console: [EXIT_BRAIN_SHADOW] messages
     - File: backend/data/exit_brain_shadow.jsonl
  ❌ PLASSERER INGEN ORDRER (shadow mode)

Legacy Moduler (AKTIVE - de styrer faktisk):
  ✅ Position Monitor (position_monitor)
     → Setter TP/SL, justerer basert på AI
     → PLASSERER ORDRER DIREKTE til Binance
     → Gateway logger conflicts men TILLATER ordrer
  
  ✅ Hybrid TP/SL System (hybrid_tpsl)
     → Dynamiske TP/SL nivåer
     → PLASSERER ORDRER DIREKTE
     → Gateway logger conflicts men TILLATER ordrer
  
  ✅ Trailing Stop Manager (trailing_stop_manager)
     → Flytter SL ved profit
     → PLASSERER ORDRER DIREKTE
     → Gateway logger conflicts men TILLATER ordrer

STATUS: Exit Brain observerer, men Legacy moduler styrer fortsatt!
        Gateway logger "OWNERSHIP CONFLICT" warnings, men tillater ordrer.
        Dette er NORMALT i SHADOW mode - vi evaluerer AI før vi gir kontroll.
            """)
    
    print(f"\n📋 SAMMENDRAG:")
    print("="*70)
    
    if exit_mode == "LEGACY":
        print("❌ Legacy mode: 3+ moduler konkurrerer (too many cooks problem)")
        print("⚠️  Exit Brain V3 ikke aktivert")
    elif exit_mode == "EXIT_BRAIN_V3":
        if fully_live:
            print("✅ LIVE MODE: Exit Brain har full kontroll")
            print("✅ Legacy moduler blokkert")
            print("✅ Single MUSCLE for exits")
        else:
            print("🟡 SHADOW MODE: Legacy moduler styrer FORTSATT")
            print("🔍 Exit Brain observerer og logger")
            print("⏳ Kjør 24-48t før LIVE mode")
            print("")
            print("LEGACY MODULER ER FORTSATT AKTIVE I SHADOW MODE!")
            print("Exit Brain lærer av deres beslutninger før den tar over.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
