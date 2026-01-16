"""
Quick TP/SL System Check
=========================
One-command check for TP/SL system status.
"""

import os
import sys

def main():
    print("=" * 60)
    print("🎯 QUICK TP/SL SYSTEM CHECK")
    print("=" * 60)
    print()
    
    # Check .env
    has_key = bool(os.getenv("BINANCE_API_KEY"))
    has_secret = bool(os.getenv("BINANCE_API_SECRET"))
    
    print("📋 SYSTEM STATUS:")
    print(f"  Backend: {'✅' if os.path.exists('backend/') else '❌'}")
    print(f"  Credentials: {'✅' if (has_key and has_secret) else '❌'}")
    print()
    
    if not (has_key and has_secret):
        print("⚠️  MISSING CREDENTIALS!")
        print()
        print("TO FIX:")
        print("1. Edit .env file")
        print("2. Add:")
        print("   BINANCE_API_KEY=your_key")
        print("   BINANCE_API_SECRET=your_secret")
        print("3. Restart: docker-compose --profile dev restart")
        print()
        return
    
    print("✅ CREDENTIALS FOUND")
    print()
    print("NEXT STEPS:")
    print()
    print("1. VERIFY BACKEND:")
    print("   python verify_backend_tpsl.py")
    print()
    print("2. FIX DASHUSDT:")
    print("   python fix_dash_tpsl.py")
    print()
    print("3. PROTECT ALL POSITIONS:")
    print("   python position_protection_service.py --once")
    print()
    print("4. RUN CONTINUOUS PROTECTION:")
    print("   python position_protection_service.py")
    print()

if __name__ == "__main__":
    main()

