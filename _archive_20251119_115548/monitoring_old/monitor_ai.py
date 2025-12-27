#!/usr/bin/env python3
"""Monitor AI activity and show when it's checking for signals."""
import asyncio
import subprocess
from datetime import datetime

async def monitor_ai():
    print("\n" + "=" * 70)
    print("🤖 AI MONITORING - LIVE ACTIVITY")
    print("=" * 70)
    print("\n⏰ Timestamp: ", datetime.now().strftime("%H:%M:%S"))
    print("\n[CHART] AI SØKER ETTER SIGNALER...")
    print("   • Sjekker 36 symbols hvert 10. sekund")
    print("   • Krever 70%+ confidence for å trade")
    print("   • Position Monitor sjekker hvert 30s")
    print("\n💡 FØLG MED I DOCKER LOGS:")
    print("   docker logs quantum_backend --tail 50 --follow")
    print("\n[SEARCH] ELLER SJEKK STATUS:")
    print("   python check_ai_status.py")
    print("\n" + "=" * 70)
    print("⏳ Venter på AI aktivitet...\n")
    
    # Show last few log lines
    try:
        result = subprocess.run(
            ["docker", "logs", "quantum_backend", "--tail", "10"],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.split('\n')
        for line in lines[-5:]:
            if line.strip():
                print(f"   {line}")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(monitor_ai())
