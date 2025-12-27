"""Check why circuit breaker is active"""

import sys
sys.path.insert(0, '/app')

def check_circuit_breaker():
    print("🔍 CHECKING CIRCUIT BREAKER STATUS")
    print("=" * 80)
    
    # Check logs for circuit breaker info
    import subprocess
    result = subprocess.run(
        ["grep", "-i", "circuit breaker", "/app/backend/logs/app.log"],
        capture_output=True,
        text=True
    )
    
    # Check if circuit breaker is active
    print(f"\n📊 Circuit Breaker Status:")
    
    if "circuit breaker active" in result.stdout.lower():
        print(f"   ❌ Active")
    else:
        print(f"   ✅ Not active - trading allowed")
        print("\n" + "=" * 80)
        return True
    
    # Alternatively, check from data files
    import json
    from pathlib import Path
    
    # Check if there's a state file
    state_files = [
        "/app/backend/data/global_risk_state.json",
        "/app/backend/data/circuit_breaker.json"
    ]
    
    found_state = False
    for path in state_files:
        p = Path(path)
        if p.exists():
            found_state = True
            print(f"\n📂 Found state file: {path}")
            try:
                data = json.loads(p.read_text())
                print(json.dumps(data, indent=2))
            except Exception as e:
                print(f"   Error reading: {e}")
    
    if not found_state:
        print("\n⚠️  No circuit breaker state files found")
        print("   Checking recent logs...")
    
    print("\n" + "=" * 80)
    print("⛔ CIRCUIT BREAKER BLOKKERER ALL TRADING!")
    print("\n💡 LØSNINGER:")
    print("   1. ⏰ Vent til cooldown periode utløper (vanligvis 30min - 1 time)")
    print("   2. 🔄 Restart backend for å resette: docker-compose restart backend")
    print("   3. ⚙️  Øk max_daily_drawdown i config (ikke anbefalt)")
    print("=" * 80)
    
    return False
    
    # Check if circuit breaker is active
    print(f"\n📊 Circuit Breaker Status:")
    print(f"   Active: {grc.circuit_breaker_active}")
    
    if grc.circuit_breaker_active:
        import datetime
        remaining = (grc.circuit_breaker_until - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 3600
        print(f"   ⏱️  Active until: {grc.circuit_breaker_until}")
        print(f"   ⏱️  Time remaining: {remaining:.2f} hours")
        print(f"\n⚠️  CIRCUIT BREAKER BLOKKERER ALL TRADING!")
        print(f"   Årsak: For store tap detektert")
        print(f"   Cooldown periode må utløpe før ny trading kan starte")
    else:
        print(f"   ✅ Not active - trading allowed")
    
    # Check recent trades
    print(f"\n📈 Recent Trade History:")
    print(f"   Total trades: {len(grc.trade_history)}")
    
    if grc.trade_history:
        recent = grc.trade_history[-10:]
        wins = sum(1 for t in recent if t.pnl_usd > 0)
        losses = sum(1 for t in recent if t.pnl_usd < 0)
        total_pnl = sum(t.pnl_usd for t in recent)
        
        print(f"   Last 10 trades:")
        print(f"     Wins: {wins}")
        print(f"     Losses: {losses}")
        print(f"     Total PnL: ${total_pnl:.2f}")
        
        print(f"\n   Last 5 trades:")
        for t in grc.trade_history[-5:]:
            emoji = "✅" if t.pnl_usd > 0 else "❌"
            print(f"     {emoji} {t.symbol}: ${t.pnl_usd:.2f} ({t.pnl_pct*100:.2f}%) @ {t.exit_time.strftime('%H:%M:%S')}")
    
    # Check drawdown
    print(f"\n📉 Drawdown Tracking:")
    print(f"   Current drawdown: {grc.current_drawdown*100:.2f}%")
    print(f"   Max allowed: {grc.config.max_daily_drawdown*100:.2f}%")
    print(f"   Cooldown duration: {grc.config.circuit_breaker_cooldown_hours}h")
    
    # Check position limits
    print(f"\n📊 Position Limits:")
    print(f"   Max open positions: {grc.config.max_open_positions}")
    print(f"   Current open: {len(grc.open_positions)}")
    
    print("\n" + "=" * 80)
    
    if grc.circuit_breaker_active:
        print("⛔ LØSNING:")
        print("   1. Vent til cooldown periode utløper")
        print("   2. ELLER restart backend for å resette (ikke anbefalt)")
        print("   3. ELLER øk max_daily_drawdown i config (farlig)")
        return False
    else:
        print("✅ Trading kan fortsette")
        return True

if __name__ == "__main__":
    success = check_circuit_breaker()
    sys.exit(0 if success else 1)
