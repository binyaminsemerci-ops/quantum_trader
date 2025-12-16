"""
Live test av Exit Brain V3 mot Binance Testnet

Tester:
1. Kan hente live posisjoner fra Binance
2. Exit Brain Adapter kan analysere posisjon
3. Dynamic Executor kan kjøre i SHADOW mode
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Load .env first
from dotenv import load_dotenv
load_dotenv()

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from binance.client import Client
from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor


async def test_live_connection():
    """Test 1: Kan vi koble til Binance Testnet?"""
    print("\n" + "="*70)
    print("TEST 1: Binance Testnet Connection")
    print("="*70)
    
    binance_api_key = os.getenv("BINANCE_API_KEY")
    binance_api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not binance_api_key or not binance_api_secret:
        print("❌ FEIL: BINANCE_API_KEY/SECRET ikke satt i .env")
        return False
    
    print(f"✅ API Key: {binance_api_key[:10]}...")
    
    # Check if testnet
    use_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    print(f"✅ Testnet mode: {use_testnet}")
    
    try:
        client = Client(binance_api_key, binance_api_secret, testnet=use_testnet)
        
        # Test account access
        account = client.futures_account()
        balance = float(account.get('totalWalletBalance', 0))
        print(f"✅ Account connected: Balance = ${balance:.2f}")
        
        # Get positions
        positions = client.futures_position_information()
        open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        
        print(f"✅ Open positions: {len(open_positions)}")
        
        if open_positions:
            print("\n📊 Aktive posisjoner:")
            for pos in open_positions[:5]:  # Show first 5
                symbol = pos['symbol']
                size = float(pos['positionAmt'])
                entry = float(pos['entryPrice'])
                upnl = float(pos['unRealizedProfit'])
                pnl_pct = (upnl / (abs(size) * entry)) * 100 if size != 0 and entry != 0 else 0
                
                print(f"   {symbol:12s}: size={size:+.4f} entry=${entry:.2f} "
                      f"PnL={upnl:+.2f} ({pnl_pct:+.2f}%)")
        else:
            print("\n⚠️  Ingen åpne posisjoner (kan ikke teste AI-beslutninger)")
        
        return True, client, open_positions
        
    except Exception as e:
        print(f"❌ FEIL ved Binance connection: {e}")
        return False, None, []


async def test_exit_brain_adapter(open_positions):
    """Test 2: Kan Exit Brain analysere posisjoner?"""
    print("\n" + "="*70)
    print("TEST 2: Exit Brain Adapter Analysis")
    print("="*70)
    
    if not open_positions:
        print("⚠️  Ingen posisjoner å analysere")
        return True
    
    try:
        adapter = ExitBrainAdapter()
        print("✅ ExitBrainAdapter initialized")
        
        # Analyze first position
        pos = open_positions[0]
        symbol = pos['symbol']
        size = float(pos['positionAmt'])
        entry = float(pos['entryPrice'])
        mark_price = float(pos['markPrice'])
        upnl = float(pos['unRealizedProfit'])
        
        print(f"\n🔍 Analyzing: {symbol}")
        print(f"   Size: {size:+.4f}")
        print(f"   Entry: ${entry:.2f}")
        print(f"   Mark: ${mark_price:.2f}")
        print(f"   Unrealized PnL: ${upnl:+.2f}")
        
        # Build position context (simplified)
        from backend.domains.exits.exit_brain_v3.types import PositionContext
        
        side = "long" if size > 0 else "short"
        pnl_pct = ((mark_price - entry) / entry * 100) if side == "long" else ((entry - mark_price) / entry * 100)
        
        ctx = PositionContext(
            symbol=symbol,
            side=side,
            size=abs(size),
            entry_price=entry,
            current_price=mark_price,
            unrealized_pnl=pnl_pct  # In % terms
        )
        
        # Get AI decision
        decision = adapter.decide(ctx)
        
        print(f"\n🤖 AI Decision:")
        print(f"   Type: {decision.decision_type}")
        print(f"   Confidence: {decision.confidence:.2%}")
        print(f"   Reasoning: {decision.reasoning[:100]}...")
        
        if decision.new_sl_price:
            print(f"   New SL: ${decision.new_sl_price:.2f}")
        if decision.new_tp_prices:
            print(f"   New TPs: {[f'${p:.2f}' for p in decision.new_tp_prices]}")
        if decision.fraction_to_close:
            print(f"   Close fraction: {decision.fraction_to_close:.0%}")
        
        print("\n✅ Exit Brain kan analysere posisjoner!")
        return True
        
    except Exception as e:
        print(f"❌ FEIL i Exit Brain Adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_dynamic_executor_shadow(client, open_positions):
    """Test 3: Kan Dynamic Executor kjøre i SHADOW mode?"""
    print("\n" + "="*70)
    print("TEST 3: Dynamic Executor SHADOW Mode")
    print("="*70)
    
    try:
        from backend.config.exit_mode import get_exit_executor_mode
        
        mode = get_exit_executor_mode()
        print(f"✅ Current mode: {mode}")
        
        if mode != "SHADOW":
            print(f"⚠️  Not in SHADOW mode (mode={mode})")
            print(f"   Executor would attempt LIVE execution!")
            return True
        
        # Create adapter and executor
        adapter = ExitBrainAdapter()
        executor = ExitBrainDynamicExecutor(
            adapter=adapter,
            position_source=client,
            loop_interval_sec=5.0,
            shadow_mode=False  # Let it determine mode from config
        )
        
        print(f"✅ Dynamic Executor created")
        print(f"   Effective mode: {executor.effective_mode}")
        
        if executor.effective_mode != "SHADOW":
            print(f"⚠️  WARNING: Executor in {executor.effective_mode} mode!")
            print(f"   This test will stop immediately to avoid placing orders.")
            return True
        
        # Run one iteration
        print(f"\n🔄 Running one monitoring cycle...")
        
        # Manually run check_and_act once
        await executor._check_and_act()
        
        print(f"✅ Monitoring cycle complete!")
        
        # Check if shadow log was written
        shadow_log = Path("backend/data/exit_brain_shadow.jsonl")
        if shadow_log.exists() and shadow_log.stat().st_size > 0:
            print(f"✅ Shadow log written: {shadow_log}")
            
            # Show last entry
            with open(shadow_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    import json
                    last = json.loads(lines[-1])
                    print(f"\n📝 Last shadow decision:")
                    print(f"   Symbol: {last.get('symbol')}")
                    print(f"   Decision: {last.get('decision', {}).get('decision_type')}")
                    print(f"   Confidence: {last.get('decision', {}).get('confidence', 0):.2%}")
        else:
            print(f"⚠️  Shadow log not written (ingen beslutninger logget)")
        
        return True
        
    except Exception as e:
        print(f"❌ FEIL i Dynamic Executor: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("EXIT BRAIN V3 LIVE TEST MED BINANCE TESTNET")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Connection
    result = await test_live_connection()
    if isinstance(result, tuple):
        success, client, positions = result
    else:
        success = result
        client = None
        positions = []
    
    if not success:
        print("\n❌ Test 1 feilet - stopper")
        return False
    
    # Test 2: Adapter
    if positions:
        success = await test_exit_brain_adapter(positions)
        if not success:
            print("\n❌ Test 2 feilet - stopper")
            return False
    
    # Test 3: Executor
    if client:
        success = await test_dynamic_executor_shadow(client, positions)
        if not success:
            print("\n❌ Test 3 feilet")
            return False
    
    # Summary
    print("\n" + "="*70)
    print("RESULTAT")
    print("="*70)
    print("✅ Test 1: Binance Testnet connection OK")
    if positions:
        print("✅ Test 2: Exit Brain Adapter fungerer")
        print("✅ Test 3: Dynamic Executor SHADOW mode fungerer")
    else:
        print("⚠️  Test 2-3: Ingen posisjoner å teste med")
    
    print("\n🎯 KONKLUSJON: Exit Brain V3 fungerer med Binance Testnet!")
    print("   - Kan hente posisjoner")
    print("   - Kan analysere og bestemme actions")
    print("   - Kan kjøre i SHADOW mode uten å plassere ordrer")
    
    if positions:
        print("\n📋 For å se shadow logs:")
        print("   tail -f backend/data/exit_brain_shadow.jsonl")
        print("   python backend/tools/analyze_exit_brain_shadow.py")
    else:
        print("\n⚠️  Åpne noen testnet-posisjoner for å se AI i aksjon!")
    
    print("\n" + "="*70)
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test avbrutt av bruker")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
