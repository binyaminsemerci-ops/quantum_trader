#!/usr/bin/env python3
"""
📊 POSITION CLOSE MONITOR
Checks which positions are close to TP/SL and will trigger learning events
"""
import os
from binance.client import Client
from datetime import datetime

print("=" * 80)
print("📊 POSITION CLOSE MONITOR - LEARNING EVENT TRIGGERS")
print("=" * 80)
print(f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()

# Get API credentials
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

if not api_key or not api_secret:
    print("❌ Binance API credentials not found!")
    print("   Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
    exit(1)

try:
    # Create client
    if testnet:
        client = Client(api_key, api_secret, testnet=True)
        print("✅ Connected to Binance Futures TESTNET")
    else:
        client = Client(api_key, api_secret)
        print("✅ Connected to Binance Futures MAINNET")
    
    print()
    
    # Get account info
    account = client.futures_account()
    
    # Filter open positions
    positions = [p for p in account['positions'] if float(p['positionAmt']) != 0]
    
    print(f"📊 FOUND {len(positions)} OPEN POSITIONS")
    print("=" * 80)
    print()
    
    if not positions:
        print("No open positions currently")
        print()
        print("💡 When positions are opened, they will be monitored here")
        print("   for proximity to TP/SL levels that trigger learning events")
        exit(0)
    
    close_to_exit = []
    
    for i, pos in enumerate(positions, 1):
        symbol = pos['symbol']
        qty = float(pos['positionAmt'])
        entry_price = float(pos['entryPrice'])
        mark_price = float(pos['markPrice'])
        unrealized_pnl = float(pos['unRealizedProfit'])
        leverage = int(pos['leverage'])
        
        # Determine side
        side = "LONG" if qty > 0 else "SHORT"
        
        # Calculate PNL %
        if entry_price > 0:
            if qty > 0:  # Long
                pnl_pct = ((mark_price - entry_price) / entry_price) * 100
            else:  # Short
                pnl_pct = ((entry_price - mark_price) / entry_price) * 100
        else:
            pnl_pct = 0
        
        # Check if close to TP/SL (typically ±3%)
        tp_target = 3.0  # Standard TP %
        sl_target = -2.0  # Standard SL %
        
        distance_to_tp = abs(pnl_pct - tp_target)
        distance_to_sl = abs(pnl_pct - sl_target)
        
        is_close = distance_to_tp < 0.5 or distance_to_sl < 0.5
        
        print(f"[{i}] {symbol}")
        print("-" * 80)
        print(f"    Side: {side} ({leverage}x)")
        print(f"    Size: {abs(qty)}")
        print(f"    Entry: ${entry_price:.4f}")
        print(f"    Mark:  ${mark_price:.4f}")
        print(f"    PNL:   ${unrealized_pnl:.2f} ({pnl_pct:+.2f}%)")
        print()
        
        # Distance analysis
        print(f"    Distance to TP (+3%): {distance_to_tp:.2f}%")
        print(f"    Distance to SL (-2%): {distance_to_sl:.2f}%")
        print()
        
        if is_close:
            close_to_exit.append({
                'symbol': symbol,
                'side': side,
                'pnl_pct': pnl_pct,
                'trigger': 'TP' if distance_to_tp < distance_to_sl else 'SL'
            })
            print(f"    🔔 CLOSE TO EXIT!")
            print(f"    🎓 WILL TRIGGER LEARNING EVENTS:")
            print(f"       1. PIL → Classify as WINNER/LOSER")
            print(f"       2. PAL → Analyze amplification opportunity")
            print(f"       3. Meta-Strategy → Update reward")
            print(f"       4. Continuous Learning → Check retraining")
            print(f"       5. TP Tracker → Record hit rate/slippage")
        else:
            print(f"    ✅ Position stable")
            print(f"       {abs(pnl_pct - tp_target):.2f}% from TP")
            print(f"       {abs(pnl_pct - sl_target):.2f}% from SL")
        
        print()
    
    # Summary
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print()
    print(f"Total positions: {len(positions)}")
    print(f"Close to exit: {len(close_to_exit)}")
    print()
    
    if close_to_exit:
        print("🔔 POSITIONS ABOUT TO TRIGGER LEARNING:")
        for pos in close_to_exit:
            print(f"   • {pos['symbol']} {pos['side']}: {pos['pnl_pct']:+.2f}% → {pos['trigger']}")
        print()
        print("✅ When these close, all learning systems will activate!")
    else:
        print("✅ No positions close to exit currently")
        print("   Learning events will trigger when positions reach TP/SL")
    
    print()
    
    # Trade logging info
    print("=" * 80)
    print("📝 TRADE LOGGING STATUS")
    print("=" * 80)
    print()
    
    # Check if TradeStore is working
    try:
        from backend.services.monitoring.position_monitor import TRADESTORE_AVAILABLE
        if TRADESTORE_AVAILABLE:
            print("✅ TradeStore active - All trades are logged to database")
            print("   Every position close is recorded with:")
            print("   • Entry/Exit prices")
            print("   • PNL")
            print("   • Hold time")
            print("   • Strategy used")
            print("   • Learning labels (PIL classification)")
        else:
            print("⚠️  TradeStore not available")
    except:
        print("⚠️  Could not check TradeStore status")
    
    print()
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
