"""Check AI signals for current positions - use same method as event executor"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def main():
    print("\n" + "=" * 80)
    print("🤖 AI SENTIMENT FOR CURRENT POSITIONS")
    print("=" * 80)
    
    try:
        from backend.services.ai_trading_engine import AITradingEngine
        from ai_engine.agents.xgb_agent import XGBAgent
        from binance.client import Client
        
        # Get API keys
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        
        if not api_key:
            with open(".env") as f:
                for line in f:
                    if line.startswith("BINANCE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                    elif line.startswith("BINANCE_API_SECRET="):
                        api_secret = line.split("=", 1)[1].strip()
        
        client = Client(api_key, api_secret)
        
        # Get current positions
        print("\n[CHART] Current Positions:")
        positions = client.futures_position_information()
        active_positions = {}
        symbols_to_check = []
        
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt != 0:
                symbol = pos['symbol']
                entry = float(pos['entryPrice'])
                current = float(pos['markPrice'])
                pnl = float(pos['unRealizedProfit'])
                pnl_pct = (pnl / abs(amt * entry)) * 100 if entry else 0
                
                active_positions[symbol] = {
                    'direction': 'LONG' if amt > 0 else 'SHORT',
                    'quantity': abs(amt),
                    'entry': entry,
                    'current': current,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                symbols_to_check.append(symbol)
                
                print(f"   {symbol}: {active_positions[symbol]['direction']} "
                      f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        if not symbols_to_check:
            print("   No active positions")
            return
        
        print(f"\n⏳ Getting AI signals for {len(symbols_to_check)} positions...")
        
        # Initialize AI engine
        agent = XGBAgent()
        ai_engine = AITradingEngine(agent=agent)
        
        # Get signals using same method as event executor
        current_positions_qty = {
            sym: active_positions[sym]['quantity'] if active_positions[sym]['direction'] == 'LONG'
            else -active_positions[sym]['quantity']
            for sym in symbols_to_check
        }
        
        signals = await ai_engine.get_trading_signals(symbols_to_check, current_positions_qty)
        
        print("\n" + "=" * 80)
        print("[TARGET] AI SENTIMENT ANALYSIS")
        print("=" * 80)
        
        for signal in signals:
            symbol = signal['symbol']
            pos = active_positions[symbol]
            
            print(f"\n[CHART] {symbol}")
            print(f"   Current Position: {pos['direction']} ${pos['pnl']:.2f} ({pos['pnl_pct']:+.2f}%)")
            print(f"   Entry: ${pos['entry']:.6f} → Current: ${pos['current']:.6f}")
            print(f"\n   🤖 AI Signal:")
            print(f"      Action:      {signal['action']}")
            print(f"      Confidence:  {signal['confidence']:.2%}")
            print(f"      Score:       {signal['score']:+.3f}")
            print(f"      Model:       {signal.get('model', 'unknown')}")
            print(f"      Reason:      {signal.get('reason', 'N/A')}")
            
            # Compare with current position
            current_dir = pos['direction']
            ai_action = signal['action']
            
            print(f"\n   💡 Analysis:")
            if ai_action == 'HOLD':
                print(f"      [WARNING] AI says HOLD (neutral)")
                if pos['pnl'] < 0:
                    print(f"      [WARNING] Position losing but AI not confident to exit")
            elif (ai_action == 'BUY' and current_dir == 'LONG') or (ai_action == 'SELL' and current_dir == 'SHORT'):
                print(f"      [OK] AI AGREES with your {current_dir} position")
                print(f"      [OK] Confidence: {signal['confidence']:.0%}")
            elif (ai_action == 'SELL' and current_dir == 'LONG'):
                print(f"      [ALERT] AI DISAGREES: Says SELL but you're LONG!")
                print(f"      [ALERT] Confidence: {signal['confidence']:.0%}")
                if pos['pnl'] < 0:
                    print(f"      [ALERT][ALERT] DOUBLE WARNING: Losing + AI disagrees!")
                    print(f"      [ALERT][ALERT] STRONG RECOMMENDATION: Close position!")
            elif (ai_action == 'BUY' and current_dir == 'SHORT'):
                print(f"      [ALERT] AI DISAGREES: Says BUY but you're SHORT!")
                print(f"      [ALERT] Confidence: {signal['confidence']:.0%}")
                if pos['pnl'] < 0:
                    print(f"      [ALERT][ALERT] DOUBLE WARNING: Losing + AI disagrees!")
                    print(f"      [ALERT][ALERT] STRONG RECOMMENDATION: Close position!")
            
            # Dynamic TP/SL recommendation
            if 'tp_percent' in signal:
                print(f"\n   [TARGET] AI's Dynamic TP/SL Recommendation:")
                print(f"      TP:      {signal['tp_percent']*100:.2f}%")
                print(f"      SL:      {signal['sl_percent']*100:.2f}%")
                print(f"      Trail:   {signal['trail_percent']*100:.2f}%")
                print(f"      Partial: {signal['partial_tp']*100:.0f}%")
        
        print("\n" + "=" * 80)
        print("[CHART] SUMMARY")
        print("=" * 80)
        
        agree_count = sum(1 for s in signals 
                         if (s['action'] == 'BUY' and active_positions[s['symbol']]['direction'] == 'LONG')
                         or (s['action'] == 'SELL' and active_positions[s['symbol']]['direction'] == 'SHORT'))
        
        disagree_count = sum(1 for s in signals 
                            if (s['action'] == 'SELL' and active_positions[s['symbol']]['direction'] == 'LONG')
                            or (s['action'] == 'BUY' and active_positions[s['symbol']]['direction'] == 'SHORT'))
        
        neutral_count = sum(1 for s in signals if s['action'] == 'HOLD')
        
        print(f"   [OK] AI Agrees:    {agree_count}")
        print(f"   [ALERT] AI Disagrees: {disagree_count}")
        print(f"   [WARNING] AI Neutral:   {neutral_count}")
        
        if disagree_count > 0:
            print(f"\n   [WARNING] WARNING: {disagree_count} position(s) where AI disagrees!")
            print(f"   Consider closing or reducing these positions.")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
