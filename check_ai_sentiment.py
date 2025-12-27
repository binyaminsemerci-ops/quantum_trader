"""Check current AI sentiment/signals for APTUSDT and SOLUSDT"""
import asyncio
import sys
import os

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai_engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def check_signals():
    print("\n" + "=" * 80)
    print("🤖 AI SENTIMENT CHECK - APTUSDT & SOLUSDT")
    print("=" * 80)
    
    try:
        from ai_engine.agents.xgb_agent import XGBAgent
        
        print("\n⏳ Initializing AI agent...")
        agent = XGBAgent()
        
        symbols = ["APTUSDT", "SOLUSDT"]
        
        for symbol in symbols:
            print(f"\n{'='*80}")
            print(f"[CHART] {symbol}")
            print(f"{'='*80}")
            
            try:
                # Get signal
                result = await agent.get_signal(symbol, timeframe="15m")
                
                if result and isinstance(result, dict):
                    signal = result.get('signal', 'NEUTRAL')
                    confidence = result.get('confidence', 0.0)
                    predicted_return = result.get('predicted_return', 0.0)
                    
                    print(f"\n   Signal:      {signal}")
                    print(f"   Confidence:  {confidence:.2%}")
                    print(f"   Predicted:   {predicted_return:+.2%}")
                    
                    # Get current position status
                    from binance.client import Client
                    
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
                    positions = client.futures_position_information(symbol=symbol)
                    
                    for pos in positions:
                        amt = float(pos['positionAmt'])
                        if amt != 0:
                            entry = float(pos['entryPrice'])
                            current = float(pos['markPrice'])
                            pnl = float(pos['unRealizedProfit'])
                            
                            print(f"\n   📍 Current Position:")
                            print(f"      Direction:   {'LONG' if amt > 0 else 'SHORT'} ({abs(amt):.1f})")
                            print(f"      Entry:       ${entry:.6f}")
                            print(f"      Current:     ${current:.6f}")
                            print(f"      P&L:         ${pnl:.2f}")
                            
                            # Compare with AI signal
                            current_direction = "BUY" if amt > 0 else "SELL"
                            
                            print(f"\n   [TARGET] AI Recommendation vs Current:")
                            if signal == current_direction:
                                print(f"      [OK] AI agrees: {signal} (confidence {confidence:.0%})")
                            elif signal == "NEUTRAL":
                                print(f"      [WARNING] AI neutral: Maybe exit or reduce")
                            else:
                                print(f"      [ALERT] AI DISAGREES: Says {signal} but you're {current_direction}!")
                                print(f"      [ALERT] Consider closing this position!")
                            
                            # Check if losing
                            if pnl < 0:
                                pnl_pct = (pnl / abs(amt * entry)) * 100
                                print(f"\n   [WARNING] Position Status:")
                                print(f"      Losing: ${pnl:.2f} ({pnl_pct:.2f}%)")
                                if signal != current_direction:
                                    print(f"      [ALERT] DOUBLE RED FLAG: Losing + AI disagrees!")
                                    print(f"      [ALERT] STRONG RECOMMENDATION: Close position NOW!")
                    
                else:
                    print(f"   ❌ No signal data available")
                    
            except Exception as e:
                print(f"   ❌ Error getting signal: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_signals())
