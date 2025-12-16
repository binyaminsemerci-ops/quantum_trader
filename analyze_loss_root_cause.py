"""Deep analysis: Why did we lose money? What went wrong?"""
import os
from binance.client import Client
from datetime import datetime, timedelta

def analyze_losses():
    """Analyze what went wrong with SOLUSDT and APTUSDT positions"""
    
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
    
    print("\n" + "=" * 100)
    print("[SEARCH] ROOT CAUSE ANALYSIS - WHY DID WE LOSE MONEY?")
    print("=" * 100)
    
    symbols = ["SOLUSDT", "APTUSDT", "DOGEUSDT", "DYMUSDT"]
    
    print("\n[CHART] 1. ENTRY SIGNALS - What did AI say when we entered?")
    print("=" * 100)
    
    # Get recent trades to see when positions were opened
    for symbol in symbols:
        try:
            trades = client.futures_account_trades(symbol=symbol, limit=50)
            
            if not trades:
                continue
            
            # Find position opening trade
            opening_trades = []
            for trade in trades:
                if trade.get('realizedPnl') == '0':  # Opening trade
                    opening_trades.append(trade)
            
            if opening_trades:
                first_trade = opening_trades[0]
                trade_time = datetime.fromtimestamp(first_trade['time'] / 1000)
                side = first_trade['side']
                price = float(first_trade['price'])
                qty = float(first_trade['qty'])
                
                print(f"\n{symbol}:")
                print(f"  Opened: {trade_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Side: {side}")
                print(f"  Entry Price: ${price:.6f}")
                print(f"  Quantity: {qty:.4f}")
                
                # Calculate how old the position is
                age = datetime.now() - trade_time
                hours = age.total_seconds() / 3600
                print(f"  Age: {hours:.1f} hours")
                
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
    
    print("\n" + "=" * 100)
    print("[CHART] 2. FUNDING FEES - Hidden cost that drains positions")
    print("=" * 100)
    
    # Get funding rate history
    for symbol in ["SOLUSDT", "APTUSDT"]:
        try:
            # Get recent funding rates
            funding = client.futures_funding_rate(symbol=symbol, limit=10)
            
            if funding:
                print(f"\n{symbol} Funding Rates (last 10):")
                total_funding = 0
                for f in funding[:5]:
                    rate = float(f['fundingRate'])
                    total_funding += rate
                    time = datetime.fromtimestamp(f['fundingTime'] / 1000)
                    print(f"  {time.strftime('%Y-%m-%d %H:%M')}: {rate*100:.4f}%")
                
                avg_funding = total_funding / len(funding[:5])
                print(f"\n  Average funding rate: {avg_funding*100:.4f}% per 8h")
                print(f"  Daily impact (3x): {avg_funding*3*100:.4f}%")
                
                # With 20x leverage, funding is magnified
                leveraged_impact = avg_funding * 3 * 20
                print(f"  With 20x leverage: {leveraged_impact*100:.2f}% daily drain!")
                
        except Exception as e:
            print(f"  Error getting funding for {symbol}: {e}")
    
    print("\n" + "=" * 100)
    print("[CHART] 3. AI CONFIDENCE - Was the signal strong enough?")
    print("=" * 100)
    
    print("\nFrom earlier AI check:")
    print("  SOLUSDT: HOLD with 36.88% confidence (LOW!)")
    print("  APTUSDT: HOLD with 41.17% confidence (LOW!)")
    print("  Model: 'rule_fallback_rsi' (NOT the ML models!)")
    print("\n  ❌ PROBLEM: Positions opened with LOW confidence signals")
    print("  ❌ PROBLEM: Using fallback rules, not trained ML models")
    
    print("\n" + "=" * 100)
    print("[CHART] 4. STRATEGY LOGIC - What's the trading threshold?")
    print("=" * 100)
    
    # Check event-driven executor config
    confidence_threshold = os.getenv("QT_CONFIDENCE_THRESHOLD", "0.65")
    
    print(f"\n  System threshold: {float(confidence_threshold)*100}% confidence required")
    print(f"  SOLUSDT confidence: 36.88% ❌ (BELOW THRESHOLD!)")
    print(f"  APTUSDT confidence: 41.17% ❌ (BELOW THRESHOLD!)")
    print(f"\n  [ALERT] CRITICAL: Positions should NOT have been opened!")
    print(f"  [ALERT] These were below the 65% confidence threshold")
    
    print("\n" + "=" * 100)
    print("[CHART] 5. MARKET CONDITIONS - Price movement analysis")
    print("=" * 100)
    
    for symbol in ["SOLUSDT", "APTUSDT"]:
        try:
            # Get recent klines to see market movement
            klines = client.futures_klines(symbol=symbol, interval="1h", limit=24)
            
            prices = [float(k[4]) for k in klines]  # Close prices
            
            first_price = prices[0]
            last_price = prices[-1]
            highest = max(prices)
            lowest = min(prices)
            
            change_24h = ((last_price - first_price) / first_price) * 100
            volatility = ((highest - lowest) / lowest) * 100
            
            print(f"\n{symbol} (24h movement):")
            print(f"  Start: ${first_price:.6f}")
            print(f"  End: ${last_price:.6f}")
            print(f"  Change: {change_24h:+.2f}%")
            print(f"  High: ${highest:.6f}")
            print(f"  Low: ${lowest:.6f}")
            print(f"  Volatility: {volatility:.2f}%")
            
            if change_24h < 0:
                print(f"  [WARNING] DOWNTREND - Why did we LONG in a falling market?")
            
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
    
    print("\n" + "=" * 100)
    print("[CHART] 6. ROOT CAUSES SUMMARY")
    print("=" * 100)
    
    print("""
[ALERT] PROBLEM #1: LOW CONFIDENCE ENTRIES
   • SOLUSDT: 36.88% confidence (BELOW 65% threshold)
   • APTUSDT: 41.17% confidence (BELOW 65% threshold)
   • Should NOT have traded these signals!

[ALERT] PROBLEM #2: FALLBACK RULES USED
   • Model: 'rule_fallback_rsi' 
   • This is NOT the trained ML ensemble
   • Fallback rules are weak - only RSI-based
   • Real ML models not being used!

[ALERT] PROBLEM #3: FUNDING FEE DRAIN
   • With 20x leverage, funding fees are magnified 20x
   • Even 0.01% funding = 0.2% loss per 8h with leverage
   • 3x per day = 0.6% daily drain
   • Over days, this accumulates significantly

[ALERT] PROBLEM #4: NO AI AGREEMENT
   • When we checked AI sentiment, it said HOLD (neutral)
   • AI was NOT confident about these positions
   • Position Monitor didn't detect this conflict

[ALERT] PROBLEM #5: POSSIBLE MANUAL ENTRIES?
   • System threshold is 65%, but these had 36-41%
   • Either:
     a) Bug in event-driven executor (ignoring threshold)
     b) Manual entries bypassing AI checks
     c) Threshold was lower when positions opened

[ALERT] PROBLEM #6: NO EXIT STRATEGY FOR WEAK SIGNALS
   • Position Monitor only checks TP/SL orders exist
   • Doesn't re-evaluate AI confidence over time
   • Doesn't close positions when AI sentiment changes to HOLD
    """)
    
    print("\n" + "=" * 100)
    print("💡 RECOMMENDATIONS")
    print("=" * 100)
    
    print("""
[OK] FIX #1: Enforce confidence threshold strictly
   • Add validation in position opening logic
   • Log warning if threshold bypassed
   • Require manual override approval for low confidence

[OK] FIX #2: Enable ML models properly
   • Check why ensemble models not loading
   • Fix catboost dependency: pip install catboost
   • Verify model files exist and are trained

[OK] FIX #3: Add dynamic position monitoring
   • Re-check AI sentiment every hour
   • Close positions if AI changes to HOLD or opposite signal
   • Add funding fee accumulation tracking

[OK] FIX #4: Add pre-trade validation
   • Before opening position:
     * Check AI confidence >= threshold
     * Verify ML models (not fallback) being used
     * Check market trend aligns with trade direction
     * Estimate funding fee impact

[OK] FIX #5: Improve Position Monitor
   • Monitor actual P&L vs expected
   • Track funding fees separately
   • Alert on P&L anomalies
   • Re-evaluate AI sentiment periodically
    """)
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    analyze_losses()
