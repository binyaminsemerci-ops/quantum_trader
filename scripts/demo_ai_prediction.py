"""
Demonstrasjon: Live AI Trading Prediction

Dette scriptet viser steg-for-steg hvordan systemet genererer trading signals
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ai_engine.agents.xgb_agent import make_default_agent
from backend.routes.external_data import binance_ohlcv, twitter_sentiment
import pandas as pd


async def demo_live_prediction():
    """
    Live demonstrasjon av AI trading decision making
    """
    print("=" * 70)
    print("🤖 QUANTUM TRADER - AI PREDICTION DEMO")
    print("=" * 70)
    
    # STEG 1: Initialisér AI Agent
    print("\n[CHART] STEG 1: Loading AI Model...")
    agent = make_default_agent()
    
    if agent.model is None:
        print("[WARNING]  No trained model found. Using fallback heuristic.")
    else:
        print("[OK] XGBoost model loaded successfully")
        
        # Show model metadata if available
        metadata = agent.get_metadata()
        if metadata:
            print(f"   Model Version: {metadata.get('version', 'N/A')}")
            print(f"   Trained: {metadata.get('trained_at', 'N/A')}")
            print(f"   Samples: {metadata.get('samples', 'N/A')}")
    
    # STEG 2: Hent Market Data
    print("\n[CHART_UP] STEG 2: Fetching Market Data...")
    symbol = "BTCUSDT"
    
    try:
        response = await binance_ohlcv(symbol=symbol, limit=120)
        candles = response.get("candles", [])
        
        if candles:
            latest = candles[-1]
            print(f"[OK] Fetched {len(candles)} candles for {symbol}")
            print(f"   Latest Price: ${latest['close']:,.2f}")
            print(f"   Volume: {latest['volume']:,.2f}")
            print(f"   Time: {latest['timestamp']}")
    except Exception as e:
        print(f"[WARNING]  Market data fetch failed: {e}")
        return
    
    # STEG 3: Hent Sentiment Data
    print("\n💬 STEG 3: Fetching Sentiment Data...")
    try:
        sentiment = await twitter_sentiment(symbol="BTC")
        sent_score = sentiment.get("score", 0.0)
        
        print(f"[OK] Sentiment Score: {sent_score:.2f}")
        if sent_score > 0.6:
            print("   Mood: 😊 Bullish (Positive)")
        elif sent_score < 0.4:
            print("   Mood: 😟 Bearish (Negative)")
        else:
            print("   Mood: 😐 Neutral")
        
        # Enrich candles with sentiment
        for candle in candles:
            candle['sentiment'] = sent_score
            
    except Exception as e:
        print(f"[WARNING]  Sentiment fetch failed: {e}")
        sent_score = 0.5
    
    # STEG 4: Feature Engineering
    print("\n🔧 STEG 4: Computing Technical Indicators...")
    try:
        df = pd.DataFrame(candles)
        
        # Beregn tekniske indikatorer (subset for demo)
        df['MA_10'] = df['close'].rolling(10).mean()
        df['MA_50'] = df['close'].rolling(50).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Display latest indicators
        latest_row = df.iloc[-1]
        print(f"[OK] Technical Indicators Computed:")
        print(f"   RSI (14): {latest_row['RSI_14']:.2f}")
        if latest_row['RSI_14'] > 70:
            print("      → Overbought (potential SELL)")
        elif latest_row['RSI_14'] < 30:
            print("      → Oversold (potential BUY)")
        else:
            print("      → Neutral range")
        
        print(f"   MA (10): ${latest_row['MA_10']:,.2f}")
        print(f"   MA (50): ${latest_row['MA_50']:,.2f}")
        
        if latest_row['close'] > latest_row['MA_10'] > latest_row['MA_50']:
            print("      → Strong uptrend (MA alignment)")
        elif latest_row['close'] < latest_row['MA_10'] < latest_row['MA_50']:
            print("      → Strong downtrend (MA alignment)")
        
    except Exception as e:
        print(f"[WARNING]  Feature computation failed: {e}")
    
    # STEG 5: AI Prediction
    print("\n🤖 STEG 5: AI Model Prediction...")
    try:
        prediction = agent.predict_for_symbol(candles)
        
        action = prediction['action']
        score = prediction['score']
        
        # Display prediction
        print(f"[OK] AI Decision: {action}")
        print(f"   Confidence: {score:.2%}")
        
        # Action indicators
        if action == "BUY":
            print("   [GREEN_CIRCLE] Recommendation: LONG (Buy) Position")
            print(f"      Model expects price to rise")
        elif action == "SELL":
            print("   [RED_CIRCLE] Recommendation: SHORT (Sell) Position")
            print(f"      Model expects price to fall")
        else:
            print("   🟡 Recommendation: HOLD (No Position)")
            print(f"      No clear signal - wait for better opportunity")
        
        # Confidence interpretation
        if score > 0.75:
            print("   💪 High confidence - Strong signal")
        elif score > 0.55:
            print("   👍 Medium confidence - Moderate signal")
        else:
            print("   🤷 Low confidence - Weak signal")
            
    except Exception as e:
        print(f"[WARNING]  Prediction failed: {e}")
        return
    
    # STEG 6: Risk Management
    print("\n⚖️  STEG 6: Risk Management Parameters...")
    
    current_price = latest['close']
    
    if action == "BUY":
        # Calculate targets based on ATR (Average True Range)
        atr = df['high'] - df['low']
        avg_atr = atr.rolling(14).mean().iloc[-1]
        
        price_target = current_price * (1 + score * 0.03)  # Max 3% target
        stop_loss = current_price * (1 - avg_atr / current_price * 1.5)
        
        print(f"   Entry: ${current_price:,.2f}")
        print(f"   Target: ${price_target:,.2f} (+{(price_target/current_price-1)*100:.2f}%)")
        print(f"   Stop Loss: ${stop_loss:,.2f} (-{(1-stop_loss/current_price)*100:.2f}%)")
        
        # Position sizing (2% risk rule)
        risk_amount = 10000 * 0.02  # $200 risk on $10k account
        price_risk = current_price - stop_loss
        position_size = risk_amount / price_risk
        
        print(f"   Position Size: {position_size:.4f} BTC (${position_size * current_price:,.2f})")
        print(f"   Risk Amount: ${risk_amount:.2f} (2% of account)")
        
        # Risk/Reward ratio
        potential_gain = price_target - current_price
        potential_loss = current_price - stop_loss
        risk_reward = potential_gain / potential_loss
        
        print(f"   Risk/Reward: 1:{risk_reward:.2f}")
        if risk_reward > 2:
            print("      [OK] Good risk/reward ratio (>2:1)")
        else:
            print("      [WARNING]  Risk/reward ratio below optimal (<2:1)")
    
    elif action == "SELL":
        print(f"   Entry: ${current_price:,.2f}")
        print(f"   Strategy: Short position or exit existing longs")
    else:
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   Action: Wait for clearer signal")
    
    # STEG 7: Final Decision
    print("\n" + "=" * 70)
    print("[CLIPBOARD] TRADING DECISION SUMMARY")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Price: ${current_price:,.2f}")
    print(f"AI Action: {action}")
    print(f"Confidence: {score:.2%}")
    print(f"Sentiment: {sent_score:.2f} ({'Bullish' if sent_score > 0.6 else 'Bearish' if sent_score < 0.4 else 'Neutral'})")
    print(f"RSI: {latest_row['RSI_14']:.2f}")
    print(f"Trend: {'Uptrend' if latest_row['MA_10'] > latest_row['MA_50'] else 'Downtrend'}")
    
    # Execute recommendation
    if action != "HOLD" and score > 0.6:
        print(f"\n[OK] EXECUTE: {action} order recommended")
        print(f"   This signal meets minimum confidence threshold (>60%)")
    else:
        print(f"\n⏸️  SKIP: Signal too weak or HOLD recommendation")
        print(f"   Wait for higher confidence signal (current: {score:.1%})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n[ROCKET] Starting AI Trading Demo...\n")
    asyncio.run(demo_live_prediction())
    print("\n✨ Demo completed!\n")
