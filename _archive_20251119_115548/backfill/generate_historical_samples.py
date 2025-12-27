"""
Generate training samples from historical liquidity snapshots
Uses 26,500 snapshots to create realistic training data
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample

def calculate_technical_indicators(prices, volumes):
    """Calculate 14 technical indicators from price series"""
    if len(prices) < 50:
        return None
    
    close = np.array(prices)
    volume = np.array(volumes)
    
    # EMA
    ema_10 = close[-10:].mean() if len(close) >= 10 else close.mean()
    ema_50 = close[-50:].mean() if len(close) >= 50 else close.mean()
    
    # RSI
    deltas = np.diff(close[-15:]) if len(close) >= 15 else np.diff(close)
    gains = np.where(deltas > 0, deltas, 0).mean()
    losses = np.where(deltas < 0, -deltas, 0).mean()
    rs = gains / losses if losses != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    # MACD (simple approximation)
    ema_12 = close[-12:].mean() if len(close) >= 12 else close.mean()
    ema_26 = close[-26:].mean() if len(close) >= 26 else close.mean()
    macd = ema_12 - ema_26
    macd_signal = macd * 0.9  # Simplified
    
    # Bollinger Bands
    sma_20 = close[-20:].mean() if len(close) >= 20 else close.mean()
    std_20 = close[-20:].std() if len(close) >= 20 else close.std()
    bb_upper = sma_20 + 2 * std_20
    bb_middle = sma_20
    bb_lower = sma_20 - 2 * std_20
    
    # ATR (simplified)
    if len(close) >= 14:
        high_low = np.abs(close[1:] - close[:-1])[-14:]
        atr = high_low.mean()
    else:
        atr = np.abs(close[1:] - close[:-1]).mean()
    
    # Volume SMA
    volume_sma_20 = volume[-20:].mean() if len(volume) >= 20 else volume.mean()
    
    # Price change
    price_change_pct = ((close[-1] - close[0]) / close[0] * 100) if close[0] != 0 else 0
    
    # High-Low range (simplified)
    high_low_range = close.max() - close.min()
    
    return {
        "Close": float(close[-1]),
        "Volume": float(volume[-1]),
        "EMA_10": float(ema_10),
        "EMA_50": float(ema_50),
        "RSI": float(rsi),
        "MACD": float(macd),
        "MACD_signal": float(macd_signal),
        "BB_upper": float(bb_upper),
        "BB_middle": float(bb_middle),
        "BB_lower": float(bb_lower),
        "ATR": float(atr),
        "volume_sma_20": float(volume_sma_20),
        "price_change_pct": float(price_change_pct),
        "high_low_range": float(high_low_range)
    }

def generate_from_liquidity_snapshots(limit=500):
    """
    Generate training samples from liquidity snapshots
    
    Strategy:
    1. Group snapshots by symbol and run_id
    2. For each symbol, get price history (50+ points)
    3. Calculate technical indicators
    4. Look ahead 1-4 runs to determine outcome
    5. Create training sample
    """
    
    print("[ROCKET] GENERERER TRAINING DATA FRA LIQUIDITY SNAPSHOTS")
    print("=" * 60)
    
    conn = sqlite3.connect('backend/data/trades.db')
    cursor = conn.cursor()
    
    # Get top traded symbols with most history
    cursor.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM liquidity_snapshots
        GROUP BY symbol
        HAVING cnt >= 50
        ORDER BY cnt DESC
        LIMIT 20
    """)
    
    symbols = cursor.fetchall()
    print(f"\n[OK] Fant {len(symbols)} symboler med 50+ datapunkter")
    for sym, cnt in symbols[:10]:
        print(f"   {sym}: {cnt} snapshots")
    
    db = SessionLocal()
    samples_created = 0
    
    for symbol, count in symbols[:10]:  # Start with top 10
        print(f"\n[CHART] Prosesserer {symbol}...")
        
        # Get all snapshots for this symbol, ordered by time
        cursor.execute("""
            SELECT ls.price, ls.quote_volume, ls.run_id, lr.fetched_at
            FROM liquidity_snapshots ls
            JOIN liquidity_runs lr ON ls.run_id = lr.id
            WHERE ls.symbol = ?
            ORDER BY lr.fetched_at
        """, (symbol,))
        
        data = cursor.fetchall()
        
        # Create samples every 10 snapshots (to avoid too much correlation)
        for i in range(50, len(data), 10):
            if samples_created >= limit:
                break
            
            # Get historical window
            window = data[i-50:i]
            prices = [row[0] for row in window]
            volumes = [row[1] for row in window]
            
            # Calculate features
            features = calculate_technical_indicators(prices, volumes)
            if features is None:
                continue
            
            # Look ahead to determine outcome (4 snapshots = ~20 minutes)
            current_price = prices[-1]
            if i + 4 < len(data):
                future_price = data[i + 4][0]
                pnl_pct = ((future_price - current_price) / current_price) * 100
                
                # Determine action and outcome - AGGRESSIVE (more BUY/SELL)
                if features["RSI"] < 50 and features["MACD"] > 0:
                    action = "BUY"
                    outcome = "WIN" if pnl_pct > 0.2 else "LOSS" if pnl_pct < -0.2 else "NEUTRAL"
                elif features["RSI"] > 50 and features["MACD"] < 0:
                    action = "SELL"
                    outcome = "WIN" if pnl_pct < -0.2 else "LOSS" if pnl_pct > 0.2 else "NEUTRAL"
                elif features["RSI"] < 40:  # Very oversold = BUY
                    action = "BUY"
                    outcome = "WIN" if pnl_pct > 0.2 else "LOSS" if pnl_pct < -0.2 else "NEUTRAL"
                elif features["RSI"] > 60:  # Very overbought = SELL
                    action = "SELL"
                    outcome = "WIN" if pnl_pct < -0.2 else "LOSS" if pnl_pct > 0.2 else "NEUTRAL"
                else:
                    action = "HOLD"
                    outcome = "NEUTRAL"
                    pnl_pct = 0
                
                # Create training sample
                timestamp = datetime.fromisoformat(window[-1][3].replace('Z', '+00:00'))
                
                sample = AITrainingSample(
                    symbol=symbol,
                    timestamp=timestamp,
                    predicted_action=action,
                    prediction_score=0.7,
                    prediction_confidence=0.7,
                    model_version="backfill_v1",
                    features=json.dumps(features),
                    feature_names=json.dumps(list(features.keys())),
                    executed=True,
                    execution_side=action if action != "HOLD" else None,
                    entry_price=current_price,
                    entry_quantity=100.0,
                    entry_time=timestamp,
                    outcome_known=True,
                    exit_price=future_price,
                    exit_time=timestamp + timedelta(minutes=20),
                    realized_pnl=pnl_pct if action != "HOLD" else 0,
                    hold_duration_seconds=1200,
                    target_label=pnl_pct,
                    target_class=outcome
                )
                
                db.add(sample)
                samples_created += 1
                
                if samples_created % 50 == 0:
                    print(f"   [OK] {samples_created} samples generert...")
        
        if samples_created >= limit:
            break
    
    db.commit()
    db.close()
    conn.close()
    
    print(f"\n🎉 FERDIG!")
    print(f"   Total samples generert: {samples_created}")
    print(f"   Fra: {len(symbols)} symboler")
    print(f"   Basert på: 26,500 historical snapshots")
    
    return samples_created

if __name__ == '__main__':
    count = generate_from_liquidity_snapshots(limit=500)
    print(f"\n[OK] Suksess! Genererte {count} training samples")
    print("[ROCKET] Start re-training for å bruke ny data!")
