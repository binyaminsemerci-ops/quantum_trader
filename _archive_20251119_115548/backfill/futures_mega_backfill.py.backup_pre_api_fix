"""
FUTURES MEGA BACKFILL - Maximum futures data extraction
Target: 100,000+ NEW futures samples
Strategy: Different time windows, higher sampling density
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
import aiohttp
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import random
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample

def calculate_indicators(df):
    """Calculate 14 technical indicators"""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['BB_upper'] = sma_20 + (std_20 * 2)
    df['BB_middle'] = sma_20
    df['BB_lower'] = sma_20 - (std_20 * 2)
    
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df['volume_sma_20'] = volume.rolling(window=20).mean()
    df['price_change_pct'] = close.pct_change() * 100
    df['high_low_range'] = high - low
    
    return df

async def fetch_futures_klines(session, symbol, interval='5m', limit=1000, start_time=None, end_time=None):
    """Fetch FUTURES klines with optional end_time"""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    try:
        async with session.get(url, params=params, timeout=30) as response:
            if response.status == 200:
                return await response.json()
    except:
        pass
    return None

def determine_action_outcome(features, future_price, current_price):
    """ULTRA aggressive futures strategy"""
    rsi = features['RSI']
    macd = features['MACD']
    ema_10 = features['EMA_10']
    ema_50 = features['EMA_50']
    close = features['Close']
    bb_upper = features['BB_upper']
    bb_lower = features['BB_lower']
    
    pnl_pct = ((future_price - current_price) / current_price) * 100
    
    # ULTRA aggressive - maximera BUY/SELL
    if rsi < 48 and ema_10 > ema_50 * 0.998:
        action = "BUY"
    elif rsi > 52 and ema_10 < ema_50 * 1.002:
        action = "SELL"
    elif macd > 0 and close < bb_upper:
        action = "BUY"
    elif macd < 0 and close > bb_lower:
        action = "SELL"
    elif rsi < 35:
        action = "BUY"
    elif rsi > 65:
        action = "SELL"
    else:
        action = "HOLD"
        return action, "NEUTRAL", 0
    
    # Futures: Lower thresholds for leverage
    if action == "BUY":
        outcome = "WIN" if pnl_pct > 0.05 else "LOSS" if pnl_pct < -0.05 else "NEUTRAL"
    elif action == "SELL":
        outcome = "WIN" if pnl_pct < -0.05 else "LOSS" if pnl_pct > 0.05 else "NEUTRAL"
    else:
        outcome = "NEUTRAL"
        pnl_pct = 0
    
    return action, outcome, pnl_pct

async def process_symbol_period(session, symbol, interval, start_time, end_time, db, source_name, max_samples=300):
    """Process one symbol with specific time period"""
    klines = await fetch_futures_klines(session, symbol, interval=interval, limit=1000, 
                                       start_time=start_time, end_time=end_time)
    
    if not klines or len(klines) < 100:
        return 0
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    
    df = calculate_indicators(df)
    df = df.dropna()
    
    if len(df) < 60:
        return 0
    
    samples_created = 0
    lookback = 8 if interval in ['1m', '3m'] else 10 if interval == '5m' else 12
    
    # AGGRESSIVE sampling - every 3-4 candles instead of 6-8
    step = 3 if interval in ['1m', '3m'] else 4 if interval in ['5m', '15m'] else 5
    
    for i in range(50, len(df) - lookback, step):
        if samples_created >= max_samples:
            break
        
        row = df.iloc[i]
        future_row = df.iloc[i + lookback]
        
        features = {
            "Close": float(row['close']),
            "Volume": float(row['volume']),
            "EMA_10": float(row['EMA_10']),
            "EMA_50": float(row['EMA_50']),
            "RSI": float(row['RSI']),
            "MACD": float(row['MACD']),
            "MACD_signal": float(row['MACD_signal']),
            "BB_upper": float(row['BB_upper']),
            "BB_middle": float(row['BB_middle']),
            "BB_lower": float(row['BB_lower']),
            "ATR": float(row['ATR']),
            "volume_sma_20": float(row['volume_sma_20']),
            "price_change_pct": float(row['price_change_pct']),
            "high_low_range": float(row['high_low_range'])
        }
        
        action, outcome, pnl_pct = determine_action_outcome(
            features, future_row['close'], row['close']
        )
        
        timestamp = datetime.fromtimestamp(row['timestamp'] / 1000, tz=timezone.utc)
        exit_time = datetime.fromtimestamp(future_row['timestamp'] / 1000, tz=timezone.utc)
        
        sample = AITrainingSample(
            symbol=symbol,
            timestamp=timestamp,
            predicted_action=action,
            prediction_score=0.75,
            prediction_confidence=0.75,
            model_version=source_name,
            features=json.dumps(features),
            feature_names=json.dumps(list(features.keys())),
            executed=True if action != "HOLD" else False,
            execution_side=action if action != "HOLD" else None,
            entry_price=row['close'],
            entry_quantity=100.0,
            entry_time=timestamp,
            outcome_known=True,
            exit_price=future_row['close'],
            exit_time=exit_time,
            realized_pnl=pnl_pct if action != "HOLD" else 0,
            hold_duration_seconds=(exit_time - timestamp).total_seconds(),
            target_label=pnl_pct,
            target_class=outcome
        )
        
        db.add(sample)
        samples_created += 1
    
    if samples_created > 0:
        db.commit()
    
    return samples_created

async def main(target_samples=100000):
    """MEGA FUTURES backfill - maximum extraction"""
    
    print("🔥🔥🔥 FUTURES MEGA BACKFILL - MAXIMUM EXTRACTION 🔥🔥🔥")
    print("=" * 80)
    print(f"[TARGET] Target: {target_samples:,} NEW futures samples")
    print("[CHART] Strategy: Aggressive sampling, diverse time windows")
    print("⚡ Higher density: 3-4 candle intervals")
    print("📅 Extended coverage: 90+ days back")
    print("=" * 80)
    
    futures_symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'SHIBUSDT',
        'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT',
        'NEARUSDT', 'ALGOUSDT', 'XLMUSDT', 'FILUSDT', 'APTUSDT',
        'ARBUSDT', 'OPUSDT', 'INJUSDT', 'SUIUSDT', 'WLDUSDT',
        'TIAUSDT', 'RENDERUSDT', 'FETUSDT', 'TRXUSDT', 'TONUSDT',
        'BCHUSDT', 'ETCUSDT', 'AAVEUSDT', 'ICPUSDT', 'IMXUSDT',
        'LDOUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT',
        'THETAUSDT', 'EGLDUSDT', 'PEPEUSDT', 'MKRUSDT', 'RUNEUSDT',
        'GMXUSDT', 'APEUSDT', 'GMTUSDT', 'BLZUSDT', 'PENDLEUSDT'
    ]
    
    now = datetime.now(timezone.utc)
    
    # MEGA periods - extensive time coverage with overlap
    mega_configs = [
        # Recent high-frequency
        ('mega_1m_ultra_recent', now - timedelta(hours=30), now - timedelta(hours=6), '1m', 300),
        ('mega_3m_recent_a', now - timedelta(days=4), now - timedelta(days=2), '3m', 280),
        ('mega_5m_recent_b', now - timedelta(days=6), now - timedelta(days=3), '5m', 260),
        
        # Weekly coverage
        ('mega_5m_week1_a', now - timedelta(days=9), now - timedelta(days=6), '5m', 250),
        ('mega_5m_week1_b', now - timedelta(days=12), now - timedelta(days=9), '5m', 240),
        ('mega_15m_week2_a', now - timedelta(days=16), now - timedelta(days=12), '15m', 220),
        ('mega_15m_week2_b', now - timedelta(days=20), now - timedelta(days=16), '15m', 210),
        
        # Monthly deep coverage
        ('mega_30m_month1_a', now - timedelta(days=28), now - timedelta(days=21), '30m', 200),
        ('mega_30m_month1_b', now - timedelta(days=35), now - timedelta(days=28), '30m', 190),
        ('mega_1h_month2_a', now - timedelta(days=45), now - timedelta(days=35), '1h', 180),
        ('mega_1h_month2_b', now - timedelta(days=55), now - timedelta(days=45), '1h', 170),
        
        # Extended historical
        ('mega_2h_month3', now - timedelta(days=70), now - timedelta(days=55), '2h', 160),
        ('mega_4h_month4', now - timedelta(days=90), now - timedelta(days=70), '4h', 150),
    ]
    
    db = SessionLocal()
    total_samples = 0
    
    async with aiohttp.ClientSession() as session:
        for source_name, start_time, end_time, interval, max_samples in mega_configs:
            if total_samples >= target_samples:
                break
            
            days_back = (now - start_time).days
            print(f"\n🔥 {source_name}")
            print(f"   Interval: {interval}, Period: {days_back}-{(now-end_time).days} days back")
            print(f"   Target: {max_samples} samples/symbol")
            
            config_samples = 0
            valid_symbols = 0
            
            # Process in smaller batches for progress tracking
            for i in range(0, len(futures_symbols), 10):
                if total_samples >= target_samples:
                    break
                
                batch = futures_symbols[i:i+10]
                tasks = [
                    process_symbol_period(session, sym, interval, start_time, end_time, 
                                        db, source_name, max_samples)
                    for sym in batch
                ]
                
                results = await asyncio.gather(*tasks)
                
                for count in results:
                    if count > 0:
                        valid_symbols += 1
                        config_samples += count
                        total_samples += count
                
                print(f"   [OK] {min(i+10, len(futures_symbols))}/{len(futures_symbols)} symbols: {config_samples:,} samples...")
                
                if total_samples >= target_samples:
                    print(f"\n[TARGET] TARGET REACHED: {total_samples:,} samples!")
                    break
            
            print(f"   [OK] Period complete: {config_samples:,} samples from {valid_symbols} pairs")
    
    db.close()
    
    print("\n" + "=" * 80)
    print(f"🔥🔥🔥 FUTURES MEGA BACKFILL COMPLETE! 🔥🔥🔥")
    print(f"   Total NEW samples: {total_samples:,}")
    print(f"   Periods covered: {len(mega_configs)} time windows")
    print(f"   Aggressive sampling: 3-4 candle intervals")
    print(f"   Historical depth: 90+ days")
    print(f"   FUTURES-OPTIMIZED for leverage trading")
    print("=" * 80)
    
    return total_samples

if __name__ == '__main__':
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    asyncio.run(main(target))
