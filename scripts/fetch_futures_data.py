"""
Fetch Binance Futures (USDT-M Perpetual) Training Data
Includes: OHLCV, funding rates, open interest, long/short ratios, liquidation data

This is CRITICAL for leverage trading - we need futures-specific metrics.
"""
import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from binance.client import Client as BinanceClient
from binance.enums import *
import time
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_top_futures_symbols(client: BinanceClient, min_volume_usdt: float = 10_000_000) -> List[str]:
    """Get top liquid USDT-M perpetual futures by 24h volume."""
    try:
        # Get all futures tickers
        tickers = client.futures_ticker()
        
        # Filter USDT perpetual futures with high volume
        usdt_futures = []
        for ticker in tickers:
            symbol = ticker['symbol']
            if symbol.endswith('USDT') and not symbol.endswith('_'):  # Perpetual only
                volume_usdt = float(ticker.get('quoteVolume', 0))
                if volume_usdt >= min_volume_usdt:
                    usdt_futures.append({
                        'symbol': symbol,
                        'volume': volume_usdt,
                        'price': float(ticker['lastPrice'])
                    })
        
        # Sort by volume
        usdt_futures.sort(key=lambda x: x['volume'], reverse=True)
        
        symbols = [f['symbol'] for f in usdt_futures[:100]]  # Top 100 for faster training
        logger.info(f"Found {len(symbols)} liquid USDT-M perpetual futures (top 100)")
        
        return symbols
    
    except Exception as e:
        logger.error(f"Failed to get futures symbols: {e}")
        # Fallback to known liquid pairs
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
            'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'NEARUSDT', 'ATOMUSDT',
            'ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT', 'INJUSDT',
            'WLDUSDT', 'TIAUSDT', 'SEIUSDT', 'FETUSDT', 'RENDERUSDT'
        ]


def fetch_futures_klines(client: BinanceClient, symbol: str, days: int = 180) -> pd.DataFrame:
    """Fetch futures OHLCV data."""
    try:
        # Calculate start time (180 days ago)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Fetch 1h klines - use futures_klines NOT futures_historical_klines
        klines = client.futures_klines(
            symbol=symbol,
            interval='1h',  # Use string, not constant
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1500
        )
        
        if not klines:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['symbol'] = symbol
        
        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
        
        logger.info(f"  {symbol}: {len(df)} candles")
        return df
    
    except Exception as e:
        logger.error(f"  {symbol}: FAILED - {e}")
        return None


def fetch_funding_rates(client: BinanceClient, symbol: str, days: int = 180) -> pd.DataFrame:
    """Fetch historical funding rates (critical for futures trading)."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Funding rate history (every 8 hours)
        funding_rates = client.futures_funding_rate(
            symbol=symbol,
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=1000
        )
        
        if not funding_rates:
            return None
        
        df = pd.DataFrame(funding_rates)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        
        # Rename for merge
        df = df.rename(columns={'fundingTime': 'timestamp', 'fundingRate': 'funding_rate'})
        df = df[['timestamp', 'funding_rate']]
        
        return df
    
    except Exception as e:
        logger.warning(f"  {symbol}: No funding rate data - {e}")
        return None


def fetch_open_interest(client: BinanceClient, symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch open interest history (last 30 days only - API limitation)."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Open interest history
        oi_history = client.futures_open_interest_hist(
            symbol=symbol,
            period='1h',
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=500
        )
        
        if not oi_history:
            return None
        
        df = pd.DataFrame(oi_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
        df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')
        
        df = df.rename(columns={
            'sumOpenInterest': 'open_interest',
            'sumOpenInterestValue': 'open_interest_usd'
        })
        df = df[['timestamp', 'open_interest', 'open_interest_usd']]
        
        return df
    
    except Exception as e:
        logger.warning(f"  {symbol}: No open interest data - {e}")
        return None


def fetch_long_short_ratio(client: BinanceClient, symbol: str, days: int = 30) -> pd.DataFrame:
    """Fetch long/short account ratio (sentiment indicator)."""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # Top trader long/short ratio
        ls_ratio = client.futures_top_longshort_account_ratio(
            symbol=symbol,
            period='1h',
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=500
        )
        
        if not ls_ratio:
            return None
        
        df = pd.DataFrame(ls_ratio)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
        df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
        df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')
        
        df = df.rename(columns={'longShortRatio': 'long_short_ratio'})
        df = df[['timestamp', 'long_short_ratio', 'longAccount', 'shortAccount']]
        
        return df
    
    except Exception as e:
        logger.warning(f"  {symbol}: No long/short ratio - {e}")
        return None


def merge_futures_data(ohlcv: pd.DataFrame, funding: pd.DataFrame, 
                       oi: pd.DataFrame, ls_ratio: pd.DataFrame) -> pd.DataFrame:
    """Merge all futures metrics into single DataFrame."""
    df = ohlcv.copy()
    
    # Round timestamps to nearest hour for alignment
    df['timestamp'] = df['timestamp'].dt.floor('H')
    
    # Merge funding rates (every 8h, forward fill)
    if funding is not None and len(funding) > 0:
        funding['timestamp'] = funding['timestamp'].dt.floor('H')
        df = df.merge(funding, on='timestamp', how='left')
        df['funding_rate'] = df['funding_rate'].fillna(method='ffill').fillna(0)
    else:
        df['funding_rate'] = 0.0
    
    # Merge open interest (1h)
    if oi is not None and len(oi) > 0:
        oi['timestamp'] = oi['timestamp'].dt.floor('H')
        df = df.merge(oi, on='timestamp', how='left')
        df['open_interest'] = df['open_interest'].fillna(method='ffill').fillna(0)
        df['open_interest_usd'] = df['open_interest_usd'].fillna(method='ffill').fillna(0)
    else:
        df['open_interest'] = 0.0
        df['open_interest_usd'] = 0.0
    
    # Merge long/short ratio (1h)
    if ls_ratio is not None and len(ls_ratio) > 0:
        ls_ratio['timestamp'] = ls_ratio['timestamp'].dt.floor('H')
        df = df.merge(ls_ratio, on='timestamp', how='left')
        df['long_short_ratio'] = df['long_short_ratio'].fillna(method='ffill').fillna(1.0)
        df['longAccount'] = df['longAccount'].fillna(0)
        df['shortAccount'] = df['shortAccount'].fillna(0)
    else:
        df['long_short_ratio'] = 1.0
        df['longAccount'] = 0.5
        df['shortAccount'] = 0.5
    
    return df


def main():
    """Main data fetching pipeline."""
    logger.info("=" * 80)
    logger.info("[ROCKET] FETCHING BINANCE FUTURES TRAINING DATA")
    logger.info("=" * 80)
    
    # Initialize client
    cfg = load_config()
    client = BinanceClient(cfg.binance_api_key, cfg.binance_api_secret)
    
    # Get top liquid futures
    logger.info("\n[CHART] Finding liquid USDT-M perpetual futures...")
    symbols = get_top_futures_symbols(client, min_volume_usdt=5_000_000)  # $5M+ daily volume
    logger.info(f"[OK] Selected {len(symbols)} symbols")
    
    # Fetch data for each symbol
    all_data = []
    total = len(symbols)
    checkpoint_path = Path("data/futures_checkpoint.csv")
    
    # Load checkpoint if exists
    processed_symbols = set()
    if checkpoint_path.exists():
        logger.info(f"\n♻️  Loading checkpoint from {checkpoint_path}")
        checkpoint_df = pd.read_csv(checkpoint_path)
        all_data.append(checkpoint_df)
        processed_symbols = set(checkpoint_df['symbol'].unique())
        logger.info(f"   Already processed: {len(processed_symbols)} symbols")
        symbols = [s for s in symbols if s not in processed_symbols]
        logger.info(f"   Remaining: {len(symbols)} symbols")
    
    logger.info(f"\n📥 Fetching data for {len(symbols)} symbols (180 days OHLCV + funding + OI + L/S ratio)...")
    logger.info("⏱️  Estimated time: ~{:.0f} minutes".format(len(symbols) * 0.5))  # ~30 sec per symbol
    
    for idx, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"\n[{idx}/{total}] {symbol}")
            
            # Fetch OHLCV (180 days)
            ohlcv = fetch_futures_klines(client, symbol, days=180)
            if ohlcv is None or len(ohlcv) < 100:
                logger.warning(f"  Skipping {symbol} - insufficient OHLCV data")
                continue
            
            # Fetch funding rates (180 days)
            funding = fetch_funding_rates(client, symbol, days=180)
            
            # Fetch open interest (30 days - API limit)
            oi = fetch_open_interest(client, symbol, days=30)
            
            # Fetch long/short ratio (30 days - API limit)
            ls_ratio = fetch_long_short_ratio(client, symbol, days=30)
            
            # Merge all data
            merged = merge_futures_data(ohlcv, funding, oi, ls_ratio)
            all_data.append(merged)
            
            logger.info(f"  [OK] Total rows: {len(merged)}")
            
            # Rate limiting (to avoid IP ban)
            if idx % 10 == 0:
                logger.info("  ⏸️  Sleeping 5s to avoid rate limit...")
                time.sleep(5)
                
                # Save checkpoint every 10 symbols
                if all_data:
                    checkpoint_df = pd.concat(all_data, ignore_index=True)
                    checkpoint_df.to_csv(checkpoint_path, index=False)
                    logger.info(f"  💾 Checkpoint saved ({len(checkpoint_df):,} rows)")
            else:
                time.sleep(0.5)
        
        except Exception as e:
            logger.error(f"  ❌ {symbol} failed: {e}")
            continue
    
    # Combine all data
    logger.info("\n🔗 Combining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by symbol and timestamp
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    logger.info(f"[OK] Total rows: {len(combined_df):,}")
    logger.info(f"[OK] Total symbols: {combined_df['symbol'].nunique()}")
    logger.info(f"[OK] Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Save to CSV
    output_path = Path("data/binance_futures_training_data.csv")
    output_path.parent.mkdir(exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f"\n💾 Saved to: {output_path}")
    logger.info(f"📦 File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Show sample
    logger.info("\n[CHART] Sample data:")
    print(combined_df.head(10))
    
    # Show column info
    logger.info("\n[CLIPBOARD] Columns:")
    for col in combined_df.columns:
        non_null = combined_df[col].notna().sum()
        pct = non_null / len(combined_df) * 100
        logger.info(f"  {col:20} - {pct:5.1f}% non-null")
    
    logger.info("\n" + "=" * 80)
    logger.info("[OK] FUTURES DATA COLLECTION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Train models: python scripts/train_all_models_futures.py")
    logger.info("2. Setup testnet: Edit config.yaml with testnet keys")
    logger.info("3. Run testnet trading: python scripts/testnet_trading.py")


if __name__ == "__main__":
    main()
