"""
EXPORT TRAINING DATA FROM BACKEND DATABASE
Fetches historical OHLCV data from PostgreSQL for TFT training
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

from backend.database import get_db_pool
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_training_data(
    output_path: str = "data/binance_training_data.csv",
    days_back: int = 30,
    min_candles_per_symbol: int = 100
):
    """
    Export OHLCV data from backend database
    
    Args:
        output_path: Output CSV file path
        days_back: How many days of history to fetch
        min_candles_per_symbol: Minimum candles required per symbol
    """
    
    logger.info("🔌 Connecting to backend database...")
    
    pool = await get_db_pool()
    
    try:
        async with pool.acquire() as conn:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"📅 Fetching data from {start_date} to {end_date}")
            
            # Query OHLCV data
            query = """
            SELECT 
                symbol,
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM ohlcv_1h
            WHERE timestamp >= :start_date
            AND timestamp <= :end_date
            ORDER BY symbol, timestamp
            """
            
            result = await conn.execute(
                text(query),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            rows = await result.fetchall()
            
            if not rows:
                logger.error("❌ No data found in database!")
                logger.info("💡 Run backend first to collect OHLCV data")
                return
            
            logger.info(f"[OK] Fetched {len(rows)} candles from database")
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Filter symbols with enough data
            symbol_counts = df['symbol'].value_counts()
            valid_symbols = symbol_counts[symbol_counts >= min_candles_per_symbol].index
            df = df[df['symbol'].isin(valid_symbols)]
            
            logger.info(f"[CHART] Found {len(valid_symbols)} symbols with >={min_candles_per_symbol} candles")
            
            # Save to CSV
            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            
            df.to_csv(output_file, index=False)
            
            logger.info(f"[OK] Training data saved to {output_file}")
            logger.info(f"   Total rows: {len(df)}")
            logger.info(f"   Symbols: {len(valid_symbols)}")
            logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise
    
    finally:
        await pool.close()


async def main():
    """Export training data from backend"""
    
    print("\n" + "="*60)
    print("📥 EXPORTING TRAINING DATA FROM BACKEND DATABASE")
    print("="*60 + "\n")
    
    await export_training_data(
        output_path="data/binance_training_data.csv",
        days_back=30,
        min_candles_per_symbol=100
    )
    
    print("\n" + "="*60)
    print("[OK] EXPORT COMPLETE!")
    print("="*60)
    print("\n[ROCKET] Next step:")
    print("   python scripts/train_tft_quantile.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
