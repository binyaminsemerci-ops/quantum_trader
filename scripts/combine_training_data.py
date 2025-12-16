"""
Combine all market_data CSV files into single training dataset
"""
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime

def combine_all_data():
    """Combine all CSV files into one training dataset."""
    print("="*60)
    print("📂 COMBINING MARKET DATA FOR TRAINING")
    print("="*60)
    
    # Find all CSV files
    csv_files = sorted(glob.glob('data/market_data/*.csv'))
    print(f"\n[OK] Found {len(csv_files)} CSV files")
    
    if not csv_files:
        print("❌ No CSV files found in data/market_data/")
        return
    
    # Load and combine
    dfs = []
    total_rows = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            symbol = Path(csv_file).stem.replace('_90d', '')
            
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            dfs.append(df)
            total_rows += len(df)
            
            if len(dfs) % 10 == 0:
                print(f"  Loaded {len(dfs)} files ({total_rows:,} rows)...")
                
        except Exception as e:
            print(f"  [WARNING]  Skipped {Path(csv_file).name}: {e}")
    
    print(f"\n[OK] Loaded {len(dfs)} files with {total_rows:,} total rows")
    
    # Combine all dataframes
    print("\n🔄 Combining data...")
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates (same timestamp + symbol)
    if 'timestamp' in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['timestamp', 'symbol'])
        print(f"  Removed {before - len(combined):,} duplicates")
    
    # Sort by symbol and timestamp
    if 'timestamp' in combined.columns:
        combined = combined.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Save
    output_path = 'data/binance_training_data_full.csv'
    combined.to_csv(output_path, index=False)
    
    print("="*60)
    print(f"[OK] SAVED: {output_path}")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Columns: {len(combined.columns)}")
    print(f"   Unique symbols: {combined['symbol'].nunique()}")
    print(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print("="*60)
    
    return combined

if __name__ == '__main__':
    combine_all_data()
