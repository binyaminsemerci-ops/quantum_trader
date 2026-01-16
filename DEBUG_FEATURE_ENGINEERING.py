#!/usr/bin/env python3
"""
DEBUG: Investigate why feature engineering fails for most trades
Testing synthetic data generation + feature calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import redis
import sys

sys.path.insert(0, "/home/qt/quantum_trader")

from scripts.train_ensemble_models_v4 import SyntheticDataGenerator, EnsembleTrainer

# Connect to Redis
redis_conn = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Get 82 trades
trades_raw = redis_conn.xrange("quantum:stream:trade.closed")

trades = []
for msg_id, data in trades_raw:
    try:
        entry = float(data.get('entry', 0))
        exit_price = float(data.get('exit', 0))
        pnl_usd = float(data.get('pnl', 0))
        pnl_pct = (pnl_usd / (entry + 1e-8)) * 100 if entry != 0 else 0
        
        trades.append({
            'id': msg_id,
            'symbol': data.get('symbol', 'UNKNOWN'),
            'entry_price': entry,
            'exit_price': exit_price,
            'pnl': pnl_usd,
            'pnl_pct': pnl_pct,
        })
    except Exception as e:
        print(f"Parse error: {e}")

print(f"Loaded {len(trades)} trades")
print()

# Test first 5 trades
trainer = EnsembleTrainer()
success_count = 0
fail_reasons = {}

for i, trade in enumerate(trades[:10]):
    print(f"\n{'='*80}")
    print(f"Trade {i+1}: {trade['symbol']} | Entry={trade['entry_price']:.2f} | Exit={trade['exit_price']:.2f} | PnL%={trade['pnl_pct']:.4f}")
    print('='*80)
    
    try:
        # Generate synthetic data
        df = SyntheticDataGenerator.generate_synthetic_window(
            trade['exit_price'],
            trade['pnl_pct'],
            n_bars=100
        )
        print(f"✅ Synthetic window: {len(df)} bars")
        print(f"   Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        # Engineer features
        df_original_len = len(df)
        df = trainer.engineer_features(df)
        print(f"✅ Features engineered: {df_original_len} → {len(df)} rows (dropped {df_original_len - len(df)})")
        
        if len(df) == 0:
            print(f"❌ All rows dropped after feature engineering!")
            fail_reason = "all_rows_dropped"
        else:
            print(f"   Columns: {', '.join(df.columns.tolist()[:10])}...")
            print(f"   Last row:\n{df.iloc[-1]}")
            success_count += 1
            
            # Check for NaN
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                print(f"   ⚠️  {nan_count} NaN values in last row")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        fail_reason = f"exception: {type(e).__name__}"
    
    fail_reasons[fail_reason] = fail_reasons.get(fail_reason, 0) + 1

print(f"\n\n{'='*80}")
print(f"SUMMARY: {success_count}/10 trades succeeded")
print(f"Failure reasons:")
for reason, count in fail_reasons.items():
    print(f"  - {reason}: {count}")
print('='*80)
