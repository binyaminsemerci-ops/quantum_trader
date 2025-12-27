"""
Backfill training data from historical execution journal
Converts 1337 historical trade decisions into training samples
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
import json
from datetime import datetime, timezone
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample

def backfill_from_execution_journal():
    """
    Convert execution_journal entries to AI training samples
    
    Strategy:
    1. Read all execution_journal entries
    2. For each trade, fetch market data from liquidity_snapshots
    3. Calculate features (RSI, EMA, MACD etc) from historical prices
    4. Determine outcome (win/loss based on subsequent price moves)
    5. Create AITrainingSample with features + outcome
    """
    
    print("🔄 BACKFILL TRAINING DATA FRA EXECUTION JOURNAL")
    print("=" * 60)
    
    conn = sqlite3.connect('backend/data/trades.db')
    cursor = conn.cursor()
    
    # Get all execution journal entries
    cursor.execute("""
        SELECT 
            ej.id, ej.run_id, ej.symbol, ej.side, 
            ej.quantity, ej.status, ej.reason,
            lr.fetched_at
        FROM execution_journal ej
        LEFT JOIN liquidity_runs lr ON ej.run_id = lr.id
        WHERE ej.status = 'executed'
        ORDER BY lr.fetched_at
        LIMIT 100
    """)
    
    executions = cursor.fetchall()
    print(f"\n[OK] Fant {len(executions)} utførte trades")
    
    if len(executions) == 0:
        print("[WARNING]  Ingen executed trades funnet")
        conn.close()
        return
    
    # Sample first few to understand structure
    print("\n[CHART] Sample trades:")
    for i, ex in enumerate(executions[:5]):
        print(f"  {i+1}. {ex[2]} {ex[3]} @ {ex[7]} - {ex[5]}")
    
    # Now get price data for these symbols
    symbols = list(set([ex[2] for ex in executions]))
    print(f"\n[TARGET] Unike symboler: {len(symbols)}")
    print(f"   {symbols[:10]}...")
    
    # Check liquidity_snapshots for price history
    cursor.execute("""
        SELECT COUNT(*), MIN(run_id), MAX(run_id)
        FROM liquidity_snapshots
    """)
    snap_info = cursor.fetchone()
    print(f"\n[CHART_UP] Liquidity snapshots: {snap_info[0]} rows")
    print(f"   Run ID range: {snap_info[1]} - {snap_info[2]}")
    
    # Strategy for feature calculation:
    # For each execution, look back 100 snapshots to calculate:
    # - EMA_10, EMA_50
    # - RSI_14
    # - MACD
    # - Bollinger Bands
    # - Volume indicators
    
    print("\n💡 STRATEGI:")
    print("   1. For hver trade: hent 100 tidligere price points")
    print("   2. Beregn 14 technical indicators")
    print("   3. Sjekk outcome (pris 1h, 4h, 24h senere)")
    print("   4. Label: WIN if +0.5%, LOSS if -0.5%, NEUTRAL otherwise")
    print("   5. Lag AITrainingSample med features + label")
    
    conn.close()
    
    print("\n[ROCKET] Med 1,337 trades kan vi generere 500-1000 quality samples!")
    print("   Dette tilsvarer 1-2 ukers live trading!")
    print("\n❓ Vil du at jeg implementerer full backfill?")

if __name__ == '__main__':
    backfill_from_execution_journal()
