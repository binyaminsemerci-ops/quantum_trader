"""
Strategy Management Tool

Check, create, and promote strategies to LIVE status.
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.database import SessionLocal
from sqlalchemy import text

def check_strategies():
    """Check existing strategies in database"""
    print("\n" + "="*70)
    print("Strategy Database Status")
    print("="*70)
    
    session = SessionLocal()
    
    try:
        # Check if table exists
        result = session.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sg_strategies'"
        )).fetchone()
        
        if not result:
            print("\n[WARNING] sg_strategies table does not exist!")
            print("          Creating table...")
            create_strategies_table(session)
            return
        
        # Count total strategies
        result = session.execute(text(
            "SELECT COUNT(*) as count FROM sg_strategies"
        )).fetchone()
        
        total = result[0] if result else 0
        print(f"\nTotal strategies: {total}")
        
        if total == 0:
            print("\n[INFO] No strategies found in database")
            print("       Need to either:")
            print("       1. Start Strategy Generator AI to create strategies")
            print("       2. Create demo strategies for testing")
            return
        
        # Count by status
        result = session.execute(text(
            "SELECT status, COUNT(*) as count FROM sg_strategies GROUP BY status"
        )).fetchall()
        
        print("\nStrategies by status:")
        for status, count in result:
            print(f"  {status}: {count}")
        
        # Show LIVE strategies
        result = session.execute(text(
            "SELECT strategy_id, name, min_confidence FROM sg_strategies WHERE status = 'LIVE' LIMIT 5"
        )).fetchall()
        
        if result:
            print("\nLIVE strategies:")
            for strategy_id, name, confidence in result:
                print(f"  • {strategy_id}: {name} (confidence >= {confidence:.2f})")
        else:
            print("\n[ACTION NEEDED] No LIVE strategies found!")
            print("                Run: python manage_strategies.py promote")
        
    finally:
        session.close()

def create_strategies_table(session):
    """Create sg_strategies table if it doesn't exist"""
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS sg_strategies (
            strategy_id VARCHAR(100) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'DRAFT',
            entry_params TEXT,
            sl_percent FLOAT,
            tp_percent FLOAT,
            min_confidence FLOAT DEFAULT 0.5,
            max_concurrent_positions INTEGER DEFAULT 1,
            regime_filter VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_executed_at TIMESTAMP
        )
    """))
    session.commit()
    print("[OK] Created sg_strategies table")

def create_demo_strategies():
    """Create demo strategies for testing"""
    print("\n" + "="*70)
    print("Creating Demo Strategies")
    print("="*70)
    
    session = SessionLocal()
    
    try:
        # Ensure table exists
        create_strategies_table(session)
        
        strategies = [
            {
                'strategy_id': 'demo-rsi-oversold-001',
                'name': 'RSI Oversold Long',
                'status': 'DRAFT',
                'entry_params': '{"rsi_threshold": 30, "timeframe": "1h"}',
                'sl_percent': 2.0,
                'tp_percent': 5.0,
                'min_confidence': 0.55,
                'max_concurrent_positions': 2,
                'regime_filter': 'TRENDING_ONLY'
            },
            {
                'strategy_id': 'demo-macd-cross-002',
                'name': 'MACD Bullish Cross',
                'status': 'DRAFT',
                'entry_params': '{"indicator": "MACD", "signal": "bullish_cross"}',
                'sl_percent': 2.5,
                'tp_percent': 6.0,
                'min_confidence': 0.60,
                'max_concurrent_positions': 1,
                'regime_filter': 'ALL'
            },
            {
                'strategy_id': 'demo-sma-golden-003',
                'name': 'SMA Golden Cross',
                'status': 'DRAFT',
                'entry_params': '{"sma_fast": 50, "sma_slow": 200}',
                'sl_percent': 3.0,
                'tp_percent': 8.0,
                'min_confidence': 0.65,
                'max_concurrent_positions': 1,
                'regime_filter': 'TRENDING_ONLY'
            },
            {
                'strategy_id': 'demo-rsi-overbought-004',
                'name': 'RSI Overbought Short',
                'status': 'DRAFT',
                'entry_params': '{"rsi_threshold": 70, "timeframe": "1h", "direction": "short"}',
                'sl_percent': 2.0,
                'tp_percent': 4.0,
                'min_confidence': 0.58,
                'max_concurrent_positions': 2,
                'regime_filter': 'ALL'
            },
            {
                'strategy_id': 'demo-bbands-squeeze-005',
                'name': 'Bollinger Bands Squeeze',
                'status': 'DRAFT',
                'entry_params': '{"indicator": "BBANDS", "pattern": "squeeze"}',
                'sl_percent': 2.5,
                'tp_percent': 7.0,
                'min_confidence': 0.62,
                'max_concurrent_positions': 1,
                'regime_filter': 'RANGING_ONLY'
            }
        ]
        
        created = 0
        for strat in strategies:
            # Check if exists
            result = session.execute(text(
                "SELECT strategy_id FROM sg_strategies WHERE strategy_id = :id"
            ), {"id": strat['strategy_id']}).fetchone()
            
            if result:
                print(f"[SKIP] {strat['strategy_id']} already exists")
                continue
            
            # Insert
            session.execute(text("""
                INSERT INTO sg_strategies 
                (strategy_id, name, status, entry_params, sl_percent, tp_percent, 
                 min_confidence, max_concurrent_positions, regime_filter, created_at, updated_at)
                VALUES 
                (:strategy_id, :name, :status, :entry_params, :sl_percent, :tp_percent,
                 :min_confidence, :max_concurrent_positions, :regime_filter, :created_at, :updated_at)
            """), {
                **strat,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            
            print(f"[CREATED] {strat['strategy_id']}: {strat['name']}")
            created += 1
        
        session.commit()
        print(f"\n[OK] Created {created} demo strategies")
        print("     Status: DRAFT (ready to be promoted to LIVE)")
        
    finally:
        session.close()

def promote_strategies(strategy_ids=None):
    """Promote strategies to LIVE status"""
    print("\n" + "="*70)
    print("Promoting Strategies to LIVE")
    print("="*70)
    
    session = SessionLocal()
    
    try:
        if strategy_ids:
            # Promote specific strategies
            for strategy_id in strategy_ids:
                result = session.execute(text(
                    "UPDATE sg_strategies SET status = 'LIVE', updated_at = :now WHERE strategy_id = :id"
                ), {"id": strategy_id, "now": datetime.utcnow()})
                
                if result.rowcount > 0:
                    print(f"[PROMOTED] {strategy_id} -> LIVE")
                else:
                    print(f"[NOT FOUND] {strategy_id}")
        else:
            # Promote all DRAFT strategies
            result = session.execute(text(
                "SELECT strategy_id, name FROM sg_strategies WHERE status = 'DRAFT'"
            )).fetchall()
            
            if not result:
                print("\n[INFO] No DRAFT strategies to promote")
                return
            
            print(f"\nFound {len(result)} DRAFT strategies:")
            for strategy_id, name in result:
                print(f"  • {strategy_id}: {name}")
            
            print("\nPromoting all to LIVE...")
            
            session.execute(text(
                "UPDATE sg_strategies SET status = 'LIVE', updated_at = :now WHERE status = 'DRAFT'"
            ), {"now": datetime.utcnow()})
            
            print(f"[OK] Promoted {len(result)} strategies to LIVE")
        
        session.commit()
        
    finally:
        session.close()

def pause_strategies(strategy_ids=None):
    """Pause LIVE strategies"""
    print("\n" + "="*70)
    print("Pausing Strategies")
    print("="*70)
    
    session = SessionLocal()
    
    try:
        if strategy_ids:
            for strategy_id in strategy_ids:
                result = session.execute(text(
                    "UPDATE sg_strategies SET status = 'PAUSED', updated_at = :now WHERE strategy_id = :id"
                ), {"id": strategy_id, "now": datetime.utcnow()})
                
                if result.rowcount > 0:
                    print(f"[PAUSED] {strategy_id}")
                else:
                    print(f"[NOT FOUND] {strategy_id}")
        else:
            # Pause all LIVE strategies
            result = session.execute(text(
                "UPDATE sg_strategies SET status = 'PAUSED', updated_at = :now WHERE status = 'LIVE'"
            ), {"now": datetime.utcnow()})
            
            print(f"[OK] Paused {result.rowcount} LIVE strategies")
        
        session.commit()
        
    finally:
        session.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Management Tool")
    parser.add_argument('action', choices=['check', 'create', 'promote', 'pause'], 
                       help='Action to perform')
    parser.add_argument('--ids', nargs='+', help='Strategy IDs (for promote/pause)')
    
    args = parser.parse_args()
    
    if args.action == 'check':
        check_strategies()
    
    elif args.action == 'create':
        create_demo_strategies()
        print("\n[NEXT] Run: python manage_strategies.py promote")
    
    elif args.action == 'promote':
        promote_strategies(args.ids)
        print("\n[NEXT] Restart backend to load LIVE strategies")
        print("       python -m backend.main")
    
    elif args.action == 'pause':
        pause_strategies(args.ids)
    
    # Show final status
    print("\n" + "="*70)
    check_strategies()

if __name__ == "__main__":
    main()
