"""Check all trades for current positions."""
import sqlite3

try:
    conn = sqlite3.connect('data/execution_journal.db')
    cursor = conn.cursor()
    
    # Get ALL trades for current positions
    symbols = ('BNBUSDT', 'BTCUSDT', 'SOLUSDT', 'DOTUSDT', 'ETHUSDT')
    
    rows = cursor.execute('''
        SELECT timestamp, symbol, side, entry_price, exit_price, pnl, exit_reason 
        FROM trades 
        WHERE symbol IN (?, ?, ?, ?, ?)
        ORDER BY timestamp DESC 
        LIMIT 30
    ''', symbols).fetchall()
    
    print('\n[CHART] ALLE TRADES FOR CURRENT POSITIONS:\n')
    print('═' * 80)
    
    if not rows:
        print('❌ Ingen trades funnet')
    else:
        for r in rows:
            timestamp, symbol, side, entry, exit, pnl, reason = r
            pnl = pnl or 0
            
            pnl_emoji = '[GREEN_CIRCLE]' if pnl > 0 else '[RED_CIRCLE]' if pnl < 0 else '⚪'
            status = reason or 'ÅPEN'
            closed = '[OK] CLOSED' if exit and exit > 0 else '🔵 OPEN'
            
            print(f'{pnl_emoji} {closed} | {timestamp}')
            print(f'   {symbol:10s} {side:5s}')
            print(f'   Entry: ${entry:.4f} | Exit: ${exit or 0:.4f}')
            print(f'   P&L: ${pnl:.2f} | Exit Reason: {status}')
            print()
    
    print('═' * 80)
    
    conn.close()
    
except Exception as e:
    print(f'❌ Feil: {e}')
