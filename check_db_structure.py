"""Check database structure."""
import sqlite3

try:
    conn = sqlite3.connect('data/execution_journal.db')
    
    # List all tables
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    
    print('\n📚 DATABASE TABLES:\n')
    print('═' * 80)
    
    for table in tables:
        table_name = table[0]
        print(f'\n🗂️  Table: {table_name}')
        
        # Get column info
        columns = conn.execute(f'PRAGMA table_info({table_name})').fetchall()
        print('   Columns:')
        for col in columns:
            print(f'     - {col[1]} ({col[2]})')
        
        # Get row count
        count = conn.execute(f'SELECT COUNT(*) FROM {table_name}').fetchone()[0]
        print(f'   Rows: {count}')
    
    print('\n' + '═' * 80)
    
    conn.close()
    
except Exception as e:
    print(f'❌ Feil: {e}')
