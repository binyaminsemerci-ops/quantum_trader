import os
import sqlite3

# Determine sqlite path from env if set, otherwise use default backend path
db_url = os.environ.get('QUANTUM_TRADER_DATABASE_URL')
if db_url and db_url.startswith('sqlite:///'):
    db_path = db_url.replace('sqlite:///', '')
else:
    db_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'data', 'trades.db')

db_path = os.path.abspath(db_path)
print('Using DB:', db_path)

if not os.path.exists(db_path):
    print('DB file not found:', db_path)
    raise SystemExit(1)

conn = sqlite3.connect(db_path)
print('\nSignals (most recent 10):')
for row in conn.execute("select id,symbol,side,qty,price,confidence,status from signals order by id desc limit 10"):
    print(row)

print('\nTrade logs (most recent 10):')
for row in conn.execute("select id,timestamp,symbol,side,qty,price,status,reason from trade_logs order by id desc limit 10"):
    print(row)

conn.close()
