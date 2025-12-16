"""Check how many trading pairs the AI model was trained on."""
import sqlite3

conn = sqlite3.connect('ai_engine/data/training_data.db')
cursor = conn.cursor()

# Get count
cursor.execute('SELECT COUNT(DISTINCT symbol) FROM candles')
count = cursor.fetchone()[0]

# Get all symbols
cursor.execute('SELECT DISTINCT symbol FROM candles ORDER BY symbol')
symbols = [s[0] for s in cursor.fetchall()]

print(f'\n[TARGET] TFT/XGBoost modellen er trent på {count} unike pairs\n')
print('De første 30 pairs:')
for i, s in enumerate(symbols[:30], 1):
    print(f'  {i:2d}. {s}')

if len(symbols) > 30:
    print(f'\n  ... og {len(symbols)-30} flere')

print(f'\n[CLIPBOARD] Totalt: {len(symbols)} trading pairs')

# Get sample count per symbol
cursor.execute('''
    SELECT symbol, COUNT(*) as samples 
    FROM candles 
    GROUP BY symbol 
    ORDER BY samples DESC 
    LIMIT 5
''')
print(f'\nTop 5 symboler med flest samples:')
for symbol, samples in cursor.fetchall():
    print(f'  • {symbol}: {samples:,} candles')

conn.close()
