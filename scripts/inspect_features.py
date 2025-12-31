import sqlite3
import pandas as pd
import json

conn = sqlite3.connect("/app/data/quantum_trader.db")
df = pd.read_sql("SELECT features, target_class FROM ai_training_samples LIMIT 5", conn)
conn.close()

print(f"Sample rows: {len(df)}")
print("\nFirst row:")
print(f"  features type: {type(df.iloc[0]['features'])}")
print(f"  features value: {df.iloc[0]['features'][:200]}...")
print(f"  target_class: {df.iloc[0]['target_class']}")

# Try parsing
feat = df.iloc[0]['features']
if isinstance(feat, str):
    parsed = json.loads(feat)
    print(f"\nParsed type: {type(parsed)}")
    print(f"Parsed value: {parsed}")
    if isinstance(parsed, list):
        print(f"Length: {len(parsed)}")
    elif isinstance(parsed, dict):
        print(f"Keys: {list(parsed.keys())}")
else:
    print(f"\nFeatures is already: {type(feat)}")
    print(f"Value: {feat}")
