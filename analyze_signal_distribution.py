"""Quick analysis of signal distribution"""
import json
from collections import Counter
from pathlib import Path

signal_file = Path("/app/data/policy_observations/signals_2025-11-22.jsonl")

symbols = []
symbol_decisions = {}

with open(signal_file) as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec.get('type') != 'signal_decision':
            continue
        
        sym = rec['signal']['symbol']
        symbols.append(sym)
        
        if sym not in symbol_decisions:
            symbol_decisions[sym] = {'allowed': 0, 'blocked': 0, 'total': 0}
        
        symbol_decisions[sym]['total'] += 1
        if rec['actual_decision'] == 'TRADE_ALLOWED':
            symbol_decisions[sym]['allowed'] += 1
        else:
            symbol_decisions[sym]['blocked'] += 1

print("="*80)
print("SIGNAL DISTRIBUTION ANALYSIS")
print("="*80)

symbol_counts = Counter(symbols)

print(f"\nTop 50 symbols by signal count:")
print(f"{'Rank':>4} | {'Symbol':15} | {'Signals':>7} | {'Allowed':>7} | {'Blocked':>7} | {'Allow %':>8}")
print("-" * 80)

for i, (sym, count) in enumerate(symbol_counts.most_common(50), 1):
    dec = symbol_decisions[sym]
    allow_pct = (dec['allowed'] / dec['total'] * 100) if dec['total'] > 0 else 0
    print(f"{i:4d} | {sym:15s} | {count:7d} | {dec['allowed']:7d} | {dec['blocked']:7d} | {allow_pct:7.1f}%")

print(f"\n{'='*80}")
print(f"Total unique symbols: {len(symbol_counts)}")
print(f"Total signals: {sum(symbol_counts.values())}")
print(f"Symbols with >= 5 signals: {len([c for c in symbol_counts.values() if c >= 5])}")
print(f"Symbols with >= 3 signals: {len([c for c in symbol_counts.values() if c >= 3])}")
print(f"Symbols with 1-2 signals: {len([c for c in symbol_counts.values() if c < 3])}")
print(f"{'='*80}\n")
