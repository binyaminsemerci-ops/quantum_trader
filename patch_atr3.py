path = '/opt/quantum/microservices/ai_engine/service.py'
with open(path, 'rb') as f:
    raw = f.read()

crlf = b'\r\n' in raw
LE = b'\r\n' if crlf else b'\n'

lines = raw.split(LE)

start_idx = None
end_idx = None

for i, line in enumerate(lines):
    s = line.strip()
    if start_idx is None and b'# Get ATR from features (for volatility-based sizing)' in s:
        start_idx = i
    if start_idx is not None and b'atr_pct = atr_value' in s:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"Markers not found: start={start_idx} end={end_idx}")
    for i, line in enumerate(lines[2510:2545], 2510):
        print(f"  {i}: {repr(line[:80])}")
    exit(1)

print(f"Replacing lines {start_idx+1}-{end_idx+1} ({end_idx-start_idx+1} lines)")

I16 = b'                '
I20 = b'                    '
I24 = b'                        '

new_lines = [
    I16 + b'# Get ATR: prefer PHASE 2D result; fallback: compute from price history',
    I16 + b'atr_value = features.get("atr")',
    I16 + b'if atr_value is None:',
    I20 + b'_prices = self._price_history.get(symbol, [])',
    I20 + b'_price = features.get("price") or (_prices[-1] if _prices else None)',
    I20 + b'if _prices and len(_prices) >= 10 and _price and _price > 0:',
    I24 + b'# Range over last 20 ticks: differentiates BTC(~0.003) vs memes(~0.05)',
    I24 + b'_win = _prices[-20:]',
    I24 + b'_range_pct = (max(_win) - min(_win)) / _price',
    I24 + b'atr_value = max(0.005, min(0.15, _range_pct))',
    I24 + b'logger.info("[ATR-INLINE] " + symbol + " pct=" + str(round(atr_value, 4)) + " ticks=" + str(len(_prices)))',
    I20 + b'else:',
    I24 + b'atr_value = 0.02  # fallback: insufficient ticks',
    I16 + b'atr_pct = atr_value  # e.g. 0.02 = 2%',
]

result_lines = lines[:start_idx] + new_lines + lines[end_idx + 1:]
result = LE.join(result_lines)

with open(path, 'wb') as f:
    f.write(result)

print(f"PATCHED OK - {end_idx-start_idx+1} lines -> {len(new_lines)} lines")
for l in new_lines:
    print(" ", l.decode())
