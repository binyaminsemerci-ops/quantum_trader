path = '/opt/quantum/microservices/ai_engine/service.py'
with open(path, 'rb') as f:
    data = f.read()

# Find the line we want to replace
target = b'                atr_value = features.get("atr", 0.02)  # Default 2% if not available'

replacement = (
    b'                atr_value = features.get("atr")\n'
    b'                if atr_value is None:\n'
    b'                    _ohlcv = self._ohlcv_history.get(symbol, [])\n'
    b'                    _price = features.get("price") or (_ohlcv[-1]["close"] if _ohlcv else None)\n'
    b'                    if _ohlcv and len(_ohlcv) >= 2 and _price and _price > 0:\n'
    b'                        _period = min(14, len(_ohlcv) - 1)\n'
    b'                        _trs = [\n'
    b'                            max(_ohlcv[i]["high"] - _ohlcv[i]["low"],\n'
    b'                                abs(_ohlcv[i]["high"] - _ohlcv[i-1]["close"]),\n'
    b'                                abs(_ohlcv[i]["low"]  - _ohlcv[i-1]["close"]))\n'
    b'                            for i in range(-_period, 0)\n'
    b'                        ]\n'
    b'                        _atr_abs = sum(_trs) / len(_trs) if _trs else _price * 0.02\n'
    b'                        atr_value = _atr_abs / _price\n'
    b'                        logger.info("[ATR-INLINE] " + symbol + " pct=" + str(round(atr_value, 4)) + " candles=" + str(len(_ohlcv)))\n'
    b'                    else:\n'
    b'                        atr_value = 0.02\n'
    b'                atr_value  # (reassigned above)'
)

# Split on lines, find and replace
lines = data.split(b'\n')
replaced = False
out = []
for line in lines:
    stripped = line.rstrip(b'\r')
    if not replaced and stripped == target:
        ending = b'\r' if line.endswith(b'\r') else b''
        block_lines = replacement.split(b'\n')
        for bl in block_lines[:-1]:
            out.append(bl + ending)
        out.append(block_lines[-1])
        replaced = True
    else:
        out.append(line)

result = b'\n'.join(out)

if replaced:
    with open(path, 'wb') as f:
        f.write(result)
    print("PATCHED OK - ATR inline calculation added")
else:
    print("NOT FOUND - searching for partial match:")
    for i, line in enumerate(lines):
        if b'atr_value = features.get' in line:
            print("  line", i, repr(line[:120]))
