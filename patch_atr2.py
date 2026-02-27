import os, sys

path = '/opt/quantum/microservices/ai_engine/service.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

OLD = (
    "                    else:\n"
    "                        atr_value = 0.02  # fallback: insufficient ticks\n"
    "                atr_pct = atr_value  # e.g. 0.02 = 2%"
)

NEW = (
    "                    else:\n"
    "                        # Fallback: fetch Binance klines (public REST, no auth)\n"
    "                        try:\n"
    "                            import urllib.request as _ur, json as _json\n"
    "                            _testnet = os.getenv('BINANCE_USE_TESTNET','false').lower()=='true'\n"
    "                            _base = 'https://testnet.binancefuture.com' if _testnet else 'https://fapi.binance.com'\n"
    "                            _url = _base + '/fapi/v1/klines?symbol=' + symbol + '&interval=1m&limit=30'\n"
    "                            with _ur.urlopen(_url, timeout=3) as _r:\n"
    "                                _klines = _json.loads(_r.read())\n"
    "                            if _klines and len(_klines) >= 5:\n"
    "                                _closes = [float(k[4]) for k in _klines]\n"
    "                                _p = _closes[-1]\n"
    "                                _rng = (max(_closes) - min(_closes)) / _p\n"
    "                                atr_value = max(0.005, min(0.15, _rng))\n"
    "                                self._price_history.setdefault(symbol, []).extend(_closes[-10:])\n"
    "                                logger.info('[ATR-KLINES] ' + symbol + ' pct=' + str(round(atr_value,4)) + ' n=' + str(len(_klines)))\n"
    "                            else:\n"
    "                                atr_value = 0.02\n"
    "                        except Exception as _e:\n"
    "                            atr_value = 0.02\n"
    "                            logger.debug('[ATR-KLINES-FAIL] ' + symbol + ' ' + str(_e))\n"
    "                atr_pct = atr_value  # e.g. 0.02 = 2%"
)

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("PATCH_OK")
else:
    print("NO_MATCH")
    idx = content.find("fallback: insufficient ticks")
    print(repr(content[max(0,idx-200):idx+200]))
    sys.exit(1)
