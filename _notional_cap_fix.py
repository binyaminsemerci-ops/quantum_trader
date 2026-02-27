import shutil, re

filepath = "/home/qt/quantum_trader/microservices/intent_bridge/main.py"
shutil.copy2(filepath, filepath + ".bak_notional_cap")

with open(filepath, "r") as f:
    content = f.read()

target = "            if not qty or qty <= 0:"
idx = content.find(target)
if idx == -1:
    print("ERROR: target not found")
    exit(1)

cap_code = (
    '            # NOTIONAL CAP: prevent -4005 on cheap coins (2026-02-25)\n'
    '            if qty and qty > 0:\n'
    '                try:\n'
    '                    import os as _os\n'
    '                    _p = locals().get("price_used") or float(payload.get("price", 0) or payload.get("entry_price", 0))\n'
    '                    if _p and _p > 0:\n'
    '                        _n = qty * _p\n'
    '                        _mx = float(_os.getenv("QT_MAX_NOTIONAL_USD", "3000"))\n'
    '                        if _n > _mx:\n'
    '                            qty = _mx / _p\n'
    '                            logger.warning(f"[NOTIONAL_CAP] {symbol}: notional ${_n:.2f} > ${_mx:.0f} capped qty={qty:.4f}")\n'
    '                except Exception:\n'
    '                    pass\n'
    '\n'
)

content = content[:idx] + cap_code + content[idx:]
with open(filepath, "w") as f:
    f.write(content)
print("SUCCESS: notional cap inserted at index", idx)
