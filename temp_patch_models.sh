#!/bin/bash
# Patch SQLAlchemy models to allow extend_existing

cd ~/quantum_trader/backend/models

# Patch trade.py
if ! grep -q "__table_args__" trade.py; then
    sed -i '/__tablename__ = "trades"/a\    __table_args__ = {'"'"'extend_existing'"'"': True}' trade.py
    echo "✅ Patched trade.py"
else
    echo "⚠️  trade.py already patched"
fi

# Patch trade_log.py
if ! grep -q "__table_args__" trade_log.py; then
    sed -i '/__tablename__/a\    __table_args__ = {'"'"'extend_existing'"'"': True}' trade_log.py
    echo "✅ Patched trade_log.py"
else
    echo "⚠️  trade_log.py already patched"
fi

# Patch policy.py
if ! grep -q "__table_args__" policy.py; then
    sed -i '/__tablename__/a\    __table_args__ = {'"'"'extend_existing'"'"': True}' policy.py
    echo "✅ Patched policy.py"
else
    echo "⚠️  policy.py already patched"
fi

echo ""
echo "🎯 Verification:"
grep -A1 "__tablename__" trade.py | head -2
echo ""
echo "✅ All patches applied!"
