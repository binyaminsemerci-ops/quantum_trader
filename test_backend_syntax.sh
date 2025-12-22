#!/bin/bash
# Test backend/main.py syntax after Position Monitor integration

echo "🧪 Testing backend/main.py syntax..."

docker exec quantum_backend python3 -c "
from backend.main import app
print('✅ Backend main.py syntax OK')
print('✅ Position Monitor import OK')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All syntax checks passed!"
    echo "🛡️ Position Monitor integration ready for deployment"
else
    echo ""
    echo "❌ Syntax error detected - fix before deployment"
    exit 1
fi
