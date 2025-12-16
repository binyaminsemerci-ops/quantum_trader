#!/bin/bash
cd ~/quantum_trader

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  COMMIT FIX                                              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "📋 Current status:"
git status microservices/ai_engine/service.py

echo ""
echo "🔍 Show diff:"
git diff microservices/ai_engine/service.py

echo ""
echo "📝 Staging file..."
git add microservices/ai_engine/service.py

echo ""
echo "💾 Creating commit..."
git commit -m "fix(ai-engine): Remove ServiceHealth import shadowing

Remove duplicate ServiceHealth import from models.py that was shadowing
the correct import from backend.core.health_contract.

Root cause: ServiceHealth was imported twice:
- Line 26: from backend.core.health_contract import ServiceHealth (has create())
- Line 35: from .models import ServiceHealth (no create() - Pydantic BaseModel)

Python used the last import, causing AttributeError when calling
ServiceHealth.create() in get_health() endpoint.

Fix: Removed ServiceHealth from models import line 35.

Result:
- Health endpoint now returns valid status without 'create' error
- ServiceHealth.create() works correctly
- No import collision"

echo ""
echo "✓ Commit created!"
echo ""
git log -1 --oneline
