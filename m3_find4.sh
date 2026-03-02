#!/bin/bash
echo "=== Search for PHASE format strings ==="
grep -rn "\[PHASE" /opt/quantum/ai_engine/ /opt/quantum/backend/ 2>/dev/null | grep -v ".pyc\|emoji_backup" | head -30

echo ""
echo "=== Search in ensemble_manager for PHASE ==="
grep -n "PHASE\|timeout" /opt/quantum/ai_engine/enhanced_data_collection.py 2>/dev/null | head -30
