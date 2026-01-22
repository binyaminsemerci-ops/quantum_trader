#!/usr/bin/env python3
"""Patch ai_strategy_router.py to fix trace_id extraction bug"""

with open('/usr/local/bin/ai_strategy_router.py', 'r') as f:
    content = f.read()

# Find and replace the buggy trace_id extraction
old_pattern = '''                        trace_id = msg_data.get('trace_id', msg_id)
                        correlation_id = msg_data.get('correlation_id', trace_id)'''

new_pattern = '''                        # FIX: AI Engine publishes empty trace_id, use correlation_id
                        correlation_id = msg_data.get('correlation_id', '')
                        trace_id = correlation_id if correlation_id else msg_id'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open('/usr/local/bin/ai_strategy_router.py', 'w') as f:
        f.write(content)
    print("✅ Patched trace_id extraction (correlation_id → trace_id fallback)")
else:
    print("❌ Pattern not found - manual inspection needed")
    print("Looking for:", repr(old_pattern))
