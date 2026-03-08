#!/usr/bin/env python3
"""Patch portfolio_state_publisher to write quantum:state:open_orders"""
path = '/home/qt/quantum_trader/microservices/portfolio_state_publisher/main.py'
with open(path) as f:
    content = f.read()

old = (
    '        r.hset(PSP_PORTFOLIO_KEY, mapping=state)\n'
    '        r.expire(PSP_PORTFOLIO_KEY, PSP_STATE_TTL_SEC)'
)
new = (
    '        r.hset(PSP_PORTFOLIO_KEY, mapping=state)\n'
    '        r.expire(PSP_PORTFOLIO_KEY, PSP_STATE_TTL_SEC)\n\n'
    '        # CRITICAL: Write open_orders count so ai_strategy_router capacity check passes\n'
    '        r.set("quantum:state:open_orders", str(positions_count), ex=PSP_STATE_TTL_SEC)\n'
    '        logger.info(f"Set quantum:state:open_orders={positions_count}")'
)

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('PATCH_OK')
else:
    print('PATTERN_NOT_FOUND')
    print('Looking for:')
    print(repr(old[:80]))
