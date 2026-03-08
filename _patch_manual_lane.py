"""Patch PSP to auto-renew quantum:manual_lane:enabled every 5s with TTL=3600."""
path = '/home/qt/quantum_trader/microservices/portfolio_state_publisher/main.py'

with open(path) as f:
    data = f.read()

old = (
    '        # CRITICAL: Write open_orders count so ai_strategy_router capacity check passes\n'
    '        r.set("quantum:state:open_orders", str(positions_count), ex=PSP_STATE_TTL_SEC)\n'
    '        logger.info(f"Set quantum:state:open_orders={positions_count}")'
)

new = (
    '        # CRITICAL: Write open_orders count so ai_strategy_router capacity check passes\n'
    '        r.set("quantum:state:open_orders", str(positions_count), ex=PSP_STATE_TTL_SEC)\n'
    '        logger.info(f"Set quantum:state:open_orders={positions_count}")\n'
    '\n'
    '        # AUTO-RENEW manual trading lane (PSP keeps it alive; DELETE this key to disable trading)\n'
    '        r.set("quantum:manual_lane:enabled", "1", ex=3600)\n'
    '        logger.debug("Renewed quantum:manual_lane:enabled (ex=3600)")'
)

if old not in data:
    print('ERROR: anchor string not found — check file manually')
    import sys; sys.exit(1)

data = data.replace(old, new, 1)

with open(path, 'w') as f:
    f.write(data)

print('PATCHED OK')
for i, line in enumerate(data.splitlines(), 1):
    if 'manual_lane' in line or 'open_orders' in line:
        print(f'  L{i}: {line.strip()}')
