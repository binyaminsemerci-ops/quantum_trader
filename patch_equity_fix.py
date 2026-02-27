"""Patch service.py to read live equity from Redis instead of hardcoded 10000."""

path = '/home/qt/quantum_trader/microservices/ai_engine/service.py'
with open(path, 'r') as f:
    src = f.read()

HELPER = (
    '    async def _get_live_equity(self, fallback: float = 1000.0) -> float:\n'
    '        """Read live account balance from Redis (quantum:account:balance).\n'
    '\n'
    '        Scales correctly for any capital size (50 USD to 1 million USD).\n'
    '        Leverage (Kelly-based) is completely independent of balance size.\n'
    '        Falls back to `fallback` if Redis is unavailable.\n'
    '        """\n'
    '        try:\n'
    '            val = await self.redis_client.hget("quantum:account:balance", "balance")\n'
    '            if val:\n'
    '                equity = float(val)\n'
    '                if equity > 0:\n'
    '                    logger.debug(f"[EQUITY] Live equity: ${equity:.2f}")\n'
    '                    return equity\n'
    '        except Exception as e:\n'
    '            logger.warning(f"[EQUITY] Redis read failed: {e} — using fallback ${fallback:.0f}")\n'
    '        return fallback\n'
    '\n'
)

ANCHOR = (
    '    # ========================================================================\n'
    '    # SIGNAL GENERATION (MAIN PIPELINE)'
)

if '_get_live_equity' not in src:
    if ANCHOR in src:
        src = src.replace(ANCHOR, HELPER + ANCHOR, 1)
        print('Helper method inserted before SIGNAL GENERATION')
    else:
        print('WARNING: anchor not found, cannot insert helper')
else:
    print('Helper already present, skipping insert')

OLD1 = (
    '                # TODO: Get real account equity from execution-service\n'
    '                equity_usd = 10000.0  # Default $10K account'
)
NEW1 = (
    '                # Live equity from Redis — scales for $50 to $1M+\n'
    '                equity_usd = await self._get_live_equity()'
)
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    print('Replaced equity_usd=10000 in RL sizing block')
else:
    print('WARNING: equity_usd RL block not found')

OLD2 = (
    '                # TODO: Get real account equity from Binance account info\n'
    '                account_equity = 10000.0  # Placeholder'
)
NEW2 = (
    '                # Live equity from Redis — scales for $50 to $1M+\n'
    '                account_equity = await self._get_live_equity()'
)
if OLD2 in src:
    src = src.replace(OLD2, NEW2, 1)
    print('Replaced account_equity=10000 in BRIDGE-PATCH block')
else:
    print('WARNING: account_equity BRIDGE-PATCH block not found')

with open(path, 'w') as f:
    f.write(src)
print('DONE')
