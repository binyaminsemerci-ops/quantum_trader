import asyncio
import redis.asyncio as aioredis
import json

async def main():
    r0 = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r1 = aioredis.Redis(host='localhost', port=6379, db=1, decode_responses=True)

    bt_keys = []
    async for k in r1.scan_iter('quantum:backtest:*'):
        bt_keys.append(k)
    print(f"Layer 3 keys (db=1): {len(bt_keys)}")
    for k in sorted(bt_keys)[:15]:
        print(f"  {k}")
    print()
    for sym in ['SOLUSDT', 'LINKUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
        l4 = await r0.hgetall(f'quantum:layer4:sizing:{sym}')
        if l4:
            print(f"L4 {sym}: wr={l4.get('metrics_wr')} pf={l4.get('metrics_pf')} sh={l4.get('metrics_sharpe')} rec={l4.get('recommendation')}")
    print()
    fe = await r0.hgetall('quantum:dag8:freeze_exit:status')
    print(f"dag8 freeze_exit: {json.dumps(fe)}")
    phase = await r0.get('quantum:dag8:current_phase')
    print(f"dag8 current_phase: {phase}")
    await r0.aclose(); await r1.aclose()

asyncio.run(main())
