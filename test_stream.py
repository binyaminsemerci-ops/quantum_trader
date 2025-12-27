import redis
r = redis.from_url('redis://redis:6379/0')
data = r.xrevrange('quantum:stream:exchange.raw', count=5)
print(f'Found {len(data)} entries')
for entry_id, fields in data:
    symbol = fields.get(b'symbol', b'').decode()
    close = fields.get(b'close', b'').decode()
    print(f'{symbol}: {close}')
