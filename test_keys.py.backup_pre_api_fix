#!/usr/bin/env python3
import os
import time
import hmac
import hashlib
from urllib.parse import urlencode

api_key = os.getenv('BINANCE_API_KEY', '')
secret = os.getenv('BINANCE_API_SECRET', '')

print(f'✅ API Key: {api_key[:10]}...{api_key[-10:]} (len={len(api_key)})')
print(f'✅ Secret: {secret[:10]}...{secret[-10:]} (len={len(secret)})')

# Test signature generation
params = {'symbol': 'BTCUSDT', 'timestamp': str(int(time.time() * 1000))}
query_string = urlencode(params)
signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
print(f'✅ Test Signature: {signature}')
print(f'✅ Query: {query_string}')
