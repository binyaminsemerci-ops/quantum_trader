#!/usr/bin/env python3
from config.config import load_config

cfg = load_config()
api_key = getattr(cfg, 'binance_api_key', 'NONE')
secret = getattr(cfg, 'binance_api_secret', 'NONE')

print(f"API Key: {api_key[:20] if api_key != 'NONE' else 'NONE'}... (len={len(api_key) if api_key != 'NONE' else 0})")
print(f"Secret: {secret[:20] if secret != 'NONE' else 'NONE'}... (len={len(secret) if secret != 'NONE' else 0})")
