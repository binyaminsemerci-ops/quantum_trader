"""Adapter smoke test for CI: instantiate exchange adapters and call spot_balance.

This script is intended to be quick and tolerant: it will print adapter availability
and won't fail the job unless an unexpected exception occurs while importing.
"""

import sys
from config.config import load_config

try:
    from backend.utils.exchanges import get_exchange_client
except Exception as e:
    print("Failed to import exchanges module:", e)
    sys.exit(2)

cfg = load_config()
print(
    "Loaded config keys:",
    {k: bool(v) for k, v in cfg.__dict__.items() if "API" in k or "EXCHANGE" in k},
)
for name in ("binance", "coinbase", "kucoin"):
    print("\nTesting adapter:", name)
    try:
        client = get_exchange_client(name=name, api_key=None, api_secret=None)
        try:
            bal = client.spot_balance()
            print("spot balance result (type):", type(bal))
        except Exception as e:
            print("Adapter spot_balance call raised (expected without creds):", repr(e))
    except Exception as e:
        print("Failed to create adapter:", repr(e))

print("\nAdapter smoke script finished")
