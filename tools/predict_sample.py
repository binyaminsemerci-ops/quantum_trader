import sys
import os
import datetime

# Ensure the repository root is on sys.path so local packages can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai_engine.agents.xgb_agent import make_default_agent  # noqa: E402

agent = make_default_agent()
print('Model loaded?', agent.model is not None)
print('Scaler loaded?', agent.scaler is not None)

rows = []
now = datetime.datetime.now()
price = 50000.0
for i in range(120):
    rows.append({'timestamp': now.isoformat()+'Z', 'open': price, 'high': price+5, 'low': price-5, 'close': price+((i%5)-2)*5, 'volume': 1000 + i*5})
    price = rows[-1]['close']

print('Sample predict:', agent.predict_for_symbol(rows))
