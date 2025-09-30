import datetime

from scripts.import_helper import import_module

make_default_agent = import_module("ai_engine.agents.xgb_agent", "make_default_agent")

agent = make_default_agent()
print("Model loaded?", agent.model is not None)
print("Scaler loaded?", agent.scaler is not None)

rows = []
now = datetime.datetime.now()
price = 50000.0
for i in range(120):
    rows.append(
        {
            "timestamp": now.isoformat() + "Z",
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price + ((i % 5) - 2) * 5,
            "volume": 1000 + i * 5,
        }
    )
    price = rows[-1]["close"]

print("Sample predict:", agent.predict_for_symbol(rows))
