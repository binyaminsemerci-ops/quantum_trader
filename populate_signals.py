#!/usr/bin/env python3
"""
Populate live_signals with real symbols from trading bot
"""
import redis
import json

r = redis.Redis(host='redis', port=6379, decode_responses=True)

# Based on trading bot's recent signals - diverse portfolio
signals = [
    {"symbol": "BTCUSDT", "action": "BUY", "confidence": 0.78, "price": 88111.0, "pnl": 0.0, "drawdown": 0.5},
    {"symbol": "ETHUSDT", "action": "SELL", "confidence": 0.65, "price": 2973.0, "pnl": 0.0, "drawdown": 0.3},
    {"symbol": "SOLUSDT", "action": "BUY", "confidence": 0.72, "price": 180.0, "pnl": 0.0, "drawdown": 0.4},
    {"symbol": "BNBUSDT", "action": "BUY", "confidence": 0.68, "price": 630.0, "pnl": 0.0, "drawdown": 0.3},
    {"symbol": "XRPUSDT", "action": "SELL", "confidence": 0.61, "price": 0.62, "pnl": 0.0, "drawdown": 0.2},
    {"symbol": "ADAUSDT", "action": "BUY", "confidence": 0.66, "price": 0.88, "pnl": 0.0, "drawdown": 0.3},
    {"symbol": "DOTUSDT", "action": "SELL", "confidence": 0.58, "price": 1.83, "pnl": 0.0, "drawdown": 0.2},
    {"symbol": "MATICUSDT", "action": "BUY", "confidence": 0.63, "price": 0.67, "pnl": 0.0, "drawdown": 0.3},
    {"symbol": "AVAXUSDT", "action": "BUY", "confidence": 0.69, "price": 35.50, "pnl": 0.0, "drawdown": 0.4},
    {"symbol": "ATOMUSDT", "action": "BUY", "confidence": 0.59, "price": 1.99, "pnl": 0.0, "drawdown": 0.2},
]

r.set('live_signals', json.dumps(signals))
print(f"✅ Updated live_signals with {len(signals)} diversified signals")
for sig in signals:
    print(f"   {sig['symbol']}: {sig['action']} conf={sig['confidence']:.0%}")
