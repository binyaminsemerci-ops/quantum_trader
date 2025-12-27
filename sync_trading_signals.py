#!/usr/bin/env python3
"""
Sync trading signals from bot logs to executor
This bridges the gap between trading bot output and auto executor input
"""
import redis
import json
import re
import subprocess
import time

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def extract_signals_from_logs():
    """Extract signals from trading bot logs"""
    try:
        result = subprocess.run(
            ['docker', 'logs', 'quantum_trading_bot', '--tail', '500'],
            capture_output=True,
            text=True
        )
        
        logs = result.stdout + result.stderr
        signals = []
        
        # Pattern: Signal: BTCUSDT BUY @ $50000 (confidence=78.00%, size=$150)
        pattern = r'Signal: (\w+) (BUY|SELL) @ \$([0-9.]+) \(confidence=([0-9.]+)%'
        
        for match in re.finditer(pattern, logs):
            symbol, action, price, confidence = match.groups()
            
            signal = {
                "symbol": symbol,
                "action": action,
                "confidence": float(confidence) / 100,  # Convert to 0-1
                "price": float(price),
                "pnl": 0.0,
                "drawdown": 0.0
            }
            
            # Only add if confidence meets threshold
            if signal['confidence'] >= 0.55:
                signals.append(signal)
        
        # Remove duplicates, keep last 20 unique signals
        unique_signals = {}
        for sig in reversed(signals):
            key = f"{sig['symbol']}_{sig['action']}"
            if key not in unique_signals:
                unique_signals[key] = sig
        
        return list(unique_signals.values())[:20]
        
    except Exception as e:
        print(f"Error extracting signals: {e}")
        return []

def update_live_signals():
    """Update Redis with latest signals"""
    signals = extract_signals_from_logs()
    
    if signals:
        r.set('live_signals', json.dumps(signals))
        print(f"✅ Updated live_signals with {len(signals)} signals:")
        for sig in signals[:5]:
            print(f"   - {sig['symbol']} {sig['action']} @ ${sig['price']} (conf={sig['confidence']:.2%})")
        if len(signals) > 5:
            print(f"   ... and {len(signals) - 5} more")
    else:
        print("⚠️  No signals extracted")

if __name__ == '__main__':
    print("🔄 Syncing trading signals...")
    update_live_signals()
