#!/usr/bin/env python3
"""
Emergency SimpleCLM Verification Script
Injects synthetic trade.closed events to Redis for CLM testing when testnet has $0 balance.
"""

import redis
from datetime import datetime, timezone, timedelta
import random

# Connect to Redis
r = redis.from_url("redis://localhost:6379", decode_responses=True)

def inject_synthetic_trade(symbol: str, side: str, win: bool):
    """
    Inject a synthetic trade.closed event with realistic parameters.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        side: "BUY" or "SELL"
        win: True for winning trade, False for losing trade
    """
    # Generate realistic PnL
    if win:
        pnl_percent = random.uniform(0.6, 2.5)  # 0.6% to 2.5% win
    else:
        pnl_percent = random.uniform(-2.0, -0.6)  # -2.0% to -0.6% loss
    
    # Random entry price (for test purposes)
    base_price = random.uniform(20000, 70000) if "BTC" in symbol else random.uniform(100, 500)
    entry_price = base_price
    exit_price = entry_price * (1 + pnl_percent / 100)
    
    # Random trade duration (5-30 minutes)
    duration_mins = random.randint(5, 30)
    entry_time = datetime.now(timezone.utc) - timedelta(minutes=duration_mins)
    exit_time = datetime.now(timezone.utc)
    
    # Build event matching SimpleCLM expectations
    event_data = {
        "event_type": "trade.closed",
        "symbol": symbol,
        "side": side,
        "entry_price": str(entry_price),
        "exit_price": str(exit_price),
        "pnl_percent": str(pnl_percent),
        "confidence": str(random.uniform(0.65, 0.85)),  # Realistic confidence
        "timestamp": exit_time.isoformat(),
        "entry_timestamp": entry_time.isoformat(),
        "model": "ensemble",
        "leverage": "5",
        "position_size_usd": "50.0",
        "exit_reason": "TP" if win else "SL",
        "trace_id": f"{symbol}_{entry_time.isoformat()}",
    }
    
    # Publish to trade.closed stream
    stream_id = r.xadd(
        "quantum:stream:trade.closed",
        event_data,
        maxlen=1000
    )
    
    result = "WIN" if win else "LOSS"
    print(f"✅ Injected {symbol} {side} → {result} (PnL={pnl_percent:+.2f}%) | Stream ID: {stream_id}")
    return stream_id


def main():
    print("="*80)
    print("SimpleCLM Test Data Injector")
    print("="*80)
    print()
    
    # Inject a mix of winning and losing trades
    test_trades = [
        ("BTCUSDT", "BUY", True),   # Win
        ("ETHUSDT", "SELL", False),  # Loss
        ("BTCUSDT", "SELL", True),   # Win
        ("XRPUSDT", "BUY", True),    # Win
        ("ETHUSDT", "BUY", False),   # Loss
        ("BTCUSDT", "BUY", False),   # Loss
        ("SOLUSDT", "SELL", True),   # Win
        ("DOGEUSDT", "BUY", True),   # Win
    ]
    
    print(f"Injecting {len(test_trades)} synthetic trades...\n")
    
    for symbol, side, win in test_trades:
        inject_synthetic_trade(symbol, side, win)
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.1)
    
    print()
    print("="*80)
    print("Injection complete!")
    print()
    
    # Check stream length
    stream_len = r.xlen("quantum:stream:trade.closed")
    print(f"📊 trade.closed stream length: {stream_len} events")
    print()
    print("Expected outcome:")
    print("  1. SimpleCLM should pick up these events")
    print("  2. File created: /home/qt/quantum_trader/data/clm_trades.jsonl")
    print("  3. Check AI Engine logs for [sCLM] messages")
    print()
    print("Verify with:")
    print("  journalctl -u quantum-ai-engine -f | grep sCLM")
    print("  cat /home/qt/quantum_trader/data/clm_trades.jsonl | jq")
    print("="*80)


if __name__ == "__main__":
    main()
