#!/usr/bin/env python3
"""
Quick check of trades database
"""
import asyncio
import sys
sys.path.insert(0, '/app')

from backend.core.trading import get_trade_store

async def main():
    print("Initializing TradeStore...")
    store = await get_trade_store()
    print(f"✓ TradeStore type: {type(store).__name__}")
    
    # Get all trades
    all_trades = await store.get_all_trades()
    print(f"\n[TRADES] Total in database: {len(all_trades)}")
    
    if len(all_trades) > 0:
        print("\nMost recent trades:")
        for trade in all_trades[-5:]:
            print(f"  - {trade.symbol} {trade.side.value} @ {trade.entry_time}")
            print(f"    Status: {trade.status.value}, Entry: ${trade.entry_price}")
            if trade.exit_price:
                print(f"    Exit: ${trade.exit_price}, PnL: {trade.pnl_usd:.2f} USD")
    else:
        print("\n[WARNING] No trades found in database yet!")
        print("This is expected if no trades have been opened since backend restart.")
    
    # Close the store
    await store.close()
    print("\n[OK] TradeStore check complete")

if __name__ == "__main__":
    asyncio.run(main())
