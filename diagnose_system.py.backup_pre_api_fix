#!/usr/bin/env python3
"""
System Diagnosis - Check trading, learning, and profit-taking
"""
import sys
sys.path.insert(0, '/app')
import asyncio
from datetime import datetime, timedelta
from backend.adapters.binance_futures_client import BinanceFuturesClient
from backend.core.trading import get_trade_store

async def main():
    print("=" * 70)
    print("QUANTUM TRADER SYSTEM DIAGNOSIS")
    print("=" * 70)
    
    # 1. Check open positions
    print("\n1. OPEN POSITIONS:")
    client = BinanceFuturesClient()
    positions = await client.get_position_risk()
    open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    if open_positions:
        print(f"   Found {len(open_positions)} open positions:")
        for p in open_positions:
            symbol = p['symbol']
            amt = float(p['positionAmt'])
            entry = float(p['entryPrice'])
            mark = float(p['markPrice'])
            pnl = float(p['unRealizedProfit'])
            pnl_pct = ((mark - entry) / entry * 100) if amt > 0 else ((entry - mark) / entry * 100)
            
            print(f"   {symbol}: {amt:+.4f} @ ${entry:.4f}")
            print(f"      Mark: ${mark:.4f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    else:
        print("   No open positions")
    
    # 2. Check recent orders
    print("\n2. RECENT ORDERS (last 24h):")
    if open_positions:
        symbol = open_positions[0]['symbol']
        orders = await client.get_all_orders(symbol=symbol, limit=10)
        recent = [o for o in orders if 
                 datetime.fromtimestamp(o['updateTime']/1000) > datetime.now() - timedelta(hours=24)]
        
        if recent:
            print(f"   Found {len(recent)} orders in last 24h for {symbol}:")
            for o in recent[-5:]:
                time_str = datetime.fromtimestamp(o['updateTime']/1000).strftime("%H:%M:%S")
                side = o['side']
                order_type = o['type']
                status = o['status']
                price = float(o.get('price', 0) or o.get('avgPrice', 0))
                qty = float(o['executedQty'])
                
                print(f"   [{time_str}] {side} {order_type} - ${price:.4f} x {qty:.4f} - {status}")
        else:
            print(f"   No orders in last 24h for {symbol}")
    
    # 3. Check TradeStore
    print("\n3. TRADESTORE DATABASE:")
    try:
        store = await get_trade_store()
        stats = await store.get_stats()
        
        print(f"   Backend: {stats['backend']}")
        print(f"   Total trades: {stats['total_trades']}")
        print(f"   Open: {stats['open_trades']} | Closed: {stats['closed_trades']}")
        print(f"   Total PnL: ${stats['total_pnl_usd']:.2f}")
        
        if stats['total_trades'] > 0:
            open_trades = await store.get_open_trades()
            if open_trades:
                print(f"\n   Open trades in database:")
                for t in open_trades[:3]:
                    print(f"   - {t.symbol} {t.side.value}: ${t.entry_price:.4f} @ {t.entry_time}")
        
        await store.close()
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # 4. Check account balance
    print("\n4. ACCOUNT STATUS:")
    account = await client.get_account()
    balance = float(account['totalWalletBalance'])
    unrealized = float(account['totalUnrealizedProfit'])
    margin = float(account['totalMarginBalance'])
    available = float(account['availableBalance'])
    
    print(f"   Wallet Balance: ${balance:.2f}")
    print(f"   Unrealized PnL: ${unrealized:+.2f}")
    print(f"   Margin Balance: ${margin:.2f}")
    print(f"   Available: ${available:.2f}")
    print(f"   Used: ${margin - available:.2f}")
    
    await client.close()
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
