import asyncio
import sys
sys.path.insert(0, '/app')

from backend.services.execution.execution import BinanceExecution

async def main():
    ex = BinanceExecution()
    positions = await ex.get_open_positions()
    
    print(f"\n=== ÅPNE POSISJONER: {len(positions)} ===\n")
    
    total_notional = 0
    for p in positions:
        symbol = p["symbol"]
        amt = float(p["positionAmt"])
        notional = abs(float(p["notional"]))
        pnl = float(p["unrealizedProfit"])
        pnl_pct = (pnl / notional * 100) if notional > 0 else 0
        
        total_notional += notional
        
        print(f"{symbol:12} | Notional: ${notional:>8.2f} | PnL: ${pnl:>7.2f} ({pnl_pct:>6.2f}%)")
    
    print(f"\n{'TOTAL':12} | Notional: ${total_notional:>8.2f}\n")

if __name__ == "__main__":
    asyncio.run(main())
