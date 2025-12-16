#!/usr/bin/env python3
"""Diagnose all issues preventing order execution."""
import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.database import SessionLocal
from sqlalchemy import text
from backend.services.execution.execution import BinanceFuturesExecutionAdapter

def check_skipped_orders():
    """Check execution journal for skipped orders."""
    print("\n" + "="*60)
    print("[CHART] CHECKING SKIPPED ORDERS")
    print("="*60)
    
    session = SessionLocal()
    try:
        result = session.execute(
            text("""
                SELECT symbol, side, status, reason, created_at 
                FROM execution_journal 
                WHERE status = 'skipped' 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
        )
        rows = result.fetchall()
        
        if rows:
            print(f"\n❌ Found {len(rows)} skipped orders:")
            for row in rows:
                print(f"  • {row.created_at}: {row.symbol} {row.side}")
                print(f"    Reason: {row.reason}")
        else:
            print("\n[OK] No skipped orders in execution journal")
    finally:
        session.close()

async def check_binance_balance():
    """Check actual Binance Futures balance."""
    print("\n" + "="*60)
    print("[MONEY] CHECKING BINANCE FUTURES BALANCE")
    print("="*60)
    
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("\n[WARNING]  WARNING: No Binance API credentials found!")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    adapter = BinanceFuturesExecutionAdapter(
        api_key=api_key,
        api_secret=api_secret,
        quote='USDC',
        market_type='usdm_perp'
    )
    
    try:
        cash = await adapter.get_cash_balance()
        positions = await adapter.get_positions()
        
        print(f"\n💵 Available cash: {cash:.2f} USDC")
        print(f"[CHART_UP] Open positions: {len(positions)}")
        
        if positions:
            total_notional = 0.0
            for sym, qty in positions.items():
                print(f"  • {sym}: {qty}")
                # Approximate notional (would need prices for exact)
                total_notional += abs(qty)
            print(f"\n[BRIEFCASE] Total equity: ~{cash + total_notional:.2f} USDC")
        else:
            print(f"\n[BRIEFCASE] Total equity: {cash:.2f} USDC (no open positions)")
        
        # Analyze if cash is sufficient for trading
        if cash < 5.0:
            print(f"\n[WARNING]  WARNING: Cash balance ({cash:.2f} USDC) is below min_notional (5.0 USDC)")
            print("Orders will be skipped due to insufficient funds!")
            print("\n💡 SOLUTION: Deposit more USDC to your Binance Futures account")
        else:
            print(f"\n[OK] Cash balance is sufficient for trading (>= 5.0 USDC)")
            
    except Exception as e:
        print(f"\n❌ ERROR checking Binance: {e}")
        import traceback
        traceback.print_exc()

def check_risk_config():
    """Check risk guard configuration."""
    print("\n" + "="*60)
    print("[SHIELD]  CHECKING RISK GUARD CONFIGURATION")
    print("="*60)
    
    from backend.config.risk import load_risk_config
    
    try:
        config = load_risk_config()
        print(f"\n[OK] Risk configuration loaded:")
        print(f"  • Max notional per trade: {config.max_notional_per_trade} USDC")
        print(f"  • Max position per symbol: {config.max_position_per_symbol} USDC")
        print(f"  • Max gross exposure: {config.max_gross_exposure} USDC")
        print(f"  • Max daily loss: {config.max_daily_loss} USDC")
        print(f"  • Kill switch: {config.kill_switch}")
        
        # Check environment variables
        print(f"\n📌 Environment variables:")
        print(f"  • QT_EXECUTION_MIN_NOTIONAL: {os.getenv('QT_EXECUTION_MIN_NOTIONAL', 'not set')}")
        print(f"  • QT_MAX_NOTIONAL_PER_TRADE: {os.getenv('QT_MAX_NOTIONAL_PER_TRADE', 'not set')}")
        
    except Exception as e:
        print(f"\n❌ ERROR loading risk config: {e}")

def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("[SEARCH] QUANTUM TRADER - FULL DIAGNOSTICS")
    print("="*60)
    
    # 1. Check skipped orders
    check_skipped_orders()
    
    # 2. Check Binance balance
    asyncio.run(check_binance_balance())
    
    # 3. Check risk configuration
    check_risk_config()
    
    print("\n" + "="*60)
    print("[OK] DIAGNOSTICS COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
