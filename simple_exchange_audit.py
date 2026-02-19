#!/usr/bin/env python3
"""
Simple Exchange Trade Fetcher
Uses existing backend Binance client configuration
"""
import sys
import os

# Add backend to path
sys.path.insert(0, '/root/quantum_trader/backend')

import json
from datetime import datetime
import numpy as np

def fetch_account_trades():
    """Fetch all trades using existing backend infrastructure"""
    print("Initializing Binance client from backend config...")
    
    from utils.exchanges import get_exchange_client
    
    # Get API credentials from environment
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in environment")
        return None
    
    print(f"✅ Found API credentials (key: {api_key[:10]}...)")
    
    try:
        exchange = get_exchange_client(
            name='binance',
            api_key=api_key,
            api_secret=api_secret
        )
        print(f"✅ Exchange adapter initialized")
        print(f"   Adapter type: {type(exchange).__name__}")
        
        # Access the underlying Binance client
        client = exchange._client
        print(f"   Client type: {type(client).__name__}")
        
        # Force testnet URL (backend uses testnet)
        if client and hasattr(client, 'API_URL'):
            client.API_URL = 'https://testnet.binancefuture.com'
            print(f"   Endpoint: {client.API_URL}")
        
        # Get account information
        print("\nFetching account info...")
        account = client.futures_account()
        
        if not account:
            print("❌ Could not fetch account info")
            return None
        
        print(f"✅ Account fetched")
        print(f"   Total Wallet Balance: {account.get('totalWalletBalance', 'N/A')} USDT")
        print(f"   Unrealized PnL: {account.get('totalUnrealizedProfit', 'N/A')} USDT")
        
        # Get all positions to find symbols
        positions = account.get('positions', [])
        symbols_with_activity = set()
        
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol:
                symbols_with_activity.add(symbol)
        
        print(f"Found {len(symbols_with_activity)} symbols with positions")
        
        # Fetch trades for each symbol
        all_trades = []
        all_income = []
        
        for symbol in sorted(symbols_with_activity):
            print(f"\nFetching trades for {symbol}...")
            
            try:
                # Get all trades for this symbol
                trades = client.futures_account_trades(symbol=symbol, limit=1000)
                print(f"   {len(trades)} fills")
                all_trades.extend(trades)
                
                # Get income history for this symbol
                income = client.futures_income_history(symbol=symbol, limit=1000)
                print(f"   {len(income)} income records")
                all_income.extend(income)
                
            except Exception as e:
                print(f"   Error: {e}")
                continue
        
        print(f"\n✅ Total trades fetched: {len(all_trades)}")
        print(f"✅ Total income records: {len(all_income)}")
        
        # Save raw dataset
        dataset = {
            'fetch_time': datetime.now().isoformat(),
            'account_info': account,
            'trades': all_trades,
            'income': all_income
        }
        
        with open('exchange_raw_data.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Raw data saved to exchange_raw_data.json")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def quick_analysis(dataset):
    """Perform quick analysis on fetched data"""
    if not dataset or not dataset.get('trades'):
        print("\n❌ No trades to analyze")
        return
    
    trades = dataset['trades']
    
    print(f"\n{'='*80}")
    print("QUICK ANALYSIS")
    print(f"{'='*80}")
    
    # Group by symbol
    by_symbol = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(trade)
    
    print(f"\nTotal symbols traded: {len(by_symbol)}")
    print(f"Total fills: {len(trades)}")
    
    for symbol, symbol_trades in sorted(by_symbol.items()):
        total_qty = sum(float(t['qty']) for t in symbol_trades)
        total_commission = sum(float(t['commission']) for t in symbol_trades)
        realized_pnl = sum(float(t.get('realizedPnl', 0)) for t in symbol_trades)
        
        print(f"\n{symbol}:")
        print(f"   Fills: {len(symbol_trades)}")
        print(f"   Total qty traded: {total_qty:.4f}")
        print(f"   Total commission: ${total_commission:.2f}")
        print(f"   Realized PnL: ${realized_pnl:.2f}")
    
    # Income analysis
    income = dataset.get('income', [])
    if income:
        print(f"\n{'='*80}")
        print("INCOME HISTORY")
        print(f"{'='*80}")
        
        income_types = {}
        for inc in income:
            inc_type = inc.get('incomeType', 'UNKNOWN')
            amount = float(inc.get('income', 0))
            
            if inc_type not in income_types:
                income_types[inc_type] = 0
            income_types[inc_type] += amount
        
        for inc_type, total in sorted(income_types.items()):
            print(f"{inc_type}: ${total:.2f}")

if __name__ == "__main__":
    dataset = fetch_account_trades()
    if dataset:
        quick_analysis(dataset)
