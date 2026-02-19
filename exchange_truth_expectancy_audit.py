#!/usr/bin/env python3
"""
Exchange Ground-Truth Expectancy Audit
=======================================

Connects directly to Binance Futures Testnet API to fetch actual trade history.
Bypasses all internal system logs and Redis data.

Provides exchange-verified statistical analysis.

Author: Forensic Exchange Auditor
Date: February 18, 2026
"""

import os
import sys
import json
import time
import hmac
import hashlib
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

# Binance Futures Testnet Configuration
BASE_URL = "https://testnet.binancefuture.com"
API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")

# Validate credentials
if not API_KEY or not API_SECRET:
    print("❌ ERROR: Missing Binance API credentials")
    print("   Set environment variables:")
    print("   - BINANCE_TESTNET_API_KEY")
    print("   - BINANCE_TESTNET_SECRET")
    sys.exit(1)


class BinanceAuditor:
    """Direct Binance API auditor"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.base_url = BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        
    def _sign(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make signed API request"""
        if params is None:
            params = {}
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Sign request
        params['signature'] = self._sign(params)
        
        # Make request
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=30)
            else:
                response = self.session.post(url, params=params, timeout=30)
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise
    
    def get_account_info(self) -> Dict:
        """Fetch account information"""
        return self._request('GET', '/fapi/v2/account')
    
    def get_all_orders(self, symbol: str = None, limit: int = 1000) -> List[Dict]:
        """Fetch all orders with pagination"""
        all_orders = []
        
        # Get list of symbols if not specified
        if symbol:
            symbols = [symbol]
        else:
            # Fetch all symbols from exchange info
            exchange_info = self._request('GET', '/fapi/v1/exchangeInfo', {})
            symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
            print(f"Fetching orders for {len(symbols)} symbols...")
        
        for sym in symbols:
            try:
                params = {
                    'symbol': sym,
                    'limit': limit
                }
                orders = self._request('GET', '/fapi/v1/allOrders', params)
                
                # Filter for filled orders only
                filled_orders = [o for o in orders if o['status'] == 'FILLED']
                all_orders.extend(filled_orders)
                
                if filled_orders:
                    print(f"  {sym}: {len(filled_orders)} filled orders")
                
                time.sleep(0.1)  # Rate limiting
            
            except Exception as e:
                print(f"⚠️  Failed to fetch orders for {sym}: {e}")
                continue
        
        return all_orders
    
    def get_income_history(self, income_type: str = 'REALIZED_PNL', limit: int = 1000) -> List[Dict]:
        """Fetch realized PnL history"""
        all_income = []
        
        params = {
            'incomeType': income_type,
            'limit': limit
        }
        
        try:
            income = self._request('GET', '/fapi/v1/income', params)
            all_income.extend(income)
            print(f"Fetched {len(income)} {income_type} records")
        
        except Exception as e:
            print(f"⚠️  Failed to fetch income history: {e}")
        
        return all_income
    
    def get_user_trades(self, symbol: str = None, limit: int = 1000) -> List[Dict]:
        """Fetch account trades"""
        all_trades = []
        
        # Get exchange info for symbols
        exchange_info = self._request('GET', '/fapi/v1/exchangeInfo', {})
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        
        if symbol:
            symbols = [symbol]
        
        print(f"Fetching user trades for {len(symbols)} symbols...")
        
        for sym in symbols:
            try:
                params = {
                    'symbol': sym,
                    'limit': limit
                }
                trades = self._request('GET', '/fapi/v1/userTrades', params)
                
                if trades:
                    all_trades.extend(trades)
                    print(f"  {sym}: {len(trades)} trades")
                
                time.sleep(0.1)  # Rate limiting
            
            except Exception as e:
                print(f"⚠️  Failed to fetch trades for {sym}: {e}")
                continue
        
        return all_trades


class TradeReconstructor:
    """Reconstruct complete trades from fills"""
    
    def __init__(self, orders: List[Dict], trades: List[Dict], income: List[Dict]):
        self.orders = sorted(orders, key=lambda x: x['time'])
        self.trades = sorted(trades, key=lambda x: x['time'])
        self.income = sorted(income, key=lambda x: x['time'])
        self.reconstructed_trades: List[Dict] = []
    
    def reconstruct(self):
        """Reconstruct complete trades from position entries and exits"""
        print("\n" + "="*80)
        print("PHASE 2: RECONSTRUCTING TRADES FROM FILLS")
        print("="*80)
        
        # Group orders by symbol
        orders_by_symbol = defaultdict(list)
        for order in self.orders:
            orders_by_symbol[order['symbol']].append(order)
        
        print(f"Processing {len(orders_by_symbol)} symbols...")
        
        total_reconstructed = 0
        
        for symbol, symbol_orders in orders_by_symbol.items():
            trades = self._reconstruct_symbol_trades(symbol, symbol_orders)
            self.reconstructed_trades.extend(trades)
            total_reconstructed += len(trades)
            
            if trades:
                print(f"  {symbol}: {len(trades)} complete trades")
        
        print(f"\n✅ Reconstructed {total_reconstructed} complete trades")
        
        return self.reconstructed_trades
    
    def _reconstruct_symbol_trades(self, symbol: str, orders: List[Dict]) -> List[Dict]:
        """Reconstruct trades for a single symbol"""
        trades = []
        
        # Track position state
        position_qty = 0.0
        position_side = None
        entry_fills = []
        
        for order in orders:
            qty = float(order['executedQty'])
            price = float(order['avgPrice'])
            side = order['side']  # BUY or SELL
            position_side_str = order.get('positionSide', 'BOTH')
            
            # Determine if this is entry or exit
            if position_side_str == 'LONG':
                # LONG position
                if side == 'BUY':
                    # Entry
                    entry_fills.append(order)
                    position_qty += qty
                    position_side = 'LONG'
                elif side == 'SELL':
                    # Exit
                    if position_qty > 0 and entry_fills:
                        # Complete trade
                        trade = self._create_trade(entry_fills, [order], symbol, 'LONG')
                        if trade:
                            trades.append(trade)
                        
                        # Update position
                        position_qty -= qty
                        if position_qty <= 0.0001:  # Effectively flat
                            entry_fills = []
                            position_qty = 0
                            position_side = None
            
            elif position_side_str == 'SHORT':
                # SHORT position
                if side == 'SELL':
                    # Entry
                    entry_fills.append(order)
                    position_qty += qty
                    position_side = 'SHORT'
                elif side == 'BUY':
                    # Exit
                    if position_qty > 0 and entry_fills:
                        # Complete trade
                        trade = self._create_trade(entry_fills, [order], symbol, 'SHORT')
                        if trade:
                            trades.append(trade)
                        
                        # Update position
                        position_qty -= qty
                        if position_qty <= 0.0001:  # Effectively flat
                            entry_fills = []
                            position_qty = 0
                            position_side = None
            
            else:
                # BOTH (hedge mode disabled) - use net position logic
                if side == 'BUY':
                    if position_qty < 0:  # Closing SHORT
                        if entry_fills:
                            trade = self._create_trade(entry_fills, [order], symbol, 'SHORT')
                            if trade:
                                trades.append(trade)
                        entry_fills = []
                        position_qty = 0
                    else:
                        entry_fills.append(order)
                        position_qty += qty
                        position_side = 'LONG'
                
                elif side == 'SELL':
                    if position_qty > 0:  # Closing LONG
                        if entry_fills:
                            trade = self._create_trade(entry_fills, [order], symbol, 'LONG')
                            if trade:
                                trades.append(trade)
                        entry_fills = []
                        position_qty = 0
                    else:
                        entry_fills.append(order)
                        position_qty -= qty
                        position_side = 'SHORT'
        
        return trades
    
    def _create_trade(self, entry_orders: List[Dict], exit_orders: List[Dict], 
                      symbol: str, side: str) -> Optional[Dict]:
        """Create trade record from entry and exit fills"""
        try:
            # Calculate weighted average entry
            total_entry_qty = sum(float(o['executedQty']) for o in entry_orders)
            total_entry_value = sum(float(o['executedQty']) * float(o['avgPrice']) for o in entry_orders)
            avg_entry_price = total_entry_value / total_entry_qty if total_entry_qty > 0 else 0
            
            # Calculate weighted average exit
            total_exit_qty = sum(float(o['executedQty']) for o in exit_orders)
            total_exit_value = sum(float(o['executedQty']) * float(o['avgPrice']) for o in exit_orders)
            avg_exit_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0
            
            # Use minimum qty (partial fills)
            trade_qty = min(total_entry_qty, total_exit_qty)
            
            # Calculate PnL
            if side == 'LONG':
                pnl = trade_qty * (avg_exit_price - avg_entry_price)
            else:  # SHORT
                pnl = trade_qty * (avg_entry_price - avg_exit_price)
            
            # Get commission
            commission = sum(float(o.get('commission', 0)) for o in entry_orders + exit_orders)
            
            # Net PnL
            net_pnl = pnl - commission
            
            # Get leverage from first entry order
            leverage = entry_orders[0].get('leverage', 1)
            
            # Timestamps
            entry_time = entry_orders[0]['time']
            exit_time = exit_orders[-1]['time']
            holding_time = (exit_time - entry_time) / 1000  # seconds
            
            return {
                'symbol': symbol,
                'side': side,
                'entry_price': avg_entry_price,
                'exit_price': avg_exit_price,
                'qty': trade_qty,
                'leverage': leverage,
                'realized_pnl': net_pnl,
                'gross_pnl': pnl,
                'commission': commission,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'holding_time_sec': holding_time,
                'entry_orders': len(entry_orders),
                'exit_orders': len(exit_orders)
            }
        
        except Exception as e:
            print(f"⚠️  Failed to create trade: {e}")
            return None


class ExpectancyAnalyzer:
    """Analyze expectancy from reconstructed trades"""
    
    def __init__(self, trades: List[Dict]):
        self.trades = trades
        self.metrics = {}
    
    def analyze(self) -> Dict:
        """Perform comprehensive analysis"""
        print("\n" + "="*80)
        print("PHASE 3: COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        if not self.trades:
            print("❌ No trades to analyze")
            return {}
        
        print(f"Analyzing {len(self.trades)} complete trades...")
        
        # Overall metrics
        self.metrics['overall'] = self._analyze_overall()
        
        # Per-symbol breakdown
        self.metrics['per_symbol'] = self._analyze_per_symbol()
        
        # Long vs Short
        self.metrics['long_vs_short'] = self._analyze_long_vs_short()
        
        # Distribution analysis
        self.metrics['distribution'] = self._analyze_distribution()
        
        # Tail risk analysis
        self.metrics['tail_risk'] = self._analyze_tail_risk()
        
        # Time-based analysis
        self.metrics['time_analysis'] = self._analyze_time_patterns()
        
        self._print_summary()
        
        return self.metrics
    
    def _analyze_overall(self) -> Dict:
        """Overall performance metrics"""
        pnl_values = [t['realized_pnl'] for t in self.trades]
        winners = [p for p in pnl_values if p > 0]
        losers = [p for p in pnl_values if p < 0]
        
        total_trades = len(self.trades)
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        total_pnl = sum(pnl_values)
        avg_pnl = np.mean(pnl_values)
        expectancy = avg_pnl
        
        std_dev = np.std(pnl_values) if len(pnl_values) > 1 else 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified, assuming 0 risk-free rate)
        sharpe = avg_pnl / std_dev if std_dev > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_win_usdt': avg_win,
            'avg_loss_usdt': avg_loss,
            'total_pnl_usdt': total_pnl,
            'avg_pnl_usdt': avg_pnl,
            'expectancy_usdt': expectancy,
            'profit_factor': profit_factor,
            'std_dev': std_dev,
            'max_drawdown_usdt': max_drawdown,
            'sharpe_ratio': sharpe,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _analyze_per_symbol(self) -> Dict:
        """Per-symbol performance"""
        by_symbol = defaultdict(list)
        for trade in self.trades:
            by_symbol[trade['symbol']].append(trade['realized_pnl'])
        
        results = {}
        for symbol, pnls in by_symbol.items():
            winners = [p for p in pnls if p > 0]
            results[symbol] = {
                'trades': len(pnls),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'win_rate': len(winners) / len(pnls) if pnls else 0,
                'best_trade': max(pnls) if pnls else 0,
                'worst_trade': min(pnls) if pnls else 0
            }
        
        return results
    
    def _analyze_long_vs_short(self) -> Dict:
        """Long vs Short performance"""
        longs = [t for t in self.trades if t['side'] == 'LONG']
        shorts = [t for t in self.trades if t['side'] == 'SHORT']
        
        def analyze_side(trades):
            if not trades:
                return {}
            pnls = [t['realized_pnl'] for t in trades]
            winners = [p for p in pnls if p > 0]
            return {
                'trades': len(trades),
                'win_count': len(winners),
                'win_rate': len(winners) / len(trades),
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'std_dev': np.std(pnls)
            }
        
        return {
            'LONG': analyze_side(longs),
            'SHORT': analyze_side(shorts)
        }
    
    def _analyze_distribution(self) -> Dict:
        """PnL distribution"""
        pnls = [t['realized_pnl'] for t in self.trades]
        
        bins = [-1000, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 1000]
        hist, edges = np.histogram(pnls, bins=bins)
        
        histogram = {}
        for i in range(len(hist)):
            bin_label = f"{edges[i]:.0f}_to_{edges[i+1]:.0f}"
            histogram[bin_label] = int(hist[i])
        
        return {
            'histogram': histogram,
            'median': float(np.median(pnls)),
            'p10': float(np.percentile(pnls, 10)),
            'p25': float(np.percentile(pnls, 25)),
            'p75': float(np.percentile(pnls, 75)),
            'p90': float(np.percentile(pnls, 90)),
            'min': float(np.min(pnls)),
            'max': float(np.max(pnls))
        }
    
    def _analyze_tail_risk(self) -> Dict:
        """Tail loss analysis"""
        pnls = [t['realized_pnl'] for t in self.trades]
        losers = [p for p in pnls if p < 0]
        
        if not losers:
            return {'tail_losses': 0, 'tail_loss_pct': 0}
        
        # Define tail as losses > 2 std devs from mean loss
        mean_loss = np.mean(losers)
        std_loss = np.std(losers)
        tail_threshold = mean_loss - 2 * std_loss
        
        tail_losses = [p for p in losers if p < tail_threshold]
        
        return {
            'tail_threshold': float(tail_threshold),
            'tail_losses': len(tail_losses),
            'tail_loss_pct': len(tail_losses) / len(self.trades) * 100,
            'worst_loss': float(min(losers)),
            'avg_tail_loss': float(np.mean(tail_losses)) if tail_losses else 0
        }
    
    def _analyze_time_patterns(self) -> Dict:
        """Time-based patterns"""
        holding_times = [t['holding_time_sec'] for t in self.trades]
        
        return {
            'avg_holding_time_sec': float(np.mean(holding_times)),
            'median_holding_time_sec': float(np.median(holding_times)),
            'min_holding_time_sec': float(np.min(holding_times)),
            'max_holding_time_sec': float(np.max(holding_times))
        }
    
    def _print_summary(self):
        """Print summary table"""
        overall = self.metrics['overall']
        
        print(f"\n{'='*80}")
        print("OVERALL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Metric':<40} {'Value':>20}")
        print("-" * 65)
        print(f"{'Total Trades':<40} {overall['total_trades']:>20,}")
        print(f"{'Winning Trades':<40} {overall['win_count']:>20,}")
        print(f"{'Losing Trades':<40} {overall['loss_count']:>20,}")
        print(f"{'Win Rate':<40} {overall['win_rate']:>19.2%}")
        print(f"{'─'*65}")
        print(f"{'Average Win (USDT)':<40} {overall['avg_win_usdt']:>20.2f}")
        print(f"{'Average Loss (USDT)':<40} {overall['avg_loss_usdt']:>20.2f}")
        print(f"{'Total PnL (USDT)':<40} {overall['total_pnl_usdt']:>20.2f}")
        print(f"{'Average PnL per Trade (USDT)':<40} {overall['avg_pnl_usdt']:>20.2f}")
        print(f"{'─'*65}")
        print(f"{'Profit Factor':<40} {overall['profit_factor']:>20.2f}")
        print(f"{'Expectancy (USDT/trade)':<40} {overall['expectancy_usdt']:>20.2f}")
        print(f"{'Standard Deviation':<40} {overall['std_dev']:>20.2f}")
        print(f"{'Max Drawdown (USDT)':<40} {overall['max_drawdown_usdt']:>20.2f}")
        print(f"{'Sharpe Ratio':<40} {overall['sharpe_ratio']:>20.2f}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("EXCHANGE GROUND-TRUTH EXPECTANCY AUDIT")
    print("Binance Futures Testnet - Direct API Query")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize auditor
    auditor = BinanceAuditor()
    
    # PHASE 1: Fetch data
    print("\n" + "="*80)
    print("PHASE 1: FETCHING HISTORICAL DATA FROM EXCHANGE")
    print("="*80)
    
    try:
        # Get account info
        print("\nFetching account information...")
        account = auditor.get_account_info()
        print(f"✅ Account balance: {account['totalWalletBalance']} USDT")
        
        # Fetch orders
        print("\nFetching all filled orders...")
        orders = auditor.get_all_orders()
        print(f"✅ Fetched {len(orders)} filled orders")
        
        # Fetch income
        print("\nFetching realized PnL history...")
        income = auditor.get_income_history()
        print(f"✅ Fetched {len(income)} realized PnL records")
        
        # Fetch user trades
        print("\nFetching account trades...")
        user_trades = auditor.get_user_trades()
        print(f"✅ Fetched {len(user_trades)} user trades")
        
    except Exception as e:
        print(f"\n❌ FAILED TO FETCH DATA: {e}")
        sys.exit(1)
    
    # PHASE 2: Reconstruct trades
    reconstructor = TradeReconstructor(orders, user_trades, income)
    trades = reconstructor.reconstruct()
    
    if not trades:
        print("\n❌ No complete trades found")
        print("   System may not have closed any positions yet")
        sys.exit(1)
    
    # PHASE 3: Analyze
    analyzer = ExpectancyAnalyzer(trades)
    metrics = analyzer.analyze()
    
    # PHASE 4: Output
    print("\n" + "="*80)
    print("PHASE 4: GENERATING REPORT")
    print("="*80)
    
    output = {
        'audit_date': datetime.now().isoformat(),
        'data_source': 'Binance Futures Testnet API',
        'total_trades_analyzed': len(trades),
        'account_balance_usdt': float(account['totalWalletBalance']),
        'metrics': {
            'overall': metrics['overall'],
            'per_symbol': metrics['per_symbol'],
            'long_vs_short': metrics['long_vs_short'],
            'distribution': metrics['distribution'],
            'tail_risk': metrics['tail_risk'],
            'time_patterns': metrics['time_analysis']
        },
        'raw_trades': trades
    }
    
    filename = "exchange_truth_expectancy_report.json"
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Report saved to: {filename}")
    print(f"   Total trades analyzed: {len(trades)}")
    print(f"   Data source: Exchange API (ground truth)")
    
    # Print verdict
    print("\n" + "="*80)
    print("EXCHANGE-VERIFIED VERDICT")
    print("="*80)
    
    overall = metrics['overall']
    expectancy = overall['expectancy_usdt']
    profit_factor = overall['profit_factor']
    win_rate = overall['win_rate']
    
    if len(trades) < 30:
        print(f"\n⚠️  SAMPLE SIZE WARNING: Only {len(trades)} trades")
        print("   Minimum 200 trades recommended for statistical significance")
    
    if expectancy > 1.0:
        print("\n✅ POSITIVE EXPECTANCY (Exchange-Verified)")
    elif expectancy > -1.0:
        print("\n⚠️  NEUTRAL EXPECTANCY (Exchange-Verified)")
    else:
        print("\n❌ NEGATIVE EXPECTANCY (Exchange-Verified)")
    
    print(f"\nExpectancy: ${expectancy:.2f} USDT per trade")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total PnL: ${overall['total_pnl_usdt']:.2f} USDT")
    
    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
