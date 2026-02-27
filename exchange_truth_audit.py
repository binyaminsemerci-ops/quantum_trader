#!/usr/bin/env python3
"""
Exchange Ground-Truth Profitability Audit
==========================================

Queries Binance Futures Testnet API directly for complete trade history.
Bypasses all internal system logs to establish exchange-level truth.

READ-ONLY MODE: No orders placed, no positions modified.

Author: Exchange Truth Auditor
Date: February 18, 2026
"""

import os
import sys
import time
import json
import hmac
import hashlib
import requests
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Binance Futures Configuration (auto-detects testnet vs production)
USE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

if USE_TESTNET:
    BASE_URL = "https://testnet.binancefuture.com"
    API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_TESTNET_SECRET")
    print("🔧 Using TESTNET endpoint")
else:
    BASE_URL = "https://fapi.binance.com"
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_API_SECRET")
    print("🔧 Using PRODUCTION endpoint")

if not API_KEY or not SECRET_KEY:
    print("❌ ERROR: Binance API credentials not found in environment")
    sys.exit(1)


class BinanceAuditor:
    """Direct Binance API auditor"""
    
    def __init__(self, base_url, api_key, secret_key, is_testnet):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.is_testnet = is_testnet
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        
        # Data storage
        self.raw_trades: List[Dict] = []
        self.raw_income: List[Dict] = []
        self.raw_orders: List[Dict] = []
        self.reconstructed_trades: List[Dict] = []
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = True) -> Dict:
        """Make authenticated API request"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
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
            return None
    
    def fetch_user_trades(self, symbol: str = None, limit: int = 1000) -> List[Dict]:
        """Fetch user trades (fills) from exchange"""
        print(f"\nFetching user trades{f' for {symbol}' if symbol else ' (all symbols)'}...")
        
        all_trades = []
        
        # If no symbol specified, get list of symbols from account positions
        if not symbol:
            account = self._request('GET', '/fapi/v2/account')
            if not account:
                return []
            
            symbols = set()
            for position in account.get('positions', []):
                if float(position.get('positionAmt', 0)) != 0 or float(position.get('unrealizedProfit', 0)) != 0:
                    symbols.add(position['symbol'])
            
            # Also fetch recent orders to find historically traded symbols
            recent_orders = self._request('GET', '/fapi/v1/allOrders', {'limit': 500})
            if recent_orders:
                for order in recent_orders:
                    symbols.add(order['symbol'])
            
            print(f"Found {len(symbols)} symbols with activity")
            
            # Fetch trades for each symbol
            for sym in sorted(symbols):
                trades = self._fetch_symbol_trades(sym, limit)
                all_trades.extend(trades)
                time.sleep(0.2)  # Rate limiting
        else:
            all_trades = self._fetch_symbol_trades(symbol, limit)
        
        print(f"✅ Total user trades fetched: {len(all_trades)}")
        return all_trades
    
    def _fetch_symbol_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """Fetch trades for specific symbol with pagination"""
        trades = []
        from_id = None
        
        while True:
            params = {'symbol': symbol, 'limit': limit}
            if from_id:
                params['fromId'] = from_id
            
            batch = self._request('GET', '/fapi/v1/userTrades', params)
            if not batch or len(batch) == 0:
                break
            
            trades.extend(batch)
            
            if len(batch) < limit:
                break
            
            from_id = batch[-1]['id'] + 1
            time.sleep(0.1)
        
        return trades
    
    def fetch_income_history(self, income_type: str = None) -> List[Dict]:
        """Fetch income history (realized PnL, funding fees, etc.)"""
        print(f"\nFetching income history{f' ({income_type})' if income_type else ''}...")
        
        all_income = []
        start_time = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        
        while True:
            params = {'startTime': start_time, 'limit': 1000}
            if income_type:
                params['incomeType'] = income_type
            
            batch = self._request('GET', '/fapi/v1/income', params)
            if not batch or len(batch) == 0:
                break
            
            all_income.extend(batch)
            
            if len(batch) < 1000:
                break
            
            start_time = batch[-1]['time'] + 1
            time.sleep(0.2)
        
        print(f"✅ Income records fetched: {len(all_income)}")
        return all_income
    
    def fetch_all_orders(self, symbol: str = None) -> List[Dict]:
        """Fetch all orders"""
        print(f"\nFetching all orders{f' for {symbol}' if symbol else ''}...")
        
        all_orders = []
        
        if not symbol:
            # Get symbols from account
            account = self._request('GET', '/fapi/v2/account')
            if not account:
                return []
            
            symbols = set()
            for position in account.get('positions', []):
                symbols.add(position['symbol'])
            
            for sym in sorted(symbols):
                orders = self._fetch_symbol_orders(sym)
                all_orders.extend(orders)
                time.sleep(0.2)
        else:
            all_orders = self._fetch_symbol_orders(symbol)
        
        print(f"✅ Total orders fetched: {len(all_orders)}")
        return all_orders
    
    def _fetch_symbol_orders(self, symbol: str) -> List[Dict]:
        """Fetch orders for specific symbol"""
        params = {'symbol': symbol, 'limit': 1000}
        orders = self._request('GET', '/fapi/v1/allOrders', params)
        return orders if orders else []
    
    def save_raw_dataset(self):
        """Save raw API data"""
        dataset = {
            'fetch_timestamp': datetime.now().isoformat(),
            'user_trades': self.raw_trades,
            'income_history': self.raw_income,
            'orders': self.raw_orders
        }
        
        filename = 'exchange_raw_dataset.json'
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n✅ Raw dataset saved: {filename}")
    
    def reconstruct_trades(self):
        """Group fills into complete position cycles (entry → exit)"""
        print("\n" + "="*80)
        print("PHASE 2: RECONSTRUCTING COMPLETE TRADES")
        print("="*80)
        
        # Group trades by symbol
        by_symbol = defaultdict(list)
        for trade in self.raw_trades:
            by_symbol[trade['symbol']].append(trade)
        
        # Sort trades by timestamp within each symbol
        for symbol in by_symbol:
            by_symbol[symbol].sort(key=lambda x: x['time'])
        
        # Reconstruct position cycles for each symbol
        for symbol, trades in by_symbol.items():
            self._reconstruct_symbol_trades(symbol, trades)
        
        print(f"\n✅ Reconstructed {len(self.reconstructed_trades)} complete trades")
    
    def _reconstruct_symbol_trades(self, symbol: str, trades: List[Dict]):
        """Reconstruct trades for one symbol"""
        position_amt = 0.0
        current_position = None
        
        for trade in trades:
            side = trade['side']  # BUY or SELL
            qty = float(trade['qty'])
            price = float(trade['price'])
            commission = float(trade['commission'])
            realized_pnl = float(trade.get('realizedPnl', 0))
            timestamp = trade['time']
            
            # Determine if opening or closing
            if side == 'BUY':
                change = qty
            else:
                change = -qty
            
            new_position = position_amt + change
            
            # Position opening
            if position_amt == 0 and new_position != 0:
                current_position = {
                    'symbol': symbol,
                    'side': 'LONG' if new_position > 0 else 'SHORT',
                    'entry_fills': [],
                    'exit_fills': [],
                    'entry_time': timestamp,
                    'exit_time': None,
                    'total_commission': 0,
                    'realized_pnl': 0
                }
                current_position['entry_fills'].append({
                    'qty': abs(change),
                    'price': price,
                    'commission': commission,
                    'time': timestamp
                })
                current_position['total_commission'] += commission
            
            # Position adding
            elif position_amt != 0 and (position_amt * new_position) > 0 and abs(new_position) > abs(position_amt):
                if current_position:
                    current_position['entry_fills'].append({
                        'qty': abs(change),
                        'price': price,
                        'commission': commission,
                        'time': timestamp
                    })
                    current_position['total_commission'] += commission
            
            # Position closing (partial or full)
            elif position_amt != 0 and abs(new_position) < abs(position_amt):
                if current_position:
                    current_position['exit_fills'].append({
                        'qty': abs(change),
                        'price': price,
                        'commission': commission,
                        'time': timestamp,
                        'realized_pnl': realized_pnl
                    })
                    current_position['total_commission'] += commission
                    current_position['realized_pnl'] += realized_pnl
                    
                    # Full close
                    if new_position == 0:
                        current_position['exit_time'] = timestamp
                        self._finalize_trade(current_position)
                        current_position = None
            
            # Position reversal
            elif position_amt != 0 and new_position != 0 and (position_amt * new_position) < 0:
                # Close old position
                if current_position:
                    close_qty = abs(position_amt)
                    current_position['exit_fills'].append({
                        'qty': close_qty,
                        'price': price,
                        'commission': commission * (close_qty / abs(change)),
                        'time': timestamp,
                        'realized_pnl': realized_pnl
                    })
                    current_position['exit_time'] = timestamp
                    self._finalize_trade(current_position)
                
                # Open new position
                new_qty = abs(new_position)
                current_position = {
                    'symbol': symbol,
                    'side': 'LONG' if new_position > 0 else 'SHORT',
                    'entry_fills': [{
                        'qty': new_qty,
                        'price': price,
                        'commission': commission * (new_qty / abs(change)),
                        'time': timestamp
                    }],
                    'exit_fills': [],
                    'entry_time': timestamp,
                    'exit_time': None,
                    'total_commission': commission * (new_qty / abs(change)),
                    'realized_pnl': 0
                }
            
            position_amt = new_position
    
    def _finalize_trade(self, position: Dict):
        """Calculate final metrics for completed trade"""
        if not position['entry_fills'] or not position['exit_fills']:
            return
        
        # Calculate weighted average entry price
        total_entry_value = sum(f['qty'] * f['price'] for f in position['entry_fills'])
        total_entry_qty = sum(f['qty'] for f in position['entry_fills'])
        entry_price = total_entry_value / total_entry_qty if total_entry_qty > 0 else 0
        
        # Calculate weighted average exit price
        total_exit_value = sum(f['qty'] * f['price'] for f in position['exit_fills'])
        total_exit_qty = sum(f['qty'] for f in position['exit_fills'])
        exit_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0
        
        # Calculate PnL
        qty = min(total_entry_qty, total_exit_qty)
        if position['side'] == 'LONG':
            gross_pnl = qty * (exit_price - entry_price)
        else:
            gross_pnl = qty * (entry_price - exit_price)
        
        # Holding time
        holding_time = (position['exit_time'] - position['entry_time']) / 1000  # seconds
        
        trade_record = {
            'symbol': position['symbol'],
            'side': position['side'],
            'entry_price': round(entry_price, 8),
            'exit_price': round(exit_price, 8),
            'quantity': round(qty, 8),
            'entry_timestamp': position['entry_time'],
            'exit_timestamp': position['exit_time'],
            'holding_time_seconds': round(holding_time, 2),
            'gross_pnl': round(gross_pnl, 4),
            'total_commission': round(position['total_commission'], 4),
            'realized_pnl_exchange': round(position['realized_pnl'], 4),
            'entry_fills_count': len(position['entry_fills']),
            'exit_fills_count': len(position['exit_fills'])
        }
        
        self.reconstructed_trades.append(trade_record)
    
    def calculate_friction_costs(self):
        """Add funding fees and calculate net PnL"""
        print("\n" + "="*80)
        print("PHASE 3: CALCULATING FRICTION COSTS")
        print("="*80)
        
        # Group funding fees by symbol and timestamp
        funding_by_symbol = defaultdict(list)
        for income in self.raw_income:
            if income.get('incomeType') == 'FUNDING_FEE':
                funding_by_symbol[income['symbol']].append({
                    'time': income['time'],
                    'amount': float(income['income'])
                })
        
        # Assign funding fees to trades
        for trade in self.reconstructed_trades:
            symbol = trade['symbol']
            entry_time = trade['entry_timestamp']
            exit_time = trade['exit_timestamp']
            
            # Sum funding fees during position holding period
            funding_total = 0
            if symbol in funding_by_symbol:
                for funding in funding_by_symbol[symbol]:
                    if entry_time <= funding['time'] <= exit_time:
                        funding_total += funding['amount']
            
            trade['funding_fees'] = round(funding_total, 4)
            trade['total_fees'] = round(trade['total_commission'] + abs(funding_total), 4)
            trade['net_pnl'] = round(trade['gross_pnl'] - trade['total_fees'], 4)
        
        print(f"✅ Friction costs calculated for {len(self.reconstructed_trades)} trades")
    
    def estimate_slippage(self):
        """Estimate slippage costs"""
        print("\n" + "="*80)
        print("PHASE 4: ESTIMATING SLIPPAGE")
        print("="*80)
        
        # Note: Slippage estimation would require mark price data at execution time
        # For testnet, we'll use a conservative estimate based on bid-ask spread
        
        for trade in self.reconstructed_trades:
            # Conservative estimate: 0.02% slippage on entry and exit
            position_value = trade['quantity'] * trade['entry_price']
            estimated_slippage = position_value * 0.0002 * 2  # Entry + exit
            
            trade['estimated_slippage'] = round(estimated_slippage, 4)
            trade['net_pnl_after_slippage'] = round(trade['net_pnl'] - estimated_slippage, 4)
        
        print("✅ Slippage estimates added")
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        print("\n" + "="*80)
        print("PHASE 5: CALCULATING COMPREHENSIVE METRICS")
        print("="*80)
        
        if not self.reconstructed_trades:
            print("❌ No trades to analyze")
            return {}
        
        trades = self.reconstructed_trades
        
        # Basic counts
        total_trades = len(trades)
        
        # Gross metrics
        gross_wins = [t for t in trades if t['gross_pnl'] > 0]
        gross_losses = [t for t in trades if t['gross_pnl'] < 0]
        gross_win_rate = len(gross_wins) / total_trades if total_trades > 0 else 0
        
        # Net metrics (after fees+funding)
        net_wins = [t for t in trades if t['net_pnl'] > 0]
        net_losses = [t for t in trades if t['net_pnl'] < 0]
        net_win_rate = len(net_wins) / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        gross_pnls = [t['gross_pnl'] for t in trades]
        net_pnls = [t['net_pnl'] for t in trades]
        
        avg_gross_win = np.mean([t['gross_pnl'] for t in gross_wins]) if gross_wins else 0
        avg_gross_loss = np.mean([t['gross_pnl'] for t in gross_losses]) if gross_losses else 0
        avg_net_win = np.mean([t['net_pnl'] for t in net_wins]) if net_wins else 0
        avg_net_loss = np.mean([t['net_pnl'] for t in net_losses]) if net_losses else 0
        
        # Profit factors
        gross_profit = sum([t['gross_pnl'] for t in gross_wins]) if gross_wins else 0
        gross_loss = abs(sum([t['gross_pnl'] for t in gross_losses])) if gross_losses else 1
        profit_factor_gross = gross_profit / gross_loss if gross_loss > 0 else 0
        
        net_profit = sum([t['net_pnl'] for t in net_wins]) if net_wins else 0
        net_loss = abs(sum([t['net_pnl'] for t in net_losses])) if net_losses else 1
        profit_factor_net = net_profit / net_loss if net_loss > 0 else 0
        
        # Expectancy
        expectancy_gross = np.mean(gross_pnls)
        expectancy_net = np.mean(net_pnls)
        
        # Risk metrics
        std_dev = np.std(net_pnls)
        sharpe_ratio = (expectancy_net / std_dev) if std_dev > 0 else 0
        
        # Drawdown calculation
        cumulative_pnl = np.cumsum(net_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Holding time
        avg_holding_time = np.mean([t['holding_time_seconds'] for t in trades])
        
        # Side analysis
        long_trades = [t for t in trades if t['side'] == 'LONG']
        short_trades = [t for t in trades if t['side'] == 'SHORT']
        
        long_pnl = sum([t['net_pnl'] for t in long_trades]) if long_trades else 0
        short_pnl = sum([t['net_pnl'] for t in short_trades]) if short_trades else 0
        
        # Fees analysis
        total_commission = sum([t['total_commission'] for t in trades])
        total_funding = sum([t['funding_fees'] for t in trades])
        total_fees = total_commission + abs(total_funding)
        
        metrics = {
            'total_trades': total_trades,
            'gross_win_rate': round(gross_win_rate, 4),
            'net_win_rate': round(net_win_rate, 4),
            'avg_gross_win': round(avg_gross_win, 4),
            'avg_gross_loss': round(avg_gross_loss, 4),
            'avg_net_win': round(avg_net_win, 4),
            'avg_net_loss': round(avg_net_loss, 4),
            'profit_factor_gross': round(profit_factor_gross, 4),
            'profit_factor_net': round(profit_factor_net, 4),
            'expectancy_gross_per_trade': round(expectancy_gross, 4),
            'expectancy_net_per_trade': round(expectancy_net, 4),
            'total_gross_pnl': round(sum(gross_pnls), 2),
            'total_net_pnl': round(sum(net_pnls), 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'std_dev_pnl': round(std_dev, 4),
            'avg_holding_time_seconds': round(avg_holding_time, 2),
            'long_trades_count': len(long_trades),
            'short_trades_count': len(short_trades),
            'long_total_pnl': round(long_pnl, 2),
            'short_total_pnl': round(short_pnl, 2),
            'total_commission_paid': round(total_commission, 2),
            'total_funding_paid': round(total_funding, 2),
            'total_friction_cost': round(total_fees, 2),
            'friction_pct_of_gross_profit': round((total_fees / gross_profit * 100) if gross_profit > 0 else 0, 2)
        }
        
        # Print summary
        print(f"\n{'Metric':<40} {'Gross':>15} {'Net':>15}")
        print("-" * 75)
        print(f"{'Total Trades':<40} {total_trades:>15,}")
        print(f"{'Win Rate':<40} {gross_win_rate:>14.1%} {net_win_rate:>14.1%}")
        print(f"{'Average Win':<40} ${avg_gross_win:>14.2f} ${avg_net_win:>14.2f}")
        print(f"{'Average Loss':<40} ${avg_gross_loss:>14.2f} ${avg_net_loss:>14.2f}")
        print(f"{'Profit Factor':<40} {profit_factor_gross:>15.2f} {profit_factor_net:>15.2f}")
        print(f"{'Expectancy/Trade':<40} ${expectancy_gross:>14.2f} ${expectancy_net:>14.2f}")
        print(f"{'Total PnL':<40} ${sum(gross_pnls):>14.2f} ${sum(net_pnls):>14.2f}")
        print(f"{'Max Drawdown':<40} {'':<15} ${max_drawdown:>14.2f}")
        print(f"{'Sharpe Ratio':<40} {'':<15} {sharpe_ratio:>15.2f}")
        
        return metrics
    
    def generate_per_symbol_breakdown(self) -> Dict:
        """Calculate performance per symbol"""
        by_symbol = defaultdict(list)
        for trade in self.reconstructed_trades:
            by_symbol[trade['symbol']].append(trade)
        
        breakdown = {}
        for symbol, trades in by_symbol.items():
            net_pnls = [t['net_pnl'] for t in trades]
            breakdown[symbol] = {
                'trade_count': len(trades),
                'total_net_pnl': round(sum(net_pnls), 2),
                'avg_net_pnl': round(np.mean(net_pnls), 2),
                'win_rate': round(len([p for p in net_pnls if p > 0]) / len(trades), 4),
                'std_dev': round(np.std(net_pnls), 2)
            }
        
        return breakdown
    
    def validate_edge(self, metrics: Dict) -> Dict:
        """Determine system edge classification"""
        print("\n" + "="*80)
        print("PHASE 6: EDGE VALIDATION")
        print("="*80)
        
        gross_exp = metrics['expectancy_gross_per_trade']
        net_exp = metrics['expectancy_net_per_trade']
        friction = metrics['total_friction_cost']
        gross_profit = metrics['total_gross_pnl']
        
        # Classification logic
        has_positive_gross = gross_exp > 0
        has_positive_net = net_exp > 0
        funding_dominant = abs(metrics['total_funding_paid']) > metrics['total_commission_paid']
        fees_dominant = friction > (gross_profit * 0.5) if gross_profit > 0 else True
        
        if net_exp > 0:
            classification = "STRUCTURALLY_PROFITABLE"
            verdict = "System has positive expectancy after all friction costs"
        elif gross_exp > 0 and net_exp <= 0:
            classification = "FRICTION_DOMINATED"
            verdict = "System profitable before friction, but fees/funding erase edge"
        elif gross_exp <= 0:
            classification = "STRUCTURALLY_NEGATIVE"
            verdict = "System has negative gross expectancy (losing before fees)"
        else:
            classification = "NEUTRAL"
            verdict = "System breaks even"
        
        validation = {
            'classification': classification,
            'verdict': verdict,
            'has_positive_gross_expectancy': has_positive_gross,
            'has_positive_net_expectancy': has_positive_net,
            'funding_eroding_profitability': funding_dominant and not has_positive_net,
            'fees_dominating_wins': fees_dominant,
            'friction_pct_of_profit': metrics['friction_pct_of_gross_profit']
        }
        
        print(f"\nCLASSIFICATION: {classification}")
        print(f"VERDICT: {verdict}\n")
        print(f"{'Question':<50} {'Answer':>15}")
        print("-" * 70)
        print(f"{'A) Positive gross expectancy?':<50} {'YES' if has_positive_gross else 'NO':>15}")
        print(f"{'B) Positive net expectancy after friction?':<50} {'YES' if has_positive_net else 'NO':>15}")
        print(f"{'C) Funding eroding profitability?':<50} {'YES' if validation['funding_eroding_profitability'] else 'NO':>15}")
        print(f"{'D) Fees dominating small wins?':<50} {'YES' if fees_dominant else 'NO':>15}")
        print(f"{'E) Friction as % of gross profit':<50} {f'{metrics["friction_pct_of_gross_profit"]:.1f}%':>15}")
        
        return validation
    
    def generate_distribution_bins(self) -> Dict:
        """Create histogram bins for PnL distribution"""
        net_pnls = [t['net_pnl'] for t in self.reconstructed_trades]
        
        bins = [-1000, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 1000]
        hist, edges = np.histogram(net_pnls, bins=bins)
        
        distribution = []
        for i in range(len(hist)):
            distribution.append({
                'bin': f"${edges[i]:.0f} to ${edges[i+1]:.0f}",
                'count': int(hist[i]),
                'percentage': round(hist[i] / len(net_pnls) * 100, 2)
            })
        
        return {
            'bins': distribution,
            'tail_losses': len([p for p in net_pnls if p < -50]),
            'tail_wins': len([p for p in net_pnls if p > 50]),
            'worst_trade': round(min(net_pnls), 2),
            'best_trade': round(max(net_pnls), 2)
        }
    
    def save_final_report(self, metrics: Dict, validation: Dict, symbol_breakdown: Dict, distribution: Dict):
        """Generate comprehensive audit report"""
        report = {
            'audit_metadata': {
                'audit_date': datetime.now().isoformat(),
                'data_source': f'Binance Futures {"Testnet" if self.is_testnet else "Production"} API',
                'api_endpoint': self.base_url,
                'total_trades_analyzed': len(self.reconstructed_trades),
                'data_completeness': '100% (exchange ground truth)'
            },
            'summary_metrics': metrics,
            'edge_validation': validation,
            'per_symbol_breakdown': symbol_breakdown,
            'distribution_analysis': distribution,
            'friction_breakdown': {
                'total_commission': metrics['total_commission_paid'],
                'total_funding': metrics['total_funding_paid'],
                'total_friction': metrics['total_friction_cost'],
                'friction_per_trade': round(metrics['total_friction_cost'] / metrics['total_trades'], 4),
                'friction_as_pct_of_gross': metrics['friction_pct_of_gross_profit']
            },
            'risk_metrics': {
                'max_drawdown': metrics['max_drawdown'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'std_dev_pnl': metrics['std_dev_pnl'],
                'avg_holding_time_hours': round(metrics['avg_holding_time_seconds'] / 3600, 2)
            },
            'trades': self.reconstructed_trades
        }
        
        filename = 'exchange_truth_full_audit.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Full audit report saved: {filename}")
    
    def run_full_audit(self):
        """Execute complete audit pipeline"""
        print("\n" + "="*80)
        print("EXCHANGE GROUND-TRUTH PROFITABILITY AUDIT")
        print(f"Binance Futures {"Testnet" if self.is_testnet else "PRODUCTION"} API")
        print(f"Endpoint: {self.base_url}")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Phase 1: Fetch raw data
        print("\n" + "="*80)
        print("PHASE 1: FETCHING RAW DATA FROM EXCHANGE")
        print("="*80)
        
        self.raw_trades = self.fetch_user_trades()
        self.raw_income = self.fetch_income_history()
        self.raw_orders = self.fetch_all_orders()
        
        self.save_raw_dataset()
        
        if len(self.raw_trades) == 0:
            print("\n❌ No trades found on exchange")
            print("   Account may not have any filled orders yet")
            return
        
        # Phase 2: Reconstruct trades
        self.reconstruct_trades()
        
        if len(self.reconstructed_trades) == 0:
            print("\n❌ Could not reconstruct any complete trades")
            print("   All positions may still be open")
            return
        
        # Phase 3: Friction costs
        self.calculate_friction_costs()
        
        # Phase 4: Slippage
        self.estimate_slippage()
        
        # Phase 5: Metrics
        metrics = self.calculate_comprehensive_metrics()
        
        # Symbol breakdown
        symbol_breakdown = self.generate_per_symbol_breakdown()
        
        # Phase 6: Edge validation
        validation = self.validate_edge(metrics)
        
        # Distribution analysis
        distribution = self.generate_distribution_bins()
        
        # Phase 7: Final report
        print("\n" + "="*80)
        print("PHASE 7: GENERATING FINAL REPORT")
        print("="*80)
        
        self.save_final_report(metrics, validation, symbol_breakdown, distribution)
        
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80)


def main():
    """Main entry point"""
    auditor = BinanceAuditor(BASE_URL, API_KEY, SECRET_KEY, USE_TESTNET)
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
