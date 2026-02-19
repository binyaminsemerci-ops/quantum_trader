#!/usr/bin/env python3
"""
Quantitative Expectancy Audit
==============================

Analyzes closed trade history to determine system expectancy.

Extracts data from:
- Redis: quantum:ledger:{symbol} trade_history
- SQLite: trade_logs table

Calculates:
- R-multiples per trade
- Win rate, profit factor, expectancy
- Distribution analysis
- Leverage sensitivity

Output: Statistical truth report (no fix proposals)

Author: Forensic Trading Systems Auditor
Date: February 18, 2026
"""

import os
import sys
import json
import redis
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import Counter

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SQLITE_PATH = os.getenv("SQLITE_PATH", "/app/backend/data/trades.db")
SQLITE_PATH_WINDOWS = os.getenv("SQLITE_PATH_WINDOWS", "c:/quantum_trader/backend/data/trades.db")

MIN_TRADES_REQUIRED = 200


class TradeRecord:
    """Structured trade record with calculated R-multiple"""
    
    def __init__(self, data: Dict):
        self.symbol = data.get('symbol', 'UNKNOWN')
        self.side = data.get('side', 'UNKNOWN')
        self.entry_price = float(data.get('entry_price', 0))
        self.exit_price = float(data.get('exit_price', 0))
        self.stop_loss_price = float(data.get('stop_loss_price', 0))
        self.leverage = float(data.get('leverage', 1))
        self.position_size = float(data.get('position_size', 0))
        self.qty = float(data.get('qty', 0))
        self.realized_pnl_usdt = float(data.get('realized_pnl', 0))
        self.entry_timestamp = data.get('entry_timestamp', 0)
        self.exit_timestamp = data.get('exit_timestamp', 0)
        
        # Calculate R-multiple
        self.r_multiple = self._calculate_r()
        
    def _calculate_r(self) -> float:
        """
        Calculate R-multiple for this trade.
        
        R = (exit_price - entry_price) / (entry_price - stop_loss) for LONG
        R = (entry_price - exit_price) / (stop_loss - entry_price) for SHORT
        
        If no stop_loss, assume 2% entry risk.
        """
        if self.entry_price == 0:
            return 0.0
        
        # Determine entry risk
        if self.stop_loss_price > 0:
            if self.side in ['BUY', 'LONG']:
                entry_risk = abs(self.entry_price - self.stop_loss_price)
            elif self.side in ['SELL', 'SHORT']:
                entry_risk = abs(self.stop_loss_price - self.entry_price)
            else:
                # Unknown side, use 2% default
                entry_risk = self.entry_price * 0.02
        else:
            # No stop-loss provided, assume 2% risk
            entry_risk = self.entry_price * 0.02
        
        if entry_risk == 0:
            return 0.0
        
        # Calculate price move
        if self.side in ['BUY', 'LONG']:
            price_move = self.exit_price - self.entry_price
        elif self.side in ['SELL', 'SHORT']:
            price_move = self.entry_price - self.exit_price
        else:
            # Fallback: use realized PnL if available
            if self.realized_pnl_usdt != 0 and self.position_size > 0:
                return self.realized_pnl_usdt / (self.position_size * entry_risk / self.entry_price)
            return 0.0
        
        # R = price_move / entry_risk
        r = price_move / entry_risk
        
        return r
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for analysis"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss_price': self.stop_loss_price,
            'leverage': self.leverage,
            'position_size': self.position_size,
            'realized_pnl_usdt': self.realized_pnl_usdt,
            'entry_timestamp': self.entry_timestamp,
            'exit_timestamp': self.exit_timestamp,
            'r_multiple': self.r_multiple,
            'is_winner': self.r_multiple > 0
        }


class ExpectancyAuditor:
    """Main audit engine"""
    
    def __init__(self):
        self.redis_client = None
        self.sqlite_conn = None
        self.trades: List[TradeRecord] = []
        
    def connect(self):
        """Connect to data sources"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"⚠️  Redis connection failed: {e}")
            self.redis_client = None
        
        # Try both Windows and Linux paths
        sqlite_path = SQLITE_PATH_WINDOWS if os.path.exists(SQLITE_PATH_WINDOWS) else SQLITE_PATH
        
        try:
            self.sqlite_conn = sqlite3.connect(sqlite_path)
            print(f"✅ Connected to SQLite at {sqlite_path}")
        except Exception as e:
            print(f"⚠️  SQLite connection failed: {e}")
            self.sqlite_conn = None
    
    def extract_trades_from_redis(self) -> List[Dict]:
        """Extract trade history from Redis ledger"""
        trades = []
        
        if not self.redis_client:
            return trades
        
        try:
            # Scan all ledger keys
            ledger_count = 0
            for key in self.redis_client.scan_iter("quantum:ledger:*"):
                if "seen_orders" in key:
                    continue
                
                ledger_count += 1
                symbol = key.replace("quantum:ledger:", "")
                ledger_data = self.redis_client.hgetall(key)
                
                if not ledger_data:
                    continue
                
                # Parse trade_history JSON
                trade_history_str = ledger_data.get('trade_history', '[]')
                try:
                    trade_history = json.loads(trade_history_str)
                except:
                    continue
                
                # Extract entry_price, stop_loss from ledger (applies to all trades in this ledger)
                entry_price = float(ledger_data.get('entry_price', 0))
                stop_loss = float(ledger_data.get('stop_loss', 0))
                leverage = float(ledger_data.get('leverage', 30))
                
                # Process each completed trade
                for trade in trade_history:
                    # Trade history format: timestamp, order_id, qty, price, side, pnl_usdt, pnl_pct, volume_usdt, is_full_close
                    exit_price = float(trade.get('price', 0))
                    qty = float(trade.get('qty', 0))
                    side = trade.get('side', 'UNKNOWN')  # This is the EXIT side
                    pnl_usdt = float(trade.get('pnl_usdt', 0))
                    timestamp = trade.get('timestamp', 0)
                    
                    # Determine position side from exit side
                    # If exit side is SELL, position was LONG
                    # If exit side is BUY, position was SHORT
                    position_side = 'LONG' if side == 'SELL' else 'SHORT'
                    
                    # Use entry_price from ledger
                    # If not available, try to infer from PnL
                    if entry_price == 0 and pnl_usdt != 0 and qty != 0:
                        # Reverse calculate entry from PnL: pnl = qty * (exit - entry) for LONG
                        if position_side == 'LONG':
                            entry_price = exit_price - (pnl_usdt / qty)
                        else:
                            entry_price = exit_price + (pnl_usdt / qty)
                    
                    trades.append({
                        'symbol': symbol,
                        'side': position_side,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss_price': stop_loss,
                        'leverage': leverage,
                        'position_size': qty * exit_price,
                        'qty': qty,
                        'realized_pnl': pnl_usdt,
                        'entry_timestamp': 0,  # Not tracked in Redis
                        'exit_timestamp': timestamp
                    })
            
            print(f"Scanned {ledger_count} ledger keys")
        
        except Exception as e:
            print(f"❌ Error extracting from Redis: {e}")
            import traceback
            traceback.print_exc()
        
        return trades
    
    def extract_trades_from_sqlite(self) -> List[Dict]:
        """Extract closed trades from SQLite database"""
        trades = []
        
        if not self.sqlite_conn:
            return trades
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Query trade_logs for completed trades (those with both entry and exit)
            query = """
                SELECT 
                    symbol,
                    side,
                    entry_price,
                    exit_price,
                    qty,
                    price,
                    realized_pnl,
                    realized_pnl_pct,
                    timestamp,
                    strategy_id
                FROM trade_logs
                WHERE status = 'FILLED'
                    AND entry_price IS NOT NULL
                    AND exit_price IS NOT NULL
                    AND entry_price > 0
                    AND exit_price > 0
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            for row in rows:
                symbol, side, entry_price, exit_price, qty, price, pnl, pnl_pct, ts, strategy = row
                
                if entry_price and exit_price:
                    trades.append({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'stop_loss_price': 0,  # Calculate from 2% default
                        'leverage': 30,  # Default assumption
                        'position_size': qty * exit_price if qty else 0,
                        'qty': qty if qty else 0,
                        'realized_pnl': pnl if pnl else 0,
                        'entry_timestamp': 0,
                        'exit_timestamp': int(datetime.fromisoformat(str(ts)).timestamp()) if ts else 0
                    })
        
        except Exception as e:
            print(f"❌ Error extracting from SQLite: {e}")
        
        return trades
    
    def extract_all_trades(self):
        """Extract from all sources and deduplicate"""
        print("\n" + "="*80)
        print("PHASE 1: EXTRACTING TRADE HISTORY")
        print("="*80)
        
        redis_trades = self.extract_trades_from_redis()
        print(f"Extracted {len(redis_trades)} trades from Redis")
        
        sqlite_trades = self.extract_trades_from_sqlite()
        print(f"Extracted {len(sqlite_trades)} trades from SQLite")
        
        # Combine and deduplicate
        all_trades = redis_trades + sqlite_trades
        
        # Convert to TradeRecord objects
        rejected_count = 0
        for i, trade_data in enumerate(all_trades):
            try:
                trade = TradeRecord(trade_data)
                # Filter out invalid trades
                if trade.entry_price > 0 and trade.exit_price > 0:
                    self.trades.append(trade)
                else:
                    rejected_count += 1
                    if i < 3:  # Debug first 3 rejections
                        print(f"⚠️  Trade rejected: entry={trade.entry_price}, exit={trade.exit_price}, symbol={trade.symbol}")
            except Exception as e:
                rejected_count += 1
                print(f"⚠️  Skipping invalid trade: {e}")
        
        if rejected_count > 0:
            print(f"⚠️  Rejected {rejected_count} invalid trades (missing entry/exit price)")
        
        print(f"\n✅ Total valid trades extracted: {len(self.trades)}")
        
        if len(self.trades) < MIN_TRADES_REQUIRED:
            print(f"⚠️  WARNING: Only {len(self.trades)} trades found (minimum {MIN_TRADES_REQUIRED} required for statistical significance)")
        
        return len(self.trades)
    
    def calculate_r_multiples(self):
        """R-multiples already calculated in TradeRecord.__init__"""
        print("\n" + "="*80)
        print("PHASE 2: R-MULTIPLE CALCULATION")
        print("="*80)
        
        r_values = [t.r_multiple for t in self.trades]
        
        print(f"R-multiples calculated for {len(r_values)} trades")
        print(f"R range: [{min(r_values):.3f}, {max(r_values):.3f}]")
        print(f"Mean R: {np.mean(r_values):.3f}")
        print(f"Median R: {np.median(r_values):.3f}")
    
    def compute_core_metrics(self) -> Dict:
        """Calculate win rate, profit factor, expectancy"""
        print("\n" + "="*80)
        print("PHASE 3: CORE METRICS")
        print("="*80)
        
        if not self.trades:
            print("❌ No trades to analyze")
            return {}
        
        r_values = [t.r_multiple for t in self.trades]
        winners = [r for r in r_values if r > 0]
        losers = [r for r in r_values if r < 0]
        
        total_trades = len(r_values)
        win_count = len(winners)
        loss_count = len(losers)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        
        # Profit Factor = Gross Profit / Gross Loss
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1  # Prevent div by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy = (Win% × Avg Win) - (Loss% × |Avg Loss|)
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
        
        # Percentiles
        median_r = np.median(r_values)
        p90_r = np.percentile(r_values, 90)
        worst_r = np.min(r_values)
        best_r = np.max(r_values)
        
        metrics = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'median_r': median_r,
            'p90_r': p90_r,
            'worst_r': worst_r,
            'best_r': best_r
        }
        
        # Print table
        print(f"\n{'Metric':<30} {'Value':>15}")
        print("-" * 50)
        print(f"{'Total Trades':<30} {total_trades:>15,}")
        print(f"{'Winning Trades':<30} {win_count:>15,}")
        print(f"{'Losing Trades':<30} {loss_count:>15,}")
        print(f"{'Win Rate':<30} {win_rate:>14.2%}")
        print(f"{'Average Win (R)':<30} {avg_win:>15.3f}")
        print(f"{'Average Loss (R)':<30} {avg_loss:>15.3f}")
        print(f"{'Profit Factor':<30} {profit_factor:>15.3f}")
        print(f"{'Expectancy (R)':<30} {expectancy:>15.3f}")
        print(f"{'Median R':<30} {median_r:>15.3f}")
        print(f"{'90th Percentile R':<30} {p90_r:>15.3f}")
        print(f"{'Worst R':<30} {worst_r:>15.3f}")
        print(f"{'Best R':<30} {best_r:>15.3f}")
        
        return metrics
    
    def distribution_analysis(self):
        """Analyze R-multiple distribution"""
        print("\n" + "="*80)
        print("PHASE 4: DISTRIBUTION ANALYSIS")
        print("="*80)
        
        r_values = [t.r_multiple for t in self.trades]
        
        # Create histogram bins
        bins = [-10, -5, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        hist, edges = np.histogram(r_values, bins=bins)
        
        print("\nR-Multiple Distribution:")
        print(f"{'Bin':<20} {'Count':>10} {'%':>10}")
        print("-" * 45)
        
        total = len(r_values)
        for i in range(len(hist)):
            bin_label = f"{edges[i]:.1f} to {edges[i+1]:.1f}"
            count = hist[i]
            pct = count / total * 100 if total > 0 else 0
            print(f"{bin_label:<20} {count:>10} {pct:>9.1f}%")
        
        # Analysis
        print("\n" + "-" * 80)
        print("DISTRIBUTION FINDINGS:")
        print("-" * 80)
        
        # Check if winners are capped
        winners = [r for r in r_values if r > 0]
        if winners:
            max_winner = max(winners)
            winners_above_2r = sum(1 for r in winners if r > 2.0)
            winner_cap_analysis = f"{winners_above_2r} trades ({winners_above_2r/len(winners)*100:.1f}%) exceed 2R"
            print(f"✓ Winner analysis: Max R = {max_winner:.2f}R, {winner_cap_analysis}")
            if max_winner < 1.5:
                print("  ⚠️  WARNING: Winners appear capped (max < 1.5R)")
        
        # Check if losses cluster at -1R
        losers = [r for r in r_values if r < 0]
        if losers:
            losers_near_minus1r = sum(1 for r in losers if -1.2 < r < -0.8)
            cluster_pct = losers_near_minus1r / len(losers) * 100 if losers else 0
            print(f"✓ Loss clustering: {losers_near_minus1r} trades ({cluster_pct:.1f}%) cluster near -1R")
            if cluster_pct > 50:
                print("  ✓ Majority of losses hitting full stop-loss (expected behavior)")
        
        # Check for fat tail losses
        fat_tail_losses = sum(1 for r in r_values if r < -1.5)
        fat_tail_pct = fat_tail_losses / len(r_values) * 100 if r_values else 0
        print(f"✓ Fat tail losses: {fat_tail_losses} trades ({fat_tail_pct:.1f}%) with R < -1.5")
        if fat_tail_pct > 10:
            print("  ⚠️  WARNING: High fat-tail risk (>10% of trades exceed -1.5R)")
    
    def leverage_sensitivity(self, metrics: Dict):
        """Recalculate expectancy at different leverage levels"""
        print("\n" + "="*80)
        print("PHASE 5: LEVERAGE SENSITIVITY")
        print("="*80)
        
        # Expectancy is leverage-neutral when expressed in R
        # But we can show PnL% impact
        
        base_expectancy_r = metrics['expectancy']
        
        print("\nExpectancy at different leverage levels:")
        print(f"{'Leverage':<15} {'Expectancy (R)':>20} {'Expectancy (% per trade)':>25}")
        print("-" * 65)
        
        for lev in [1, 5, 10, 20, 30]:
            # Assume 2% risk per trade (entry_risk / capital)
            risk_pct = 0.02
            expectancy_pct = base_expectancy_r * risk_pct * lev
            
            print(f"{lev}x{'':<12} {base_expectancy_r:>20.3f} {expectancy_pct:>24.2%}")
        
        print("\n" + "-" * 80)
        print("LEVERAGE ANALYSIS:")
        print("-" * 80)
        
        if base_expectancy_r < 0:
            print("⚠️  System has NEGATIVE expectancy in R-multiples")
            print("   → Leverage AMPLIFIES structural loss")
            print("   → Higher leverage = faster capital depletion")
        elif base_expectancy_r > 0:
            print("✓ System has POSITIVE expectancy in R-multiples")
            print("  → Leverage amplifies gains proportionally")
            print("  → Risk management determines optimal leverage")
        else:
            print("⚠️  System has NEUTRAL expectancy (break-even)")
            print("   → Leverage irrelevant, will bleed through fees")
    
    def generate_verdict(self, metrics: Dict):
        """Final classification and diagnosis"""
        print("\n" + "="*80)
        print("PHASE 6: VERDICT")
        print("="*80)
        
        expectancy = metrics.get('expectancy', 0)
        win_rate = metrics.get('win_rate', 0)
        avg_win = metrics.get('avg_win', 0)
        avg_loss = metrics.get('avg_loss', 0)
        profit_factor = metrics.get('profit_factor', 0)
        worst_r = metrics.get('worst_r', 0)
        
        # Classification
        if expectancy > 0.05:
            classification = "A) POSITIVE EXPECTANCY ✅"
        elif expectancy > -0.05:
            classification = "B) NEUTRAL EXPECTANCY ⚠️"
        else:
            classification = "C) NEGATIVE EXPECTANCY ❌"
        
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION: {classification}")
        print(f"{'='*80}\n")
        
        print(f"Expectancy: {expectancy:.3f}R per trade")
        print(f"Profit Factor: {profit_factor:.2f}\n")
        
        if expectancy < 0:
            print("ROOT CAUSE ANALYSIS:")
            print("-" * 80)
            
            issues = []
            
            # Diagnose specific issues
            if win_rate < 0.45:
                issues.append(f"❌ LOW WIN RATE ({win_rate:.1%}) - System wins less than 45% of trades")
            
            if abs(avg_loss) > abs(avg_win):
                ratio = abs(avg_loss) / abs(avg_win) if avg_win != 0 else 999
                issues.append(f"❌ LARGE AVERAGE LOSS ({avg_loss:.2f}R vs {avg_win:.2f}R win) - Losses {ratio:.1f}x larger")
            
            if avg_win < 0.5:
                issues.append(f"❌ SMALL AVERAGE WIN ({avg_win:.2f}R) - Winners exited too early")
            
            if worst_r < -2.0:
                issues.append(f"❌ FAT-TAIL LOSS (worst: {worst_r:.2f}R) - Risk management failure")
            
            # Check for exit asymmetry (harvest ladder signature)
            winners = [t.r_multiple for t in self.trades if t.r_multiple > 0]
            if winners:
                winners_below_1r = sum(1 for r in winners if r < 1.0)
                early_exit_pct = winners_below_1r / len(winners) * 100
                if early_exit_pct > 50:
                    issues.append(f"❌ EXIT ASYMMETRY ({early_exit_pct:.0f}% of winners < 1R) - Premature profit-taking")
            
            for issue in issues:
                print(issue)
            
            if not issues:
                print("⚠️  Negative expectancy present but no single dominant cause identified")
                print("   Likely: Combination of low win rate + poor reward:risk ratio")
        
        else:
            print("✓ System demonstrates positive edge")
            print("  Continue monitoring for regime stability")
        
        print("\n" + "="*80)
        print("AUDIT COMPLETE")
        print("="*80)
    
    def run_full_audit(self):
        """Execute full audit pipeline"""
        print("\n" + "="*80)
        print("QUANTITATIVE EXPECTANCY AUDIT - LIVE TRADING SYSTEM")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        self.connect()
        
        trade_count = self.extract_all_trades()
        
        if trade_count == 0:
            print("\n❌ AUDIT FAILED: No trades found in data sources")
            print("   Check that:")
            print("   - Trade history logger is running")
            print("   - Positions have been closed (not just opened)")
            print("   - Database/Redis contains historical data")
            return
        
        self.calculate_r_multiples()
        metrics = self.compute_core_metrics()
        self.distribution_analysis()
        self.leverage_sensitivity(metrics)
        self.generate_verdict(metrics)
        
        # Save results
        self.save_results(metrics)
    
    def save_results(self, metrics: Dict):
        """Save audit results to file"""
        output_file = "QUANTITATIVE_EXPECTANCY_AUDIT_RESULTS_FEB18_2026.json"
        
        results = {
            'audit_date': datetime.now().isoformat(),
            'total_trades': len(self.trades),
            'metrics': metrics,
            'trades': [t.to_dict() for t in self.trades]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")


def main():
    """Main entry point"""
    auditor = ExpectancyAuditor()
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
