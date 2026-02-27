#!/usr/bin/env python3
"""
Limited Data Expectancy Assessment
===================================

Analyzes available trade data from production system.
WARNING: Statistical significance requires 200+ trades.
Current data: ~11 trades (INSUFFICIENT for robust conclusions)

This assessment provides preliminary indicators only.

Author: Forensic Trading Systems Auditor
Date: February 18, 2026
"""

import redis
import json
import numpy as np
from datetime import datetime
from typing import List, Dict


class LimitedDataAssessor:
    """Assess expectancy with limited trade history"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True
        )
        self.trades: List[Dict] = []
    
    def extract_pnl_history(self):
        """Extract PnL history from Redis ledger"""
        print("\n" + "="*80)
        print("DATA EXTRACTION FROM PRODUCTION SYSTEM")
        print("="*80)
        
        ledger_count = 0
        total_trades = 0
        
        for key in self.redis_client.scan_iter("quantum:ledger:*"):
            if "seen_orders" in key:
                continue
            
            ledger_count += 1
            symbol = key.replace("quantum:ledger:", "")
            ledger_data = self.redis_client.hgetall(key)
            
            # Parse aggregate stats
            total_pnl = float(ledger_data.get('total_pnl_usdt', 0))
            winning_trades = int(ledger_data.get('winning_trades', 0))
            losing_trades = int(ledger_data.get('losing_trades', 0))
            trades_count = int(ledger_data.get('total_trades', 0))
            
            # Parse trade_history
            trade_history_str = ledger_data.get('trade_history', '[]')
            try:
                trade_history = json.loads(trade_history_str)
            except:
                trade_history = []
            
            for trade in trade_history:
                timestamp = trade.get('timestamp', 0)
                pnl_usdt = float(trade.get('pnl_usdt', 0))
                pnl_pct = float(trade.get('pnl_pct', 0))
                
                self.trades.append({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'pnl_usdt': pnl_usdt,
                    'pnl_pct': pnl_pct,
                    'is_winner': pnl_usdt > 0
                })
                total_trades += 1
        
        print(f"Ledger keys found: {ledger_count}")
        print(f"Total trades extracted: {total_trades}")
        
        if total_trades < 30:
            print(f"\n⚠️  CRITICAL WARNING: Only {total_trades} trades found")
            print("   Statistical significance requires minimum 200 trades")
            print("   Results below are PRELIMINARY INDICATORS ONLY")
            print("   DO NOT base trading decisions on this sample size")
        
        return total_trades
    
    def calculate_metrics(self) -> Dict:
        """Calculate what we can from limited data"""
        print("\n" + "="*80)
        print("PRELIMINARY STATISTICAL ANALYSIS")
        print("="*80)
        
        if not self.trades:
            print("❌ NO DATA AVAILABLE")
            return {}
        
        pnl_values = [t['pnl_usdt'] for t in self.trades]
        winners = [p for p in pnl_values if p > 0]
        losers = [p for p in pnl_values if p < 0]
        
        total_trades = len(self.trades)
        win_count = len(winners)
        loss_count = len(losers)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        avg_win_usdt = np.mean(winners) if winners else 0
        avg_loss_usdt = np.mean(losers) if losers else 0
        total_pnl = sum(pnl_values)
        avg_pnl = np.mean(pnl_values)
        
        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy in USDT (NOT R, since we don't have entry risk data)
        expectancy_usdt = avg_pnl
        
        # Statistical confidence metrics
        std_dev = np.std(pnl_values) if len(pnl_values) > 1 else 0
        std_error = std_dev / np.sqrt(total_trades) if total_trades > 0 else 0
        
        # Min sample size for 95% confidence (rough estimate)
        if std_dev > 0 and avg_pnl != 0:
            # n = (Z * σ / E)² where Z=1.96 for 95%, E = desired precision
            desired_precision = abs(avg_pnl) * 0.2  # 20% precision
            min_sample = ((1.96 * std_dev) / desired_precision) ** 2 if desired_precision > 0 else 999
        else:
            min_sample = 200
        
        metrics = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'avg_win_usdt': avg_win_usdt,
            'avg_loss_usdt': avg_loss_usdt,
            'total_pnl_usdt': total_pnl,
            'avg_pnl_usdt': avg_pnl,
            'profit_factor': profit_factor,
            'expectancy_usdt': expectancy_usdt,
            'std_dev': std_dev,
            'std_error': std_error,
            'min_sample_required': int(min_sample)
        }
        
        # Print results
        print(f"\n{'Metric':<35} {'Value':>20} {'Confidence':>15}")
        print("-" * 75)
        print(f"{'Total Trades':<35} {total_trades:>20,} {'❌ INSUFFICIENT'}")
        print(f"{'Minimum Required (95% conf.)':<35} {int(min_sample):>20,}")
        print(f"{'─'*75}")
        print(f"{'Winning Trades':<35} {win_count:>20,}")
        print(f"{'Losing Trades':<35} {loss_count:>20,}")
        print(f"{'Win Rate':<35} {win_rate:>19.1%}")
        print(f"{'─'*75}")
        print(f"{'Average Win (USDT)':<35} {avg_win_usdt:>19.2f}")
        print(f"{'Average Loss (USDT)':<35} {avg_loss_usdt:>19.2f}")
        print(f"{'Total PnL (USDT)':<35} {total_pnl:>19.2f}")
        print(f"{'Average PnL per Trade (USDT)':<35} {avg_pnl:>19.2f}")
        print(f"{'─'*75}")
        print(f"{'Profit Factor':<35} {profit_factor:>20.2f}")
        print(f"{'Expectancy (USDT/trade)':<35} {expectancy_usdt:>19.2f}")
        print(f"{'Standard Deviation':<35} {std_dev:>19.2f}")
        print(f"{'Standard Error':<35} {std_error:>19.2f}")
        
        return metrics
    
    def preliminary_verdict(self, metrics: Dict):
        """Issue verdict with strong caveats"""
        print("\n" + "="*80)
        print("PRELIMINARY ASSESSMENT (STATISTICALLY INVALID)")
        print("="*80)
        
        total_trades = metrics.get('total_trades', 0)
        min_required = metrics.get('min_sample_required', 200)
        expectancy = metrics.get('expectancy_usdt', 0)
        profit_factor = metrics.get('profit_factor', 0)
        win_rate = metrics.get('win_rate', 0)
        
        print(f"\n⚠️  SAMPLE SIZE: {total_trades} trades")
        print(f"⚠️  REQUIRED: {min_required:.0f} trades for 95% confidence")
        print(f"⚠️  DATA COMPLETENESS: {total_trades/min_required*100:<.1f}%\n")
        
        print("="*80)
        print("PROVISIONAL CLASSIFICATION (SUBJECT TO REVISION)")
        print("="*80)
        
        if expectancy > 1.0:
            classification = "POTENTIALLY POSITIVE"
            emoji = "⚠️"
        elif expectancy > -1.0:
            classification = "POTENTIALLY NEUTRAL"
            emoji = "⚠️"
        else:
            classification = "POTENTIALLY NEGATIVE"
            emoji = "⚠️"
        
        print(f"\n{emoji} {classification} EXPECTANCY\n")
        print(f"Expectancy: ${expectancy:.2f} USDT per trade")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Win Rate: {win_rate:.1%}\n")
        
        print("="*80)
        print("CRITICAL LIMITATIONS")
        print("="*80)
        
        limitations = [
            f"1. Sample size ({total_trades}) is {min_required/total_trades:.0f}x below statistical minimum",
            "2. Cannot calculate R-multiples (entry/exit/stop-loss data unavailable)",
            "3. Cannot assess risk-adjusted returns",
            "4. Cannot analyze distribution (insufficient data points)",
            "5. Cannot determine leverage sensitivity",
            "6. High variance expected with small sample",
            "7. Results may NOT be representative of system behavior",
            "8. Conclusions are PRELIMINARY and UNRELIABLE"
        ]
        
        for limitation in limitations:
            print(f"   {limitation}")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        recommendations = [
            "1. CONTINUE TRADING to accumulate data (target: 200+ closed trades)",
            "2. ENABLE detailed trade logging (entry/exit/stop-loss prices)",
            "3. VERIFY trade_history_logger service is running correctly",
            "4. RE-RUN audit after 200+ trades completed",
            "5. DO NOT draw firm conclusions from current dataset",
            "6. MONITOR for at least 100 more trades before system tuning"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n" + "="*80)
        print("WHAT WE CAN SAY WITH CURRENT DATA")
        print("="*80)
        
        observations = []
        
        if total_trades > 0:
            observations.append(f"✓ System is executing and closing positions ({total_trades} completed)")
        
        if win_rate > 0:
            observations.append(f"✓ {int(win_rate*100)}% of trades profitable (but sample too small)")
        
        if expectancy > 0:
            observations.append(f"✓ Early PnL trend appears positive (+${expectancy:.2f}/trade)")
        elif expectancy < 0:
            observations.append(f"⚠️  Early PnL trend appears negative (${expectancy:.2f}/trade)")
        
        if profit_factor > 1.0:
            observations.append(f"✓ Profit factor {profit_factor:.2f} > 1.0 (but not statistically significant)")
        else:
            observations.append(f"⚠️  Profit factor {profit_factor:.2f} < 1.0 (but not statistically significant)")
        
        observations.append("❌ NO RELIABLE CONCLUSIONS POSSIBLE YET")
        
        for obs in observations:
            print(f"   {obs}")
        
        print("\n" + "="*80)
        print("AUDIT CONCLUDED - DATA INSUFFICIENT")
        print("="*80)
        
        # Save results
        self.save_limited_results(metrics)
    
    def save_limited_results(self, metrics: Dict):
        """Save preliminary results"""
        output = {
            'audit_date': datetime.now().isoformat(),
            'data_status': 'INSUFFICIENT',
            'sample_size': len(self.trades),
            'min_required': metrics.get('min_sample_required', 200),
            'metrics': metrics,
            'trades': self.trades,
            'warnings': [
                'Sample size below statistical minimum',
                'Results not statistically significant',
                'Cannot calculate R-multiples without entry/exit/SL data',
                'Conclusions are preliminary only'
            ]
        }
        
        filename = "LIMITED_DATA_EXPECTANCY_ASSESSMENT_FEB18_2026.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Preliminary results saved to: {filename}")
    
    def run(self):
        """Execute limited data assessment"""
        print("\n" + "="*80)
        print("LIMITED DATA EXPECTANCY ASSESSMENT")
        print("Production Trading System - Real Data Analysis")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        trade_count = self.extract_pnl_history()
        
        if trade_count == 0:
            print("\n❌ NO TRADE DATA FOUND")
            print("   System may not have closed any positions yet")
            print("   Or trade_history_logger may not be running")
            return
        
        metrics = self.calculate_metrics()
        self.preliminary_verdict(metrics)


def main():
    assessor = LimitedDataAssessor()
    assessor.run()


if __name__ == "__main__":
    main()
