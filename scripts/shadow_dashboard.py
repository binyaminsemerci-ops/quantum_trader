#!/usr/bin/env python3
"""
SHADOW MODEL REAL-TIME MONITORING DASHBOARD

Usage:
    python scripts/shadow_dashboard.py
    python scripts/shadow_dashboard.py --json  # JSON output for automation
    python scripts/shadow_dashboard.py --once  # Single run (no loop)

Features:
- Real-time champion and challenger metrics
- Promotion status and scoring
- Trade progress tracking
- Alert notifications
- Historical trends
"""

import requests
import time
import argparse
import json
import sys
from datetime import datetime
from typing import Dict, Any, List


class ShadowDashboard:
    """Real-time dashboard for shadow model monitoring"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.refresh_interval = 60  # seconds
        self.last_champion = None
        self.promotion_count = 0
    
    def fetch_status(self) -> Dict[str, Any]:
        """Fetch current shadow model status"""
        try:
            resp = requests.get(f"{self.api_url}/shadow/status", timeout=5)
            resp.raise_for_status()
            return resp.json()['data']
        except Exception as e:
            print(f"❌ Error fetching status: {e}")
            return None
    
    def fetch_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Fetch promotion history"""
        try:
            resp = requests.get(f"{self.api_url}/shadow/history?n={n}", timeout=5)
            resp.raise_for_status()
            return resp.json()['data']
        except Exception as e:
            return []
    
    def fetch_comparison(self, challenger_name: str) -> Dict[str, Any]:
        """Fetch detailed comparison for challenger"""
        try:
            resp = requests.get(
                f"{self.api_url}/shadow/comparison/{challenger_name}",
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()['data']
        except Exception as e:
            return None
    
    def print_dashboard(self, data: Dict[str, Any], history: List[Dict[str, Any]]):
        """Print formatted dashboard"""
        
        # Clear screen (ANSI escape code)
        print("\033[2J\033[H", end="")
        
        # Header
        print("=" * 100)
        print(f"{'SHADOW MODEL MONITORING DASHBOARD':^100}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^100}")
        print("=" * 100)
        
        # Champion Section
        champ = data['champion']
        print(f"\n🏆 CHAMPION: {champ['model_name']}")
        print("─" * 100)
        
        if champ['metrics']:
            metrics = champ['metrics']
            print(f"  Trades:        {champ['trade_count']:>6} trades")
            print(f"  Win Rate:      {metrics['win_rate']:>6.2%}  {'🟢' if metrics['win_rate'] >= 0.56 else '🔴'}")
            print(f"  Sharpe Ratio:  {metrics['sharpe_ratio']:>6.2f}  {'🟢' if metrics['sharpe_ratio'] >= 1.5 else '🔴'}")
            print(f"  Mean PnL:      ${metrics['mean_pnl']:>6.2f}")
            print(f"  Total PnL:     ${metrics['total_pnl']:>8,.2f}")
            print(f"  Max Drawdown:  ${metrics['max_drawdown']:>8,.2f}")
            print(f"  Sortino Ratio: {metrics['sortino_ratio']:>6.2f}")
            
            # Health indicator
            health_score = 0
            if metrics['win_rate'] >= 0.56:
                health_score += 33
            if metrics['sharpe_ratio'] >= 1.5:
                health_score += 33
            if metrics['max_drawdown'] <= 5000:
                health_score += 34
            
            if health_score >= 90:
                health_emoji = "🟢 EXCELLENT"
            elif health_score >= 70:
                health_emoji = "🟡 GOOD"
            elif health_score >= 50:
                health_emoji = "🟠 FAIR"
            else:
                health_emoji = "🔴 POOR"
            
            print(f"  Health:        {health_score:>3}/100  {health_emoji}")
        else:
            print("  [No metrics yet - accumulating trades]")
        
        # Challengers Section
        challengers = data['challengers']
        print(f"\n🎯 CHALLENGERS: {len(challengers)}")
        print("─" * 100)
        
        if not challengers:
            print("  No challengers currently testing.")
            print("  💡 Deploy a challenger to begin shadow testing.")
        else:
            for i, chal in enumerate(challengers, 1):
                print(f"\n  [{i}] {chal['model_name']}")
                print(f"      Status:  {chal['promotion_status']:>12}  ", end="")
                
                # Status emoji
                if chal['promotion_status'] == 'APPROVED':
                    print("✅ READY TO PROMOTE")
                elif chal['promotion_status'] == 'PENDING':
                    print("⚠️  MANUAL REVIEW NEEDED")
                elif chal['promotion_status'] == 'REJECTED':
                    print("❌ REJECTED")
                else:
                    print("⏳ TESTING...")
                
                # Progress bar
                progress = min(chal['trade_count'] / 500, 1.0)
                bar_width = 40
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"      Progress: [{bar}] {chal['trade_count']}/500 trades")
                
                # Score
                score = chal['promotion_score']
                print(f"      Score:    {score:>5.1f}/100  ", end="")
                if score >= 70:
                    print("🟢 AUTO-PROMOTE")
                elif score >= 50:
                    print("🟡 MANUAL REVIEW")
                else:
                    print("🔴 LIKELY REJECT")
                
                # Metrics (if available)
                if chal['metrics']:
                    metrics = chal['metrics']
                    print(f"      WR:       {metrics['win_rate']:>6.2%}  ", end="")
                    
                    # Compare to champion
                    if champ['metrics']:
                        diff = metrics['win_rate'] - champ['metrics']['win_rate']
                        if diff > 0:
                            print(f"(+{diff:.2%} vs champion 🟢)")
                        else:
                            print(f"({diff:.2%} vs champion 🔴)")
                    else:
                        print()
                    
                    print(f"      Sharpe:   {metrics['sharpe_ratio']:>6.2f}")
                    print(f"      Mean PnL: ${metrics['mean_pnl']:>6.2f}")
                
                # Reason (if rejected/pending)
                if chal['reason']:
                    print(f"      Reason:   {chal['reason']}")
        
        # Recent Promotions
        print(f"\n📊 RECENT PROMOTIONS (Last {len(history)})")
        print("─" * 100)
        
        if not history:
            print("  No promotions yet.")
        else:
            for event in history[:5]:  # Show last 5
                timestamp = event.get('timestamp', 'N/A')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        pass
                
                old_champ = event.get('old_champion', 'N/A')
                new_champ = event.get('new_champion', 'N/A')
                score = event.get('promotion_score', 0)
                
                print(f"  • {timestamp}: {new_champ} → Champion (score: {score:.1f}/100)")
                print(f"    Replaced: {old_champ}")
                
                # Performance improvement
                if 'performance_improvement' in event:
                    improvement = event['performance_improvement']
                    wr_imp = improvement.get('wr', 0) * 100
                    sharpe_imp = improvement.get('sharpe', 0)
                    print(f"    Improvement: WR +{wr_imp:.2f}pp, Sharpe +{sharpe_imp:.2f}")
        
        # System Stats
        print(f"\n⚙️  SYSTEM STATS")
        print("─" * 100)
        print(f"  API Endpoint:      {self.api_url}")
        print(f"  Refresh Interval:  {self.refresh_interval}s")
        print(f"  Total Promotions:  {len(history)}")
        print(f"  Active Challengers: {len(challengers)}")
        
        # Alerts
        alerts = []
        
        # Check for APPROVED challengers
        for chal in challengers:
            if chal['promotion_status'] == 'APPROVED':
                alerts.append(f"🔔 {chal['model_name']} ready for promotion!")
        
        # Check for PENDING challengers
        for chal in challengers:
            if chal['promotion_status'] == 'PENDING':
                alerts.append(f"⚠️  {chal['model_name']} needs manual review")
        
        # Check champion health
        if champ['metrics']:
            if champ['metrics']['win_rate'] < 0.52:
                alerts.append(f"🚨 Champion WR critically low: {champ['metrics']['win_rate']:.2%}")
            elif champ['metrics']['win_rate'] < 0.55:
                alerts.append(f"⚠️  Champion WR below target: {champ['metrics']['win_rate']:.2%}")
        
        if alerts:
            print(f"\n🔔 ALERTS ({len(alerts)})")
            print("─" * 100)
            for alert in alerts:
                print(f"  {alert}")
        
        # Footer
        print("\n" + "=" * 100)
        print(f"Press Ctrl+C to exit  |  Next refresh in {self.refresh_interval}s")
        print("=" * 100)
    
    def print_json(self, data: Dict[str, Any]):
        """Print JSON format for automation"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'champion': data['champion'],
            'challengers': data['challengers'],
            'alerts': []
        }
        
        # Add alerts
        for chal in data['challengers']:
            if chal['promotion_status'] == 'APPROVED':
                output['alerts'].append({
                    'level': 'HIGH',
                    'message': f"{chal['model_name']} ready for promotion",
                    'model': chal['model_name']
                })
        
        print(json.dumps(output, indent=2))
    
    def run_once(self, json_output: bool = False):
        """Run dashboard once (no loop)"""
        data = self.fetch_status()
        if data is None:
            sys.exit(1)
        
        if json_output:
            self.print_json(data)
        else:
            history = self.fetch_history(n=10)
            self.print_dashboard(data, history)
    
    def run_loop(self):
        """Run dashboard in continuous loop"""
        print("Starting Shadow Model Dashboard...")
        print(f"Connecting to {self.api_url}...")
        
        try:
            while True:
                data = self.fetch_status()
                if data is not None:
                    history = self.fetch_history(n=10)
                    self.print_dashboard(data, history)
                    
                    # Detect promotions
                    current_champ = data['champion']['model_name']
                    if self.last_champion and current_champ != self.last_champion:
                        # Champion changed!
                        print(f"\n🎉 PROMOTION DETECTED: {self.last_champion} → {current_champ}")
                        self.promotion_count += 1
                    
                    self.last_champion = current_champ
                
                time.sleep(self.refresh_interval)
        
        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user.")
            print(f"Promotions detected during session: {self.promotion_count}")
        except Exception as e:
            print(f"\n\n❌ Fatal error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Shadow Model Monitoring Dashboard"
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API base URL (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Refresh interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output JSON format (for automation)'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (no continuous loop)'
    )
    
    args = parser.parse_args()
    
    dashboard = ShadowDashboard(api_url=args.api_url)
    dashboard.refresh_interval = args.interval
    
    if args.once:
        dashboard.run_once(json_output=args.json)
    else:
        dashboard.run_loop()


if __name__ == '__main__':
    main()
