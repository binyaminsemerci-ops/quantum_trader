#!/usr/bin/env python3
"""
Monitor AdaptiveLeverageEngine in Production
Tracks adaptive level calculations, harvest schemes, and SL clamps
"""
import redis
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List

REDIS_URL = "redis://redis:6379"
STREAM_KEY = "quantum:stream:adaptive_levels"


class AdaptiveLevelMonitor:
    """Monitor adaptive leverage calculations in production"""
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.stats = defaultdict(int)
        self.symbol_stats = defaultdict(lambda: {
            'count': 0,
            'avg_lsf': 0.0,
            'avg_leverage': 0.0,
            'sl_clamps': 0,
            'tp_minimums': 0,
            'harvest_schemes': defaultdict(int)
        })
    
    def get_recent_levels(self, count: int = 100) -> List[Dict]:
        """Get recent adaptive level calculations"""
        try:
            results = self.redis.xrevrange(STREAM_KEY, count=count)
            levels = []
            
            for msg_id, data in results:
                data['harvest_scheme'] = json.loads(data.get('harvest_scheme', '[]'))
                data['timestamp'] = float(data['timestamp'])
                data['leverage'] = float(data['leverage'])
                data['lsf'] = float(data['lsf'])
                levels.append(data)
            
            return levels
        except Exception as e:
            print(f"❌ Error fetching levels: {e}")
            return []
    
    def analyze_levels(self, levels: List[Dict]):
        """Analyze adaptive level patterns"""
        if not levels:
            print("⚠️  No adaptive levels found in stream")
            return
        
        print("\n" + "="*70)
        print("📊 ADAPTIVE LEVERAGE ENGINE - PRODUCTION MONITORING")
        print("="*70)
        
        # Reset stats
        self.stats = defaultdict(int)
        self.symbol_stats = defaultdict(lambda: {
            'count': 0,
            'avg_lsf': 0.0,
            'avg_leverage': 0.0,
            'sl_clamps': 0,
            'tp_minimums': 0,
            'harvest_schemes': defaultdict(int)
        })
        
        # Process all levels
        for level in levels:
            symbol = level.get('symbol', 'UNKNOWN')
            leverage = float(level.get('leverage', 0))
            lsf = float(level.get('lsf', 0))
            sl_clamped = level.get('sl_clamped', 'false') == 'true'
            tp_minimum = level.get('tp_minimum_enforced', 'false') == 'true'
            harvest = tuple(level.get('harvest_scheme', []))
            
            stats = self.symbol_stats[symbol]
            stats['count'] += 1
            stats['avg_lsf'] += lsf
            stats['avg_leverage'] += leverage
            
            if sl_clamped:
                stats['sl_clamps'] += 1
                self.stats['total_sl_clamps'] += 1
            
            if tp_minimum:
                stats['tp_minimums'] += 1
                self.stats['total_tp_minimums'] += 1
            
            if harvest:
                harvest_key = str(harvest)
                stats['harvest_schemes'][harvest_key] += 1
        
        # Compute averages
        for symbol, stats in self.symbol_stats.items():
            count = stats['count']
            stats['avg_lsf'] /= count
            stats['avg_leverage'] /= count
        
        # Overall stats
        print(f"\n📈 Overall Statistics (last {len(levels)} calculations):")
        print(f"  Total calculations: {len(levels)}")
        print(f"  Unique symbols: {len(self.symbol_stats)}")
        print(f"  SL clamps triggered: {self.stats['total_sl_clamps']} ({self.stats['total_sl_clamps']/len(levels)*100:.1f}%)")
        print(f"  TP minimums enforced: {self.stats['total_tp_minimums']} ({self.stats['total_tp_minimums']/len(levels)*100:.1f}%)")
        
        # Per-symbol breakdown
        print(f"\n📊 Per-Symbol Breakdown:")
        print(f"{'Symbol':<12} {'Count':>6} {'Avg Leverage':>13} {'Avg LSF':>9} {'SL Clamps':>11} {'Top Harvest Scheme':<20}")
        print("-" * 70)
        
        for symbol, stats in sorted(self.symbol_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]:
            count = stats['count']
            avg_lev = stats['avg_leverage']
            avg_lsf = stats['avg_lsf']
            sl_clamps = stats['sl_clamps']
            
            # Find most common harvest scheme
            if stats['harvest_schemes']:
                top_scheme = max(stats['harvest_schemes'].items(), key=lambda x: x[1])
                scheme_str = f"{top_scheme[0]} ({top_scheme[1]}x)"
            else:
                scheme_str = "N/A"
            
            print(f"{symbol:<12} {count:>6} {avg_lev:>13.1f}x {avg_lsf:>9.4f} {sl_clamps:>11} {scheme_str:<20}")
        
        # Recent examples
        print(f"\n🔍 Recent Examples (last 5):")
        for i, level in enumerate(levels[:5], 1):
            ts = datetime.fromtimestamp(level['timestamp']).strftime('%H:%M:%S')
            symbol = level['symbol']
            side = level['side']
            leverage = float(level['leverage'])
            lsf = float(level['lsf'])
            tp1 = float(level['tp1_pct']) * 100
            tp2 = float(level['tp2_pct']) * 100
            tp3 = float(level['tp3_pct']) * 100
            sl = float(level['sl_pct']) * 100
            harvest = level['harvest_scheme']
            
            print(f"  [{i}] {ts} | {symbol} {side.upper()} {leverage:.1f}x")
            print(f"      LSF={lsf:.4f} | TP: {tp1:.2f}%/{tp2:.2f}%/{tp3:.2f}% | SL: {sl:.2f}%")
            print(f"      Harvest: {harvest[0]*100:.0f}%/{harvest[1]*100:.0f}%/{harvest[2]*100:.0f}%")
            
            # Highlight warnings
            if level.get('sl_clamped') == 'true':
                print(f"      ⚠️  SL clamped to safety limit!")
            if level.get('tp_minimum_enforced') == 'true':
                print(f"      ⚠️  TP minimum enforced!")
        
        # Harvest scheme distribution
        print(f"\n📋 Harvest Scheme Distribution:")
        all_schemes = defaultdict(int)
        for stats in self.symbol_stats.values():
            for scheme, count in stats['harvest_schemes'].items():
                all_schemes[scheme] += count
        
        for scheme, count in sorted(all_schemes.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(levels) * 100
            print(f"  {scheme:<30} {count:>4} ({pct:>5.1f}%)")
        
        print("\n" + "="*70)
    
    def watch_stream(self, interval: int = 30):
        """Watch stream in real-time"""
        print(f"\n👀 Watching adaptive levels stream (refresh every {interval}s)...")
        print("Press Ctrl+C to stop\n")
        
        last_id = '$'  # Start from now
        
        try:
            while True:
                # Check for new entries
                results = self.redis.xread({STREAM_KEY: last_id}, block=interval * 1000, count=10)
                
                if results:
                    for stream_name, messages in results:
                        for msg_id, data in messages:
                            last_id = msg_id
                            
                            ts = datetime.fromtimestamp(float(data['timestamp'])).strftime('%H:%M:%S')
                            symbol = data['symbol']
                            side = data['side']
                            leverage = float(data['leverage'])
                            lsf = float(data['lsf'])
                            tp1 = float(data['tp1_pct']) * 100
                            sl = float(data['sl_pct']) * 100
                            
                            print(f"[{ts}] {symbol} {side.upper()} {leverage:.1f}x | LSF={lsf:.4f} | TP1={tp1:.2f}% | SL={sl:.2f}%", end='')
                            
                            if data.get('sl_clamped') == 'true':
                                print(" ⚠️ SL_CLAMPED", end='')
                            if data.get('tp_minimum_enforced') == 'true':
                                print(" ⚠️ TP_MIN", end='')
                            
                            print()
                
        except KeyboardInterrupt:
            print("\n\n✅ Stopped watching")


def main():
    import sys
    
    monitor = AdaptiveLevelMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'watch':
        # Real-time watch mode
        monitor.watch_stream()
    else:
        # Analysis mode
        count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
        levels = monitor.get_recent_levels(count=count)
        monitor.analyze_levels(levels)


if __name__ == "__main__":
    main()
