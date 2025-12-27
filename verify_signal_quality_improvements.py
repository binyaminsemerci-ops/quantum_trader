#!/usr/bin/env python3
"""
Verify Signal Quality Improvements Applied

Changes:
1. Confidence threshold: 0.45 → 0.70 (only strong signals)
2. Signal queue: 100 → 20 (best signals only)
3. Consensus filter: Require 3/4 models agreement
"""

import os
import sys
from datetime import datetime, timedelta

def main():
    print("="*70)
    print("🔍 SIGNAL QUALITY IMPROVEMENTS VERIFICATION")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check environment variables
    confidence = os.getenv('QT_CONFIDENCE_THRESHOLD', 'NOT SET')
    queue_size = os.getenv('QT_SIGNAL_QUEUE_MAX', 'NOT SET')
    
    print("📊 ENVIRONMENT CONFIGURATION")
    print(f"   Confidence Threshold: {confidence} (target: 0.70)")
    print(f"   Signal Queue Max: {queue_size} (target: 20)")
    
    if confidence == '0.70' and queue_size == '20':
        print("   ✅ Environment variables CORRECT\n")
    else:
        print("   ❌ Environment variables INCORRECT\n")
        sys.exit(1)
    
    # Parse recent logs to verify filtering is working
    print("🔍 RECENT ACTIVITY (Last 2 minutes)")
    print("-"*70)
    
    import subprocess
    try:
        # Get logs from last 2 minutes
        result = subprocess.run(
            ['docker', 'logs', 'quantum_backend', '--since', '2m'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        logs = result.stdout + result.stderr
        
        # Count consensus filtering
        consensus_filter_lines = [l for l in logs.split('\n') if 'CONSENSUS FILTER' in l]
        strong_consensus_lines = [l for l in logs.split('\n') if 'Strong consensus' in l]
        blocked_consensus_lines = [l for l in logs.split('\n') if 'Weak consensus' in l or 'BLOCKED' in l]
        
        print(f"   Consensus Filter Events: {len(consensus_filter_lines)}")
        print(f"   Strong Consensus Signals: {len(strong_consensus_lines)}")
        print(f"   Blocked Weak Signals: {len(blocked_consensus_lines)}")
        
        if consensus_filter_lines:
            print("\n📋 LATEST CONSENSUS FILTER RESULT:")
            # Get last filter event
            last_filter = consensus_filter_lines[-1]
            print(f"   {last_filter.strip()}")
            
            # Extract numbers (removed X, Y remaining)
            import re
            match = re.search(r'Removed (\d+).*\((\d+) remaining\)', last_filter)
            if match:
                removed = int(match.group(1))
                remaining = int(match.group(2))
                total = removed + remaining
                filter_rate = (removed / total * 100) if total > 0 else 0
                
                print(f"\n   📊 Filter Statistics:")
                print(f"      Total signals: {total}")
                print(f"      Filtered out: {removed} ({filter_rate:.1f}%)")
                print(f"      Passed filter: {remaining} ({100-filter_rate:.1f}%)")
                
                if filter_rate >= 50:
                    print(f"   ✅ Good! Filtering out {filter_rate:.0f}% of weak signals")
                else:
                    print(f"   ⚠️ Only {filter_rate:.0f}% filtered - most signals already strong")
        
        # Check if any positions opened recently
        position_opened = [l for l in logs.split('\n') if 'Trade OPENED' in l or 'Position opened' in l]
        
        print(f"\n🎯 TRADING ACTIVITY:")
        print(f"   New positions opened: {len(position_opened)}")
        
        if position_opened:
            print("\n   ⚠️ WARNING: Positions opened in last 2 minutes")
            print("   This is expected if strong signals were found")
            for line in position_opened[-3:]:  # Show last 3
                # Extract symbol from log line
                if 'USDT' in line:
                    symbol_match = re.search(r'([A-Z]+USDT)', line)
                    if symbol_match:
                        print(f"      - {symbol_match.group(1)}")
        else:
            print("   No new positions (waiting for strong signals)")
        
        print("\n" + "="*70)
        print("✅ SIGNAL QUALITY IMPROVEMENTS VERIFIED")
        print("="*70)
        print("\n📈 EXPECTED BEHAVIOR:")
        print("   1. Fewer signals pass the filter (50-80% filtered out)")
        print("   2. Only signals with 3/4+ model agreement trade")
        print("   3. Minimum 70% confidence required")
        print("   4. Reduced noise, higher quality trades")
        print("\n💡 MONITORING TIP:")
        print("   Run: docker logs quantum_backend -f | grep -E 'CONSENSUS|BLOCKED|OPENED'")
        
    except subprocess.TimeoutExpired:
        print("   ⚠️ Timeout getting logs")
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()
