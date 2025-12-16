"""
Monitor Exit Brain V3 logs in real-time
"""

import time
import json
from pathlib import Path
from datetime import datetime

SHADOW_LOG = Path("backend/data/exit_brain_shadow.jsonl")
MAIN_LOG = Path("backend/logs/quantum_trader.log")

def tail_file(file_path, n=20):
    """Get last n lines from file."""
    if not file_path.exists():
        return []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        return lines[-n:]

def monitor_logs():
    """Monitor EXIT_BRAIN logs."""
    print("\n" + "="*60)
    print("EXIT BRAIN V3 LOG MONITOR")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check main log for EXIT messages
    print(f"\n📋 Recent EXIT messages in main log:")
    if MAIN_LOG.exists():
        lines = tail_file(MAIN_LOG, 100)
        exit_lines = [l for l in lines if 'EXIT' in l.upper()]
        
        if exit_lines:
            for line in exit_lines[-10:]:  # Show last 10
                print(f"   {line.strip()[:150]}")
        else:
            print(f"   No EXIT messages found in last 100 lines")
            print(f"   Log file: {MAIN_LOG}")
            print(f"   Last modified: {datetime.fromtimestamp(MAIN_LOG.stat().st_mtime)}")
    else:
        print(f"   ❌ Main log not found: {MAIN_LOG}")
    
    # Check shadow log
    print(f"\n🔍 Shadow mode log status:")
    if SHADOW_LOG.exists():
        size = SHADOW_LOG.stat().st_size
        modified = datetime.fromtimestamp(SHADOW_LOG.stat().st_mtime)
        print(f"   ✅ Shadow log exists: {SHADOW_LOG}")
        print(f"   Size: {size} bytes")
        print(f"   Last modified: {modified}")
        
        if size > 0:
            lines = tail_file(SHADOW_LOG, 5)
            print(f"\n   Recent shadow decisions:")
            for line in lines:
                try:
                    data = json.loads(line)
                    symbol = data.get('symbol', 'Unknown')
                    decision_type = data.get('decision', {}).get('decision_type', 'Unknown')
                    confidence = data.get('decision', {}).get('confidence', 0.0)
                    print(f"   - {symbol}: {decision_type} (confidence: {confidence:.2f})")
                except:
                    pass
        else:
            print(f"   ⚠️  Shadow log is empty (no decisions logged yet)")
    else:
        print(f"   ❌ Shadow log not found: {SHADOW_LOG}")
        print(f"   Expected location: {SHADOW_LOG.absolute()}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        monitor_logs()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
