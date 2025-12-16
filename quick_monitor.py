"""
Quick System Monitor - Watch live trading activity
"""
import time
from datetime import datetime
import subprocess
import sys

def run_command(cmd):
    """Run PowerShell command and return output."""
    try:
        result = subprocess.run(
            ['pwsh', '-Command', cmd],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def get_recent_signals():
    """Get last signal detection."""
    cmd = r'docker logs quantum_backend --tail 100 2>&1 | Select-String -Pattern "Found [0-9]+ high-confidence" | Select-Object -Last 1'
    return run_command(cmd)

def get_recent_execution():
    """Get last execution result."""
    cmd = r'docker logs quantum_backend --tail 100 2>&1 | Select-String -Pattern "orders_submitted" | Select-Object -Last 1'
    return run_command(cmd)

def get_positions():
    """Get active positions count."""
    cmd = r'curl -s http://localhost:8000/api/futures_positions 2>$null | ConvertFrom-Json | Where-Object {[double]$_.positionAmt -ne 0} | Measure-Object | Select-Object -ExpandProperty Count'
    return run_command(cmd)

def main():
    print("\n" + "="*60)
    print("  🤖 QUANTUM TRADER - LIVE MONITOR")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n[{timestamp}] Update #{iteration}")
            print("-" * 60)
            
            # Get signals
            signals = get_recent_signals()
            if "Found" in signals:
                signal_count = signals.split("Found ")[1].split(" ")[0]
                print(f"[CHART] Last Signal Scan: {signal_count} high-confidence detected")
            else:
                print(f"[CHART] Signals: {signals[:80]}")
            
            # Get execution
            execution = get_recent_execution()
            if "orders_submitted" in execution:
                try:
                    submitted = execution.split("'orders_submitted': ")[1].split(",")[0]
                    print(f"[OK] Last Execution: {submitted} orders submitted")
                except:
                    print(f"[OK] Execution: {execution[:80]}")
            else:
                print(f"[OK] Execution: {execution[:80]}")
            
            # Get positions
            pos_count = get_positions()
            try:
                count = int(pos_count) if pos_count.isdigit() else 0
                print(f"[BRIEFCASE] Active Positions: {count}")
            except:
                print(f"[BRIEFCASE] Positions: {pos_count[:50]}")
            
            print("-" * 60)
            
            # Wait before next update
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n[WARNING] Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
