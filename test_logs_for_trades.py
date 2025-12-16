"""
Quick test: Count closed trades in Docker logs
"""
import subprocess
import re

print("\n🔍 Checking for closed trades in logs...\n")

try:
    result = subprocess.run(
        ["docker", "logs", "quantum_backend"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore',
        timeout=30
    )
    logs = (result.stdout or "") + (result.stderr or "")
    
    # Count MEMO CLOSE LOG entries
    close_logs = logs.count('[MEMO] CLOSE LOG:')
    
    # Count position closed detections
    pos_closed = len(re.findall(r'Position.*CLOSED detected', logs))
    
    # Count RL updates
    rl_updates = logs.count('Q-table updated')
    
    print(f"📊 LOG STATISTICS:")
    print(f"   [MEMO] CLOSE LOG entries: {close_logs}")
    print(f"   Position CLOSED detections: {pos_closed}")
    print(f"   Q-table updates: {rl_updates}")
    print()
    
    if close_logs > 0:
        print(f"✅ Found {close_logs} closed trades - RL replay is possible!")
        
        # Show first few
        lines = logs.split('\n')
        count = 0
        for i, line in enumerate(lines):
            if '[MEMO] CLOSE LOG:' in line and count < 3:
                print(f"\nExample trade {count+1}:")
                for j in range(i, min(i+10, len(lines))):
                    if lines[j].strip():
                        print(f"  {lines[j]}")
                count += 1
    else:
        print("❌ No closed trades found in current logs")
        print("   Logs might have been cleared by backend restart")
    
except Exception as e:
    print(f"❌ Error: {e}")
