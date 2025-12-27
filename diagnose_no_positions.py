#!/usr/bin/env python3
"""
🔍 DIAGNOSE: Hvorfor åpnes ingen posisjoner?
"""
import subprocess
import json

def check_logs(pattern, label):
    """Check docker logs for pattern"""
    print(f"\n{'='*80}")
    print(f"🔍 {label}")
    print('='*80)
    
    result = subprocess.run(
        ["docker", "logs", "quantum_backend", "--tail", "500"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    lines = [line for line in result.stdout.split('\n') + result.stderr.split('\n')
             if pattern.lower() in line.lower()]
    
    if lines:
        for line in lines[-10:]:  # Last 10 matches
            print(line)
    else:
        print(f"❌ No lines found with '{pattern}'")

def main():
    print("🔍 DIAGNOSING: Why are no positions opening?\n")
    
    # Check 1: Signal generation
    check_logs("signal", "1. SIGNAL GENERATION")
    
    # Check 2: Confidence levels
    check_logs("confidence", "2. CONFIDENCE LEVELS")
    
    # Check 3: Position opening attempts
    check_logs("opening position", "3. POSITION OPENING ATTEMPTS")
    
    # Check 4: Errors
    check_logs("error", "4. ERRORS")
    
    # Check 5: Entry signals
    check_logs("entry", "5. ENTRY SIGNALS")
    
    # Check 6: Trading system status
    check_logs("trading", "6. TRADING SYSTEM")
    
    # Check 7: Strategy execution
    check_logs("strategy", "7. STRATEGY EXECUTION")
    
    # Check 8: Risk management
    check_logs("risk", "8. RISK MANAGEMENT")
    
    print(f"\n{'='*80}")
    print("💡 SUMMARY")
    print('='*80)
    print("Check the sections above for:")
    print("  • Are signals being generated?")
    print("  • Are confidence levels too low?")
    print("  • Are there errors blocking position opening?")
    print("  • Is the trading system active?")
    print("  • Is risk management blocking entries?")

if __name__ == "__main__":
    main()
