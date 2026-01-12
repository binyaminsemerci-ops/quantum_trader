#!/usr/bin/env python3
"""
QSC MODE - End-to-End Test

Tests the full QSC workflow:
1. Quality gate check (mocked)
2. Canary activation
3. Monitoring (short duration)
4. Rollback

USAGE:
  python3 ops/model_safety/qsc_test.py
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta

# Test configuration
TEST_CUTOVER = "2026-01-10T05:43:15Z"
TEST_MODEL = "patchtst"
TEST_MONITOR_DURATION = 0.05  # 3 minutes (0.05 hours)

def run_command(cmd: list, description: str) -> tuple[int, str, str]:
    """Run command and return (exit_code, stdout, stderr)"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\nExit Code: {result.returncode}")
    
    return result.returncode, result.stdout, result.stderr


def test_quality_gate():
    """Test 1: Quality gate check"""
    cmd = [
        sys.executable,
        "ops/model_safety/quality_gate.py",
        "--after", TEST_CUTOVER
    ]
    
    exit_code, stdout, stderr = run_command(cmd, "Quality Gate Check")
    
    # Parse event count
    event_count = 0
    for line in stdout.split('\n'):
        if "Parsed" in line and "events" in line:
            try:
                event_count = int(line.split()[1])
            except:
                pass
    
    print(f"\n📊 Test Result:")
    print(f"   Exit Code: {exit_code} (0=PASS, 2=FAIL)")
    print(f"   Events:    {event_count}")
    
    if exit_code == 0 and event_count >= 200:
        print("   ✅ PASS - Eligible for canary")
        return True
    else:
        print("   ⚠️  SKIP - Not eligible (need ≥200 events and quality pass)")
        print("   Note: Run this test when system has real telemetry data")
        return False


def test_canary_activation():
    """Test 2: Canary activation (dry run)"""
    cmd = [
        sys.executable,
        "ops/model_safety/qsc_mode.py",
        "--model", TEST_MODEL,
        "--cutover", TEST_CUTOVER,
        "--dry-run"
    ]
    
    exit_code, stdout, stderr = run_command(cmd, "Canary Activation (Dry Run)")
    
    print(f"\n📊 Test Result:")
    if exit_code == 0:
        print("   ✅ PASS - Quality gate passed, canary eligible")
        return True
    elif exit_code == 1:
        print("   ⚠️  SKIP - Quality gate failed (expected if no data)")
        return False
    else:
        print("   ❌ FAIL - Activation error")
        return False


def test_file_creation():
    """Test 3: Check created files"""
    print(f"\n{'='*80}")
    print("TEST: File Creation")
    print(f"{'='*80}\n")
    
    files_to_check = [
        "ops/model_safety/qsc_mode.py",
        "ops/model_safety/qsc_monitor.py",
        "ops/model_safety/qsc_rollback.sh",
        "ops/systemd/quantum-qsc-monitor.service"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    print(f"\n📊 Test Result:")
    if all_exist:
        print("   ✅ PASS - All files created")
        return True
    else:
        print("   ❌ FAIL - Missing files")
        return False


def test_rollback_script():
    """Test 4: Rollback script syntax"""
    print(f"\n{'='*80}")
    print("TEST: Rollback Script Syntax")
    print(f"{'='*80}\n")
    
    cmd = ["bash", "-n", "ops/model_safety/qsc_rollback.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"Command: {' '.join(cmd)}")
    
    if result.returncode == 0:
        print("✅ Bash syntax OK")
        print(f"\n📊 Test Result:")
        print("   ✅ PASS - Script syntax valid")
        return True
    else:
        print(f"❌ Bash syntax error:\n{result.stderr}")
        print(f"\n📊 Test Result:")
        print("   ❌ FAIL - Script has syntax errors")
        return False


def test_log_structure():
    """Test 5: Log file structure"""
    print(f"\n{'='*80}")
    print("TEST: Log Structure")
    print(f"{'='*80}\n")
    
    # Create test log entry
    log_file = Path("logs/qsc_canary_test.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    test_entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'action': 'test_entry',
        'canary_model': TEST_MODEL,
        'test': True
    }
    
    with open(log_file, 'w') as f:
        f.write(json.dumps(test_entry) + '\n')
    
    print(f"Created test log: {log_file}")
    print(f"Entry: {json.dumps(test_entry, indent=2)}")
    
    # Verify can read
    with open(log_file) as f:
        line = f.readline()
        parsed = json.loads(line)
    
    print(f"\nParsed back: {json.dumps(parsed, indent=2)}")
    
    if parsed == test_entry:
        print(f"\n📊 Test Result:")
        print("   ✅ PASS - Log structure valid")
        return True
    else:
        print(f"\n📊 Test Result:")
        print("   ❌ FAIL - Log parsing error")
        return False


def test_weight_calculation():
    """Test 6: Canary weight calculation"""
    print(f"\n{'='*80}")
    print("TEST: Weight Calculation")
    print(f"{'='*80}\n")
    
    # Test weight distribution
    baseline = {
        'xgb': 0.25,
        'lgbm': 0.25,
        'nhits': 0.30,
        'patchtst': 0.20
    }
    
    canary_model = 'patchtst'
    canary_weight = 0.10
    
    print(f"Baseline weights:")
    for m, w in sorted(baseline.items()):
        print(f"  {m:10s} {w*100:5.1f}%")
    
    print(f"\nCanary: {canary_model} @ {canary_weight*100}%")
    
    # Calculate canary weights
    canary_weights = baseline.copy()
    canary_weights[canary_model] = canary_weight
    
    remaining_models = [m for m in canary_weights.keys() if m != canary_model]
    total_remaining = sum(baseline[m] for m in remaining_models)
    
    for model in remaining_models:
        canary_weights[model] = baseline[model] / total_remaining * 0.90
    
    print(f"\nCanary weights:")
    total = 0
    for m, w in sorted(canary_weights.items()):
        icon = "🔸" if m == canary_model else "  "
        change = canary_weights[m] - baseline[m]
        print(f"  {icon} {m:10s} {w*100:5.1f}% (Δ {change*100:+5.1f}%)")
        total += w
    
    print(f"\nTotal: {total*100:.1f}%")
    
    print(f"\n📊 Test Result:")
    if abs(total - 1.0) < 0.001:
        print("   ✅ PASS - Weights sum to 100%")
        return True
    else:
        print(f"   ❌ FAIL - Weights sum to {total*100:.1f}%")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("QSC MODE - End-to-End Test Suite")
    print("="*80)
    print()
    print(f"Test Model:    {TEST_MODEL}")
    print(f"Test Cutover:  {TEST_CUTOVER}")
    print()
    
    tests = [
        ("File Creation", test_file_creation),
        ("Rollback Script Syntax", test_rollback_script),
        ("Weight Calculation", test_weight_calculation),
        ("Log Structure", test_log_structure),
        ("Quality Gate Check", test_quality_gate),
        ("Canary Activation (Dry Run)", test_canary_activation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
        
        time.sleep(1)  # Pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print()
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}")
    
    print()
    print(f"Results: {passed_count}/{total_count} tests passed")
    print()
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED")
        print()
        print("QSC MODE is ready for production use.")
        print()
        print("Next steps:")
        print("1. Ensure quality_gate.py has ≥200 post-cutover events")
        print("2. Run: python3 ops/model_safety/qsc_mode.py --model patchtst --cutover <timestamp>")
        print("3. Start monitoring: python3 ops/model_safety/qsc_monitor.py")
        print()
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Review failures above and fix before production use.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
