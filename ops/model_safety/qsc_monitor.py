#!/usr/bin/env python3
"""
QSC MONITOR - 6-Hour Canary Violation Detection

MONITORING RULES:
- Duration: 6 hours from start
- Check interval: Every 30 seconds
- Data source: scoreboard.py telemetry
- Violations trigger immediate rollback

VIOLATION CRITERIA (from quality_gate.py):
1. Majority class >70% (action collapse)
2. Confidence std <0.05 (flat predictions)
3. P10-P90 range <0.12 (narrow range)
4. HOLD >85% (dead zone)
5. Constant output (std<0.01 OR p10==p90)
6. Agreement <55% or >80% (ensemble dysfunction)
7. Hard disagree >20% (model chaos)

EXIT CODES:
  0 = Monitoring completed, no violations (canary safe)
  1 = Violation detected, rollback executed
  2 = Monitoring error

USAGE:
  python3 ops/model_safety/qsc_monitor.py
  python3 ops/model_safety/qsc_monitor.py --duration 3  # Custom duration (hours)
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Paths
SCOREBOARD_SCRIPT = Path(__file__).parent / "scoreboard.py"
QSC_LOG_FILE = Path("logs/qsc_canary.jsonl")
ROLLBACK_SCRIPT = Path(__file__).parent / "qsc_rollback.sh"

# Monitoring parameters
CHECK_INTERVAL_SECONDS = 30
MONITORING_DURATION_HOURS = 6


def get_latest_canary_config() -> dict:
    """Read latest canary activation from log"""
    if not QSC_LOG_FILE.exists():
        raise RuntimeError(f"No canary log found: {QSC_LOG_FILE}")
    
    with open(QSC_LOG_FILE) as f:
        lines = f.readlines()
    
    if not lines:
        raise RuntimeError("Canary log is empty")
    
    latest = json.loads(lines[-1])
    
    if latest.get('action') != 'canary_activated':
        raise RuntimeError(f"Latest log entry is not canary_activated: {latest.get('action')}")
    
    return latest


def run_scoreboard() -> dict:
    """
    Run scoreboard.py and parse output.
    
    Returns dict with:
    - models: {model_name: {status, stats}}
    - overall_status: GO/WAIT/NO-GO
    - agreement: float
    - hard_disagree: float
    """
    cmd = [sys.executable, str(SCOREBOARD_SCRIPT)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    
    if result.returncode != 0:
        raise RuntimeError(f"Scoreboard failed: {result.stderr}")
    
    # Parse scoreboard markdown report (stored in reports/safety/scoreboard_latest.md)
    report_path = Path("reports/safety/scoreboard_latest.md")
    
    if not report_path.exists():
        raise RuntimeError(f"Scoreboard report not found: {report_path}")
    
    return parse_scoreboard_report(report_path)


def parse_scoreboard_report(report_path: Path) -> dict:
    """
    Parse scoreboard markdown report.
    
    Extracts:
    - Overall status (GO/WAIT/NO-GO)
    - Agreement percentage
    - Hard disagree percentage
    - Per-model status and stats
    """
    content = report_path.read_text()
    lines = content.split('\n')
    
    data = {
        'overall_status': None,
        'agreement': None,
        'hard_disagree': None,
        'models': {}
    }
    
    current_model = None
    
    for i, line in enumerate(lines):
        # Overall status
        if line.startswith('**') and any(x in line for x in ['ALL-GO', 'WAIT', 'NO-GO']):
            if 'ALL-GO' in line:
                data['overall_status'] = 'GO'
            elif 'NO-GO' in line:
                data['overall_status'] = 'NO-GO'
            else:
                data['overall_status'] = 'WAIT'
        
        # Agreement
        if line.startswith('- Agreement:'):
            try:
                data['agreement'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        
        # Hard disagree
        if line.startswith('- Hard Disagree:'):
            try:
                data['hard_disagree'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        
        # Model sections
        if line.startswith('###') and any(x in line for x in ['GO', 'WAIT', 'NO-GO']):
            # Extract model name and status
            parts = line.split()
            if len(parts) >= 3:
                model_name = parts[1]
                status = parts[-1]
                
                current_model = model_name
                data['models'][model_name] = {
                    'status': status,
                    'action_pcts': {},
                    'conf_std': None,
                    'p10_p90_range': None,
                    'gate_pass': None
                }
        
        # Model stats (when inside model section)
        if current_model:
            # Action distribution
            if line.startswith('- BUY:') or line.startswith('- SELL:') or line.startswith('- HOLD:'):
                try:
                    action = line.split(':')[0].strip('- ')
                    pct = float(line.split(':')[1].strip().rstrip('%'))
                    data['models'][current_model]['action_pcts'][action] = pct
                except:
                    pass
            
            # Confidence std
            if line.startswith('- Std:'):
                try:
                    data['models'][current_model]['conf_std'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            # P10-P90 range
            if line.startswith('- P10-P90 Range:'):
                try:
                    data['models'][current_model]['p10_p90_range'] = float(line.split(':')[1].strip())
                except:
                    pass
            
            # Quality gate status
            if '✅ PASS' in line:
                data['models'][current_model]['gate_pass'] = True
            elif '❌ FAIL' in line:
                data['models'][current_model]['gate_pass'] = False
    
    return data


def check_violations(scoreboard: dict, canary_model: str) -> list[str]:
    """
    Check for quality violations in canary model.
    
    Returns list of violation descriptions (empty if no violations).
    """
    violations = []
    
    # Check canary model exists in scoreboard
    if canary_model not in scoreboard['models']:
        violations.append(f"Canary model {canary_model} not found in scoreboard")
        return violations
    
    model_data = scoreboard['models'][canary_model]
    
    # Violation 1: NO-GO status
    if model_data['status'] == 'NO-GO':
        violations.append("Model status: NO-GO (quality gate failed)")
    
    # Violation 2: Quality gate fail
    if model_data['gate_pass'] is False:
        violations.append("Quality gate: FAIL")
    
    # Violation 3: Majority class >70%
    for action, pct in model_data['action_pcts'].items():
        if pct > 70:
            violations.append(f"Action collapse: {action}={pct:.1f}% (>70%)")
    
    # Violation 4: HOLD >85%
    if model_data['action_pcts'].get('HOLD', 0) > 85:
        violations.append(f"Dead zone: HOLD={model_data['action_pcts']['HOLD']:.1f}% (>85%)")
    
    # Violation 5: Confidence std <0.05
    if model_data['conf_std'] is not None and model_data['conf_std'] < 0.05:
        violations.append(f"Flat predictions: conf_std={model_data['conf_std']:.4f} (<0.05)")
    
    # Violation 6: P10-P90 range <0.12
    if model_data['p10_p90_range'] is not None and model_data['p10_p90_range'] < 0.12:
        violations.append(f"Narrow range: p10_p90={model_data['p10_p90_range']:.4f} (<0.12)")
    
    # Violation 7: Ensemble agreement out of range
    if scoreboard['agreement'] is not None:
        if scoreboard['agreement'] < 55 or scoreboard['agreement'] > 80:
            violations.append(f"Agreement out of range: {scoreboard['agreement']:.1f}% (target 55-80%)")
    
    # Violation 8: Hard disagree >20%
    if scoreboard['hard_disagree'] is not None and scoreboard['hard_disagree'] > 20:
        violations.append(f"Hard disagree too high: {scoreboard['hard_disagree']:.1f}% (>20%)")
    
    return violations


def execute_rollback(canary_model: str, violations: list[str]):
    """
    Execute rollback script and log the event.
    """
    print()
    print("=" * 80)
    print("🚨 VIOLATION DETECTED - EXECUTING ROLLBACK")
    print("=" * 80)
    print()
    print(f"Canary Model: {canary_model}")
    print()
    print("Violations:")
    for v in violations:
        print(f"  ❌ {v}")
    print()
    print(f"Rollback Script: {ROLLBACK_SCRIPT}")
    print()
    
    # Execute rollback
    try:
        result = subprocess.run(
            ["bash", str(ROLLBACK_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            print(f"⚠️  Rollback script exited with code {result.returncode}")
        else:
            print("✅ Rollback completed successfully")
        
    except Exception as e:
        print(f"❌ Rollback execution failed: {e}")
    
    # Log rollback event
    rollback_ts = datetime.utcnow().isoformat() + 'Z'
    log_entry = {
        'timestamp': rollback_ts,
        'action': 'rollback_executed',
        'canary_model': canary_model,
        'violations': violations,
        'trigger': 'qsc_monitor_violation'
    }
    
    with open(QSC_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print()
    print(f"📋 Rollback logged: {QSC_LOG_FILE}")
    print()


def format_time_remaining(seconds: int) -> str:
    """Format seconds as human-readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"


def main():
    """Main monitoring loop"""
    import argparse
    
    parser = argparse.ArgumentParser(description='QSC Monitor - Canary Violation Detection')
    parser.add_argument(
        '--duration',
        type=float,
        default=MONITORING_DURATION_HOURS,
        help=f'Monitoring duration in hours (default: {MONITORING_DURATION_HOURS})'
    )
    args = parser.parse_args()
    
    print()
    print("=" * 80)
    print("QSC MONITOR - Canary Violation Detection")
    print("=" * 80)
    print()
    
    # Load canary config
    try:
        canary_config = get_latest_canary_config()
    except Exception as e:
        print(f"❌ Failed to load canary config: {e}")
        return 2
    
    canary_model = canary_config['canary_model']
    canary_start = datetime.fromisoformat(canary_config['timestamp'].rstrip('Z'))
    
    print(f"Canary Model:      {canary_model}")
    print(f"Start Time:        {canary_config['timestamp']}")
    print(f"Monitoring For:    {args.duration} hours")
    print(f"Check Interval:    {CHECK_INTERVAL_SECONDS}s")
    print()
    
    # Calculate end time
    end_time = canary_start + timedelta(hours=args.duration)
    total_checks = int(args.duration * 3600 / CHECK_INTERVAL_SECONDS)
    
    print(f"End Time:          {end_time.isoformat()}Z")
    print(f"Total Checks:      ~{total_checks}")
    print()
    print("=" * 80)
    print()
    
    check_count = 0
    
    try:
        while datetime.utcnow() < end_time:
            check_count += 1
            now = datetime.utcnow()
            remaining_seconds = int((end_time - now).total_seconds())
            
            print(f"[Check {check_count}/{total_checks}] {now.isoformat()}Z - {format_time_remaining(remaining_seconds)} remaining")
            
            try:
                # Run scoreboard
                scoreboard = run_scoreboard()
                
                # Check for violations
                violations = check_violations(scoreboard, canary_model)
                
                if violations:
                    # VIOLATION: Execute rollback
                    execute_rollback(canary_model, violations)
                    return 1
                
                # Report status
                status = scoreboard['models'][canary_model]['status']
                print(f"  Status: {status} ✅")
                
            except Exception as e:
                print(f"  ⚠️  Check failed: {e}")
                print(f"     Will retry in {CHECK_INTERVAL_SECONDS}s")
            
            # Wait for next check
            time.sleep(CHECK_INTERVAL_SECONDS)
        
        # Monitoring completed without violations
        print()
        print("=" * 80)
        print("✅ MONITORING COMPLETED - NO VIOLATIONS")
        print("=" * 80)
        print()
        print(f"Canary Model:      {canary_model}")
        print(f"Duration:          {args.duration} hours")
        print(f"Total Checks:      {check_count}")
        print()
        print("Canary is safe to promote to full production.")
        print()
        
        # Log success
        success_ts = datetime.utcnow().isoformat() + 'Z'
        log_entry = {
            'timestamp': success_ts,
            'action': 'monitoring_completed',
            'canary_model': canary_model,
            'duration_hours': args.duration,
            'checks_performed': check_count,
            'violations': None,
            'result': 'safe'
        }
        
        with open(QSC_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        return 0
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Monitoring interrupted by user")
        print()
        return 2


if __name__ == '__main__':
    sys.exit(main())
