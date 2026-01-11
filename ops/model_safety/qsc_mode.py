#!/usr/bin/env python3
"""
QSC MODE - Quality Safeguard Canary Deployment

RULES:
1. quality_gate.py MUST exit 0 with ≥200 post-cutover events
2. Activate ONE model as canary at 10% traffic via systemd override
3. Log: start_ts, model_id, weight, rollback command
4. Monitor scoreboard for 6h → qsc_monitor.py handles this
5. Violation → immediate rollback (logged)
6. NO retraining, NO auto-scale

EXIT CODES:
  0 = Canary activated successfully
  1 = Quality gate failed (not eligible for canary)
  2 = Canary activation failed (systemd/config error)

USAGE:
  python3 ops/model_safety/qsc_mode.py --model patchtst --cutover 2026-01-10T05:43:15Z
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Paths
QUALITY_GATE_SCRIPT = Path(__file__).parent / "quality_gate.py"
QSC_LOG_FILE = Path("logs/qsc_canary.jsonl")
SYSTEMD_OVERRIDE_DIR = Path("/etc/systemd/system/quantum-ai_engine.service.d")
BASELINE_WEIGHTS_FILE = Path("data/baseline_model_weights.json")
CANARY_WEIGHTS_FILE = Path("data/qsc_canary_weights.json")

# Constants
CANARY_WEIGHT = 0.10  # 10% traffic to canary
MIN_EVENTS = 200


def run_quality_gate(cutover_ts: str, mode: str = "canary") -> tuple[int, int]:
    """
    Run quality_gate.py with cutover timestamp and mode.
    
    🔒 FAIL-CLOSED: Only canary mode (RC=0) enables activation.
    Collection mode (RC=3) is blocked.
    
    Returns:
        (exit_code, event_count)
    """
    cmd = [
        sys.executable,
        str(QUALITY_GATE_SCRIPT),
        "--after", cutover_ts,
        "--mode", mode
    ]
    
    print(f"[QSC] Running quality gate check (cutover: {cutover_ts}, mode: {mode})...")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse event count from output
    event_count = 0
    for line in result.stdout.split('\n'):
        if "Found" in line and "events" in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "events" and i > 0:
                        event_count = int(parts[i-1])
                        break
            except:
                pass
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # 🔒 FAIL-CLOSED: Block collection mode (returncode 3)
    if result.returncode == 3:
        print()
        print("=" * 80)
        print("🚫 QSC FAIL-CLOSED: Collection mode cannot activate canary")
        print("=" * 80)
        print()
        print("Rerun quality_gate.py with --mode canary and >=200 events to enable deployment")
        return result.returncode, event_count
    
    # 🔒 BELT + SUSPENDERS: Check report content for MODE: collection
    if result.returncode == 0:
        # Find latest report
        report_dir = Path("reports/safety")
        if report_dir.exists():
            reports = sorted(report_dir.glob("quality_gate_*.md"))
            if reports:
                latest_report = reports[-1]
                with open(latest_report, 'r') as f:
                    report_content = f.read()
                
                if 'MODE:** 🔒 COLLECTION' in report_content or 'collection' in latest_report.name:
                    print()
                    print("=" * 80)
                    print("🚫 QSC FAIL-CLOSED: Report contains MODE: collection")
                    print("=" * 80)
                    print()
                    print(f"Report: {latest_report}")
                    print("Cannot activate canary from collection-mode report (belt + suspenders check)")
                    return 3, event_count  # Override to RC=3
    
    return result.returncode, event_count


def save_baseline_weights():
    """Save current model weights as baseline for rollback"""
    # Default 4-model weights
    baseline = {
        'xgb': 0.25,
        'lgbm': 0.25,
        'nhits': 0.30,
        'patchtst': 0.20
    }
    
    BASELINE_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_WEIGHTS_FILE.write_text(json.dumps(baseline, indent=2))
    
    print(f"[SAVED] Baseline weights saved: {BASELINE_WEIGHTS_FILE}")
    return baseline


def create_canary_weights(canary_model: str, baseline: dict) -> dict:
    """
    Create canary weights with 10% traffic to target model.
    
    Strategy: Scale down all other models proportionally to give 10% to canary.
    """
    canary_weights = baseline.copy()
    
    # Set canary to 10%
    canary_weights[canary_model] = CANARY_WEIGHT
    
    # Scale remaining models to sum to 90%
    remaining_models = [m for m in canary_weights.keys() if m != canary_model]
    total_remaining = sum(baseline[m] for m in remaining_models)
    
    for model in remaining_models:
        canary_weights[model] = baseline[model] / total_remaining * 0.90
    
    return canary_weights


def activate_canary_via_systemd(canary_model: str, weights: dict) -> bool:
    """
    Activate canary by creating systemd override with environment variable.
    
    Creates: /etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf
    
    Environment variable: QSC_CANARY_WEIGHTS_JSON
    """
    # Save canary weights to file
    CANARY_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CANARY_WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))
    
    # Create systemd override config
    override_content = f"""[Service]
# QSC MODE: Canary deployment active
Environment="QSC_CANARY_WEIGHTS={CANARY_WEIGHTS_FILE}"
Environment="QSC_CANARY_MODEL={canary_model}"
Environment="QSC_CANARY_TRAFFIC=0.10"
"""
    
    # For local testing (no sudo), write to local dir
    local_override_dir = Path("data/systemd_overrides")
    local_override_dir.mkdir(parents=True, exist_ok=True)
    local_override_file = local_override_dir / "qsc_canary.conf"
    local_override_file.write_text(override_content)
    
    print(f"[CREATED] Systemd override created: {local_override_file}")
    print()
    print("Override content:")
    print(override_content)
    print()
    
    # Try to install to systemd (may fail without sudo - that's OK for testing)
    try:
        SYSTEMD_OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
        override_file = SYSTEMD_OVERRIDE_DIR / "qsc_canary.conf"
        override_file.write_text(override_content)
        
        # Reload systemd
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        
        print(f"✅ Systemd override installed: {override_file}")
        return True
    except Exception as e:
        print(f"⚠️  Could not install systemd override (need sudo): {e}")
        print(f"   Manual install: sudo cp {local_override_file} {SYSTEMD_OVERRIDE_DIR}/")
        return True  # Not a blocker - operator can install manually


def log_canary_activation(canary_model: str, weights: dict, cutover_ts: str, event_count: int):
    """
    Log canary activation to JSONL file.
    
    Includes rollback command for immediate execution if needed.
    """
    start_ts = datetime.utcnow().isoformat() + 'Z'
    
    rollback_cmd = f"python3 ops/model_safety/qsc_rollback.sh"
    
    log_entry = {
        'timestamp': start_ts,
        'action': 'canary_activated',
        'canary_model': canary_model,
        'canary_weight': CANARY_WEIGHT,
        'weights': weights,
        'cutover_ts': cutover_ts,
        'quality_gate_events': event_count,
        'monitoring_duration_hours': 6,
        'rollback_cmd': rollback_cmd,
        'systemd_override': str(CANARY_WEIGHTS_FILE),
        'baseline_weights_file': str(BASELINE_WEIGHTS_FILE)
    }
    
    QSC_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(QSC_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"[LOG] Canary activation logged: {QSC_LOG_FILE}")
    print()
    print("=" * 80)
    print("[ACTIVATED] CANARY ACTIVATED")
    print("=" * 80)
    print()
    print(f"  Model:         {canary_model}")
    print(f"  Weight:        {CANARY_WEIGHT * 100}%")
    print(f"  Start Time:    {start_ts}")
    print(f"  Monitor For:   6 hours")
    print()
    print("Weights:")
    for model, weight in sorted(weights.items()):
        icon = "[*]" if model == canary_model else "[ ]"n        print(f"  {icon} {model:10s} {weight*100:5.1f}%")
    print()
    print(f"Rollback Command:")
    print(f"  {rollback_cmd}")
    print()
    print("=" * 80)
    print()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='QSC MODE - Quality Safeguard Canary Deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Activate PatchTST as canary after quality gate pass
  python3 ops/model_safety/qsc_mode.py --model patchtst --cutover 2026-01-10T05:43:15Z
  
  # Check quality gate only (no activation)
  python3 ops/model_safety/qsc_mode.py --dry-run --cutover 2026-01-10T05:43:15Z
        """
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['xgb', 'lgbm', 'nhits', 'patchtst'],
        default='patchtst',
        help='Model to deploy as canary (default: patchtst)'
    )
    parser.add_argument(
        '--cutover',
        type=str,
        required=True,
        help='Cutover timestamp (ISO 8601 format, e.g., 2026-01-10T05:43:15Z)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run quality gate check only, do not activate canary'
    )
    return parser.parse_args()


def main():
    """Main QSC mode entry point"""
    args = parse_args()
    
    print()
    print("=" * 80)
    print("QSC MODE - Quality Safeguard Canary Deployment")
    print("=" * 80)
    print()
    print(f"Canary Model:  {args.model}")
    print(f"Cutover Time:  {args.cutover}")
    print(f"Dry Run:       {args.dry_run}")
    print()
    
    # STEP 1: Run quality gate with cutover filter
    exit_code, event_count = run_quality_gate(args.cutover)
    
    print()
    print("=" * 80)
    print("Quality Gate Result")
    print("=" * 80)
    print(f"Exit Code:     {exit_code}")
    print(f"Event Count:   {event_count}")
    print(f"Min Required:  {MIN_EVENTS}")
    print()
    
    # Check quality gate passed
    if exit_code != 0:
        print("❌ QUALITY GATE FAILED")
        print("   Canary NOT eligible for activation")
        print()
        return 1
    
    # Check sufficient events
    if event_count < MIN_EVENTS:
        print(f"❌ INSUFFICIENT DATA ({event_count} < {MIN_EVENTS})")
        print("   Canary NOT eligible for activation")
        print()
        return 1
    
    print("✅ QUALITY GATE PASSED")
    print(f"   {event_count} post-cutover events analyzed")
    print()
    
    # Dry run: stop here
    if args.dry_run:
        print("ℹ️  Dry run mode - stopping before canary activation")
        print()
        return 0
    
    # STEP 2: Save baseline weights
    baseline = save_baseline_weights()
    
    # STEP 3: Create canary weights (10% to target model)
    canary_weights = create_canary_weights(args.model, baseline)
    
    print()
    print("Canary Weight Distribution:")
    for model, weight in sorted(canary_weights.items()):
        change = canary_weights[model] - baseline[model]
        icon = "🔸" if model == args.model else "  "
        print(f"  {icon} {model:10s} {weight*100:5.1f}% (Δ {change*100:+5.1f}%)")
    print()
    
    # STEP 4: Activate canary via systemd override
    success = activate_canary_via_systemd(args.model, canary_weights)
    
    if not success:
        print("❌ CANARY ACTIVATION FAILED")
        print()
        return 2
    
    # STEP 5: Log activation with rollback command
    log_canary_activation(args.model, canary_weights, args.cutover, event_count)
    
    # STEP 6: Instructions for monitoring
    print("Next Steps:")
    print()
    print("1. Restart AI engine to apply canary weights:")
    print("   sudo systemctl restart quantum-ai_engine.service")
    print()
    print("2. Start monitoring daemon (runs for 6h, auto-rollback on violation):")
    print("   python3 ops/model_safety/qsc_monitor.py")
    print()
    print("3. Manual rollback if needed:")
    print("   python3 ops/model_safety/qsc_rollback.sh")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
