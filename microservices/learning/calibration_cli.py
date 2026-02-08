#!/usr/bin/env python3
"""
Calibration CLI - Command Line Interface for Calibration Operations

Usage:
    python calibration_cli.py check              # Check if ready for calibration
    python calibration_cli.py run [--force]      # Run calibration analysis
    python calibration_cli.py approve <job_id>   # Approve and deploy
    python calibration_cli.py rollback [version] # Rollback to previous version
    python calibration_cli.py status             # Show current status
    python calibration_cli.py list               # List all calibration jobs
"""
import sys
import argparse
import logging
from typing import Optional
from pathlib import Path

from microservices.learning.calibration_orchestrator import CalibrationOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def cmd_check(orchestrator: CalibrationOrchestrator) -> int:
    """Check if calibration is ready"""
    
    try:
        readiness = orchestrator.check_readiness()
        
        print("\n" + "=" * 60)
        print("📊 LEARNING CADENCE READINESS CHECK")
        print("=" * 60)
        
        ready = readiness.get('ready', False)
        reason = readiness.get('reason', 'unknown')
        actions = readiness.get('actions', [])
        
        status_emoji = "🟢" if ready else "⏸️"
        
        print(f"\n{status_emoji} Status: {'READY' if ready else 'NOT READY'}")
        print(f"   Reason: {reason}")
        print(f"   Allowed actions: {actions if actions else 'None'}")
        
        if ready and 'calibration' in actions:
            print("\n✅ CALIBRATION AUTHORIZED")
            print("   You can run: python calibration_cli.py run")
        elif ready:
            print(f"\n⚠️  Learning Cadence is ready, but only for: {actions}")
            print("   Calibration not authorized yet")
        else:
            print("\n⏸️  NOT READY YET")
            print(f"   {reason}")
            print("   Wait for more trades to accumulate")
        
        print("=" * 60)
        print()
        
        return 0 if ready and 'calibration' in actions else 1
        
    except Exception as e:
        logger.error(f"❌ Check failed: {e}")
        return 1


def cmd_run(orchestrator: CalibrationOrchestrator, force: bool = False) -> int:
    """Run calibration analysis"""
    
    try:
        job = orchestrator.start_calibration(dry_run=False, force=force)
        
        # Show next steps
        if job.status.value == "pending_approval":
            print("\n📋 NEXT STEPS:")
            print(f"\n1. Review report:")
            print(f"   cat {job.report_path}")
            print(f"\n2. If satisfied, approve and deploy:")
            print(f"   python calibration_cli.py approve {job.id}")
            print(f"\n3. Monitor for 24-48 hours")
            print(f"\n4. If issues occur, rollback:")
            print(f"   python calibration_cli.py rollback")
            print()
            
            return 0
        else:
            print(f"\n❌ Calibration failed with status: {job.status.value}")
            if job.error:
                print(f"   Error: {job.error}")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Run failed: {e}")
        return 1


def cmd_approve(orchestrator: CalibrationOrchestrator, job_id: str) -> int:
    """Approve and deploy calibration"""
    
    try:
        # Load job to show summary
        job = orchestrator.get_job(job_id)
        
        if not job:
            print(f"\n❌ Job {job_id} not found")
            print("\nAvailable jobs:")
            for jid, j in orchestrator.list_jobs().items():
                print(f"  - {jid}: {j.status.value}")
            print()
            return 1
        
        # Show summary and ask for confirmation
        print("\n" + "=" * 60)
        print("⚠️  CALIBRATION DEPLOYMENT CONFIRMATION")
        print("=" * 60)
        print(f"\nJob ID: {job_id}")
        print(f"Status: {job.status.value}")
        print(f"Based on: {job.calibration.based_on_trades} trades")
        print(f"Report: {job.report_path}")
        
        cal = job.calibration
        
        if cal.confidence.enabled:
            print(f"\n🎯 Confidence Calibration:")
            print(f"   Improvement: {cal.confidence.improvement_pct:+.1f}%")
            print(f"   MSE: {cal.confidence.mse_before:.4f} → {cal.confidence.mse_after:.4f}")
        
        if cal.weights.enabled:
            print(f"\n⚖️  Ensemble Weights:")
            for change in cal.weights.changes:
                if abs(change.delta) > 0.01:
                    arrow = "⬆️" if change.delta > 0 else "⬇️"
                    print(f"   {change.model:10s}: {change.before:.3f} → {change.after:.3f} ({change.delta_pct:+.1f}%) {arrow}")
        
        print(f"\n🔍 Validation:")
        passed = len([c for c in cal.validation_checks if c.passed])
        total = len(cal.validation_checks)
        print(f"   Passed: {passed}/{total} checks")
        print(f"   Risk: {cal.metadata.risk_score:.1%}")
        
        print("\n⚠️  THIS WILL DEPLOY TO PRODUCTION")
        print("   The AI Engine will immediately start using this calibration")
        print("   You can rollback if needed (< 2 minutes)")
        print("=" * 60)
        
        # Ask for confirmation
        response = input("\nType 'yes' to deploy: ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Deployment cancelled")
            return 1
        
        # Deploy
        success = orchestrator.approve_and_deploy(job_id, approved_by="manual_cli")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Approval failed: {e}")
        return 1


def cmd_rollback(orchestrator: CalibrationOrchestrator, version: Optional[str] = None) -> int:
    """Rollback to previous calibration"""
    
    try:
        # List available versions
        versions = orchestrator.deployer.list_versions()
        
        if not versions:
            print("\n❌ No backup versions available")
            return 1
        
        print("\n" + "=" * 60)
        print("🔙 CALIBRATION ROLLBACK")
        print("=" * 60)
        
        if version:
            print(f"\nRolling back to: {version}")
        else:
            print("\nRolling back to most recent backup")
            print("\nAvailable versions:")
            for v in versions[:5]:
                vtype = "🔄 backup" if v['type'] == 'backup' else "✅ deployed"
                print(f"  {vtype} {v['version']} ({v['modified_at']})")
        
        print("\n⚠️  THIS WILL CHANGE PRODUCTION CONFIG")
        print("   AI Engine will immediately reload previous calibration")
        print("=" * 60)
        
        # Ask for confirmation
        response = input("\nType 'yes' to rollback: ").strip().lower()
        
        if response != 'yes':
            print("\n❌ Rollback cancelled")
            return 1
        
        # Rollback
        success = orchestrator.rollback(version)
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        return 1


def cmd_status(orchestrator: CalibrationOrchestrator) -> int:
    """Show current calibration status"""
    
    try:
        print("\n" + "=" * 60)
        print("📊 CALIBRATION SYSTEM STATUS")
        print("=" * 60)
        
        # Current version
        try:
            current_version = orchestrator.deployer.get_current_version()
            if current_version:
                print(f"\n✅ Active Calibration:")
                print(f"   Version: {current_version}")
                
                # Try to get deployment time from archive
                versions = orchestrator.deployer.list_versions()
                deployed_versions = [v for v in versions if v['version'] == current_version and v['type'] == 'deployed']
                if deployed_versions:
                    print(f"   Deployed: {deployed_versions[0]['modified_at']}")
            else:
                print("\n⏸️  No calibration deployed yet")
                print("   AI Engine using baseline configuration")
        except Exception as e:
            print(f"\n⚠️  Could not read current calibration: {e}")
        
        # Recent jobs
        jobs = orchestrator.list_jobs()
        if jobs:
            print(f"\n📋 Recent Analysis Jobs:")
            for job_id, job in sorted(jobs.items(), key=lambda x: x[1].created_at, reverse=True)[:5]:
                status_emoji = {
                    "pending_approval": "⏸️",
                    "approved": "✅",
                    "deployed": "🚀",
                    "rejected": "❌",
                    "rolled_back": "🔙"
                }.get(job.status.value, "❓")
                
                print(f"   {status_emoji} {job.id}: {job.status.value}")
        
        # Learning Cadence readiness
        try:
            readiness = orchestrator.check_readiness()
            ready = readiness.get('ready', False)
            reason = readiness.get('reason', 'unknown')
            
            print(f"\n📊 Learning Cadence:")
            print(f"   Ready: {'🟢 YES' if ready else '⏸️ NO'}")
            print(f"   Reason: {reason}")
            
        except Exception as e:
            print(f"\n⚠️  Learning Cadence: API unavailable ({e})")
        
        # Version history
        versions = orchestrator.deployer.list_versions()
        if versions:
            print(f"\n📚 Version History: ({len(versions)} total)")
            for v in versions[:3]:
                vtype = "🔄" if v['type'] == 'backup' else "✅"
                print(f"   {vtype} {v['version']} ({v['modified_at']})")
        
        print("=" * 60)
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return 1


def cmd_list(orchestrator: CalibrationOrchestrator) -> int:
    """List all calibration jobs"""
    
    try:
        jobs = orchestrator.list_jobs()
        
        if not jobs:
            print("\n⏸️  No calibration jobs yet")
            print("\nRun analysis first:")
            print("   python calibration_cli.py run")
            print()
            return 0
        
        print("\n" + "=" * 60)
        print("📋 CALIBRATION JOBS")
        print("=" * 60)
        
        for job_id, job in sorted(jobs.items(), key=lambda x: x[1].created_at, reverse=True):
            status_emoji = {
                "pending_approval": "⏸️",
                "approved": "✅",
                "deployed": "🚀",
                "rejected": "❌",
                "rolled_back": "🔙"
            }.get(job.status.value, "❓")
            
            print(f"\n{status_emoji} {job.id}")
            print(f"   Status: {job.status.value}")
            print(f"   Created: {job.created_at.isoformat()}")
            print(f"   Trades: {job.calibration.based_on_trades}")
            
            if job.report_path:
                print(f"   Report: {job.report_path}")
            
            if job.error:
                print(f"   Error: {job.error}")
        
        print("=" * 60)
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ List failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Calibration-Only Learning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibration_cli.py check                # Check if ready
  python calibration_cli.py run                  # Run calibration analysis
  python calibration_cli.py run --force          # Force run (skip auth)
  python calibration_cli.py approve cal_20260210_143022  # Deploy calibration
  python calibration_cli.py rollback             # Rollback to previous
  python calibration_cli.py status               # Show current status
  python calibration_cli.py list                 # List all jobs
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # check command
    subparsers.add_parser('check', help='Check if calibration is ready')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Run calibration analysis')
    run_parser.add_argument('--force', action='store_true', help='Force run (skip authorization check)')
    
    # approve command
    approve_parser = subparsers.add_parser('approve', help='Approve and deploy calibration')
    approve_parser.add_argument('job_id', help='Job ID to approve')
    
    # rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous calibration')
    rollback_parser.add_argument('version', nargs='?', help='Specific version to rollback to')
    
    # status command
    subparsers.add_parser('status', help='Show current calibration status')
    
    # list command
    subparsers.add_parser('list', help='List all calibration jobs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize orchestrator
    orchestrator = CalibrationOrchestrator()
    
    # Execute command
    if args.command == 'check':
        return cmd_check(orchestrator)
    
    elif args.command == 'run':
        return cmd_run(orchestrator, force=args.force)
    
    elif args.command == 'approve':
        return cmd_approve(orchestrator, args.job_id)
    
    elif args.command == 'rollback':
        return cmd_rollback(orchestrator, args.version)
    
    elif args.command == 'status':
        return cmd_status(orchestrator)
    
    elif args.command == 'list':
        return cmd_list(orchestrator)
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
