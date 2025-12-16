#!/usr/bin/env python3
"""
HEALTH CHECK SCRIPT
===================

Quick script to check system health and detect issues:
- Model Supervisor mode mismatch
- Model bias (SHORT/LONG >70%)
- Failed models
- Retraining issues

Usage:
    python check_system_health.py              # Basic check
    python check_system_health.py --fix        # Auto-heal issues
    python check_system_health.py --watch      # Continuous monitoring
"""

import requests
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any


def print_banner():
    """Print health check banner"""
    print("\n" + "=" * 80)
    print("🏥 QUANTUM TRADER - SYSTEM HEALTH CHECK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def check_health(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Check system health via API"""
    try:
        response = requests.get(f"{base_url}/health/monitor", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend (is it running?)")
        print(f"   URL: {base_url}")
        return {"status": "error", "error": "Connection refused"}
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "error": str(e)}


def trigger_auto_heal(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Trigger auto-healing via API"""
    try:
        response = requests.post(f"{base_url}/health/monitor/auto-heal", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "error": str(e)}


def print_health_status(health_data: Dict[str, Any]):
    """Pretty print health status"""
    if health_data.get("status") == "disabled":
        print("⚠️  Health Monitor: DISABLED")
        print("   Enable with: QT_HEALTH_MONITOR_ENABLED=true")
        return
    
    if health_data.get("status") == "error":
        print(f"❌ Error: {health_data.get('error')}")
        return
    
    overall = health_data.get("overall_health", "UNKNOWN")
    summary = health_data.get("summary", {})
    recommendations = health_data.get("recommendations", [])
    
    # Overall status
    emoji = {
        "HEALTHY": "✅",
        "DEGRADED": "🟡",
        "CRITICAL": "🔴",
        "FAILED": "❌"
    }.get(overall, "❓")
    
    print(f"{emoji} OVERALL STATUS: {overall}")
    print()
    
    # Summary stats
    total_issues = summary.get("total_issues", 0)
    by_severity = summary.get("issues_by_severity", {})
    
    if total_issues == 0:
        print("✅ All systems healthy - no issues detected")
        print()
        return
    
    print(f"⚠️  {total_issues} issues detected:")
    if by_severity.get("CRITICAL", 0) > 0:
        print(f"   🔴 CRITICAL: {by_severity['CRITICAL']}")
    if by_severity.get("DEGRADED", 0) > 0:
        print(f"   🟡 DEGRADED: {by_severity['DEGRADED']}")
    if by_severity.get("FAILED", 0) > 0:
        print(f"   ❌ FAILED: {by_severity['FAILED']}")
    print()
    
    # Expected vs actual config
    expected = summary.get("expected_config", {})
    if expected:
        print("📋 Expected Configuration:")
        print(f"   Model Supervisor Mode: {expected.get('model_supervisor_mode')}")
        print(f"   Bias Threshold: {expected.get('bias_threshold', 0):.0%}")
        print(f"   Min Samples: {expected.get('min_samples')}")
        print()
    
    # Detailed issues
    if recommendations:
        print("🔍 DETECTED ISSUES:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations, 1):
            severity_emoji = {
                "CRITICAL": "🔴",
                "DEGRADED": "🟡",
                "FAILED": "❌"
            }.get(rec.get("severity"), "⚠️")
            
            print(f"\n{i}. [{rec.get('component')}] {severity_emoji} {rec.get('severity')}")
            print(f"   Problem: {rec.get('problem')}")
            
            if rec.get('auto_fixable'):
                print(f"   ✅ Auto-fixable: {rec.get('fix_action')}")
            else:
                print(f"   ⚠️  Requires manual intervention")
                if rec.get('fix_action'):
                    print(f"   Action: {rec.get('fix_action')}")
        
        print("\n" + "-" * 80)


def watch_health(base_url: str = "http://localhost:8000", interval: int = 60):
    """Continuous health monitoring"""
    print(f"👁️  Watching system health (checking every {interval} seconds)")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            print_banner()
            health_data = check_health(base_url)
            print_health_status(health_data)
            
            print(f"\n⏰ Next check in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Check Quantum Trader system health"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix detected issues"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuous monitoring mode"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    if args.watch:
        watch_health(args.url, args.interval)
        return
    
    print_banner()
    
    # Check health
    health_data = check_health(args.url)
    print_health_status(health_data)
    
    # Auto-fix if requested
    if args.fix and health_data.get("status") == "ok":
        recommendations = health_data.get("recommendations", [])
        fixable_count = sum(1 for r in recommendations if r.get("auto_fixable"))
        
        if fixable_count > 0:
            print(f"\n🔧 Attempting to auto-fix {fixable_count} issues...")
            
            heal_result = trigger_auto_heal(args.url)
            
            if heal_result.get("status") == "ok":
                fixes_applied = heal_result.get("fixes_applied", 0)
                remaining = heal_result.get("remaining_issues", [])
                
                if fixes_applied > 0:
                    print(f"✅ Successfully fixed {fixes_applied} issues")
                else:
                    print("⚠️  No auto-fixes could be applied")
                
                if remaining:
                    print(f"\n⚠️  {len(remaining)} issues require manual intervention:")
                    for issue in remaining:
                        print(f"   - [{issue['component']}] {issue['problem']}")
            else:
                print(f"❌ Auto-heal failed: {heal_result.get('error')}")
        else:
            print("\n✅ No auto-fixable issues")
    
    print()


if __name__ == "__main__":
    main()
