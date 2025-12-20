#!/usr/bin/env python3
"""
Quantum Trader V3 – Real-Time Monitoring Daemon
Purpose: Continuously monitors Docker logs, detects anomalies, triggers auto-heal
Execution: Runs as systemd service, checks logs hourly
"""

import os
import re
import subprocess
import time
import datetime
import pathlib
import json
import signal
import sys
from typing import List, Tuple, Dict

# Import audit logger
try:
    from audit_logger import log_action
except ImportError:
    # Fallback if audit_logger not available
    def log_action(action: str) -> None:
        pass

# VPS Configuration
BASE_DIR = pathlib.Path("/home/qt/quantum_trader")
REPORT_DIR = BASE_DIR / "status"
LOGFILE = REPORT_DIR / "REALTIME_MONITOR.log"
WEBHOOK = os.getenv("QT_ALERT_WEBHOOK", "")
CHECK_INTERVAL = 3600  # 1 hour in seconds

# Thresholds
MAX_ERRORS_PER_HOUR = 50
MAX_CRITICAL_PER_HOUR = 5
MAX_CONTAINER_RESTARTS = 2

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    shutdown_flag = True
    log("🛑 Shutdown signal received, stopping daemon...")
    sys.exit(0)


def log(message: str) -> None:
    """Write timestamped log message"""
    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"[{timestamp}] {message}\n"
    
    # Console output
    print(log_entry.strip())
    
    # File logging
    try:
        with open(LOGFILE, "a") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")


def run(cmd: str) -> str:
    """Execute shell command and return output"""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return p.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"Error: {str(e)}"


def get_container_logs(since: str = "60m") -> List[Tuple[str, str]]:
    """Fetch logs from all Quantum Trader containers"""
    containers = run("docker ps --format '{{.Names}}'").splitlines()
    collected = []
    
    for container in containers:
        if not container or container.startswith("Error"):
            continue
        logs = run(f"docker logs --since {since} {container} 2>&1")
        if logs and len(logs) > 0:
            collected.append((container, logs))
    
    return collected


def analyze(logs: List[Tuple[str, str]]) -> Dict:
    """Analyze logs for anomalies and issues"""
    issues = []
    errors = []
    warnings = []
    critical = []
    restarts = []
    
    for cname, data in logs:
        lines = data.splitlines()
        
        for line in lines:
            # Critical events
            if re.search(r'\b(critical|fatal|panic)\b', line, re.I):
                critical.append((cname, line[:200]))
            
            # Errors and exceptions
            elif re.search(r'\b(error|exception|failed|traceback)\b', line, re.I):
                errors.append((cname, line[:200]))
            
            # Warnings
            elif re.search(r'\b(warn|warning)\b', line, re.I):
                warnings.append((cname, line[:200]))
            
            # Container restarts
            if re.search(r'(recreated|restarting|restart)', line, re.I):
                restarts.append((cname, line[:200]))
    
    return {
        "critical": critical,
        "errors": errors,
        "warnings": warnings,
        "restarts": restarts,
        "total_issues": len(critical) + len(errors)
    }


def should_trigger_heal(analysis: Dict) -> Tuple[bool, str]:
    """Determine if auto-heal should be triggered"""
    critical_count = len(analysis["critical"])
    error_count = len(analysis["errors"])
    restart_count = len(analysis["restarts"])
    
    reasons = []
    
    if critical_count >= MAX_CRITICAL_PER_HOUR:
        reasons.append(f"{critical_count} critical events (threshold: {MAX_CRITICAL_PER_HOUR})")
    
    if error_count >= MAX_ERRORS_PER_HOUR:
        reasons.append(f"{error_count} errors (threshold: {MAX_ERRORS_PER_HOUR})")
    
    if restart_count >= MAX_CONTAINER_RESTARTS:
        reasons.append(f"{restart_count} container restarts (threshold: {MAX_CONTAINER_RESTARTS})")
    
    should_heal = len(reasons) > 0
    reason_text = "; ".join(reasons) if reasons else "No issues detected"
    
    return should_heal, reason_text


def append_incident_report(analysis: Dict, triggered_heal: bool) -> None:
    """Append incident report to weekly health report"""
    now = datetime.datetime.now().isoformat()
    today = datetime.date.today()
    report_file = REPORT_DIR / f"WEEKLY_HEALTH_REPORT_{today}.md"
    
    # Build incident report
    lines = [
        "",
        "---",
        "",
        f"### ⚡ Real-Time Incident Report",
        "",
        f"**Timestamp:** {now}  ",
        f"**Auto-Heal Triggered:** {'✅ Yes' if triggered_heal else '❌ No'}  ",
        "",
        "#### Issue Summary",
        "",
        f"| Category | Count |",
        f"|----------|-------|",
        f"| Critical Events | {len(analysis['critical'])} |",
        f"| Errors | {len(analysis['errors'])} |",
        f"| Warnings | {len(analysis['warnings'])} |",
        f"| Container Restarts | {len(analysis['restarts'])} |",
        "",
    ]
    
    # Add sample issues
    if analysis["critical"]:
        lines.append("#### Critical Events (Top 5)")
        lines.append("```")
        for cname, line in analysis["critical"][:5]:
            lines.append(f"[{cname}] {line}")
        lines.append("```")
        lines.append("")
    
    if analysis["errors"] and len(analysis["errors"]) >= MAX_ERRORS_PER_HOUR:
        lines.append("#### Errors (Top 10)")
        lines.append("```")
        for cname, line in analysis["errors"][:10]:
            lines.append(f"[{cname}] {line}")
        lines.append("```")
        lines.append("")
    
    section = "\n".join(lines)
    
    # Append to report
    try:
        if not report_file.exists():
            report_file.write_text(f"# Quantum Trader V3 - Weekly Health Report\n\n**Generated:** {now}\n\n")
        
        existing = report_file.read_text()
        updated = existing + section
        report_file.write_text(updated)
        
        log(f"📝 Incident report appended to {report_file}")
    except Exception as e:
        log(f"⚠️ Could not append to report: {e}")


def notify(message: str) -> None:
    """Send webhook notification"""
    log(message)
    
    if not WEBHOOK:
        return
    
    try:
        import requests
        payload = {"content": message}
        response = requests.post(WEBHOOK, json=payload, timeout=5)
        if response.status_code == 200:
            log("📢 Alert sent to webhook")
        else:
            log(f"⚠️ Webhook failed: {response.status_code}")
    except ImportError:
        log("⚠️ requests library not available, skipping webhook")
    except Exception as e:
        log(f"⚠️ Webhook error: {str(e)}")


def trigger_auto_heal() -> bool:
    """Execute weekly self-heal script"""
    log("🩺 Triggering auto-heal process...")
    log_action("Auto-heal: Triggered by realtime monitor threshold breach")
    
    try:
        result = subprocess.run(
            ["python3", f"{BASE_DIR}/tools/weekly_self_heal.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            log("✅ Auto-heal completed successfully")
            log_action("Auto-heal: Completed successfully (exit code 0)")
            return True
        else:
            log(f"⚠️ Auto-heal completed with warnings: {result.returncode}")
            log_action(f"Auto-heal: Completed with warnings (exit code {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        log("❌ Auto-heal timed out after 5 minutes")
        log_action("Auto-heal: FAILED - Timeout after 5 minutes")
        return False
    except Exception as e:
        log(f"❌ Auto-heal failed: {e}")
        log_action(f"Auto-heal: FAILED - {str(e)[:100]}")
        return False


def monitor_cycle() -> None:
    """Execute one monitoring cycle"""
    log("🔍 Starting monitoring cycle...")
    
    # Fetch logs
    logs = get_container_logs("60m")
    if not logs:
        log("⚠️ No logs collected, skipping cycle")
        return
    
    log(f"📊 Collected logs from {len(logs)} containers")
    
    # Analyze logs
    analysis = analyze(logs)
    log(f"📈 Analysis: {len(analysis['critical'])} critical, {len(analysis['errors'])} errors, {len(analysis['warnings'])} warnings")
    
    # Check if healing is needed
    should_heal, reason = should_trigger_heal(analysis)
    
    if should_heal:
        log(f"⚠️ Threshold breach detected: {reason}")
        
        # Send alert
        alert_msg = f"""🚨 **Quantum Trader - Auto-Heal Triggered**

**Reason:** {reason}
**Critical Events:** {len(analysis['critical'])}
**Errors:** {len(analysis['errors'])}
**Container Restarts:** {len(analysis['restarts'])}

Auto-heal process initiated."""
        notify(alert_msg)
        
        # Append incident report
        append_incident_report(analysis, triggered_heal=True)
        
        # Execute heal
        heal_success = trigger_auto_heal()
        
        if heal_success:
            notify("✅ Auto-heal completed successfully")
        else:
            notify("⚠️ Auto-heal completed with warnings")
    else:
        log(f"✅ System healthy: {analysis['total_issues']} issues detected (below thresholds)")
        
        # Still log if there were any issues
        if analysis['total_issues'] > 0:
            append_incident_report(analysis, triggered_heal=False)


def monitor_loop() -> None:
    """Main monitoring loop"""
    log("=" * 70)
    log("🧠 QUANTUM TRADER V3 - REAL-TIME MONITORING DAEMON")
    log("=" * 70)
    log(f"Check Interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60} minutes)")
    log(f"Max Errors/Hour: {MAX_ERRORS_PER_HOUR}")
    log(f"Max Critical/Hour: {MAX_CRITICAL_PER_HOUR}")
    log(f"Max Restarts/Hour: {MAX_CONTAINER_RESTARTS}")
    log(f"Log File: {LOGFILE}")
    log(f"Webhook: {'Configured' if WEBHOOK else 'Not configured'}")
    log("=" * 70)
    
    notify("🧠 Real-Time Monitor started – Quantum Trader V3")
    
    cycle_count = 0
    
    while not shutdown_flag:
        cycle_count += 1
        log(f"\n{'=' * 70}")
        log(f"Monitoring Cycle #{cycle_count}")
        log(f"{'=' * 70}")
        
        try:
            monitor_cycle()
        except Exception as e:
            log(f"❌ Monitoring cycle failed: {e}")
            notify(f"❌ Real-Time Monitor error: {str(e)[:100]}")
        
        # Sleep until next cycle
        if not shutdown_flag:
            log(f"💤 Sleeping for {CHECK_INTERVAL}s until next cycle...")
            time.sleep(CHECK_INTERVAL)
    
    log("🛑 Daemon stopped")


def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Ensure directories exist
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start monitoring
    try:
        monitor_loop()
    except KeyboardInterrupt:
        log("🛑 Keyboard interrupt received, stopping...")
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
