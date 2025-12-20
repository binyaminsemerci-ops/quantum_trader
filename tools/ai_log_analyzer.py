#!/usr/bin/env python3
"""
Quantum Trader V3 - AI Log Analyzer
Purpose: Automatically interpret logs from Docker containers and summarize anomalies.
Analyzes: Container logs, health events, RL agent activity, Exit Brain operations
Output: Appends intelligence section to weekly health report
"""

import os
import re
import datetime
import subprocess
import pathlib
from statistics import mean
from typing import Dict, List, Tuple

# VPS Configuration
BASE_DIR = pathlib.Path("/home/qt/quantum_trader")
REPORT_DIR = BASE_DIR / "status"
WEBHOOK_URL = os.getenv("QT_ALERT_WEBHOOK", "")

def run(cmd: str) -> str:
    """Execute shell command and return output"""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return p.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"Error: {str(e)}"

def extract_logs() -> str:
    """Collect last 7 days of logs from all Quantum Trader containers"""
    since = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
    containers = run("docker ps -a --format '{{.Names}}'").splitlines()
    
    all_logs = []
    for container in containers:
        if not container or container.startswith("Error"):
            continue
        logs = run(f"docker logs --since {since} {container} 2>&1")
        if logs and len(logs) > 0:
            all_logs.append(f"## Container: {container}\n{logs}\n")
    
    # Also collect self-heal logs
    selfheal_log = BASE_DIR / "status" / "selfheal.log"
    if selfheal_log.exists():
        with open(selfheal_log, 'r') as f:
            all_logs.append(f"## Self-Heal Log\n{f.read()}\n")
    
    return "\n".join(all_logs)

def analyze_logs(log_data: str) -> Tuple[Dict, List[str], float, str]:
    """Identify anomalies, restarts, RL errors, and exceptions"""
    lines = log_data.splitlines()
    
    # Pattern matching
    warnings = [l for l in lines if re.search(r'\b(warn|warning)\b', l, re.I)]
    errors = [l for l in lines if re.search(r'\b(error|exception|failed|traceback)\b', l, re.I)]
    critical = [l for l in lines if re.search(r'\b(critical|fatal|panic)\b', l, re.I)]
    exits = [l for l in lines if re.search(r'\b(exit|exited|stopped|crashed)\b', l, re.I)]
    
    # AI subsystem events
    rl_events = [l for l in lines if re.search(r'(rlagent|rl_v3|reinforcement)', l, re.I)]
    exit_brain = [l for l in lines if re.search(r'(exitbrain|exit_brain|dynamic_executor)', l, re.I)]
    tp_optimizer = [l for l in lines if re.search(r'(tp_optimizer|takeprofit|take.?profit)', l, re.I)]
    event_bus = [l for l in lines if re.search(r'(eventbus|event_bus|stream|consumer)', l, re.I)]
    model_supervisor = [l for l in lines if re.search(r'(model.?supervisor|drift|calibration)', l, re.I)]
    
    # Health indicators
    startup_success = [l for l in lines if re.search(r'(startup complete|started successfully|✅.*started)', l, re.I)]
    restarts = [l for l in lines if re.search(r'(recreated|restarting|restart)', l, re.I)]
    
    metrics = {
        "total_lines": len(lines),
        "warnings": len(warnings),
        "errors": len(errors),
        "critical": len(critical),
        "container_exits": len(exits),
        "restarts": len(restarts),
        "rl_events": len(rl_events),
        "exit_brain_events": len(exit_brain),
        "tp_optimizer_events": len(tp_optimizer),
        "event_bus_events": len(event_bus),
        "model_supervisor_events": len(model_supervisor),
        "successful_startups": len(startup_success)
    }
    
    # Calculate health score
    score = 100.0
    score -= metrics["warnings"] * 0.3
    score -= metrics["errors"] * 1.0
    score -= metrics["critical"] * 5.0
    score -= metrics["container_exits"] * 2.0
    score -= metrics["restarts"] * 1.5
    score = max(0, min(100, score))
    
    # Determine health status
    if score >= 90:
        health = "✅ Excellent"
    elif score >= 80:
        health = "✅ Stable"
    elif score >= 70:
        health = "⚠️ Minor Issues"
    elif score >= 50:
        health = "⚠️ Moderate Issues"
    else:
        health = "❌ Critical"
    
    # Collect anomalies for detailed reporting
    anomalies = []
    
    if metrics["critical"] > 0:
        anomalies.append(f"🔴 {metrics['critical']} critical events detected")
        anomalies.extend([f"  - {l[:120]}" for l in critical[:3]])
    
    if metrics["errors"] > 10:
        anomalies.append(f"⚠️ High error rate: {metrics['errors']} errors in 7 days")
        anomalies.extend([f"  - {l[:120]}" for l in errors[:3]])
    
    if metrics["restarts"] > 0:
        anomalies.append(f"🔄 {metrics['restarts']} container restart(s) detected")
        anomalies.extend([f"  - {l[:120]}" for l in restarts[:3]])
    
    return metrics, anomalies, score, health

def generate_summary(metrics: Dict, anomalies: List[str], score: float, health: str) -> str:
    """Generate markdown summary section"""
    today = datetime.date.today()
    
    summary = [
        "",
        "---",
        "",
        f"## 🧠 AI Log Analysis Summary",
        "",
        f"**Analysis Date:** {today}  ",
        f"**Health Score:** {score:.1f}/100 → {health}  ",
        f"**Log Period:** Last 7 days  ",
        "",
        "### Metrics Overview",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Log Lines Analyzed | {metrics['total_lines']:,} |",
        f"| Warnings | {metrics['warnings']} |",
        f"| Errors | {metrics['errors']} |",
        f"| Critical Events | {metrics['critical']} |",
        f"| Container Exits | {metrics['container_exits']} |",
        f"| Container Restarts | {metrics['restarts']} |",
        f"| Successful Startups | {metrics['successful_startups']} |",
        "",
        "### AI Subsystem Activity",
        "",
        f"| Subsystem | Events |",
        f"|-----------|--------|",
        f"| RL Agent V3 | {metrics['rl_events']} |",
        f"| Exit Brain V3 | {metrics['exit_brain_events']} |",
        f"| TP Optimizer V3 | {metrics['tp_optimizer_events']} |",
        f"| EventBus | {metrics['event_bus_events']} |",
        f"| Model Supervisor | {metrics['model_supervisor_events']} |",
        "",
    ]
    
    if anomalies:
        summary.append("### 🔍 Detected Anomalies")
        summary.append("")
        for anomaly in anomalies:
            summary.append(anomaly)
        summary.append("")
    else:
        summary.append("### ✅ No Anomalies Detected")
        summary.append("")
        summary.append("System is operating within normal parameters.")
        summary.append("")
    
    # System intelligence
    summary.append("### 💡 System Intelligence")
    summary.append("")
    
    if score >= 90:
        summary.append("- System is performing optimally with minimal issues")
        summary.append("- All AI subsystems show healthy activity patterns")
        summary.append("- No intervention required")
    elif score >= 80:
        summary.append("- System is stable with minor warnings")
        summary.append("- AI subsystems functioning normally")
        summary.append("- Monitor for recurring patterns")
    elif score >= 70:
        summary.append("- System experiencing minor issues")
        summary.append("- Review anomalies for potential patterns")
        summary.append("- Consider investigating high-frequency errors")
    else:
        summary.append("- ⚠️ System requires attention")
        summary.append("- Multiple critical events or high error rates detected")
        summary.append("- Manual review and intervention recommended")
    
    summary.append("")
    summary.append("---")
    summary.append("")
    summary.append(f"*Generated by AI Log Analyzer - {datetime.datetime.now().isoformat()}*")
    summary.append("")
    
    return "\n".join(summary)

def write_to_report(section: str) -> None:
    """Append summary to the current weekly health report"""
    report_file = REPORT_DIR / f"WEEKLY_HEALTH_REPORT_{datetime.date.today()}.md"
    
    if not report_file.exists():
        report_file.write_text("# Quantum Trader V3 - Weekly Health Report\n\n")
    
    existing = report_file.read_text()
    
    # Remove old AI Log Analysis section if exists
    if "## 🧠 AI Log Analysis Summary" in existing:
        parts = existing.split("## 🧠 AI Log Analysis Summary")
        existing = parts[0].rstrip()
    
    # Append new section
    updated = existing + "\n" + section
    report_file.write_text(updated)
    
    print(f"🧩 AI Log Analyzer: Section appended to {report_file}")

def send_alert(score: float, health: str, anomalies: List[str]) -> None:
    """Send webhook alert if health is degraded"""
    if not WEBHOOK_URL or score >= 80:
        return
    
    try:
        import requests
        
        anomaly_summary = "\n".join(anomalies[:5]) if anomalies else "See report for details"
        message = f"""🚨 **Quantum Trader Health Alert**

**Health Score:** {score:.1f}/100 → {health}
**Date:** {datetime.date.today()}

**Top Anomalies:**
{anomaly_summary}

Review weekly health report for full analysis.
"""
        
        payload = {"content": message}
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("📢 Alert sent to webhook")
        else:
            print(f"⚠️ Webhook failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Alert failed: {str(e)}")

def main():
    """Main execution flow"""
    print("=" * 70)
    print("QUANTUM TRADER V3 - AI LOG ANALYZER")
    print("=" * 70)
    print(f"Analysis Date: {datetime.date.today()}")
    print(f"Log Period: Last 7 days")
    print()
    
    # Extract logs
    print("[1/4] Extracting container logs...")
    log_data = extract_logs()
    
    if not log_data or len(log_data) < 100:
        print("⚠️ Insufficient log data for analysis")
        return
    
    print(f"      Collected {len(log_data):,} characters of log data")
    
    # Analyze logs
    print("[2/4] Analyzing logs for anomalies...")
    metrics, anomalies, score, health = analyze_logs(log_data)
    print(f"      Health Score: {score:.1f}/100 → {health}")
    print(f"      Anomalies: {len(anomalies)}")
    
    # Generate summary
    print("[3/4] Generating intelligence summary...")
    summary = generate_summary(metrics, anomalies, score, health)
    
    # Write to report
    print("[4/4] Appending to weekly health report...")
    write_to_report(summary)
    
    # Send alert if needed
    if score < 80:
        send_alert(score, health, anomalies)
    
    print()
    print("=" * 70)
    print(f"✅ AI Log Analysis Complete - Score: {score:.1f}/100")
    print("=" * 70)

if __name__ == "__main__":
    main()
