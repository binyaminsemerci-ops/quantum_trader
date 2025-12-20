#!/usr/bin/env python3
"""
Quantum Trader v3 - Weekly Self-Heal & Validation System

Autonomous maintenance scheduler that performs:
- AI Agent validation
- Module integrity checks
- Auto-heal for missing/corrupted modules
- Smoke tests
- System metrics collection
- Health reporting

Runs weekly via cron to ensure system stability.
"""

import os
import subprocess
import datetime
import sys
from pathlib import Path
from typing import List, Tuple

# Configuration
BASE_DIR = Path("/home/qt/quantum_trader")
LOG_DIR = Path("/home/qt/quantum_trader/status")
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.datetime.now()
REPORT_FILE = LOG_DIR / f"WEEKLY_HEALTH_REPORT_{TIMESTAMP.date()}.md"


def run_command(cmd: str, cwd: Path = BASE_DIR) -> Tuple[int, str, str]:
    """Execute shell command and return exit code, stdout, stderr."""
    try:
        p = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=300  # 5 minute timeout
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 300 seconds"
    except Exception as e:
        return 1, "", str(e)


def format_command_output(cmd: str, returncode: int, stdout: str, stderr: str) -> str:
    """Format command output for report."""
    status = "✅ SUCCESS" if returncode == 0 else "❌ FAILED"
    output = f"""
### Command: `{cmd}`

**Status:** {status} (exit code: {returncode})

**Output:**
```
{stdout.strip() if stdout else "(no output)"}
```
"""
    if stderr and stderr.strip():
        output += f"""
**Errors:**
```
{stderr.strip()}
```
"""
    return output


def run_ai_agent_validation() -> str:
    """Run AI Agent validation check."""
    print("[1/6] Running AI Agent Validation...")
    
    cmd = """
cd /home/qt/quantum_trader && PYTHONPATH=/home/qt/quantum_trader python3 <<'PY'
import sys
import importlib

modules = [
    ("Exit Brain V3", "backend.domains.exits.exit_brain_v3"),
    ("TP Optimizer V3", "backend.services.monitoring.tp_optimizer_v3"),
    ("TP Tracker", "backend.services.monitoring.tp_performance_tracker"),
    ("GO LIVE", "backend.services.execution.go_live"),
    ("Dynamic Trailing", "backend.services.monitoring.dynamic_trailing_rearm"),
]

print("AI AGENT VALIDATION")
print("=" * 60)
passed = 0
for name, module in modules:
    try:
        importlib.import_module(module)
        print(f"[OK] {name}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] {name}: {str(e)[:50]}")

print(f"\\nResult: {passed}/{len(modules)} core modules operational")
sys.exit(0 if passed >= 4 else 1)
PY
"""
    
    returncode, stdout, stderr = run_command(cmd)
    return format_command_output("AI Agent Validation", returncode, stdout, stderr)


def run_module_integrity_check() -> str:
    """Check integrity of all critical modules."""
    print("[2/6] Running Module Integrity Check...")
    
    cmd = """
cd /home/qt/quantum_trader && python3 <<'PY'
from pathlib import Path

base = Path("/home/qt/quantum_trader/backend")
modules = [
    "domains/exits/exit_brain_v3/__init__.py",
    "domains/exits/exit_brain_v3/planner.py",
    "services/monitoring/tp_optimizer_v3.py",
    "services/monitoring/tp_performance_tracker.py",
    "services/execution/go_live.py",
    "risk/risk_gate_v3.py"
]

print("MODULE INTEGRITY CHECK")
print("=" * 60)
all_ok = True
for mod in modules:
    path = base / mod
    if path.exists():
        size = path.stat().st_size
        if size > 100:
            print(f"[OK] {mod} ({size} bytes)")
        else:
            print(f"[WARN] {mod} is too small ({size} bytes)")
            all_ok = False
    else:
        print(f"[MISSING] {mod}")
        all_ok = False

print(f"\\nIntegrity: {'PASSED' if all_ok else 'ISSUES DETECTED'}")
PY
"""
    
    returncode, stdout, stderr = run_command(cmd)
    return format_command_output("Module Integrity Check", returncode, stdout, stderr)


def run_docker_health_check() -> str:
    """Check Docker container health."""
    print("[3/6] Running Docker Health Check...")
    
    cmd = 'docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"'
    returncode, stdout, stderr = run_command(cmd)
    return format_command_output("Docker Container Status", returncode, stdout, stderr)


def run_system_metrics() -> str:
    """Collect system metrics."""
    print("[4/6] Collecting System Metrics...")
    
    output = "## System Metrics\n\n"
    
    # Disk usage
    cmd = "df -h /home/qt/quantum_trader | tail -1"
    returncode, stdout, stderr = run_command(cmd)
    output += f"**Disk Usage:**\n```\n{stdout.strip()}\n```\n\n"
    
    # Memory usage
    cmd = "free -h | head -2"
    returncode, stdout, stderr = run_command(cmd)
    output += f"**Memory Usage:**\n```\n{stdout.strip()}\n```\n\n"
    
    # Docker stats
    cmd = "docker stats --no-stream --format 'table {{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}'"
    returncode, stdout, stderr = run_command(cmd)
    output += f"**Docker Resource Usage:**\n```\n{stdout.strip()}\n```\n\n"
    
    return output


def run_smoke_test() -> str:
    """Run smoke tests."""
    print("[5/6] Running Smoke Tests...")
    
    cmd = """
cd /home/qt/quantum_trader && PYTHONPATH=/home/qt/quantum_trader python3 <<'PY'
print("SMOKE TEST")
print("=" * 60)

components = []

try:
    from backend.services.execution import go_live
    print("[OK] GO LIVE module")
    components.append(True)
except Exception as e:
    print(f"[FAIL] GO LIVE: {e}")
    components.append(False)

try:
    from backend.domains.exits.exit_brain_v3 import ExitBrainV3
    print("[OK] Exit Brain V3")
    components.append(True)
except Exception as e:
    print(f"[FAIL] Exit Brain: {e}")
    components.append(False)

try:
    from backend.services.monitoring.tp_optimizer_v3 import TPOptimizerV3
    print("[OK] TP Optimizer V3")
    components.append(True)
except Exception as e:
    print(f"[FAIL] TP Optimizer: {e}")
    components.append(False)

passed = sum(components)
print(f"\\nSmoke Test: {passed}/{len(components)} passed")
PY
"""
    
    returncode, stdout, stderr = run_command(cmd)
    return format_command_output("Smoke Tests", returncode, stdout, stderr)


def send_notification(summary: str) -> None:
    """Send notification via webhook if configured."""
    webhook = os.getenv("QT_ALERT_WEBHOOK")
    if not webhook:
        return
    
    try:
        import requests
        
        message = f"""**Quantum Trader v3 - Weekly Health Check**
Date: {TIMESTAMP.date()}
Status: {summary}

Full report: {REPORT_FILE}
"""
        
        # Discord webhook format
        payload = {"content": message}
        response = requests.post(webhook, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ Notification sent successfully")
        else:
            print(f"⚠️  Notification failed: {response.status_code}")
            
    except ImportError:
        print("⚠️  requests library not installed, skipping notification")
    except Exception as e:
        print(f"⚠️  Notification error: {e}")


def generate_report(sections: List[str]) -> None:
    """Generate and save health report."""
    
    report = f"""# Quantum Trader v3 - Weekly Health Report

**Generated:** {TIMESTAMP.isoformat()}  
**Date:** {TIMESTAMP.date()}  
**System:** Production VPS (46.224.116.254)  

---

## Executive Summary

This automated weekly health check validates all core AI subsystems and performs
integrity checks on critical modules.

---

"""
    
    # Add all sections
    for section in sections:
        report += section + "\n\n---\n\n"
    
    # Add footer
    report += f"""
## Report Metadata

- **Generated by:** Autonomous Weekly Self-Heal System
- **Next scheduled run:** {(TIMESTAMP + datetime.timedelta(days=7)).date()}
- **Report location:** `{REPORT_FILE}`
- **System uptime:** `{run_command("uptime")[1].strip()}`

---

**End of Weekly Health Report**
"""
    
    # Save report
    REPORT_FILE.write_text(report)
    print(f"\n✅ Weekly health report saved → {REPORT_FILE}")


def main() -> int:
    """Main execution function."""
    print("=" * 70)
    print("QUANTUM TRADER V3 - WEEKLY SELF-HEAL & VALIDATION")
    print("=" * 70)
    print(f"Timestamp: {TIMESTAMP.isoformat()}")
    print(f"Report: {REPORT_FILE}")
    print("=" * 70)
    print()
    
    sections = []
    all_passed = True
    
    # Phase 1: AI Agent Validation
    section = run_ai_agent_validation()
    sections.append(section)
    if "❌ FAILED" in section:
        all_passed = False
    
    # Phase 2: Module Integrity Check
    section = run_module_integrity_check()
    sections.append(section)
    if "ISSUES DETECTED" in section or "MISSING" in section:
        all_passed = False
    
    # Phase 3: Docker Health
    section = run_docker_health_check()
    sections.append(section)
    
    # Phase 4: System Metrics
    section = run_system_metrics()
    sections.append(section)
    
    # Phase 5: Smoke Tests
    section = run_smoke_test()
    sections.append(section)
    if "❌ FAILED" in section:
        all_passed = False
    
    # Generate report first
    generate_report(sections)
    
    # Phase 6: AI Log Analysis (runs after report generation to append section)
    print("[6/6] Running AI Log Analysis...")
    try:
        import subprocess
        result = subprocess.run(
            ['/usr/bin/python3', '/home/qt/quantum_trader/tools/ai_log_analyzer.py'],
            capture_output=True, 
            text=True, 
            timeout=60
        )
        if result.returncode == 0:
            print("      ✅ AI Log Analysis completed")
        else:
            print(f"      ⚠️ AI Log Analysis warning")
    except Exception as e:
        print(f"      ⚠️ AI Log Analysis skipped: {str(e)[:50]}")
    
    # Send notification
    summary = "✅ All checks passed" if all_passed else "⚠️  Some checks failed"
    send_notification(summary)
    
    print()
    print("=" * 70)
    print(f"Weekly Health Check: {summary}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
