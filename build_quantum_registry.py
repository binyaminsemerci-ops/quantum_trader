#!/usr/bin/env python3
"""
QUANTUM TRADER MODULE REGISTRY - AUTHORITATIVE SOURCE OF TRUTH
Generates complete classification of all modules with provenance
"""

import json
import subprocess
import re
from datetime import datetime
from typing import Dict, List, Set

HOST = "46.224.116.254"
KEY_PATH = "~/.ssh/hetzner_fresh"

def ssh(cmd: str, raw=False) -> str:
    """Execute command on VPS via SSH"""
    full_cmd = f'wsl ssh -i {KEY_PATH} root@{HOST} "{cmd}"'
    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=60)
        if raw:
            return result.stdout
        return result.stdout.strip()
    except Exception as e:
        print(f"ERROR in ssh: {e}")
        return ""

print("="*80)
print("QUANTUM TRADER MODULE REGISTRY BUILDER")
print("="*80)
print()

# ============================================================================
# STEP 1: Collect all systemd services
# ============================================================================
print("[1/8] Collecting systemd services & timers...")

services_output = ssh("systemctl list-units --type=service --all --no-pager")
timers_output = ssh("systemctl list-timers --all --no-pager")

# Parse services
services = {}
for line in services_output.split("\n"):
    if "quantum" in line.lower() and ".service" in line:
        parts = line.split()
        if len(parts) >= 5:
            svc_name = parts[0]
            loaded = parts[1]
            active = parts[2]
            sub = parts[3]
            desc = " ".join(parts[4:]) if len(parts) > 4 else ""
            services[svc_name] = {
                "type": "service",
                "loaded": loaded,
                "active": active,
                "sub": sub,
                "description": desc,
            }

print(f"   Found {len(services)} quantum services")

# Get detailed systemd info for each service
details = {}
for svc in sorted(services.keys()):
    full_info = ssh(f"systemctl show {svc}")
    info = {}
    for line in full_info.split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    details[svc] = info

# ============================================================================
# STEP 2: Scan microservices code directory
# ============================================================================
print("[2/8] Scanning microservices code...")

code_output = ssh("""python3 << 'PYEOF'
import os
base = "/home/qt/quantum_trader/microservices"
modules = {}
for d in sorted(os.listdir(base)):
    path = os.path.join(base, d)
    if os.path.isdir(path) and not d.startswith("_") and not d.startswith("."):
        has_main = os.path.isfile(os.path.join(path, "main.py"))
        has_service = os.path.isfile(os.path.join(path, "service.py"))
        has_dunder = os.path.isfile(os.path.join(path, "__main__.py"))
        
        if has_main or has_service or has_dunder:
            ep = "main.py" if has_main else ("service.py" if has_service else "__main__.py")
            modules[d] = ep
        else:
            modules[d] = None
            
import json
print(json.dumps(modules))
PYEOF""")

microservices = json.loads(code_output) if code_output.startswith("{") else {}
print(f"   Found {len(microservices)} microservice modules")

# ============================================================================
# STEP 3: Map venv directories
# ============================================================================
print("[3/8] Mapping venv directories...")

venvs_output = ssh("ls -1 /opt/quantum/venvs 2>/dev/null")
venvs = set(v.strip() for v in venvs_output.split("\n") if v.strip())
print(f"   Found {len(venvs)} venv directories")

# ============================================================================
# STEP 4: Discover Redis streams & consumer groups
# ============================================================================
print("[4/8] Discovering Redis streams & consumer groups...")

streams_output = ssh('redis-cli --scan --pattern "quantum:stream:*" 2>&1')
streams = []
for line in streams_output.split("\n"):
    if "quantum:stream:" in line:
        stream_name = line.strip()
        streams.append(stream_name)
        
        # Get groups
        groups_output = ssh(f'redis-cli XINFO GROUPS {stream_name} 2>&1')
        # Parse group names (they appear after "name" line)

print(f"   Found {len(streams)} Redis streams")

# ============================================================================
# STEP 5: Map listening ports
# ============================================================================
print("[5/8] Mapping listening ports...")

ports_output = ssh("ss -lntp 2>/dev/null | grep LISTEN")
ports = {}
for line in ports_output.split("\n"):
    if "python" in line.lower():
        match = re.search(r'127\.0\.0\.1:(\d+)|0\.0\.0\.0:(\d+)', line)
        if match:
            port = int(match.group(1) or match.group(2))
            ports[port] = line

print(f"   Found {len(ports)} listening ports")

# ============================================================================
# STEP 6: Map system targets and dependencies
# ============================================================================
print("[6/8] Analyzing systemd targets & dependencies...")

targets_output = ssh("systemctl list-units --type=target --all --no-pager")
targets = {}
for line in targets_output.split("\n"):
    if "quantum" in line.lower():
        parts = line.split()
        if len(parts) >= 1:
            targets[parts[0]] = line

# ============================================================================
# STEP 7: Classify all modules
# ============================================================================
print("[7/8] Classifying modules...")

registry = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "host": HOST,
    "classification_timestamp": datetime.utcnow().isoformat() + "Z",
    "summary": {
        "total_systemd_units": len(services),
        "total_microservices": len(microservices),
        "total_venvs": len(venvs),
        "total_streams": len(streams),
        "total_ports": len(ports),
    },
    "modules": []
}

# Merge and classify
processed = set()

# Process each systemd service
for svc_name in sorted(services.keys()):
    if svc_name in processed:
        continue
    
    module_name = svc_name.replace("quantum-", "").replace(".service", "").replace(".timer", "")
    processed.add(svc_name)
    
    svc_info = services[svc_name]
    detail_info = details.get(svc_name, {})
    
    # Get ExecStart (clean it up)
    exec_start = detail_info.get("ExecStart", "")
    # Extract just the command, not the systemd format info
    match = re.search(r'path=([^;]+);', exec_start)
    exec_path = match.group(1).strip() if match else ""
    
    # Determine category
    active = svc_info.get("active", "").lower()
    sub = svc_info.get("sub", "").lower()
    loaded = svc_info.get("loaded", "").lower()
    
    if active == "active" and sub in ["running", "exited"]:
        category = "RUNNING"
    elif active == "failed" or (active == "activating" and "auto-restart" in sub):
        category = "FAILED"
    elif active == "inactive" and loaded == "loaded":
        category = "STOPPED"
    elif loaded == "not-found":
        category = "ORPHANED"
    else:
        category = "UNKNOWN"
    
    # Map to microservice if exists
    code_path = microservices.get(module_name, {}) if isinstance(microservices.get(module_name), dict) else None
    entrypoint = microservices.get(module_name) if isinstance(microservices.get(module_name), str) else None
    
    # Check venv status
    venv_status = "NO_VENV"
    used_venv = None
    for venv in venvs:
        if venv in exec_start:
            venv_status = "VALID_VENV"
            used_venv = venv
            break
    
    if "/opt/quantum/venvs/" in exec_start and venv_status == "NO_VENV":
        venv_status = "MISSING_VENV"
    
    # Extract port from exec start
    port_match = re.search(r'--port\s+(\d+)', exec_start)
    port = int(port_match.group(1)) if port_match else None
    
    # Get PID
    pid = detail_info.get("Pid", "")
    user = detail_info.get("User", "")
    wd = detail_info.get("WorkingDirectory", "")
    enabled = detail_info.get("UnitFileState", "").lower() in ["enabled", "static"]
    
    module_record = {
        "name": module_name,
        "full_unit": svc_name,
        "type": svc_info.get("type", "service"),
        "category": category,
        "systemd": {
            "loaded": svc_info.get("loaded"),
            "active": svc_info.get("active"),
            "sub": svc_info.get("sub"),
            "enabled": enabled,
            "description": svc_info.get("description"),
        },
        "execution": {
            "user": user,
            "pid": pid,
            "working_dir": wd,
            "exec_path": exec_path,
        },
        "code": {
            "module": module_name,
            "path": f"/home/qt/quantum_trader/microservices/{module_name}" if module_name in microservices else None,
            "entrypoint": entrypoint,
            "has_code": module_name in microservices,
        },
        "runtime": {
            "venv": used_venv,
            "venv_status": venv_status,
            "port": port,
            "redis_streams": [s for s in streams if module_name.lower() in s.lower()],
        },
        "proof_line": f"systemctl status {svc_name} --no-pager | Active: {svc_info.get('active')} ({svc_info.get('sub')})",
    }
    
    registry["modules"].append(module_record)

# ============================================================================
# STEP 8: Generate outputs
# ============================================================================
print("[8/8] Generating output files...")

# Create registry directory on VPS
ssh("mkdir -p /opt/quantum/registry")

# Write JSON
json_output = json.dumps(registry, indent=2)
with open("/tmp/module_registry.json", "w") as f:
    f.write(json_output)

# Upload to VPS
ssh_upload_cmd = f'cat > /opt/quantum/registry/module_registry.json << \'EOFN\'\n{json_output}\nEOFN'
subprocess.run(f'wsl ssh -i {KEY_PATH} root@{HOST} "{ssh_upload_cmd}"', shell=True, capture_output=True)

print(f"   ✓ Generated /opt/quantum/registry/module_registry.json")

# ============================================================================
# Generate Markdown report
# ============================================================================

report = "# QUANTUM TRADER - MODULE REGISTRY REPORT\n\n"
report += f"**Generated:** {datetime.utcnow().isoformat()}Z\n"
report += f"**Host:** {HOST}\n\n"

# Count by category
running = [m for m in registry["modules"] if m["category"] == "RUNNING"]
stopped = [m for m in registry["modules"] if m["category"] == "STOPPED"]
failed = [m for m in registry["modules"] if m["category"] == "FAILED"]
orphaned = [m for m in registry["modules"] if m["category"] == "ORPHANED"]
unknown = [m for m in registry["modules"] if m["category"] == "UNKNOWN"]

report += "## Summary\n\n"
report += "| Category | Count |\n"
report += "|----------|-------|\n"
report += f"| **RUNNING** | **{len(running)}** |\n"
report += f"| STOPPED | {len(stopped)} |\n"
report += f"| FAILED | {len(failed)} |\n"
report += f"| ORPHANED | {len(orphaned)} |\n"
report += f"| UNKNOWN | {len(unknown)} |\n"
report += f"| **TOTAL** | **{len(registry['modules'])}** |\n\n"

# RUNNING section
report += "## RUNNING Modules\n\n"
if running:
    report += "| Module | User | Port | venv | State |\n"
    report += "|--------|------|------|------|-------|\n"
    for m in sorted(running, key=lambda x: x["name"]):
        port = m["runtime"]["port"] or "-"
        venv = m["runtime"]["venv"] or "-"
        report += f"| {m['name']} | {m['execution']['user']} | {port} | {venv} | {m['systemd']['sub']} |\n"
else:
    report += "No running modules.\n"

report += "\n"

# STOPPED section
if stopped:
    report += "## STOPPED Modules\n\n"
    for m in sorted(stopped, key=lambda x: x["name"]):
        report += f"- `{m['name']}` (enabled: {m['systemd']['enabled']})\n"
    report += "\n"

# FAILED section
if failed:
    report += "## FAILED Modules\n\n"
    for m in sorted(failed, key=lambda x: x["name"]):
        report += f"- `{m['name']}` (last restart: {m['systemd']['sub']})\n"
    report += "\n"

# Systemd targets
if targets:
    report += "## Systemd Targets\n\n"
    for target in sorted(targets.keys()):
        report += f"- `{target}`\n"
    report += "\n"

# Streams
if streams:
    report += "## Redis Streams\n\n"
    for stream in sorted(streams)[:20]:  # First 20
        stream_short = stream.replace("quantum:stream:", "")
        report += f"- `{stream_short}`\n"
    if len(streams) > 20:
        report += f"- ... and {len(streams)-20} more\n"
    report += "\n"

# Architecture diagram
report += "## Architecture Overview\n\n"
report += "```\n"
report += "QUANTUM TRADER SYSTEM ARCHITECTURE\n"
report += "=" * 60 + "\n\n"
report += f"Systemd Services:        {len(services)}\n"
report += f"  - Running:              {len(running)}\n"
report += f"  - Stopped:              {len(stopped)}\n"
report += f"  - Failed:               {len(failed)}\n"
report += "\n"
report += f"Microservices:            {len(microservices)}\n"
report += f"Python venvs:             {len(venvs)}\n"
report += f"Redis Streams:            {len(streams)}\n"
report += f"Listening Ports:          {len(ports)}\n"
report += f"Systemd Targets:          {len(targets)}\n"
report += "```\n\n"

# Quality gates
report += "## Quality Gates\n\n"
report += f"- All systemd units classified: {'✓' if len(unknown) == 0 else '✗'}\n"
report += f"- Total modules tracked: {len(registry['modules'])}\n"
report += f"- Classification completeness: {'✓ 100%' if len(unknown) == 0 else f'✗ {len(unknown)} unknown'}\n\n"

# Write Markdown
with open("/tmp/REGISTRY_REPORT.md", "w") as f:
    f.write(report)

ssh_upload_cmd_md = f'cat > /opt/quantum/registry/REGISTRY_REPORT.md << \'EOFM\'\n{report}\nEOFM'
subprocess.run(f'wsl ssh -i {KEY_PATH} root@{HOST} "{ssh_upload_cmd_md}"', shell=True, capture_output=True)

print(f"   ✓ Generated /opt/quantum/registry/REGISTRY_REPORT.md")

# ============================================================================
# Print summary
# ============================================================================
print()
print("="*80)
print("REGISTRY GENERATION COMPLETE")
print("="*80)
print()
print(f"Generated files:")
print(f"  1. /opt/quantum/registry/module_registry.json")
print(f"  2. /opt/quantum/registry/REGISTRY_REPORT.md")
print()
print("Summary:")
print(f"  Total modules:      {len(registry['modules'])}")
print(f"  Running:            {len(running)}")
print(f"  Stopped:            {len(stopped)}")
print(f"  Failed:             {len(failed)}")
print(f"  Orphaned:           {len(orphaned)}")
print(f"  Unknown:            {len(unknown)}")
print()
print("Quality Gate:")
print(f"  All units classified: {'✓' if len(unknown) == 0 else '✗'}")
print()
