#!/usr/bin/env python3
"""
Quantum Trader Module Registry Builder
Authoritative Source of Truth for all modules, services, and components
"""

import json
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Set
import socket

class RegistryBuilder:
    def __init__(self):
        self.host = "46.224.116.254"
        self.registry = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "host": self.host,
            "modules": []
        }
        self.redis_streams = {}
        self.listening_ports = {}
        self.microservices = {}
        self.systemd_units = {}
        self.venvs = set()
        
    def ssh_cmd(self, cmd: str) -> str:
        """Execute SSH command on VPS"""
        full_cmd = f'wsl ssh -i ~/.ssh/hetzner_fresh root@{self.host} "{cmd}"'
        try:
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=30)
            return result.stdout.strip()
        except Exception as e:
            print(f"SSH Error: {e}")
            return ""
    
    def build_registry(self):
        print("[1] Collecting systemd services and timers...")
        self._collect_systemd()
        
        print("[2] Scanning microservices code...")
        self._scan_microservices()
        
        print("[3] Mapping venvs...")
        self._map_venvs()
        
        print("[4] Discovering Redis streams...")
        self._discover_redis_streams()
        
        print("[5] Mapping listening ports...")
        self._map_ports()
        
        print("[6] Classifying modules...")
        self._classify_modules()
        
        print("[7] Generating output files...")
        self._generate_outputs()
        
        print("[8] Running quality gates...")
        self._quality_gates()
    
    def _collect_systemd(self):
        """Collect all systemd service and timer information"""
        # Get all services
        cmd = "systemctl list-units --type=service --all --no-pager"
        output = self.ssh_cmd(cmd)
        
        for line in output.split("\n"):
            if "quantum" in line and "service" in line:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    loaded = parts[1]
                    active = parts[2]
                    sub = parts[3]
                    self.systemd_units[name] = {
                        "type": "service",
                        "loaded": loaded,
                        "active": active,
                        "sub": sub,
                    }
        
        # Get timers
        cmd = "systemctl list-timers --all --no-pager"
        output = self.ssh_cmd(cmd)
        
        for line in output.split("\n"):
            if "quantum" in line and "timer" in line:
                parts = line.split()
                if len(parts) >= 3:
                    # Timer line format is different
                    for i, part in enumerate(parts):
                        if ".timer" in part:
                            name = part
                            self.systemd_units[name] = {
                                "type": "timer",
                                "loaded": "loaded",
                                "active": "active",
                                "sub": "running",
                            }
                            break
        
        # Get detailed info for each service
        for svc_name in list(self.systemd_units.keys()):
            cmd = f"systemctl show {svc_name} -p ExecStart -p WorkingDirectory -p User -p Type -p Pid -p ActiveState -p SubState"
            output = self.ssh_cmd(cmd)
            
            info = {}
            for line in output.split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k] = v
            
            self.systemd_units[svc_name].update(info)
    
    def _scan_microservices(self):
        """Scan microservices directories"""
        cmd = 'python3 << \'EOF\'\nimport os\nbase = "/home/qt/quantum_trader/microservices"\nfor d in sorted(os.listdir(base)):\n    path = os.path.join(base, d)\n    if os.path.isdir(path) and not d.startswith("_"):\n        has_main = os.path.isfile(os.path.join(path, "main.py"))\n        has_service = os.path.isfile(os.path.join(path, "service.py"))\n        has_dunder = os.path.isfile(os.path.join(path, "__main__.py"))\n        if has_main or has_service or has_dunder:\n            ep = "main.py" if has_main else ("service.py" if has_service else "__main__.py")\n            print(f"{d}:{ep}")\n        else:\n            print(f"{d}:NO_ENTRYPOINT")\nEOF\n'
        
        output = self.ssh_cmd(cmd)
        
        for line in output.split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    module_name = parts[0].strip()
                    entrypoint = parts[1].strip()
                    self.microservices[module_name] = {
                        "code_path": f"/home/qt/quantum_trader/microservices/{module_name}",
                        "entrypoint": entrypoint if entrypoint != "NO_ENTRYPOINT" else None,
                    }
    
    def _map_venvs(self):
        """Map venv directories"""
        cmd = "ls -1 /opt/quantum/venvs"
        output = self.ssh_cmd(cmd)
        
        for venv in output.split("\n"):
            venv = venv.strip()
            if venv:
                self.venvs.add(venv)
    
    def _discover_redis_streams(self):
        """Discover Redis streams and consumer groups"""
        # List streams
        cmd = 'redis-cli --scan --pattern "quantum:stream:*"'
        output = self.ssh_cmd(cmd)
        
        streams = []
        for stream in output.split("\n"):
            stream = stream.strip()
            if stream and "quantum:stream:" in stream:
                streams.append(stream)
        
        # Get consumer groups for each stream
        for stream in streams:
            cmd = f'redis-cli XINFO GROUPS {stream} 2>&1'
            output = self.ssh_cmd(cmd)
            
            groups = []
            lines = output.split("\n")
            i = 0
            while i < len(lines):
                if "name" in lines[i] and i + 1 < len(lines):
                    group_name = lines[i + 1].strip()
                    groups.append(group_name)
                    i += 2
                else:
                    i += 1
            
            self.redis_streams[stream] = groups
    
    def _map_ports(self):
        """Map listening ports to services"""
        cmd = "ss -lntp 2>/dev/null | grep LISTEN"
        output = self.ssh_cmd(cmd)
        
        for line in output.split("\n"):
            if "LISTEN" in line and "python" in line.lower():
                parts = line.split()
                for part in parts:
                    if ":" in part and ("127.0.0.1" in part or "0.0.0.0" in part):
                        port = part.split(":")[-1]
                        if port.isdigit():
                            self.listening_ports[int(port)] = line
    
    def _classify_modules(self):
        """Classify each module based on evidence"""
        # Map service names to module info
        module_map = {}
        
        # Add all systemd services
        for svc_name in self.systemd_units.keys():
            if svc_name.endswith(".service") or svc_name.endswith(".timer"):
                clean_name = svc_name.replace(".service", "").replace(".timer", "")
                if not clean_name.startswith("quantum-"):
                    continue
                    
                clean_name = clean_name.replace("quantum-", "")
                module_name = clean_name
                
                module_map[svc_name] = {
                    "name": module_name,
                    "systemd_unit": svc_name,
                    "type": self.systemd_units[svc_name].get("type", "service"),
                }
        
        # Classify each
        for svc_name, module_info in module_map.items():
            svc_data = self.systemd_units.get(svc_name, {})
            
            active = svc_data.get("active", "unknown").lower()
            sub = svc_data.get("sub", "unknown").lower()
            loaded = svc_data.get("loaded", "unknown").lower()
            
            # Determine category
            if active == "active" and (sub == "running" or sub == "exited"):
                category = "RUNNING"
            elif active == "inactive" and loaded == "loaded":
                category = "STOPPED"
            elif active == "failed":
                category = "STOPPED"
            elif loaded == "not-found":
                category = "ORPHANED"
            else:
                category = "UNKNOWN"
            
            # Build module record
            module_record = {
                "name": module_info["name"],
                "category": category,
                "systemd_unit": svc_name,
                "enabled": svc_data.get("UnitFileState", "unknown") in ["enabled", "static"],
                "pid": svc_data.get("Pid", ""),
                "exec": svc_data.get("ExecStart", ""),
                "working_dir": svc_data.get("WorkingDirectory", ""),
                "user": svc_data.get("User", ""),
                "code_path": self.microservices.get(module_info["name"], {}).get("code_path", ""),
                "entrypoint": self.microservices.get(module_info["name"], {}).get("entrypoint", ""),
                "venv_status": self._check_venv_status(svc_data),
                "redis_streams": self._get_redis_streams_for_module(module_info["name"]),
                "ports": self._get_ports_for_module(svc_data),
                "proof_line": f"systemctl status {svc_name} | Active: {active} ({sub})",
            }
            
            self.registry["modules"].append(module_record)
    
    def _check_venv_status(self, svc_data: dict) -> str:
        """Check if venv in ExecStart is valid"""
        exec_start = svc_data.get("ExecStart", "")
        if not exec_start:
            return "NO_VENV"
        
        for venv in self.venvs:
            if venv in exec_start:
                return f"VALID_VENV: {venv}"
        
        if "/opt/quantum/venvs/" in exec_start:
            return "MISSING_VENV"
        
        return "NO_VENV"
    
    def _get_redis_streams_for_module(self, module_name: str) -> List[str]:
        """Get Redis streams consumed by module"""
        streams = []
        for stream, groups in self.redis_streams.items():
            for group in groups:
                if module_name.lower() in group.lower() or "engine" in group.lower():
                    if stream not in streams:
                        streams.append(stream)
        return streams
    
    def _get_ports_for_module(self, svc_data: dict) -> List[int]:
        """Extract port from service ExecStart"""
        ports = []
        exec_start = svc_data.get("ExecStart", "")
        
        import re
        port_matches = re.findall(r'--port\s+(\d+)', exec_start)
        ports.extend([int(p) for p in port_matches])
        
        return ports
    
    def _generate_outputs(self):
        """Generate JSON and Markdown output files"""
        # Create registry directory
        cmd = "mkdir -p /opt/quantum/registry && echo 'OK'"
        self.ssh_cmd(cmd)
        
        # Write JSON
        json_path = "/opt/quantum/registry/module_registry.json"
        json_data = json.dumps(self.registry, indent=2)
        
        cmd = f"cat > {json_path} << 'JSONEOF'\n{json_data}\nJSONEOF"
        self.ssh_cmd(cmd)
        print(f"✓ Generated: {json_path}")
        
        # Generate Markdown report
        self._generate_markdown_report()
    
    def _generate_markdown_report(self):
        """Generate human-readable Markdown report"""
        report = "# Quantum Trader - Module Registry Report\n\n"
        report += f"**Generated:** {self.registry['generated_at']}\n"
        report += f"**Host:** {self.registry['host']}\n\n"
        
        # Summary stats
        running = sum(1 for m in self.registry['modules'] if m['category'] == 'RUNNING')
        stopped = sum(1 for m in self.registry['modules'] if m['category'] == 'STOPPED')
        disabled = sum(1 for m in self.registry['modules'] if m['category'] == 'DISABLED')
        library = sum(1 for m in self.registry['modules'] if m['category'] == 'LIBRARY_ONLY')
        no_ep = sum(1 for m in self.registry['modules'] if m['category'] == 'NO_ENTRYPOINT')
        orphaned = sum(1 for m in self.registry['modules'] if m['category'] == 'ORPHANED')
        unknown = sum(1 for m in self.registry['modules'] if m['category'] == 'UNKNOWN')
        
        report += "## Executive Summary\n\n"
        report += f"| Status | Count |\n"
        report += f"|--------|-------|\n"
        report += f"| RUNNING | {running} |\n"
        report += f"| STOPPED | {stopped} |\n"
        report += f"| DISABLED | {disabled} |\n"
        report += f"| LIBRARY_ONLY | {library} |\n"
        report += f"| NO_ENTRYPOINT | {no_ep} |\n"
        report += f"| ORPHANED | {orphaned} |\n"
        report += f"| UNKNOWN | {unknown} |\n"
        report += f"| **TOTAL** | **{len(self.registry['modules'])}** |\n\n"
        
        # Running modules
        report += "## RUNNING Modules\n\n"
        running_modules = [m for m in self.registry['modules'] if m['category'] == 'RUNNING']
        if running_modules:
            report += "| Name | User | Port(s) | Redis Streams |\n"
            report += "|------|------|---------|---------------|\n"
            for m in sorted(running_modules, key=lambda x: x['name']):
                ports = ", ".join(str(p) for p in m['ports']) if m['ports'] else "-"
                streams = len(m['redis_streams'])
                report += f"| {m['name']} | {m['user']} | {ports} | {streams} |\n"
        else:
            report += "No running modules.\n"
        
        report += "\n"
        
        # Stopped modules
        if stopped > 0:
            report += "## STOPPED Modules\n\n"
            stopped_modules = [m for m in self.registry['modules'] if m['category'] == 'STOPPED']
            for m in sorted(stopped_modules, key=lambda x: x['name']):
                report += f"- `{m['name']}`\n"
            report += "\n"
        
        # Other categories
        for category in ['DISABLED', 'LIBRARY_ONLY', 'NO_ENTRYPOINT', 'ORPHANED', 'UNKNOWN']:
            modules = [m for m in self.registry['modules'] if m['category'] == category]
            if modules:
                report += f"## {category}\n\n"
                for m in sorted(modules, key=lambda x: x['name']):
                    report += f"- `{m['name']}`\n"
                report += "\n"
        
        # Architecture
        report += "## Architecture Overview\n\n"
        report += "```\n"
        report += "┌─ Quantum Trader (systemd) ──────────────────────┐\n"
        report += "│                                                  │\n"
        report += f"│ RUNNING Services:        {running:2d}                    │\n"
        report += f"│ STOPPED Services:        {stopped:2d}                    │\n"
        report += f"│ Data Streams (Redis):    {len(self.redis_streams):2d}                    │\n"
        report += f"│ Listening Ports:         {len(self.listening_ports):2d}                    │\n"
        report += f"│ Python venvs:            {len(self.venvs):2d}                    │\n"
        report += "│                                                  │\n"
        report += "└──────────────────────────────────────────────────┘\n"
        report += "```\n\n"
        
        # Quality gate
        report += "## Quality Gates\n\n"
        report += f"- Systemd units classified: {len(self.systemd_units)}\n"
        report += f"- Microservices scanned: {len(self.microservices)}\n"
        report += f"- Modules in registry: {len(self.registry['modules'])}\n"
        report += f"- All modules categorized: {'✓ YES' if unknown == 0 else '✗ NO'}\n\n"
        
        # Write to file
        md_path = "/opt/quantum/registry/REGISTRY_REPORT.md"
        cmd = f"cat > {md_path} << 'MDEOF'\n{report}\nMDEOF"
        self.ssh_cmd(cmd)
        print(f"✓ Generated: {md_path}")
    
    def _quality_gates(self):
        """Run quality assurance checks"""
        print("\n=== QUALITY GATE CHECKS ===\n")
        
        total_svc = len(self.systemd_units)
        total_classified = len(self.registry['modules'])
        
        print(f"Systemd units found:       {total_svc}")
        print(f"Modules classified:        {total_classified}")
        print(f"Match:                     {'✓' if total_svc == total_classified else '✗'}\n")
        
        running = sum(1 for m in self.registry['modules'] if m['category'] == 'RUNNING')
        stopped = sum(1 for m in self.registry['modules'] if m['category'] == 'STOPPED')
        unknown = sum(1 for m in self.registry['modules'] if m['category'] == 'UNKNOWN')
        
        print(f"RUNNING:                   {running}")
        print(f"STOPPED:                   {stopped}")
        print(f"DISABLED:                  {sum(1 for m in self.registry['modules'] if m['category'] == 'DISABLED')}")
        print(f"LIBRARY_ONLY:              {sum(1 for m in self.registry['modules'] if m['category'] == 'LIBRARY_ONLY')}")
        print(f"NO_ENTRYPOINT:             {sum(1 for m in self.registry['modules'] if m['category'] == 'NO_ENTRYPOINT')}")
        print(f"ORPHANED:                  {sum(1 for m in self.registry['modules'] if m['category'] == 'ORPHANED')}")
        print(f"UNKNOWN:                   {unknown}")
        
        print(f"\n✓ Registry Complete!" if unknown == 0 else f"\n⚠ {unknown} modules classified as UNKNOWN")

if __name__ == "__main__":
    builder = RegistryBuilder()
    builder.build_registry()
    print("\n✓ Registry generation complete!")
