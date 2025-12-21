#!/usr/bin/env python3
"""
QUANTUM TRADER - System Inventory Manager
Maintains up-to-date map of all files, containers, and deployments
"""
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class SystemInventory:
    """Manage system inventory and provide quick lookups"""
    
    def __init__(self, inventory_file: str = "SYSTEM_INVENTORY.yaml"):
        self.inventory_file = Path(inventory_file)
        self.data = self._load_inventory()
    
    def _load_inventory(self) -> Dict:
        """Load inventory from YAML file"""
        if not self.inventory_file.exists():
            return {}
        
        with open(self.inventory_file, 'r') as f:
            return yaml.safe_load(f)
    
    def find_file(self, filename: str) -> List[Dict]:
        """Find where a file is located"""
        results = []
        
        # Search in modules
        for module_name, module_data in self.data.get('modules', {}).items():
            if 'files' in module_data:
                for file in module_data['files']:
                    if isinstance(file, dict):
                        if filename in file.get('name', ''):
                            results.append({
                                'module': module_name,
                                'location': module_data.get('location'),
                                'containers': module_data.get('in_containers', []),
                                'file': file
                            })
                    elif isinstance(file, str) and filename in file:
                        results.append({
                            'module': module_name,
                            'location': module_data.get('location'),
                            'containers': module_data.get('in_containers', []),
                            'file': file
                        })
        
        return results
    
    def list_containers(self) -> Dict:
        """Get all container information"""
        return self.data.get('containers', {})
    
    def get_container_contents(self, container_name: str) -> Optional[Dict]:
        """Get what's inside a specific container"""
        return self.data.get('containers', {}).get(container_name)
    
    def get_redis_streams(self) -> Dict:
        """Get Redis stream information"""
        return self.data.get('redis_streams', {})
    
    def check_vps_status(self) -> Dict:
        """Check VPS deployment status"""
        try:
            result = subprocess.run(
                ['ssh', '-i', '~/.ssh/hetzner_fresh', 'qt@46.224.116.254', 
                 'docker ps --format "{{.Names}}\t{{.Status}}"'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                containers = {}
                for line in lines:
                    if '\t' in line:
                        name, status = line.split('\t', 1)
                        containers[name] = status
                return {'status': 'connected', 'containers': containers}
            else:
                return {'status': 'error', 'error': result.stderr}
        
        except Exception as e:
            return {'status': 'disconnected', 'error': str(e)}
    
    def find_module(self, module_name: str) -> Optional[Dict]:
        """Find a module by name"""
        modules = self.data.get('modules', {})
        
        # Exact match
        if module_name in modules:
            return modules[module_name]
        
        # Partial match
        for name, data in modules.items():
            if module_name.lower() in name.lower():
                return {name: data}
        
        return None
    
    def get_troubleshooting(self, issue: str) -> Optional[Dict]:
        """Get troubleshooting info for an issue"""
        issues = self.data.get('troubleshooting', {})
        
        for issue_name, solution in issues.items():
            if issue.lower() in issue_name.lower():
                return {issue_name: solution}
        
        return None
    
    def print_summary(self):
        """Print system summary"""
        print("=" * 70)
        print("QUANTUM TRADER - SYSTEM INVENTORY")
        print("=" * 70)
        print()
        
        # Containers
        containers = self.list_containers()
        print(f"📦 CONTAINERS: {len(containers)}")
        for name, info in containers.items():
            status = info.get('status', 'unknown')
            purpose = info.get('purpose', 'N/A')
            print(f"  {'✅' if status == 'running' else '❌'} {name}: {purpose}")
        print()
        
        # Modules
        modules = self.data.get('modules', {})
        print(f"📁 MODULES: {len(modules)}")
        for name, info in modules.items():
            location = info.get('location', 'N/A')
            status = info.get('status', '')
            print(f"  • {name}: {location} {status}")
        print()
        
        # Redis Streams
        streams = self.get_redis_streams()
        print(f"📊 REDIS STREAMS: {len(streams)}")
        for stream_name, info in streams.items():
            entries = info.get('current_entries', 'N/A')
            print(f"  • {stream_name}: {entries} entries")
        print()


def main():
    """CLI interface for system inventory"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Trader System Inventory Tool')
    parser.add_argument('--find', type=str, help='Find a file by name')
    parser.add_argument('--module', type=str, help='Find a module by name')
    parser.add_argument('--containers', action='store_true', help='List all containers')
    parser.add_argument('--streams', action='store_true', help='List Redis streams')
    parser.add_argument('--vps', action='store_true', help='Check VPS status')
    parser.add_argument('--troubleshoot', type=str, help='Get troubleshooting info')
    parser.add_argument('--summary', action='store_true', help='Print system summary')
    
    args = parser.parse_args()
    
    inventory = SystemInventory()
    
    if args.summary:
        inventory.print_summary()
    
    elif args.find:
        results = inventory.find_file(args.find)
        if results:
            print(f"📍 Found {len(results)} location(s) for '{args.find}':")
            for r in results:
                print(f"\n  Module: {r['module']}")
                print(f"  Location: {r['location']}")
                print(f"  Containers: {', '.join(r['containers'])}")
                print(f"  File: {r['file']}")
        else:
            print(f"❌ File '{args.find}' not found in inventory")
    
    elif args.module:
        result = inventory.find_module(args.module)
        if result:
            print(f"📦 Module info:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Module '{args.module}' not found")
    
    elif args.containers:
        containers = inventory.list_containers()
        print("📦 CONTAINERS:")
        print(json.dumps(containers, indent=2))
    
    elif args.streams:
        streams = inventory.get_redis_streams()
        print("📊 REDIS STREAMS:")
        print(json.dumps(streams, indent=2))
    
    elif args.vps:
        status = inventory.check_vps_status()
        print("📡 VPS STATUS:")
        print(json.dumps(status, indent=2))
    
    elif args.troubleshoot:
        solution = inventory.get_troubleshooting(args.troubleshoot)
        if solution:
            print("🔧 TROUBLESHOOTING:")
            print(json.dumps(solution, indent=2))
        else:
            print(f"❌ No troubleshooting info for '{args.troubleshoot}'")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
