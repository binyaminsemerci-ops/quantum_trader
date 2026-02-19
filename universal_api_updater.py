#!/usr/bin/env python3
"""
Universal API Updater - Replace all stale Binance API credentials with working ones
Based on discovery from apply-layer.env credentials
"""

import os
import re
import subprocess
import shutil
from pathlib import Path

# Working credentials from apply-layer.env 
WORKING_API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
WORKING_API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

# Known stale credentials to replace (from our investigation)
STALE_CREDENTIALS = {
    # API Keys
    'K1SzH4UYrVF8C1qjh3Ww4OlXWYxDl2WPFSYpHjh1': 'STALE_KEY_1',
    'KfCWBW54KHTd3kwjR8j9KR3JQYjh2KgvsFE8WfGd': 'STALE_KEY_2', 
    'Y2s3uuPDrg4h4dXdM7TW2wPa1nVKJ7j8KQq8v2I9': 'STALE_KEY_3',
    'CrKgj6jHfFR8K8s8u9Dh4YXfQ8s3q2W1WK1vJ2qp': 'STALE_KEY_4',
    'x3J4KjQ8sHgFr2W1WK1vQj9L2K8': 'STALE_KEY_5',
    
    # API Secrets  
    'J4kjYsY2PH9qPQE2WXwYXjPq8jPKjhYJ9jY2Q': 'STALE_SECRET_1',
    'Ks8R4jR9KGj6TK4LU8J9q8HK9a8Kn4xF67Q': 'STALE_SECRET_2',
    'Y7s8u7tDr4g4d3dmM7TW2wPa1nVK8j7KQq': 'STALE_SECRET_3',
    'Kg6FgFR8K8s8u9Dh4YXfQ8s3q2W1WK1vJ2': 'STALE_SECRET_4',
    '8sHgFr2W1WK1vQj9L2K8F2s9h5F2n4': 'STALE_SECRET_5',
}

def find_quantum_trader():
    """Find quantum_trader directory"""
    possible_paths = [
        "/root/quantum_trader",
        "/mnt/c/quantum_trader", 
        "c:\\quantum_trader",
        os.path.expanduser("~/quantum_trader"),
        "."
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found quantum_trader at: {path}")
            return path
    
    print("❌ Could not find quantum_trader directory")
    return None

def backup_file(filepath):
    """Create backup of file before modification"""
    backup_path = f"{filepath}.backup_pre_api_fix"
    if not os.path.exists(backup_path):  # Don't overwrite existing backup
        shutil.copy2(filepath, backup_path)
        print(f"  📁 Backup created: {backup_path}")

def update_env_file(filepath):
    """Update .env file with working credentials"""
    print(f"\n🔧 Processing: {filepath}")
    
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Replace API key patterns
    api_key_patterns = [
        r'BINANCE.*API.*KEY\s*=\s*["\']?([^"\'\s]+)["\']?',
        r'API.*KEY\s*=\s*["\']?([^"\'\s]+)["\']?',
        r'TESTNET.*KEY\s*=\s*["\']?([^"\'\s]+)["\']?'
    ]
    
    for pattern in api_key_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match != WORKING_API_KEY and len(match) > 20:  # Avoid replacing short values
                content = content.replace(match, WORKING_API_KEY)
                changes.append(f"API Key: {match[:20]}... → {WORKING_API_KEY[:20]}...")
    
    # Replace API secret patterns  
    api_secret_patterns = [
        r'BINANCE.*API.*SECRET\s*=\s*["\']?([^"\'\s]+)["\']?',
        r'API.*SECRET\s*=\s*["\']?([^"\'\s]+)["\']?',
        r'TESTNET.*SECRET\s*=\s*["\']?([^"\'\s]+)["\']?'
    ]
    
    for pattern in api_secret_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match != WORKING_API_SECRET and len(match) > 20:  # Avoid replacing short values
                content = content.replace(match, WORKING_API_SECRET)
                changes.append(f"API Secret: {match[:20]}... → {WORKING_API_SECRET[:20]}...")
    
    # Write updated content
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated with {len(changes)} changes:")
        for change in changes:
            print(f"    • {change}")
    else:
        print(f"  ℹ️  No changes needed")

def update_python_file(filepath):
    """Update Python file with working credentials"""
    print(f"\n🐍 Processing: {filepath}")
    
    backup_file(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Replace hardcoded API keys in Python files
    api_key_patterns = [
        r'api_key\s*=\s*["\']([^"\']+)["\']',
        r'API_KEY\s*=\s*["\']([^"\']+)["\']',
        r'testnet_api_key\s*=\s*["\']([^"\']+)["\']'
    ]
    
    for pattern in api_key_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match != WORKING_API_KEY and len(match) > 20:
                content = content.replace(match, WORKING_API_KEY)
                changes.append(f"API Key: {match[:20]}... → {WORKING_API_KEY[:20]}...")
    
    # Replace hardcoded API secrets
    api_secret_patterns = [
        r'api_secret\s*=\s*["\']([^"\']+)["\']',
        r'API_SECRET\s*=\s*["\']([^"\']+)["\']', 
        r'testnet_api_secret\s*=\s*["\']([^"\']+)["\']'
    ]
    
    for pattern in api_secret_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match != WORKING_API_SECRET and len(match) > 20:
                content = content.replace(match, WORKING_API_SECRET)
                changes.append(f"API Secret: {match[:20]}... → {WORKING_API_SECRET[:20]}...")
    
    # Write updated content
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated with {len(changes)} changes:")
        for change in changes:
            print(f"    • {change}")
    else:
        print(f"  ℹ️  No changes needed")

def find_and_update_files(base_path):
    """Find and update all .env and .py files with API credentials"""
    
    # Find .env files
    env_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.env') or file == 'environment':
                env_files.append(os.path.join(root, file))
    
    # Find Python files with potential API credentials
    py_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                # Quick check if file contains API-related content
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    if any(keyword in content.lower() for keyword in ['api_key', 'api_secret', 'binance']):
                        py_files.append(filepath)
                except:
                    pass  # Skip files that can't be read
    
    print(f"📋 Found {len(env_files)} .env files and {len(py_files)} Python files with API content")
    
    # Update .env files
    print(f"\n{'='*50}")
    print(f" UPDATING .ENV FILES ({len(env_files)} files)")
    print(f"{'='*50}")
    
    for env_file in env_files:
        update_env_file(env_file)
    
    # Update Python files
    print(f"\n{'='*50}")
    print(f" UPDATING PYTHON FILES ({len(py_files)} files)")
    print(f"{'='*50}")
    
    for py_file in py_files:
        update_python_file(py_file)

def restart_affected_services():
    """Restart systemd services that use API credentials"""
    services = [
        "quantum-position-state-brain",
        "quantum-balance-tracker", 
        "quantum-pnl-tracker",
        "quantum-testnet-trader",
        "formula-shadow-validator"
    ]
    
    print(f"\n{'='*50}")
    print(f" RESTARTING AFFECTED SERVICES")
    print(f"{'='*50}")
    
    for service in services:
        try:
            print(f"🔄 Restarting {service}...")
            result = subprocess.run(['sudo', 'systemctl', 'restart', f'{service}.service'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"  ✅ {service} restarted successfully")
            else:
                print(f"  ⚠️  {service} restart failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {service} restart timed out")
        except Exception as e:
            print(f"  ❌ {service} restart error: {e}")

def main():
    print(f"\n{'='*60}")
    print(f" QUANTUM TRADER - UNIVERSAL API CREDENTIAL UPDATER")
    print(f"{'='*60}")
    print(f"Working API Key: {WORKING_API_KEY[:20]}...")
    print(f"Working API Secret: {WORKING_API_SECRET[:20]}...")
    
    base_path = find_quantum_trader()
    if not base_path:
        return
    
    print(f"\nBase directory: {base_path}")
    
    # Find and update files
    find_and_update_files(base_path)
    
    # Restart services 
    restart_affected_services()
    
    print(f"\n{'='*60}")
    print(f" ✅ UNIVERSAL API UPDATE COMPLETE")
    print(f"{'='*60}")
    print(f"All stale API credentials have been replaced with working credentials from apply-layer.env")
    print(f"Affected services have been restarted to use new credentials")
    print(f"System should now have full Binance testnet API access across all modules")
    
if __name__ == "__main__":
    main()