#!/usr/bin/env python3
"""
Add Docker log rotation to all services in docker-compose files.
Prevents unbounded log growth and disk exhaustion.
"""
import sys
import re
from pathlib import Path

LOG_CONFIG = """    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
"""

def add_logging_to_service(service_block: str) -> str:
    """Add logging config to a service block if not present."""
    # Check if logging already exists
    if 'logging:' in service_block:
        print(f"  ⏭️  Logging already configured")
        return service_block
    
    # Find last line with content (before next service or end)
    lines = service_block.rstrip().split('\n')
    
    # Add logging config before the last line
    result = '\n'.join(lines) + '\n' + LOG_CONFIG
    print(f"  ✅ Added log rotation")
    return result

def process_compose_file(filepath: Path) -> tuple[bool, str]:
    """Process a docker-compose file and add logging to all services."""
    print(f"\n📄 Processing: {filepath.name}")
    
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False, f"Error reading {filepath}: {e}"
    
    # Split into services
    # Pattern: service name at start of line with colon
    service_pattern = re.compile(r'^  ([a-z0-9_-]+):\s*$', re.MULTILINE)
    
    matches = list(service_pattern.finditer(content))
    if not matches:
        print(f"  ⚠️  No services found")
        return False, "No services found"
    
    # Process from end to start to preserve positions
    modified = content
    changes = 0
    
    for i in range(len(matches) - 1, -1, -1):
        match = matches[i]
        service_name = match.group(1)
        start_pos = match.start()
        
        # Find end position (next service or end of file)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(modified)
        
        service_block = modified[start_pos:end_pos]
        
        # Skip if logging already exists
        if 'logging:' in service_block:
            print(f"  ⏭️  {service_name}: Already has logging config")
            continue
        
        print(f"  🔧 {service_name}: Adding log rotation...")
        
        # Find the last meaningful line (not empty, not a comment)
        lines = service_block.split('\n')
        insert_index = len(lines) - 1
        
        # Go backwards to find last line with content
        while insert_index > 0 and not lines[insert_index].strip():
            insert_index -= 1
        
        # Insert logging config
        lines.insert(insert_index + 1, LOG_CONFIG.rstrip())
        new_block = '\n'.join(lines)
        
        # Replace in content
        modified = modified[:start_pos] + new_block + modified[end_pos:]
        changes += 1
        print(f"      ✅ Done")
    
    if changes > 0:
        # Write back
        try:
            filepath.write_text(modified, encoding='utf-8')
            print(f"\n  💾 Saved {changes} changes to {filepath.name}")
            return True, f"Modified {changes} services"
        except Exception as e:
            print(f"  ❌ Error writing file: {e}")
            return False, f"Error writing {filepath}: {e}"
    else:
        print(f"  ℹ️  No changes needed")
        return True, "No changes needed"

def main():
    """Process all docker-compose files."""
    compose_files = [
        'docker-compose.yml',
        'docker-compose.trade-intent-consumer.yml',
        'docker-compose.vps.yml',
        'docker-compose.services.yml',
        'docker-compose.prod.yml',
    ]
    
    base_dir = Path(__file__).parent
    results = []
    
    print("=" * 70)
    print("🔧 ADDING DOCKER LOG ROTATION TO ALL SERVICES")
    print("=" * 70)
    
    for filename in compose_files:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"\n📄 {filename}: ⏭️  File not found, skipping")
            continue
        
        success, message = process_compose_file(filepath)
        results.append((filename, success, message))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    for filename, success, message in results:
        status = "✅" if success else "❌"
        print(f"{status} {filename}: {message}")
    
    print("\n" + "=" * 70)
    print("🎉 LOG ROTATION CONFIGURATION COMPLETE")
    print("=" * 70)
    print("\n🚀 Next steps:")
    print("1. Review changes: git diff docker-compose*.yml")
    print("2. Commit: git add docker-compose*.yml && git commit -m 'Add log rotation'")
    print("3. Deploy: ssh VPS 'cd ~/quantum_trader && git pull && docker compose up -d'")
    print("4. Verify: docker inspect quantum_backend | jq '.[0].HostConfig.LogConfig'")

if __name__ == '__main__':
    main()
