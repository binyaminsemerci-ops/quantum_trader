#!/usr/bin/env python3
"""Add Docker log rotation to all services in docker-compose files."""
import re
from pathlib import Path

def add_logging_to_compose(filepath):
    print(f'\n🔧 Processing {filepath.name}...')
    content = filepath.read_text(encoding='utf-8')
    
    # Split into lines
    lines = content.split('\n')
    result = []
    i = 0
    services_modified = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a service definition (2 spaces, name, colon)
        service_match = re.match(r'^  ([a-z0-9_-]+):\s*$', line)
        
        if service_match:
            service_name = service_match.group(1)
            result.append(line)
            i += 1
            
            # Collect all lines for this service
            service_lines = []
            has_logging = False
            
            while i < len(lines):
                current = lines[i]
                # Next service detected
                if re.match(r'^  [a-z0-9_-]+:\s*$', current):
                    break
                service_lines.append(current)
                if 'logging:' in current:
                    has_logging = True
                i += 1
            
            # Add service lines
            result.extend(service_lines)
            
            # Add logging if missing
            if not has_logging:
                result.append('    logging:')
                result.append('      driver: json-file')
                result.append('      options:')
                result.append('        max-size: 10m')
                result.append('        max-file: \'3\'')
                services_modified += 1
                print(f'  ✅ {service_name}: Added log rotation')
            else:
                print(f'  ⏭️  {service_name}: Already has logging')
        else:
            result.append(line)
            i += 1
    
    if services_modified > 0:
        filepath.write_text('\n'.join(result), encoding='utf-8')
        print(f'\n💾 Saved {services_modified} changes to {filepath.name}')
        return services_modified
    else:
        print(f'ℹ️  No changes needed')
        return 0

# Main
if __name__ == '__main__':
    files = [
        Path('docker-compose.yml'),
        Path('docker-compose.trade-intent-consumer.yml'),
    ]
    
    print('=' * 70)
    print('🔧 ADDING DOCKER LOG ROTATION')
    print('=' * 70)
    
    total = 0
    for filepath in files:
        if filepath.exists():
            total += add_logging_to_compose(filepath)
    
    print('\n' + '=' * 70)
    print(f'🎉 Total services updated: {total}')
    print('=' * 70)
