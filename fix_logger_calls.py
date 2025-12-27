#!/usr/bin/env python3
"""Fix logger calls that use keyword arguments."""
import re
import sys

def fix_logger_calls(content):
    """Replace logger calls with keyword args to f-strings."""
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a logger call with potential keyword args
        if 'self.logger.' in line and '(' in line and not line.strip().endswith(')'):
            # Start of multi-line logger call
            match = re.match(r'(\s*)(self\.logger\.(info|warning|error))\(\s*$', line)
            if match:
                indent = match.group(1)
                logger_call = match.group(2)
                
                # Collect all lines until closing paren
                call_lines = [line]
                i += 1
                while i < len(lines) and ')' not in lines[i]:
                    call_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    call_lines.append(lines[i])  # Include closing paren line
                
                # Parse the call
                full_call = '\n'.join(call_lines)
                
                # Try to extract message and kwargs
                msg_match = re.search(r'[\'"]([^"\']+)[\'"]', full_call)
                if msg_match:
                    msg = msg_match.group(1)
                    
                    # Find all key=value pairs
                    kwargs = re.findall(r'(\w+)=', full_call)
                    
                    if kwargs and not full_call.startswith(indent + logger_call + '(f'):
                        # Build f-string replacement
                        kwargs_str = ' '.join([f'{k}={{{k}}}' for k in kwargs])
                        result.append(f'{indent}{logger_call}(f"{msg} | {kwargs_str}")')
                        i += 1
                        continue
                
                # Couldn't parse, keep original
                result.extend(call_lines)
                i += 1
                continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result)

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/original.py'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/fixed.py'
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    fixed = fix_logger_calls(content)
    
    with open(output_file, 'w') as f:
        f.write(fixed)
    
    print(f"Fixed: {input_file} -> {output_file}")
