#!/usr/bin/env python3
"""
P1.1 Hotfix for Safety Telemetry Exporter
- Fix timestamp normalization (ms/s/entry-id)
- Add per-rank symbol gauges
- Add build info
"""

import re

# Read current file
with open("/home/qt/quantum_trader/microservices/safety_telemetry/main.py", "r") as f:
    content = f.read()

# 1. Add timestamp normalization helper function after imports
normalize_ts_function = '''
def normalize_timestamp(value, fallback_entry_id=None):
    """
    P1.1: Normalize timestamp to Unix seconds
    - If value >= 1e12: milliseconds, convert to seconds
    - If value >= 1e9 and < 1e12: seconds, use as-is
    - If value < 1e9 or missing: try to derive from Redis stream entry-id
    """
    try:
        ts = int(value)
        if ts >= 1_000_000_000_000:  # Milliseconds (>= 1e12)
            return ts // 1000
        elif ts >= 1_000_000_000:  # Seconds (>= 1e9)
            return ts
        else:
            # Too small, invalid
            raise ValueError("Timestamp too small")
    except (ValueError, TypeError):
        # Try fallback from entry-id
        if fallback_entry_id:
            try:
                # Entry-id format: "1737841080000-0" (milliseconds-sequence)
                entry_ms = int(fallback_entry_id.split("-")[0])
                if entry_ms >= 1_000_000_000_000:
                    return entry_ms // 1000
                elif entry_ms >= 1_000_000_000:
                    return entry_ms
            except (ValueError, IndexError):
                pass
        return 0  # Fallback to 0 if all else fails

'''

# Insert normalize function after logger definition, before class definition
insert_point = content.find('class SafetyTelemetryExporter:')
if insert_point > 0:
    content = content[:insert_point] + normalize_ts_function + "\n" + content[insert_point:]
    print("✅ Added normalize_timestamp function")
else:
    print("⚠️ Could not find class definition")

# 2. Add build info metric after existing gauge definitions
# Find last_fault_info definition
last_fault_info_pos = content.find('last_fault_info = Info("quantum_safety_last_fault"')
if last_fault_info_pos > 0:
    # Find end of that line
    newline_pos = content.find('\n', last_fault_info_pos)
    
    build_info_code = '''

# P1.1: Build metadata
exporter_build_info = Info("quantum_safety_exporter_build", "Exporter build information")
exporter_build_info.info({"version": "P1.1", "git": "unknown", "deployment": "2026-01-19"})

# P1.1: Individual rank gauges for top symbols
safety_rate_symbol_rank_gauge = Gauge("quantum_safety_rate_symbol_top", "Top symbol by rank", ["rank", "symbol"])
'''
    
    content = content[:newline_pos] + build_info_code + content[newline_pos:]
    print("✅ Added build info and rank gauge metrics")

# 3. Update collect_faults to use normalize_timestamp
# Find the line: timestamp = int(fault_data.get("timestamp", 0))
old_timestamp_line = '                    timestamp = int(fault_data.get("timestamp", 0))'
new_timestamp_line = '                    # P1.1: Use normalized timestamp (handles ms/s/entry-id)\n                    timestamp = normalize_timestamp(\n                        fault_data.get("timestamp", 0),\n                        fallback_entry_id=fault_id\n                    )'

if old_timestamp_line in content:
    content = content.replace(old_timestamp_line, new_timestamp_line)
    print("✅ Updated collect_faults to use normalize_timestamp")

# 4. Update collect_safety_rate_counters to set individual rank gauges
# Find the section where top5 symbols are processed
# Look for: if top5:
top5_section_start = content.find('            # Top 5 symbols\n            top5 = sorted(symbol_counts')
if top5_section_start > 0:
    # Find the info_dict section
    info_dict_start = content.find('            if top5:\n                info_dict = {}', top5_section_start)
    if info_dict_start > 0:
        # Find end of that if block (next else or except)
        else_pos = content.find('            else:\n                safety_rate_symbol_info.info({"symbols": "none"})', info_dict_start)
        
        if else_pos > 0:
            # Insert rank gauge updates before the info() call
            # Find the safety_rate_symbol_info.info(info_dict) line
            info_call_pos = content.find('                safety_rate_symbol_info.info(info_dict)', info_dict_start)
            
            if info_call_pos > 0:
                rank_gauge_code = '''
                
                # P1.1: Set individual rank gauges
                for i, (symbol, count) in enumerate(top5):
                    safety_rate_symbol_rank_gauge.labels(rank=str(i+1), symbol=symbol).set(count)
                
                # Clear unused ranks if less than 5
                for rank in range(len(top5) + 1, 6):
                    try:
                        safety_rate_symbol_rank_gauge.labels(rank=str(rank), symbol="").set(0)
                    except:
                        pass  # Skip if label doesn't exist yet
                '''
                
                content = content[:info_call_pos] + rank_gauge_code + '\n' + content[info_call_pos:]
                print("✅ Added rank gauge updates in collect_safety_rate_counters")

# Write updated file
with open("/home/qt/quantum_trader/microservices/safety_telemetry/main.py", "w") as f:
    f.write(content)

print("\n✅ P1.1 Hotfix applied successfully!")
print("Next: python3 -m py_compile main.py")
