#!/usr/bin/env python3
"""Add debug logging to harvest_brain.py"""
import sys

file_path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''            # Handle both flat dict and JSON payload formats
            if 'payload' in msg_data:
                payload_str = msg_data.get('payload', '{}')
                exec_event = json.loads(payload_str)
            elif 'data' in msg_data:
                # EventBus format: {"data": json_string}
                data_str = msg_data.get('data', '{}')
                if isinstance(data_str, bytes):
                    data_str = data_str.decode('utf-8')
                exec_event = json.loads(data_str)
            else:
                # Flat dict format (direct from stream)
                exec_event = {k.decode() if isinstance(k, bytes) else k: 
                             v.decode() if isinstance(v, bytes) else v 
                             for k, v in msg_data.items()}'''

new_code = '''            # DEBUG: Log msg_data structure
            logger.debug(f"🔍 msg_data keys: {list(msg_data.keys())}")
            logger.debug(f"🔍 msg_data type: {type(msg_data)}")
            if 'data' in msg_data:
                logger.debug(f"🔍 data field type: {type(msg_data.get('data'))}")
                logger.debug(f"🔍 data field value (first 100 chars): {str(msg_data.get('data'))[:100]}")
            
            # Handle both flat dict and JSON payload formats
            if 'payload' in msg_data:
                payload_str = msg_data.get('payload', '{}')
                exec_event = json.loads(payload_str)
            elif 'data' in msg_data:
                # EventBus format: {"data": json_string}
                data_str = msg_data.get('data', '{}')
                if isinstance(data_str, bytes):
                    data_str = data_str.decode('utf-8')
                exec_event = json.loads(data_str)
                logger.debug(f"🔍 Parsed exec_event type: {type(exec_event)}")
                logger.debug(f"🔍 exec_event keys: {list(exec_event.keys()) if isinstance(exec_event, dict) else 'NOT A DICT'}")
            else:
                # Flat dict format (direct from stream)
                exec_event = {k.decode() if isinstance(k, bytes) else k: 
                             v.decode() if isinstance(v, bytes) else v 
                             for k, v in msg_data.items()}'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added debug logging to harvest_brain.py")
    sys.exit(0)
else:
    print("❌ Could not find code to replace")
    sys.exit(1)
