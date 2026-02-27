#!/usr/bin/env python3
"""
P0 FIX: Add SKIP decision handling to execution_service.py
Prevents SKIP events from falling through to TradeIntent parsing
"""

TARGET_FILE = '/opt/quantum/services/execution_service.py'
INSERT_AFTER_LINE = 1149  # After signal_data.pop('side', None) in EXECUTE block

skip_patch = """
            # P0 FIX Feb 19: Handle SKIP decisions (do not parse as TradeIntent)
            if signal_data.get("decision") == "SKIP":
                logger.debug(f"[PATH1B] ACK SKIP {symbol}: {signal_data.get('error', 'no_error')}")
                if stream_name and group_name:
                    try:
                        await eventbus.redis.xack(stream_name, group_name, msg_id)
                    except Exception as ack_err:
                        logger.error(f"❌ Failed to ACK SKIP {msg_id}: {ack_err}")
                continue
"""

def main():
    with open(TARGET_FILE, 'r') as f:
        lines = f.readlines()
    
    # Insert patch
    lines.insert(INSERT_AFTER_LINE, skip_patch)
    
    with open(TARGET_FILE, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ Patch applied: SKIP handling added after line {INSERT_AFTER_LINE}")
    print(f"Backup available at: {TARGET_FILE}.backup_feb19_skip_fix")

if __name__ == '__main__':
    main()
