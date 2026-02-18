#!/usr/bin/env python3
"""Fix _count_active_positions to exclude ledger/snapshot keys"""
import sys

# Read the file
with open('/home/qt/quantum_trader/microservices/apply_layer/main.py', 'r') as f:
    content = f.read()

# Find and replace the _count_active_positions function
old_func = '''    def _count_active_positions(self) -> int:
        """Count active positions based on nonzero size to avoid snapshot-only keys."""
        count = 0
        for key in self.redis.scan_iter("quantum:position:*"):
            try:
                pos = self.redis.hgetall(key)
                if not pos:
                    continue
                amt_raw = pos.get("position_amt") or pos.get("quantity") or pos.get("positionAmt")
                if amt_raw is None:
                    continue
                amt = float(amt_raw)
                if abs(amt) > POSITION_EPSILON:
                    count += 1
            except Exception:
                continue
        return count'''

new_func = '''    def _count_active_positions(self) -> int:
        """Count active positions based on nonzero size, excluding ledger/snapshot keys."""
        count = 0
        for key in self.redis.scan_iter("quantum:position:*"):
            # Explicitly exclude ledger and snapshot keys
            key_str = key.decode() if isinstance(key, bytes) else key
            if ':snapshot:' in key_str or ':ledger:' in key_str:
                continue
            
            try:
                pos = self.redis.hgetall(key)
                if not pos:
                    continue
                amt_raw = pos.get("position_amt") or pos.get("quantity") or pos.get("positionAmt")
                if amt_raw is None:
                    continue
                amt = float(amt_raw)
                if abs(amt) > POSITION_EPSILON:
                    count += 1
            except Exception:
                continue
        return count'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('/home/qt/quantum_trader/microservices/apply_layer/main.py', 'w') as f:
        f.write(content)
    print("SUCCESS: _count_active_positions fixed to exclude ledger/snapshot keys")
else:
    print("ERROR: Could not find exact match for _count_active_positions function")
    sys.exit(1)
