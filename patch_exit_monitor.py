#!/usr/bin/env python3
"""P0.EXIT_GUARD - Patch exit_monitor_service.py with dedup guard"""

print("=== P0.EXIT_GUARD: Surgical patch ===")

source = "/home/qt/quantum_trader/services/exit_monitor_service.py"
output = "/tmp/exit_monitor_guarded.py"

# Read original
with open(source, "r") as f:
    content = f.read()

# Define exit guard functions to insert
guard_functions = '''
# === EXIT GUARD FUNCTIONS (P0.EXIT_GUARD) ===
_exit_processed = set()  # Global dedup tracker

def check_exit_dedup(position_symbol, order_id):
    """Prevent duplicate exits - returns True if should skip"""
    key = f"{position_symbol}_{order_id}"
    if key in _exit_processed:
        logger.info(f"🔴 EXIT_DEDUP skip key={key}")
        return True
    _exit_processed.add(key)
    return False

'''

# Insert guard functions before send_close_order
content = content.replace(
    "async def send_close_order(position: TrackedPosition, reason: str):",
    guard_functions + "\nasync def send_close_order(position: TrackedPosition, reason: str):"
)

# Add dedup check at start of send_close_order
old_start = '''async def send_close_order(position: TrackedPosition, reason: str):
    """Send close order to execution service"""
    try:'''

new_start = '''async def send_close_order(position: TrackedPosition, reason: str):
    """Send close order to execution service"""
    # === EXIT GUARD: Deduplication ===
    if check_exit_dedup(position.symbol, position.order_id):
        return
    
    try:'''

content = content.replace(old_start, new_start)

# Change log message to EXIT_PUBLISH
content = content.replace(
    'f"🎯 EXIT TRIGGERED:',
    'f"📤 EXIT_PUBLISH:'
)

# Write patched file
with open(output, "w") as f:
    f.write(content)

print(f"✅ Patched file created: {output}")
print(f"   Size: {len(content)} bytes")

# Compile check
import py_compile
try:
    py_compile.compile(output, doraise=True)
    print("✅ Syntax validation passed")
except py_compile.PyCompileError as e:
    print(f"❌ Syntax error: {e}")
    exit(1)

print("\n=== Changes summary ===")
print("1. Added check_exit_dedup() function")
print("2. Inserted dedup guard in send_close_order()")
print("3. Changed log tag to EXIT_PUBLISH")
print("\nReady to deploy!")
