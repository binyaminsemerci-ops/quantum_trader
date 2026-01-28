#!/usr/bin/env python3
"""Update manual-close endpoint with force capability and token auth"""

fp = "/home/qt/quantum_trader/services/exit_monitor_service.py"
with open(fp, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update send_close_order signature to accept force parameter
old_sig = "async def send_close_order(position: TrackedPosition, reason: str):"
new_sig = "async def send_close_order(position: TrackedPosition, reason: str, force: bool = False):"

content = content.replace(old_sig, new_sig)

# 2. Update dedup checks to respect force flag
# Find the dedup checks and wrap them in "if not force:"
old_checks = """    # V3 guards
    if not is_position_open(position.symbol, position.order_id):
        logger.info(f"🔴 EXIT_ALREADY_CLOSED {position.symbol}")
        return
    if exit_rate_limited(position.symbol):
        return

    if check_exit_dedup(position.symbol,position.order_id):return
    if check_exit_cooldown(position.symbol,position.side):return"""

new_checks = """    # V3 guards (skipped if force=True for manual testing)
    if not force:
        if not is_position_open(position.symbol, position.order_id):
            logger.info(f"🔴 EXIT_ALREADY_CLOSED {position.symbol}")
            return
        if exit_rate_limited(position.symbol):
            return
        if check_exit_dedup(position.symbol,position.order_id):return
        if check_exit_cooldown(position.symbol,position.side):return
    else:
        logger.warning(f"🧪 FORCE MODE: Bypassing dedup guards for {position.symbol} reason={reason}")"""

content = content.replace(old_checks, new_checks)

# 3. Replace the manual-close endpoint with gated version
# Find the endpoint
import re
pattern = r'@app\.post\("/manual-close/\{symbol\}"\)\s+async def manual_close_position\([^)]+\):[^}]+return \{[^}]+\}'
match = re.search(pattern, content, re.DOTALL)

if not match:
    print("❌ Could not find manual-close endpoint")
    exit(1)

new_endpoint = '''@app.post("/manual-close/{symbol}")
async def manual_close_position(
    symbol: str,
    reason: str = "MANUAL_PROOF",
    force: bool = False,
    x_exit_token: str = Header(None),
):
    """
    P0.4C: Manual close endpoint for testing exit proof chain (GATED)
    
    Requires:
    - EXIT_MONITOR_MANUAL_CLOSE_ENABLED=true
    - Valid X-Exit-Token header
    - force=true to bypass dedup guards
    
    Usage: 
    curl -X POST "http://localhost:8007/manual-close/SANDUSDT?reason=P04C_PROOF&force=true" \\
         -H "X-Exit-Token: <token>"
    """
    # Gate 1: Feature must be enabled
    if os.getenv("EXIT_MONITOR_MANUAL_CLOSE_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Manual close disabled")
    
    # Gate 2: Token authentication
    expected_token = os.getenv("EXIT_MONITOR_MANUAL_CLOSE_TOKEN", "")
    if not expected_token or x_exit_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Exit-Token")
    
    symbol = symbol.upper().strip()
    
    # Check if position exists
    if symbol not in tracked_positions:
        return {
            "status": "error",
            "message": f"Symbol {symbol} not tracked",
            "tracked_symbols": list(tracked_positions.keys())[:10]
        }
    
    position = tracked_positions[symbol]
    
    # Prefix reason with FORCE if force mode
    final_reason = f"MANUAL_PROOF_FORCE:{reason}" if force else reason
    
    if force:
        logger.warning(
            f"🧪 MANUAL_CLOSE_FORCE symbol={symbol} reason={final_reason} "
            f"qty={position.quantity} entry={position.entry_price}"
        )
    
    # Call send_close_order with force flag
    await send_close_order(position, final_reason, force=force)
    
    return {
        "status": "success",
        "message": f"Close order sent for {symbol}",
        "symbol": symbol,
        "reason": final_reason,
        "force": force,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "side": position.side,
        "proof": "Check logs: EXIT_SENT → CLOSE_EXECUTED chain"
    }'''

content = re.sub(pattern, new_endpoint, content, count=1, flags=re.DOTALL)

# 4. Add missing imports at the top
if "from fastapi import Header" not in content:
    # Find the FastAPI import line
    content = content.replace(
        "from fastapi import FastAPI",
        "from fastapi import FastAPI, Header, HTTPException, Query"
    )

with open(fp, "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Updated send_close_order with force parameter")
print("✅ Updated manual-close endpoint with token auth and force mode")
print("✅ Added dedup bypass for force mode")
