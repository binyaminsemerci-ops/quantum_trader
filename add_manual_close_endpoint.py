#!/usr/bin/env python3
"""Add manual close endpoint to exit_monitor_service.py for P0.4C testing"""

fp = "/home/qt/quantum_trader/services/exit_monitor_service.py"
with open(fp, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the last @app.get or @app.post endpoint to insert after
insert_after = -1
for i, line in enumerate(lines):
    if "@app.get" in line or "@app.post" in line:
        insert_after = i

if insert_after == -1:
    print("❌ Could not find endpoint to insert after")
    exit(1)

# Find the end of that endpoint (next blank line or next @app)
insert_at = insert_after + 1
while insert_at < len(lines):
    if lines[insert_at].strip() == "" or "@app." in lines[insert_at]:
        break
    insert_at += 1

# Skip blank lines
while insert_at < len(lines) and lines[insert_at].strip() == "":
    insert_at += 1

# Insert the new endpoint
new_endpoint = '''
@app.post("/manual-close/{symbol}")
async def manual_close_position(symbol: str, reason: str = "MANUAL_PROOF"):
    """
    P0.4C: Manual close endpoint for testing exit proof chain
    
    Usage: POST /manual-close/SANDUSDT?reason=MANUAL_PROOF
    """
    symbol = symbol.upper()
    
    if symbol not in tracked_positions:
        return {
            "status": "error",
            "message": f"Symbol {symbol} not tracked",
            "tracked_symbols": list(tracked_positions.keys())
        }
    
    position = tracked_positions[symbol]
    
    # Use existing send_close_order function (already has P0.4C fields)
    await send_close_order(position, reason)
    
    return {
        "status": "success",
        "message": f"Close order sent for {symbol}",
        "symbol": symbol,
        "reason": reason,
        "entry_price": position.entry_price,
        "quantity": position.quantity,
        "side": position.side,
        "proof": "Check logs for EXIT_SENT → CLOSE_EXECUTED chain"
    }


'''

lines.insert(insert_at, new_endpoint)

with open(fp, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"✅ Added manual close endpoint at line {insert_at}")
print("✅ Restart exit-monitor to activate")
