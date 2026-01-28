#!/usr/bin/env python3
import os

fp = "/home/qt/quantum_trader/services/exit_monitor_service.py"

with open(fp) as f:
    lines = f.readlines()

# 1. Update FastAPI import (line 31)
for i, line in enumerate(lines):
    if line.strip() == "from fastapi import FastAPI":
        lines[i] = "from fastapi import FastAPI, Header, HTTPException\n"
        print(f"✅ Updated FastAPI import at line {i+1}")
        break

# 2. Update send_close_order signature (around line 294)
for i, line in enumerate(lines):
    if "async def send_close_order(position: TrackedPosition, reason: str):" in line:
        lines[i] = line.replace(
            "async def send_close_order(position: TrackedPosition, reason: str):",
            "async def send_close_order(position: TrackedPosition, reason: str, force: bool = False):"
        )
        print(f"✅ Updated send_close_order signature at line {i+1}")
        break

# 3. Wrap dedup guards in "if not force:" (around line 296-302)
for i, line in enumerate(lines):
    if "# V3 guards" in line and "if not is_position_open" in lines[i+1]:
        # Insert force check before guards
        indent = "    "
        lines[i] = f"{indent}# V3 guards (skipped if force=True)\n"
        lines.insert(i+1, f"{indent}if not force:\n")
        
        # Find the end of guards (before "try:")
        j = i + 2
        while j < len(lines) and "try:" not in lines[j]:
            # Add extra indent to guard lines
            if lines[j].strip() and not lines[j].strip().startswith("#"):
                lines[j] = "    " + lines[j]
            j += 1
        
        # Add else clause before try
        lines.insert(j, f"{indent}else:\n")
        lines.insert(j+1, f"{indent}    logger.warning(f\"🧪 FORCE MODE: Bypassing guards for {{position.symbol}} reason={{reason}}\")\n")
        
        print(f"✅ Wrapped dedup guards with force check at lines {i+1}-{j}")
        break

# 4. Add manual close endpoint at end (before if __name__)
for i, line in enumerate(lines):
    if 'if __name__ == "__main__":' in line:
        # Insert endpoint before main
        endpoint_code = '''
@app.post("/manual-close/{symbol}")
async def manual_close_position(
    symbol: str,
    reason: str = "MANUAL_PROOF",
    force: bool = False,
    x_exit_token: str = Header(None)
):
    """P0.4C: Gated manual close for testing (requires token + force=true)"""
    # Gate 1: Feature enabled
    if os.getenv("EXIT_MONITOR_MANUAL_CLOSE_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Manual close disabled")
    
    # Gate 2: Token auth
    expected = os.getenv("EXIT_MONITOR_MANUAL_CLOSE_TOKEN", "")
    if not expected or x_exit_token != expected:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    symbol = symbol.upper().strip()
    
    if symbol not in tracked_positions:
        return {"status": "error", "message": f"{symbol} not tracked"}
    
    position = tracked_positions[symbol]
    final_reason = f"MANUAL_PROOF_FORCE:{reason}" if force else reason
    
    if force:
        logger.warning(f"🧪 MANUAL_CLOSE_FORCE {symbol} reason={final_reason} qty={position.quantity}")
    
    await send_close_order(position, final_reason, force=force)
    
    return {
        "status": "success",
        "symbol": symbol,
        "reason": final_reason,
        "force": force,
        "proof": "Check: EXIT_SENT → CLOSE_EXECUTED"
    }


'''
        lines.insert(i, endpoint_code)
        print(f"✅ Added manual-close endpoint at line {i}")
        break

with open(fp, "w") as f:
    f.writelines(lines)

print("✅ All updates complete")
