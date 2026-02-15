#!/usr/bin/env python3
"""Patch exit_manager.py to add local R-based fallback"""
import re

FILE_PATH = "/home/qt/quantum_trader/microservices/autonomous_trader/exit_manager.py"

# The fallback code to insert BEFORE "# Default: hold"
FALLBACK_CODE = '''
        # LOCAL FALLBACK: R-based exit when AI unavailable
        # Ensures exits happen even when AI Engine times out
        age_hours = position.age_sec / 3600.0
        
        # R > 2.0: Full take profit (great winner)
        if position.R_net > 2.0:
            logger.info(f"[LOCAL EXIT] {position.symbol}: R={position.R_net:.2f} > 2.0 -> CLOSE 100%")
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"local_fallback_take_profit (R={position.R_net:.2f})",
                hold_score=0,
                exit_score=8,
                factors={"local_fallback": True, "R_net": position.R_net, "trigger": "R>2.0"}
            )
        
        # R > 1.0: Partial take profit (good winner)
        if position.R_net > 1.0:
            logger.info(f"[LOCAL EXIT] {position.symbol}: R={position.R_net:.2f} > 1.0 -> PARTIAL 50%")
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.5,
                reason=f"local_fallback_partial_profit (R={position.R_net:.2f})",
                hold_score=2,
                exit_score=6,
                factors={"local_fallback": True, "R_net": position.R_net, "trigger": "R>1.0"}
            )
        
        # R < -1.0 and age > 4 hours: Cut losses
        if position.R_net < -1.0 and age_hours > 4:
            logger.info(f"[LOCAL EXIT] {position.symbol}: R={position.R_net:.2f} < -1.0, age={age_hours:.1f}h > 4h -> CLOSE")
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"local_fallback_cut_loss (R={position.R_net:.2f}, age={age_hours:.1f}h)",
                hold_score=0,
                exit_score=8,
                factors={"local_fallback": True, "R_net": position.R_net, "age_hours": age_hours, "trigger": "R<-1.0+age>4h"}
            )

'''

def main():
    # Read the file
    with open(FILE_PATH, "r") as f:
        content = f.read()
    
    # Check if already patched
    if "LOCAL FALLBACK" in content:
        print("ALREADY PATCHED - fallback already exists")
        return
    
    # Find the marker and insert before it
    marker = "        # Default: hold"
    if marker not in content:
        print(f"ERROR: Could not find marker '{marker}'")
        return
    
    # Insert fallback before the marker
    new_content = content.replace(marker, FALLBACK_CODE + marker)
    
    # Write back
    with open(FILE_PATH, "w") as f:
        f.write(new_content)
    
    print("PATCH APPLIED SUCCESSFULLY")
    
    # Verify
    with open(FILE_PATH, "r") as f:
        verify = f.read()
    if "LOCAL FALLBACK" in verify and "local_fallback_take_profit" in verify:
        print("VERIFIED: Fallback code confirmed in file")
    else:
        print("WARNING: Verification failed")

if __name__ == "__main__":
    main()
