#!/usr/bin/env python3
"""
Fix Exit Manager:
1. Increase timeout to 60s (API takes ~48s)
2. Add cut-loss logic for losing positions
"""

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/exit_manager.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# Fix 1: Increase timeout from 30s to 60s
old1 = "self.http_client = httpx.AsyncClient(timeout=30.0)"
new1 = "self.http_client = httpx.AsyncClient(timeout=60.0)  # FIX: AI Engine needs ~48s"
if old1 in content:
    content = content.replace(old1, new1)
    changes += 1
    print("✅ Fix 1: Increased HTTP timeout to 60s")

# Fix 2: Add cut-loss logic before the final HOLD
old2 = '''        # Default: hold
        return ExitDecision(
            symbol=position.symbol,
            action="HOLD",
            percentage=0.0,
            reason="fallback_hold",
            hold_score=5,
            exit_score=0,
            factors={"fallback": True, "R_net": R}
        )'''

new2 = '''        # === CUT LOSS for old losing positions ===
        # After 6+ hours with R < -1, cut losses to free capital
        if age_hours > 6 and R < -1:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"fallback_cut_loss (R={R:.2f}, age={age_hours:.1f}h)",
                hold_score=1,
                exit_score=7,
                factors={"fallback": True, "R_net": R, "age_hours": age_hours}
            )
        
        # After 12+ hours with ANY loss, cut losses
        if age_hours > 12 and R < 0:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"fallback_max_age_loss (R={R:.2f}, age={age_hours:.1f}h)",
                hold_score=0,
                exit_score=9,
                factors={"fallback": True, "R_net": R, "age_hours": age_hours}
            )

        # Default: hold
        return ExitDecision(
            symbol=position.symbol,
            action="HOLD",
            percentage=0.0,
            reason="fallback_hold",
            hold_score=5,
            exit_score=0,
            factors={"fallback": True, "R_net": R}
        )'''

if old2 in content:
    content = content.replace(old2, new2)
    changes += 1
    print("✅ Fix 2: Added cut-loss logic for old losing positions")

# Write changes
if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to exit_manager.py")
else:
    print("\n⚠️ No changes applied")
