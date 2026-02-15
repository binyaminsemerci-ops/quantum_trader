#!/usr/bin/env python3
"""
Fix Exit Manager to properly take profit:
1. Increase timeout to 60s (API takes ~48s)
2. Improve fallback logic to actually make exit decisions
3. Add LOCAL exit logic when AI Engine is slow
"""

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/exit_manager.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# Fix 1: Increase timeout from 30s to 60s
old1 = "self.http_client = httpx.AsyncClient(timeout=30.0)"
new1 = "self.http_client = httpx.AsyncClient(timeout=60.0)  # Increased: AI Engine takes ~48s"
if old1 in content:
    content = content.replace(old1, new1)
    changes += 1
    print("✅ Fix 1: Increased HTTP timeout to 60s")

# Fix 2: Find and improve _get_fallback_exit_decision
# This is the critical fix - the fallback needs to make actual exit decisions

old_fallback = '''    def _get_fallback_exit_decision(self, position: Position) -> ExitDecision:
        """
        Fallback exit decision when AI Engine is unavailable.
        Uses R-threshold logic with fee protection (simulates AI Engine logic).
        """
        R = position.R_net
        age_hours = position.age_sec / 3600'''

new_fallback = '''    def _get_fallback_exit_decision(self, position: Position) -> ExitDecision:
        """
        Fallback exit decision when AI Engine is unavailable.
        Uses R-threshold logic with fee protection (simulates AI Engine logic).
        
        🔧 FIX: Now makes actual exit decisions instead of always HOLD
        """
        R = position.R_net
        age_hours = position.age_sec / 3600
        
        # 🚨 EMERGENCY: Extreme profit - CLOSE immediately
        if R > 8:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"LOCAL_FALLBACK_extreme_profit_R={R:.2f}",
                hold_score=0,
                exit_score=10,
                factors={"R_net": R, "trigger": "extreme_profit"}
            )
        
        # 🎯 HIGH PROFIT: R > 3 - Take partial profit
        if R > 3:
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.5,
                reason=f"LOCAL_FALLBACK_high_profit_R={R:.2f}",
                hold_score=2,
                exit_score=6,
                factors={"R_net": R, "trigger": "high_profit"}
            )
        
        # 💰 DECENT PROFIT: R > 2 - Consider exit
        if R > 2:
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.3,
                reason=f"LOCAL_FALLBACK_decent_profit_R={R:.2f}",
                hold_score=3,
                exit_score=5,
                factors={"R_net": R, "trigger": "decent_profit"}
            )
        
        # ⏰ OLD POSITION: >6h with any profit - take it
        if age_hours > 6 and R > 0.5:
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.5,
                reason=f"LOCAL_FALLBACK_old_position_{age_hours:.1f}h_R={R:.2f}",
                hold_score=2,
                exit_score=5,
                factors={"R_net": R, "age_hours": age_hours, "trigger": "old_with_profit"}
            )
        
        # 📉 LOSING FOR TOO LONG: >12h with loss - cut loss
        if age_hours > 12 and R < -1:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"LOCAL_FALLBACK_cut_loss_{age_hours:.1f}h_R={R:.2f}",
                hold_score=1,
                exit_score=8,
                factors={"R_net": R, "age_hours": age_hours, "trigger": "cut_loss"}
            )'''

if old_fallback in content:
    content = content.replace(old_fallback, new_fallback)
    changes += 1
    print("✅ Fix 2: Improved fallback exit logic with actual decisions")
else:
    print("⚠️ Fix 2: Could not find fallback pattern - trying alternative")
    # Try to find just the method signature
    if "def _get_fallback_exit_decision" in content:
        print("   Found method, but pattern doesn't match exactly")

# Write changes
if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to exit_manager.py")
else:
    print("\n⚠️ No changes applied - patterns may have changed")
    print("   Manual inspection needed")
