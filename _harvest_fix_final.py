"""
Comprehensive fix for harvest_brain.py:
FIX-A: Add R_net field to Position dataclass
FIX-B: Set position.R_net = r_net after P2 runs in evaluate()
FIX-C: Use pnl_per_unit in PositionSnapshot (FIX1 was never written to disk)
"""
import sys

path = "/home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py"
content = open(path).read()
original_len = len(content)

# ── FIX-A: Add R_net field to Position dataclass ─────────────────────────
# Insert "    R_net: float = 0.0" after "    trough_price: float = 0.0"
OLD_A = (
    "    trough_price: float = 0.0  # Lowest price reached (LONG) or highest (SHORT)\n"
    "\n"
    "    def r_level(self) -> float:\n"
)
NEW_A = (
    "    trough_price: float = 0.0  # Lowest price reached (LONG) or highest (SHORT)\n"
    "    R_net: float = 0.0          # Current R (set by evaluate() after P2 runs)\n"
    "\n"
    "    def r_level(self) -> float:\n"
)
if OLD_A not in content:
    print("ERROR FIX-A: trough_price+r_level context not found")
    sys.exit(1)
content = content.replace(OLD_A, NEW_A, 1)
print("FIX-A applied: R_net field added to Position dataclass")

# ── FIX-B: Set position.R_net after computing P2 result ──────────────────
# After "r_net = p2_result['R_net']", add "position.R_net = r_net"
OLD_B = (
    "        harvest_action = p2_result['harvest_action']\n"
    "        r_net = p2_result['R_net']\n"
    "        kill_score = p2_result['kill_score']\n"
    "        reason_codes = p2_result['reason_codes']\n"
)
NEW_B = (
    "        harvest_action = p2_result['harvest_action']\n"
    "        r_net = p2_result['R_net']\n"
    "        kill_score = p2_result['kill_score']\n"
    "        reason_codes = p2_result['reason_codes']\n"
    "        # Make P2 R_net available for _calculate_dynamic_fraction\n"
    "        position.R_net = r_net\n"
)
if OLD_B not in content:
    print("ERROR FIX-B: P2 result block not found")
    sys.exit(1)
content = content.replace(OLD_B, NEW_B, 1)
print("FIX-B applied: position.R_net = r_net set after P2")

# ── FIX-C: Use pnl_per_unit in PositionSnapshot ──────────────────────────
OLD_C = (
    "        # Build P2 PositionSnapshot\n"
    "        pos_snapshot = PositionSnapshot(\n"
    "            symbol=position.symbol,\n"
    "            side=position.side,\n"
    "            entry_price=position.entry_price,\n"
    "            current_price=position.current_price,\n"
)
NEW_C = (
    "        # Build P2 PositionSnapshot\n"
    "        # unrealized_pnl must be PER-UNIT so P2 R_net is dimensionally\n"
    "        # consistent with risk_unit = entry_price * stop_dist_pct.\n"
    "        pnl_per_unit = position.unrealized_pnl / max(position.qty, 1e-9)\n"
    "        pos_snapshot = PositionSnapshot(\n"
    "            symbol=position.symbol,\n"
    "            side=position.side,\n"
    "            entry_price=position.entry_price,\n"
    "            current_price=position.current_price,\n"
)
if OLD_C not in content:
    print("ERROR FIX-C: PositionSnapshot build header not found")
    sys.exit(1)
content = content.replace(OLD_C, NEW_C, 1)
print("FIX-C: header replaced, now fixing unrealized_pnl field...")

# Also replace the unrealized_pnl line inside the snapshot
OLD_C2 = (
    "            age_sec=position.age_sec,\n"
    "            unrealized_pnl=position.unrealized_pnl,\n"
    "            current_sl=position.stop_loss,\n"
)
NEW_C2 = (
    "            age_sec=position.age_sec,\n"
    "            unrealized_pnl=pnl_per_unit,\n"
    "            current_sl=position.stop_loss,\n"
)
if OLD_C2 not in content:
    print("ERROR FIX-C2: unrealized_pnl=position.unrealized_pnl not found")
    sys.exit(1)
content = content.replace(OLD_C2, NEW_C2, 1)
print("FIX-C applied: pnl_per_unit used in PositionSnapshot")

# Write the file
open(path, "w").write(content)
new_len = len(content)
print(f"SUCCESS: all 3 fixes applied ({original_len} → {new_len} bytes)")
