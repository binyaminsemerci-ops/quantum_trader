"""
FIX2: Replace lines 644-682 in harvest_brain.py with real SL-based stop_dist_pct.
"""
path = "/home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py"
lines = open(path).readlines()

# Lines 644-682 (1-indexed) → 0-indexed: 643 to 681 inclusive
# start = first line to REPLACE, end = first line to KEEP after the block
start = 643   # "        # Build P1 proposal ..."
end   = 682   # line AFTER "        )\n" (the p1_proposal closing paren)

new_block = (
    "        # P1: use actual SL from Redis position hash (accurate & direct).\n"
    "        # Falls back to ATR-based estimate only when stop_loss is missing.\n"
    "        if position.stop_loss and position.stop_loss > 0 and position.entry_price > 0:\n"
    "            stop_dist_pct = abs(position.entry_price - position.stop_loss) / position.entry_price\n"
    "        else:\n"
    "            # Fallback: ATR-based dynamic stop (only used if no SL stored)\n"
    "            import os as _os, sys as _sys\n"
    "            _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))\n"
    "            from common.exit_math import compute_dynamic_stop, ExitPosition as _EP\n"
    "            from common.exit_math import Account as _Acc, Market as _Mkt\n"
    "            from common.risk_settings import DEFAULT_SETTINGS as _DS\n"
    "            _exit = _EP(\n"
    "                symbol=position.symbol,\n"
    "                side='BUY' if position.side == 'LONG' else 'SELL',\n"
    "                entry_price=position.entry_price, size=position.qty,\n"
    "                leverage=position.leverage,\n"
    "                highest_price=position.peak_price if position.peak_price > 0 else position.current_price,\n"
    "                lowest_price=position.trough_price if position.trough_price > 0 else position.current_price,\n"
    "                time_in_trade=position.age_sec, distance_to_liq=None\n"
    "            )\n"
    "            _dyn_stop = compute_dynamic_stop(\n"
    "                _exit, _Acc(equity=10000.0),\n"
    "                _Mkt(current_price=position.current_price, atr=position.entry_price * 0.02),\n"
    "                _DS\n"
    "            )\n"
    "            stop_dist_pct = abs(position.entry_price - _dyn_stop) / position.entry_price\n"
    "\n"
    "        p1_proposal = P1Proposal(\n"
    "            stop_dist_pct=max(stop_dist_pct, 0.001)  # floor at 0.1%\n"
    "        )\n"
    "\n"
)

print(f"Total lines in file: {len(lines)}")
print(f"Line {start+1}: {repr(lines[start][:80])}")
print(f"Line {end}:   {repr(lines[end-1][:80])}")

new_lines = lines[:start] + [new_block] + lines[end:]
open(path, "w").writelines(new_lines)
print(f"SUCCESS: FIX2 applied — replaced lines {start+1}-{end}")
print(f"New file has {len(new_lines)} lines (was {len(lines)})")
