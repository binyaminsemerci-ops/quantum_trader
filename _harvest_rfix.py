"""
Patch harvest_brain.py with two fixes:
1. Pass pnl_per_unit (not total dollar pnl) to PositionSnapshot
   so P2 R_net has correct units (per-unit basis).
2. Use actual SL price distance for stop_dist_pct instead of
   ATR-based estimate — direct, accurate, no extra imports needed.
"""
import sys

path = "/home/qt/quantum_trader/microservices/harvest_brain/harvest_brain.py"
content = open(path).read()

# ── FIX 1: PositionSnapshot unrealized_pnl ───────────────────────────────
OLD1 = (
    "        # Build P2 PositionSnapshot\n"
    "        pos_snapshot = PositionSnapshot(\n"
    "            symbol=position.symbol,\n"
    "            side=position.side,\n"
    "            entry_price=position.entry_price,\n"
    "            current_price=position.current_price,\n"
    "            peak_price=position.peak_price if position.peak_price > 0 else position.current_price,\n"
    "            trough_price=position.trough_price if position.trough_price > 0 else position.current_price,\n"
    "            age_sec=position.age_sec,\n"
    "            unrealized_pnl=position.unrealized_pnl,\n"
    "            current_sl=position.stop_loss,\n"
    "            current_tp=position.take_profit\n"
    "        )\n"
)

NEW1 = (
    "        # Build P2 PositionSnapshot\n"
    "        # unrealized_pnl must be PER-UNIT so P2 R_net is dimensionally\n"
    "        # consistent with risk_unit = entry_price * stop_dist_pct.\n"
    "        pnl_per_unit = position.unrealized_pnl / max(position.qty, 1e-9)\n"
    "        pos_snapshot = PositionSnapshot(\n"
    "            symbol=position.symbol,\n"
    "            side=position.side,\n"
    "            entry_price=position.entry_price,\n"
    "            current_price=position.current_price,\n"
    "            peak_price=position.peak_price if position.peak_price > 0 else position.current_price,\n"
    "            trough_price=position.trough_price if position.trough_price > 0 else position.current_price,\n"
    "            age_sec=position.age_sec,\n"
    "            unrealized_pnl=pnl_per_unit,\n"
    "            current_sl=position.stop_loss,\n"
    "            current_tp=position.take_profit\n"
    "        )\n"
)

if OLD1 not in content:
    print("ERROR: FIX1 target not found — check indentation")
    sys.exit(1)

content = content.replace(OLD1, NEW1, 1)
print("FIX1 applied: per-unit pnl in PositionSnapshot")

# ── FIX 2: Replace ATR-based stop_dist_pct with real SL distance ─────────
OLD2 = (
    "        # Build P1 proposal (stop distance) using FORMULA-BASED calculation\n"
    "        # Import exit_math for dynamic stop calculation\n"
    "        import sys\n"
    "        import os\n"
    "        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))\n"
    "        from common.exit_math import compute_dynamic_stop, ExitPosition, Account, Market\n"
    "        from common.risk_settings import DEFAULT_SETTINGS\n"
    "\n"
    "        # Build ExitPosition for stop calculation\n"
    "        exit_position = ExitPosition(\n"
    "            symbol=position.symbol,\n"
    "            side=\"BUY\" if position.side == \"LONG\" else \"SELL\",\n"
    "            entry_price=position.entry_price,\n"
    "            size=position.qty,\n"
    "            leverage=position.leverage,\n"
    "            highest_price=position.peak_price if position.peak_price > 0 else position.current_price,\n"
    "            lowest_price=position.trough_price if position.trough_price > 0 else position.current_price,\n"
    "            time_in_trade=position.age_sec,\n"
    "            distance_to_liq=None  # Not available in harvest brain context\n"
    "        )\n"
    "\n"
    "        # Get account equity (fallback to reasonable value)\n"
    "        # TODO: Fetch from Redis if available\n"
    "        account_equity = 10000.0  # USD\n"
    "        account = Account(equity=account_equity)\n"
    "\n"
    "        # Get market data (use current price and estimate ATR)\n"
    "        # Estimate ATR as 1% of entry price (fallback)\n"
    "        atr_estimate = position.entry_price * 0.01\n"
    "        market = Market(current_price=position.current_price, atr=atr_estimate)\n"
    "\n"
    "        # Calculate FORMULA-BASED dynamic stop distance\n"
    "        dynamic_stop = compute_dynamic_stop(exit_position, account, market, DEFAULT_SETTINGS)\n"
    "        stop_dist_pct = abs(position.entry_price - dynamic_stop) / position.entry_price\n"
    "\n"
    "        # Build P1 proposal with formula-based stop distance (NO HARDCODED 0.02)\n"
    "        p1_proposal = P1Proposal(\n"
    "            stop_dist_pct=stop_dist_pct\n"
    "        )\n"
)

NEW2 = (
    "        # Build P1 proposal: use the actual SL from Redis hash (accurate)\n"
    "        # Fall back to ATR-based estimate only when SL is missing.\n"
    "        if position.stop_loss and position.stop_loss > 0 and position.entry_price > 0:\n"
    "            stop_dist_pct = abs(position.entry_price - position.stop_loss) / position.entry_price\n"
    "        else:\n"
    "            # Fallback: 2% ATR estimate\n"
    "            import os, sys as _sys\n"
    "            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))\n"
    "            from common.exit_math import compute_dynamic_stop, ExitPosition, Account, Market\n"
    "            from common.risk_settings import DEFAULT_SETTINGS\n"
    "            exit_position = ExitPosition(\n"
    "                symbol=position.symbol,\n"
    "                side='BUY' if position.side == 'LONG' else 'SELL',\n"
    "                entry_price=position.entry_price, size=position.qty,\n"
    "                leverage=position.leverage,\n"
    "                highest_price=position.peak_price if position.peak_price > 0 else position.current_price,\n"
    "                lowest_price=position.trough_price if position.trough_price > 0 else position.current_price,\n"
    "                time_in_trade=position.age_sec, distance_to_liq=None\n"
    "            )\n"
    "            account = Account(equity=10000.0)\n"
    "            market = Market(current_price=position.current_price, atr=position.entry_price * 0.02)\n"
    "            dynamic_stop = compute_dynamic_stop(exit_position, account, market, DEFAULT_SETTINGS)\n"
    "            stop_dist_pct = abs(position.entry_price - dynamic_stop) / position.entry_price\n"
    "\n"
    "        p1_proposal = P1Proposal(\n"
    "            stop_dist_pct=max(stop_dist_pct, 0.001)  # floor at 0.1%\n"
    "        )\n"
)

if OLD2 not in content:
    print("ERROR: FIX2 target not found — printing snippet for diagnosis:")
    # Find "Build P1 proposal" and print nearby lines
    idx = content.find("Build P1 proposal")
    print(repr(content[idx:idx+1200]))
    sys.exit(1)

content = content.replace(OLD2, NEW2, 1)
print("FIX2 applied: real SL-based stop_dist_pct")

open(path, "w").write(content)
print("SUCCESS: harvest_brain.py patched with R_net unit fix")
