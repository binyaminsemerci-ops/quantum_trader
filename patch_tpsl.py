import sys

path = '/opt/quantum/backend/services/ai/rl_position_sizing_agent.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

# --- Patch 1: Add ATR override after tpsl_params line ---
OLD1 = (
    "        # Get TP/SL parameters from selected strategy\n"
    "        tpsl_params = self.tpsl_strategies[tpsl_strategy]\n"
    "        \n"
    "        # Get Q-value for logging"
)
NEW1 = (
    "        # Get TP/SL parameters from selected strategy\n"
    "        tpsl_params = self.tpsl_strategies[tpsl_strategy]\n"
    "        \n"
    "        # 🔥 ATR-BASED TP/SL OVERRIDE\n"
    "        # SL = 1.5×ATR (Wilder's classic), TP = 3.0×ATR → always 2:1 reward/risk\n"
    "        # This ensures Kelly, leverage and exit targets are one consistent system:\n"
    "        #   Kelly uses avg_win=TP, avg_loss=SL → ratio determines sizing\n"
    "        #   LeverageEngine uses same ATR → leverage is calibrated to same SL width\n"
    "        _sl_atr = round(max(tpsl_params['sl'], 1.5 * atr_pct), 4)  # never tighter than strategy floor\n"
    "        _tp_atr = round(_sl_atr * 2.0, 4)                           # 2:1 always\n"
    "        _partial_atr = round(_sl_atr * 1.0, 4)                      # partial TP at 1:1 (risk-free)\n"
    "        logger.info(\n"
    "            f'[ATR-TPSL] {symbol}: ATR={atr_pct*100:.2f}% '\n"
    "            f'SL={_sl_atr*100:.2f}% TP={_tp_atr*100:.2f}% (2:1 Kelly-consistent)'\n"
    "        )\n"
    "        # Get Q-value for logging"
)

# --- Patch 2: Use ATR-based values in SizingDecision return ---
OLD2 = (
    "            # 🔥 TP/SL Management\n"
    "            tp_percent=tpsl_params['full_tp'],\n"
    "            sl_percent=tpsl_params['sl'],\n"
    "            partial_tp_enabled=tpsl_params['partial_enabled'],\n"
    "            partial_tp_percent=tpsl_params['partial_tp'],\n"
    "            partial_tp_size=tpsl_params['partial_size']"
)
NEW2 = (
    "            # 🔥 TP/SL Management (ATR-based, Kelly-consistent)\n"
    "            tp_percent=_tp_atr,\n"
    "            sl_percent=_sl_atr,\n"
    "            partial_tp_enabled=tpsl_params['partial_enabled'],\n"
    "            partial_tp_percent=_partial_atr,\n"
    "            partial_tp_size=tpsl_params['partial_size']"
)

ok = 0
if OLD1 in content:
    content = content.replace(OLD1, NEW1, 1)
    ok += 1
else:
    print("NO_MATCH patch1")
    idx = content.find("tpsl_params = self.tpsl_strategies")
    print(repr(content[max(0,idx-50):idx+200]))
    sys.exit(1)

if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    ok += 1
else:
    print("NO_MATCH patch2")
    idx = content.find("tp_percent=tpsl_params")
    print(repr(content[max(0,idx-50):idx+200]))
    sys.exit(1)

# --- Patch 3: ATR-based Kelly defaults when no winning outcomes yet ---
OLD3 = (
    "                    avg_win_pct = sum(o.reward for o in winning_outcomes) / len(winning_outcomes) if winning_outcomes else 0.03\n"
    "                    avg_loss_pct = abs(sum(o.reward for o in losing_outcomes) / len(losing_outcomes)) if losing_outcomes else 0.015"
)
NEW3 = (
    "                    avg_win_pct = sum(o.reward for o in winning_outcomes) / len(winning_outcomes) if winning_outcomes else round(3.0 * atr_pct, 4)\n"
    "                    avg_loss_pct = abs(sum(o.reward for o in losing_outcomes) / len(losing_outcomes)) if losing_outcomes else round(1.5 * atr_pct, 4)"
)
if OLD3 in content:
    content = content.replace(OLD3, NEW3, 1)
    ok += 1
else:
    print("NO_MATCH patch3 (non-fatal, skipping)")

# --- Patch 4: ATR-based Kelly defaults when zero trades ---
OLD4 = (
    "                    # Defaults for new agent\n"
    "                    win_rate = 0.55\n"
    "                    avg_win_pct = 0.03\n"
    "                    avg_loss_pct = 0.015\n"
    "                    profit_factor = 1.5"
)
NEW4 = (
    "                    # Defaults for new agent — ATR-based so Kelly matches TP/SL\n"
    "                    win_rate = 0.55\n"
    "                    avg_win_pct = round(3.0 * atr_pct, 4)   # = TP target\n"
    "                    avg_loss_pct = round(1.5 * atr_pct, 4)  # = SL target\n"
    "                    profit_factor = 1.5"
)
if OLD4 in content:
    content = content.replace(OLD4, NEW4, 1)
    ok += 1
else:
    print("NO_MATCH patch4 (non-fatal, skipping)")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"PATCH_OK: {ok} patches applied")
