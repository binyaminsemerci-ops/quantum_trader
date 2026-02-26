import sys

path = '/opt/quantum/backend/services/ai/rl_position_sizing_agent.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('\r\n', '\n')

# Patch: override tp_pct / sl_pct in MATH-AI SizingDecision with ATR-based values
OLD = (
    "                # Return Math AI decision\n"
    "                decision = SizingDecision(\n"
    "                    position_size_usd=position_size_usd,\n"
    "                    leverage=leverage,\n"
    "                    risk_pct=risk_pct,\n"
    "                    confidence=optimal.confidence_score,\n"
    "                    reasoning=reasoning,\n"
    "                    state_key=\"math_ai_mode\",\n"
    "                    action_key=f\"math_ai_{position_size_usd:.0f}_{leverage:.1f}x\",\n"
    "                    q_value=0.0,  # Math AI doesn't use Q-values\n"
    "                    tp_percent=optimal.tp_pct,\n"
    "                    sl_percent=optimal.sl_pct,\n"
    "                    partial_tp_enabled=True,\n"
    "                    partial_tp_percent=optimal.partial_tp_pct,\n"
    "                    partial_tp_size=0.5,  # Always take 50% at partial TP\n"
    "                )"
)

NEW = (
    "                # 🔥 ATR-BASED TP/SL: SL=1.5xATR, TP=3.0xATR (2:1 Kelly-consistent)\n"
    "                # Floor: never tighter than math_ai computed values\n"
    "                _ma_sl = round(max(optimal.sl_pct, 1.5 * atr_pct), 4)\n"
    "                _ma_tp = round(_ma_sl * 2.0, 4)\n"
    "                _ma_partial = round(_ma_sl * 1.0, 4)\n"
    "                logger.info(\n"
    "                    f'[ATR-TPSL] {symbol}: ATR={atr_pct*100:.2f}% '\n"
    "                    f'SL={_ma_sl*100:.2f}% TP={_ma_tp*100:.2f}% '\n"
    "                    f'(math_ai_base SL={optimal.sl_pct*100:.2f}% TP={optimal.tp_pct*100:.2f}%)'\n"
    "                )\n"
    "                # Return Math AI decision\n"
    "                decision = SizingDecision(\n"
    "                    position_size_usd=position_size_usd,\n"
    "                    leverage=leverage,\n"
    "                    risk_pct=risk_pct,\n"
    "                    confidence=optimal.confidence_score,\n"
    "                    reasoning=reasoning,\n"
    "                    state_key=\"math_ai_mode\",\n"
    "                    action_key=f\"math_ai_{position_size_usd:.0f}_{leverage:.1f}x\",\n"
    "                    q_value=0.0,  # Math AI doesn't use Q-values\n"
    "                    tp_percent=_ma_tp,\n"
    "                    sl_percent=_ma_sl,\n"
    "                    partial_tp_enabled=True,\n"
    "                    partial_tp_percent=_ma_partial,\n"
    "                    partial_tp_size=0.5,  # Always take 50% at partial TP\n"
    "                )"
)

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("PATCH_OK")
else:
    print("NO_MATCH")
    idx = content.find("# Return Math AI decision")
    print(repr(content[max(0,idx-50):idx+400]))
    sys.exit(1)
