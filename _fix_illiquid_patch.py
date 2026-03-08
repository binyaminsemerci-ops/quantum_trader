#!/usr/bin/env python3
"""Patch service.py: map ILLIQUID regime -> UNKNOWN to fix pydantic crash"""

path = '/opt/quantum/microservices/ai_engine/service.py'

with open(path, 'r') as f:
    content = f.read()

# Find and replace the regime assignment line
old = '                    regime = _regime_result.regime\n                    logger.info(\n                        f"[AI-ENGINE] \U0001f30d Regime: {symbol} \u2192 {regime.value} "'
new = ('                    regime = _regime_result.regime\n'
       '                    # FIX: MarketRegime.ILLIQUID not in AIDecisionMadeEvent enum → map to UNKNOWN\n'
       '                    if hasattr(regime, "value") and regime.value == "illiquid":\n'
       '                        regime = MarketRegime.UNKNOWN\n'
       '                    logger.info(\n'
       '                        f"[AI-ENGINE] \U0001f30d Regime: {symbol} \u2192 {regime.value} "')

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print('PATCHED OK - ILLIQUID mapped to UNKNOWN')
else:
    # Try with escaped emoji (different file encoding)
    idx = content.find('regime = _regime_result.regime')
    if idx == -1:
        print('ERROR: Could not find target string')
    else:
        # Show context for debugging
        print(f'Found at index {idx}')
        print(repr(content[idx:idx+300]))
