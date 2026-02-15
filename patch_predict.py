#!/usr/bin/env python3
"""Patch unified_agents.py to add TorchRegressorWrapper support"""

# Read file
with open('/home/qt/quantum_trader/ai_engine/agents/unified_agents.py', 'r') as f:
    content = f.read()

# Pattern for the fallback section
old_pattern = '''        # Fallback: dummy prediction
        self.logger.w(f"{sym} \u2192 HOLD (dummy fallback, no model)")
        return {"symbol":sym,"action":"HOLD","confidence":0.5,"confidence_std":0.1,"version":self.version}'''

new_code = '''        # Try sklearn-like wrapper model (TorchRegressorWrapper)
        if self.model is not None and hasattr(self.model, 'predict'):
            try:
                pnl_pred = self.model.predict(X)[0]  # Returns PnL%
                # Map PnL to action: <-0.5=SELL, >0.5=BUY, else HOLD
                if pnl_pred < -0.5:
                    act = 'SELL'
                elif pnl_pred > 0.5:
                    act = 'BUY'
                else:
                    act = 'HOLD'
                c = min(abs(pnl_pred) / 3.0 + 0.5, 0.95)
                self.logger.i(f"{sym} \u2192 {act} (pnl={pnl_pred:.3f}, c={c:.3f})")
                return {'symbol':sym,'action':act,'confidence':c,'confidence_std':0.1,'version':self.version}
            except Exception as e:
                self.logger.e(f'Wrapper predict failed: {e}')

        # Fallback: dummy prediction
        self.logger.w(f"{sym} \u2192 HOLD (dummy fallback, no model)")
        return {'symbol':sym,'action':'HOLD','confidence':0.5,'confidence_std':0.1,'version':self.version}'''

# Count occurrences
count = content.count('# Fallback: dummy prediction')
print(f'Found {count} fallback sections')

# Replace
content = content.replace(old_pattern, new_code)

# Save
with open('/home/qt/quantum_trader/ai_engine/agents/unified_agents.py', 'w') as f:
    f.write(content)

# Verify
with open('/home/qt/quantum_trader/ai_engine/agents/unified_agents.py', 'r') as f:
    new_content = f.read()
    wrapper_count = new_content.count('TorchRegressorWrapper')
    print(f'TorchRegressorWrapper references added: {wrapper_count}')
