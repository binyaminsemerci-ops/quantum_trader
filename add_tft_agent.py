#!/usr/bin/env python3
"""Add TFTAgent to unified_agents.py and EnsembleManager"""

# Step 1: Add TFTAgent class to unified_agents.py
with open('/home/qt/quantum_trader/ai_engine/agents/unified_agents.py', 'r') as f:
    content = f.read()

# TFTAgent class - same pattern as NHiTSAgent
tft_agent_code = '''
# ---------- TFT (Temporal Fusion Transformer) ----------
class TFTAgent(BaseAgent):
    def __init__(self):
        super().__init__("TFT-Agent","tft_v")
        self.pytorch_model = None
        self._load()

    def predict(self,sym,feat):
        df=self._align(feat)

        # Scale features
        if self.scaler:
            X=self.scaler.transform(df)
        else:
            X=df.values

        # Try sklearn-like wrapper model (TorchRegressorWrapper)
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
                self.logger.i(f"{sym} → {act} (pnl={pnl_pred:.3f}, c={c:.3f})")
                return {'symbol':sym,'action':act,'confidence':c,'confidence_std':0.1,'version':self.version}
            except Exception as e:
                self.logger.e(f'Wrapper predict failed: {e}')

        # Fallback: dummy prediction
        self.logger.w(f"{sym} → HOLD (dummy fallback, no model)")
        return {'symbol':sym,'action':'HOLD','confidence':0.5,'confidence_std':0.1,'version':self.version}

'''

# Insert before "# Backward compatibility"
if 'class TFTAgent' not in content:
    content = content.replace(
        '# Backward compatibility',
        tft_agent_code + '# Backward compatibility'
    )
    print("✅ Added TFTAgent class")
else:
    print("⚠️ TFTAgent already exists")

with open('/home/qt/quantum_trader/ai_engine/agents/unified_agents.py', 'w') as f:
    f.write(content)

# Step 2: Add TFT to EnsembleManager imports
with open('/home/qt/quantum_trader/ai_engine/ensemble_manager.py', 'r') as f:
    em_content = f.read()

# Add import
if 'TFTAgent' not in em_content:
    # Check for both single line and multi-line import formats
    old_import = 'from ai_engine.agents.unified_agents import XGBoostAgent, LightGBMAgent, NHiTSAgent, PatchTSTAgent'
    new_import = 'from ai_engine.agents.unified_agents import XGBoostAgent, LightGBMAgent, NHiTSAgent, PatchTSTAgent, TFTAgent'
    if old_import in em_content:
        em_content = em_content.replace(old_import, new_import)
        print("✅ Added TFTAgent import")
    else:
        # Try variant with line break
        old_import2 = 'from ai_engine.agents.unified_agents import XGBoostAgent, LightGBMAgent, NHiT\nSAgent, PatchTSTAgent'
        if old_import2 in em_content:
            em_content = em_content.replace(old_import2, new_import)
            print("✅ Added TFTAgent import (multiline)")
        else:
            print("⚠️ Could not find import line to modify")
else:
    print("⚠️ TFTAgent import already exists")

# Add default weight
if "'tft'" not in em_content:
    em_content = em_content.replace(
        "'patchtst': 0.25    # ✅ ACTIVATED: Transformer-based forecasting",
        "'patchtst': 0.20,   # Transformer-based forecasting\n                'tft': 0.20         # ✅ Temporal Fusion Transformer"
    )
    # Also update other weights
    em_content = em_content.replace("'xgb': 0.25", "'xgb': 0.20")
    em_content = em_content.replace("'lgbm': 0.25", "'lgbm': 0.20")
    em_content = em_content.replace("'nhits': 0.25", "'nhits': 0.20")
    print("✅ Added TFT weight")

# Add default enabled model
if "'tft'" not in em_content or "enabled_models = ['xgb', 'lgbm', 'nhits', 'patchtst']" in em_content:
    em_content = em_content.replace(
        "enabled_models = ['xgb', 'lgbm', 'nhits', 'patchtst']",
        "enabled_models = ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft']"
    )
    print("✅ Added TFT to default enabled models")

# Add TFT agent loading
tft_load_code = '''
        # TFT Agent (Temporal Fusion Transformer)
        self.tft_agent = None
        if 'tft' in enabled_models:
            try:
                self.tft_agent = TFTAgent()
                logger.info(f"[✅ ACTIVATED] TFT agent loaded (weight: {self.weights.get('tft', 0.2)*100}%)")
            except Exception as e:
                logger.warning(f"[⚠️  FALLBACK] TFT loading failed: {e}")
                logger.info("   └─ Will use consensus from other models")
                self.tft_agent = None
        else:
            logger.info("[⏭️  SKIP] TFT agent disabled (not in enabled_models)")
'''

if 'self.tft_agent = None' not in em_content:
    # Insert after PatchTST skip line
    em_content = em_content.replace(
        '            logger.info("[⏭️  SKIP] PatchTST agent disabled (not in enabled_models)")\n\n        # Meta Agent',
        f'            logger.info("[⏭️  SKIP] PatchTST agent disabled (not in enabled_models)")\n{tft_load_code}\n        # Meta Agent'
    )
    # Also try alt format
    if 'self.tft_agent = None' not in em_content:
        em_content = em_content.replace(
            '        # Meta Agent (5th agent',
            f'{tft_load_code}\n        # Meta Agent (5th agent'
        )
    print("✅ Added TFT agent loading")
else:
    print("⚠️ TFT agent loading already exists")

# Update active_models count
em_content = em_content.replace(
    'self.xgb_agent, self.lgbm_agent, self.nhits_agent, self.patchtst_agent]',
    'self.xgb_agent, self.lgbm_agent, self.nhits_agent, self.patchtst_agent, self.tft_agent]'
)

# Add TFT to predict method - after patchtst
tft_predict_code = '''
        # TFT: Only predict if agent is loaded
        if self.tft_agent is not None:
            try:
                predictions['tft'] = self.tft_agent.predict(symbol, features)
            except Exception as e:
                logger.error(f"TFT prediction failed: {e} - excluding from ensemble (FAIL-CLOSED)")
'''
if "predictions['tft']" not in em_content:
    em_content = em_content.replace(
        "                # Don't add to predictions - let ensemble work with remaining models\n\n        # 🔍 QSC FAIL-CLOSED",
        f"                # Don't add to predictions - let ensemble work with remaining models\n{tft_predict_code}\n        # 🔍 QSC FAIL-CLOSED"
    )
    print("✅ Added TFT to predict method")

# Add TFT to model status check
if "'tft': self.tft_agent" not in em_content:
    em_content = em_content.replace(
        "'patchtst': self.patchtst_agent.model is not None if self.patchtst_agent else False",
        "'patchtst': self.patchtst_agent.model is not None if self.patchtst_agent else False,\n            'tft': self.tft_agent.model is not None if self.tft_agent else False"
    )
    print("✅ Added TFT to model status")

with open('/home/qt/quantum_trader/ai_engine/ensemble_manager.py', 'w') as f:
    f.write(em_content)

print("✅ Updated EnsembleManager")
print("Done!")
