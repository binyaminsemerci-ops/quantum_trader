#!/usr/bin/env python3
"""Patch XGBoostAgent.predict() in unified_agents.py to handle native xgb.Booster via DMatrix."""
import re

AGENT_FILE = "/opt/quantum/ai_engine/agents/unified_agents.py"

with open(AGENT_FILE, "r") as f:
    content = f.read()

# Find the exact block using regex
# Match from 'class XGBoostAgent' to the next 'class LightGBMAgent'
xgb_block_re = re.compile(
    r'(class XGBoostAgent\(BaseAgent\):\n.*?)\n(# ---------- LIGHTGBM ----------)',
    re.DOTALL
)

m = xgb_block_re.search(content)
if not m:
    print("ERROR: Could not find XGBoostAgent block")
    print("File snippet around 'XGBoost':")
    idx = content.find("class XGBoostAgent")
    print(repr(content[idx:idx+800]))
    exit(1)

old_block = m.group(1)
print("Found block:")
print(repr(old_block[:200]))

NEW_BLOCK = '''class XGBoostAgent(BaseAgent):
    def __init__(self): super().__init__("XGB-Agent","xgboost_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        # Handle both native xgb.Booster (needs DMatrix) and sklearn XGBClassifier
        if hasattr(self.model, 'predict_proba'):
            # sklearn XGBClassifier API
            class_pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
        else:
            # Native xgb.Booster - wrap in DMatrix
            import xgboost as xgb
            import numpy as _np
            dmat = xgb.DMatrix(X, feature_names=[str(c) for c in df.columns])
            raw = self.model.predict(dmat)
            # multi:softprob -> shape (n, 3); multi:softmax -> shape (n,) with class indices
            if raw.ndim == 2 or (raw.ndim == 1 and len(raw) >= 3 and all(0 <= v <= 1 for v in raw)):
                proba = raw[0] if raw.ndim == 2 else raw
                class_pred = int(_np.argmax(proba))
            else:
                class_pred = int(raw[0])
                proba = [0.1, 0.1, 0.1]; proba[class_pred] = 0.8
        actions = ["SELL", "HOLD", "BUY"]
        act = actions[int(class_pred)]
        c = float(proba[int(class_pred)])
        self.logger.i(f"{sym} \u2192 {act} (class={class_pred}, conf={c:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}'''

patched = content.replace(old_block, NEW_BLOCK, 1)
if patched == content:
    print("ERROR: Replace failed (no change)")
    exit(1)

with open(AGENT_FILE, "w") as f:
    f.write(patched)

print("SUCCESS: XGBoostAgent.predict() patched to handle native Booster via DMatrix")

# Verify
with open(AGENT_FILE, "r") as f:
    verify = f.read()
if "DMatrix" in verify:
    print("VERIFIED: DMatrix handling present in file")
else:
    print("ERROR: DMatrix not found in patched file")
