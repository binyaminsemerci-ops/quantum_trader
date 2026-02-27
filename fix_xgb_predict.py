#!/usr/bin/env python3
"""
Patch unified_agents.py XGBoostAgent.predict() to handle native xgb.Booster.
Run on VPS: python3 /tmp/fix_xgb_predict.py
"""
import re

AGENT_FILE = "/opt/quantum/ai_engine/agents/unified_agents.py"

# Read current content
with open(AGENT_FILE, "r") as f:
    content = f.read()

OLD = '''    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        # Multi-class classification: classes = [0:SELL, 1:HOLD, 2:BUY]
        class_pred = self.model.predict(X)[0]  # Returns class index (0, 1, or 2)
        proba = self.model.predict_proba(X)[0]  # Returns [p_SELL, p_HOLD, p_BUY]

        # Map class to action
        actions = ["SELL", "HOLD", "BUY"]
        act = actions[int(class_pred)]
        c = float(proba[int(class_pred)])  # Confidence = probability of predicted class

        self.logger.i(f"{sym} \\u2192 {act} (class={class_pred}, conf={c:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}
# ---------- LIGHTGBM ----------'''

NEW = '''    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        # Handle both native xgb.Booster (requires DMatrix) and sklearn XGBClassifier
        if hasattr(self.model, 'predict_proba'):
            # sklearn XGBClassifier API
            class_pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
        else:
            # Native xgb.Booster - wrap in DMatrix
            import xgboost as xgb
            import numpy as np
            dmat = xgb.DMatrix(X, feature_names=[str(c) for c in df.columns])
            raw = self.model.predict(dmat)
            if raw.ndim == 2 or (raw.ndim == 1 and len(raw) == 3):
                # multi:softprob - shape (1,3) or (3,)
                proba = raw[0] if raw.ndim == 2 else raw
                class_pred = int(np.argmax(proba))
            else:
                # multi:softmax - returns class index
                class_pred = int(raw[0])
                proba = [0.1, 0.1, 0.1]; proba[class_pred] = 0.8
        # Map class to action: [0:SELL, 1:HOLD, 2:BUY]
        actions = ["SELL", "HOLD", "BUY"]
        act = actions[int(class_pred)]
        c = float(proba[int(class_pred)])
        self.logger.i(f"{sym} \\u2192 {act} (class={class_pred}, conf={c:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}
# ---------- LIGHTGBM ----------'''

if OLD in content:
    patched = content.replace(OLD, NEW, 1)
    with open(AGENT_FILE, "w") as f:
        f.write(patched)
    print("PATCH APPLIED: XGBoostAgent.predict() now handles native Booster")
else:
    # Try without the unicode escape - file may have actual arrow
    OLD2 = OLD.replace('\\u2192', '\u2192')
    NEW2 = NEW.replace('\\u2192', '\u2192')
    if OLD2 in content:
        patched = content.replace(OLD2, NEW2, 1)
        with open(AGENT_FILE, "w") as f:
            f.write(patched)
        print("PATCH APPLIED (unicode variant)")
    else:
        # Show context around predict to debug
        idx = content.find("class XGBoostAgent")
        print("NOT FOUND. Context around XGBoostAgent:")
        print(repr(content[idx:idx+600]))
