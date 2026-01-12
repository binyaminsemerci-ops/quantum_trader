#!/usr/bin/env python3
"""
Quantum Trader AI Engine v5 – Unified Ensemble System
------------------------------------------------------
All agents share common BaseAgent + Logger classes.
Systemd-ready dual logging to journald and /var/log/quantum/*.log
"""
import os, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path

# ---------- LOGGER ----------
class Logger:
    def __init__(self, name):
        self.name = name
        self.logfile = Path(f"/var/log/quantum/{name.lower().replace(' ','_')}.log")
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
    def _w(self, lvl, msg):
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{self.name}] [{lvl}] {ts} | {msg}"
        print(line, flush=True)
        try:
            with open(self.logfile, "a", encoding="utf-8") as f: f.write(line + "\n")
        except Exception: pass
    def i(self,m): self._w("INFO",m)
    def w(self,m): self._w("WARN",m)
    def e(self,m): self._w("ERROR",m)

# ---------- BASE ----------
class BaseAgent:
    def __init__(self, name, prefix):
        self.name, self.prefix = name, prefix
        self.logger = Logger(name)
        base = os.path.dirname(os.path.dirname(__file__))
        self.model_dir = os.path.join(base,"models")
        self.model=None; self.scaler=None; self.features=[]
        self.ready=False; self.version="unknown"

    def _find_latest(self):
        f=[os.path.join(self.model_dir,x) for x in os.listdir(self.model_dir)
           if x.startswith(self.prefix) and x.endswith(".pkl") and "_scaler" not in x and "_meta" not in x]
        return max(f,key=os.path.getmtime) if f else None

    def _load(self, model_path=None, scaler_path=None):
        model_path = model_path or self._find_latest()
        if not model_path: raise FileNotFoundError(f"No {self.name} model found")
        scaler_path = scaler_path or model_path.replace(".pkl","_scaler.pkl")
        meta_path   = model_path.replace(".pkl","_meta.json")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        if os.path.exists(meta_path):
            meta=json.load(open(meta_path))
            self.features=meta.get("features",[])
            self.version=meta.get("version","unknown")
        else:
            self.features=[f"f{i}" for i in range(self.scaler.n_features_in_)]
        self.ready=True
        self.logger.i(f"✅ Loaded {os.path.basename(model_path)} ({len(self.features)} features)")

    def _align(self, feats:dict):
        df=pd.DataFrame([feats])
        drop=[c for c in df if c not in self.features]
        if drop: self.logger.w(f"Dropping extras: {drop[:3]}{'...' if len(drop)>3 else ''}")
        for m in [f for f in self.features if f not in df]: df[m]=0.0
        return df[self.features]

    def is_ready(self): return self.ready

# ---------- XGBOOST ----------
class XGBoostAgent(BaseAgent):
    def __init__(self): super().__init__("XGB-Agent","xgb_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        p=self.model.predict_proba(X); i=int(np.argmax(p,axis=1)[0])
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]; c,s=float(np.max(p)),float(np.std(p))
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# ---------- LIGHTGBM ----------
class LightGBMAgent(BaseAgent):
    def __init__(self): super().__init__("LGBM-Agent","lightgbm_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        p=self.model.predict(X); i=int(np.argmax(p,axis=1)[0])
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]; c,s=float(np.max(p)),float(np.std(p))
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# ---------- PATCHTST ----------
class PatchTSTAgent(BaseAgent):
    def __init__(self): super().__init__("PatchTST-Agent","patchtst_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        p=self.model.predict_proba(X); i=int(np.argmax(p,axis=1)[0])
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]; c,s=float(np.max(p)),float(np.std(p))
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# ---------- N-HiTS ----------
class NHiTSAgent(BaseAgent):
    def __init__(self): super().__init__("NHiTS-Agent","nhits_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        p=self.model.predict_proba(X); i=int(np.argmax(p,axis=1)[0])
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]; c,s=float(np.max(p)),float(np.std(p))
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# Backward compatibility
XGBAgent = XGBoostAgent
