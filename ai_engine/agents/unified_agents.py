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
    def __init__(self, name, prefix, model_dir=None):
        self.name, self.prefix = name, prefix
        self.logger = Logger(name)
        # FIX: Use absolute path or environment variable for models directory
        if model_dir:
            self.model_dir = model_dir
        else:
            # Try to resolve from __file__, but fallback to absolute path
            try:
                base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                self.model_dir = os.path.join(base, "models")
            except:
                # Fallback to hardcoded path for systemd services
                self.model_dir = "/home/qt/quantum_trader/models"
        
        if not os.path.exists(self.model_dir):
            self.logger.w(f"Model directory not found: {self.model_dir}, trying fallback")
            self.model_dir = "/home/qt/quantum_trader/models"
            
        self.model=None; self.scaler=None; self.features=[]
        self.ready=False; self.version="unknown"

    def _find_latest(self):
        # Support both .pkl and .pth formats
        f=[]
        for ext in [".pkl", ".pth"]:
            files = [os.path.join(self.model_dir,x) for x in os.listdir(self.model_dir)
                    if x.startswith(self.prefix) and x.endswith(ext) and "_scaler" not in x and "_meta" not in x]
            f.extend(files)
        return max(f,key=os.path.getmtime) if f else None

    def _load(self, model_path=None, scaler_path=None):
        model_path = model_path or self._find_latest()
        if not model_path: raise FileNotFoundError(f"No {self.name} model found")
        
        # Determine extension and set scaler/meta paths
        ext = os.path.splitext(model_path)[1]
        base_path = model_path.replace(ext, "")
        scaler_path = scaler_path or f"{base_path}_scaler.pkl"
        meta_path   = f"{base_path}_meta.json"
        
        # Load model (pkl with joblib, pth with torch)
        try:
            if ext == ".pkl":
                loaded = joblib.load(model_path)
                # FIX: Check if loaded object is a dict (checkpoint) or direct model
                if isinstance(loaded, dict):
                    if 'model_state_dict' in loaded:
                        # PyTorch checkpoint saved as .pkl - reload with torch
                        self.logger.w(f"Found PyTorch checkpoint in .pkl format, skipping (needs .pth loader)")
                        self.model = None
                    elif 'model' in loaded:
                        # Dict with 'model' key
                        self.model = loaded['model']
                    else:
                        # Unknown dict format
                        self.logger.e(f"Loaded dict without 'model' key: {list(loaded.keys())}")
                        self.model = None
                else:
                    # Direct model object
                    self.model = loaded
                    
            elif ext == ".pth":
                try:
                    import torch
                    loaded = torch.load(model_path, map_location='cpu', weights_only=False)
                    # Check if it's a checkpoint dict or direct model
                    if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                        # Need to reconstruct model architecture - NOT SUPPORTED HERE
                        self.logger.e(f"PyTorch checkpoint requires model architecture reconstruction")
                        self.model = None
                    else:
                        self.model = loaded
                except Exception as e:
                    self.logger.w(f"PyTorch load failed: {e}, trying joblib")
                    self.model = joblib.load(model_path)
            else:
                raise ValueError(f"Unknown model format: {ext}")
                
            if self.model is None:
                raise ValueError(f"Model loaded but is None or unsupported format")
                
        except Exception as e:
            self.logger.e(f"Model load error: {e}")
            raise
        
        # Load scaler (MUST exist for sklearn models)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            self.logger.w(f"Scaler not found at {scaler_path}")
            self.scaler = None
        
        # Load metadata
        if os.path.exists(meta_path):
            meta=json.load(open(meta_path))
            self.features=meta.get("features",[])
            self.version=meta.get("version","unknown")
        else:
            self.features=[f"f{i}" for i in range(self.scaler.n_features_in_ if self.scaler else 14)]
        
        self.ready=True
        self.logger.i(f"✅ Loaded {os.path.basename(model_path)} (model={type(self.model).__name__}, features={len(self.features)})")

    def _align(self, feats:dict):
        df=pd.DataFrame([feats])
        drop=[c for c in df if c not in self.features]
        if drop: self.logger.w(f"Dropping extras: {drop[:3]}{'...' if len(drop)>3 else ''}")
        for m in [f for f in self.features if f not in df]: df[m]=0.0
        return df[self.features]

    def is_ready(self): return self.ready

# ---------- XGBOOST ----------
class XGBoostAgent(BaseAgent):
    def __init__(self): super().__init__("XGB-Agent","xgboost_v"); self._load()
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
        df=self._align(feat)
        if self.scaler:
            X=self.scaler.transform(df)
        else:
            X=df.values
        
        # Handle PyTorch models
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
                X_t = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else torch.tensor(X).float()
                with torch.no_grad():
                    logits = self.model(X_t)
                p = torch.softmax(logits, dim=1).numpy() if len(logits.shape) > 1 else np.array([[0,0,1]])
            else:
                p=self.model.predict_proba(X)
        except:
            p=self.model.predict_proba(X)
        
        i=int(np.argmax(p[0] if len(p.shape)>1 else p, axis=0))
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]
        c,s=float(np.max(p)),float(np.std(p)) if len(p.shape)>1 else (0.7, 0.1)
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# ---------- N-HiTS ----------
class NHiTSAgent(BaseAgent):
    def __init__(self): super().__init__("NHiTS-Agent","nhits_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat)
        if self.scaler:
            X=self.scaler.transform(df)
        else:
            X=df.values
        
        # Handle PyTorch models
        try:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model.eval()
                X_t = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else torch.tensor(X).float()
                with torch.no_grad():
                    logits = self.model(X_t)
                p = torch.softmax(logits, dim=1).numpy() if len(logits.shape) > 1 else np.array([[0,0,1]])
            else:
                p=self.model.predict_proba(X)
        except:
            p=self.model.predict_proba(X)
        
        i=int(np.argmax(p[0] if len(p.shape)>1 else p, axis=0))
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]
        c,s=float(np.max(p)),float(np.std(p)) if len(p.shape)>1 else (0.7, 0.1)
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# Backward compatibility
XGBAgent = XGBoostAgent
