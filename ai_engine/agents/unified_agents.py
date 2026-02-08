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

# Import PyTorch for N-HiTS and PatchTST
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---------- PYTORCH MODEL ARCHITECTURES ----------
if TORCH_AVAILABLE:
    class NHiTSModel(nn.Module):
        """N-HiTS architecture matching train_nhits_v5.py"""
        def __init__(self, input_size=18, hidden_size=128, num_stacks=4, num_classes=3):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # Stack of blocks
            self.stacks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ) for i in range(num_stacks)
            ])
            
            # Classification head
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_classes)
            )
            
        def forward(self, x):
            # x: [batch, features]
            for stack in self.stacks:
                x = stack(x) + (x if x.shape[-1] == self.hidden_size else 0)
            return self.fc(x)
    
    class PatchTSTModel(nn.Module):
        """PatchTST architecture matching train_patchtst_v5.py"""
        def __init__(self, input_dim=18, d_model=128, n_heads=8, n_layers=4, dropout=0.1, num_classes=3):
            super().__init__()
            
            # Embedding
            self.embedding = nn.Linear(input_dim, d_model)
            
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Classification head
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes)
            )
            
        def forward(self, x):
            # x: [batch, features]
            x = x.unsqueeze(1)  # [batch, 1, features]
            x = self.embedding(x)  # [batch, 1, d_model]
            x = self.transformer(x)  # [batch, 1, d_model]
            x = x.mean(dim=1)  # [batch, d_model]
            return self.fc(x)  # [batch, num_classes]
else:
    # Fallback if torch not available
    NHiTSModel = None
    PatchTSTModel = None

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
                # Go up 3 levels: unified_agents.py -> agents -> ai_engine -> quantum_trader
                agents_dir = os.path.dirname(os.path.abspath(__file__))
                ai_engine_dir = os.path.dirname(agents_dir)
                project_root = os.path.dirname(ai_engine_dir)
                self.model_dir = os.path.join(project_root, "models")
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
                    
                    # Check if this agent supports PyTorch model reconstruction
                    if hasattr(self, '_load_pytorch_model') and isinstance(loaded, dict):
                        # This is a state_dict (OrderedDict), reconstruct the model
                        self.logger.i("Detected state_dict, attempting model reconstruction")
                        self.pytorch_model = self._load_pytorch_model(loaded, meta_path)
                        if self.pytorch_model:
                            self.model = loaded  # Keep state_dict for reference
                            self.logger.i("✅ PyTorch model reconstructed successfully")
                        else:
                            self.logger.e("Failed to reconstruct PyTorch model")
                            self.model = loaded  # Fallback to state_dict (will use dummy predictions)
                    elif isinstance(loaded, dict) and 'model_state_dict' in loaded:
                        # Checkpoint format with explicit key
                        self.logger.w(f"PyTorch checkpoint format not fully supported yet")
                        self.model = loaded
                    else:
                        # Direct model object or other format
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
        # Multi-class classification: classes = [0:SELL, 1:HOLD, 2:BUY]
        class_pred = self.model.predict(X)[0]  # Returns class index (0, 1, or 2)
        proba = self.model.predict_proba(X)[0]  # Returns [p_SELL, p_HOLD, p_BUY]
        
        # Map class to action
        actions = ["SELL", "HOLD", "BUY"]
        act = actions[int(class_pred)]
        c = float(proba[int(class_pred)])  # Confidence = probability of predicted class
        
        self.logger.i(f"{sym} → {act} (class={class_pred}, conf={c:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}

# ---------- LIGHTGBM ----------
class LightGBMAgent(BaseAgent):
    def __init__(self): super().__init__("LGBM-Agent","lightgbm_v"); self._load()
    def predict(self,sym,feat):
        df=self._align(feat); X=self.scaler.transform(df)
        # Multi-class classification: LightGBM Booster.predict() returns probabilities
        proba = self.model.predict(X)[0]  # Returns [p_SELL, p_HOLD, p_BUY]
        class_pred = int(np.argmax(proba))  # Get class with highest probability
        
        # Map class to action
        actions = ["SELL", "HOLD", "BUY"]
        act = actions[class_pred]
        c = float(proba[class_pred])  # Confidence = probability of predicted class
        
        self.logger.i(f"{sym} → {act} (class={class_pred}, conf={c:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}

# ---------- PATCHTST ----------
class PatchTSTAgent(BaseAgent):
    def __init__(self): 
        super().__init__("PatchTST-Agent","patchtst_v")
        self.pytorch_model = None  # Will hold reconstructed nn.Module
        self._load()
    
    def _load_pytorch_model(self, state_dict, meta_path):
        """Reconstruct PyTorch model from state_dict using metadata"""
        if not TORCH_AVAILABLE or PatchTSTModel is None:
            self.logger.e("PyTorch not available, cannot reconstruct model")
            return None
        
        # Load architecture params from metadata
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            arch = meta.get('architecture', {})
            d_model = arch.get('d_model', 128)
            n_heads = arch.get('n_heads', 8)
            n_layers = arch.get('n_layers', 4)
            dropout = arch.get('dropout', 0.1)
            num_features = meta.get('num_features', 18)
            
            self.logger.i(f"Reconstructing PatchTST: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        except Exception as e:
            self.logger.w(f"Could not read metadata, using defaults: {e}")
            d_model, n_heads, n_layers, dropout, num_features = 128, 8, 4, 0.1, 18
        
        # Instantiate model
        model = PatchTSTModel(
            input_dim=num_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            num_classes=3
        )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            # Validate model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.logger.e("Model has zero parameters")
                return None
            self.logger.i(f"✅ State dict loaded ({param_count:,} parameters)")
            
            # Validate model produces non-constant output
            with torch.no_grad():
                x1 = torch.randn(1, num_features)
                x2 = torch.randn(1, num_features)
                logits1 = model(x1)
                logits2 = model(x2)
                
                # Check if outputs are identical (would indicate broken model)
                if torch.allclose(logits1, logits2, atol=1e-6):
                    self.logger.e("Model output is constant - reconstruction failed")
                    return None
            
            self.logger.i("✅ Model validation passed (non-constant output)")
            return model
        except Exception as e:
            self.logger.e(f"Failed to load/validate model: {e}")
            return None
    
    def predict(self,sym,feat):
        df=self._align(feat)
        
        # Scale features
        if self.scaler:
            X=self.scaler.transform(df)
        else:
            X=df.values
        
        # Check if we have a reconstructed PyTorch model
        if self.pytorch_model is not None and TORCH_AVAILABLE:
            try:
                # Convert to tensor
                X_tensor = torch.FloatTensor(X)
                
                # Forward pass (no gradient needed)
                with torch.no_grad():
                    logits = self.pytorch_model(X_tensor)  # [batch, 3]
                    probs = torch.softmax(logits, dim=1)  # [batch, 3]
                    class_pred = torch.argmax(probs, dim=1).item()  # 0, 1, or 2
                    confidence = probs[0, class_pred].item()  # Probability of predicted class
                
                # Map class to action
                actions = ["SELL", "HOLD", "BUY"]
                act = actions[class_pred]
                c = float(confidence)
                
                self.logger.i(f"{sym} → {act} (class={class_pred}, conf={c:.3f})")
                return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}
                
            except Exception as e:
                self.logger.e(f"PyTorch prediction failed: {e}")
                # Fall through to dummy
        
        # Fallback: dummy prediction
        self.logger.w(f"{sym} → HOLD (dummy fallback, no model)")
        return {"symbol":sym,"action":"HOLD","confidence":0.5,"confidence_std":0.1,"version":self.version}

# ---------- N-HiTS ----------
class NHiTSAgent(BaseAgent):
    def __init__(self): 
        super().__init__("NHiTS-Agent","nhits_v")
        self.pytorch_model = None  # Will hold reconstructed nn.Module
        self._load()
    
    def _load_pytorch_model(self, state_dict, meta_path):
        """Reconstruct PyTorch model from state_dict using metadata"""
        if not TORCH_AVAILABLE or NHiTSModel is None:
            self.logger.e("PyTorch not available, cannot reconstruct model")
            return None
        
        # Load architecture params from metadata
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            arch = meta.get('architecture', {})
            hidden_size = arch.get('hidden_size', 128)
            num_stacks = arch.get('num_stacks', 4)
            num_features = meta.get('num_features', 18)
            
            self.logger.i(f"Reconstructing N-HiTS: hidden_size={hidden_size}, num_stacks={num_stacks}")
        except Exception as e:
            self.logger.w(f"Could not read metadata, using defaults: {e}")
            hidden_size, num_stacks, num_features = 128, 4, 18
        
        # Instantiate model
        model = NHiTSModel(
            input_size=num_features,
            hidden_size=hidden_size,
            num_stacks=num_stacks,
            num_classes=3
        )
        
        # Load state dict
        try:
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            # Validate model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.logger.e("Model has zero parameters")
                return None
            self.logger.i(f"✅ State dict loaded ({param_count:,} parameters)")
            
            # Validate model produces non-constant output
            with torch.no_grad():
                x1 = torch.randn(1, num_features)
                x2 = torch.randn(1, num_features)
                logits1 = model(x1)
                logits2 = model(x2)
                
                # Check if outputs are identical (would indicate broken model)
                if torch.allclose(logits1, logits2, atol=1e-6):
                    self.logger.e("Model output is constant - reconstruction failed")
                    return None
            
            self.logger.i("✅ Model validation passed (non-constant output)")
            return model
        except Exception as e:
            self.logger.e(f"Failed to load/validate model: {e}")
            return None
    
    def predict(self,sym,feat):
        df=self._align(feat)
        
        # Scale features
        if self.scaler:
            X=self.scaler.transform(df)
        else:
            X=df.values
        
        # Check if we have a reconstructed PyTorch model
        if self.pytorch_model is not None and TORCH_AVAILABLE:
            try:
                # Convert to tensor
                X_tensor = torch.FloatTensor(X)
                
                # Forward pass (no gradient needed)
                with torch.no_grad():
                    logits = self.pytorch_model(X_tensor)  # [batch, 3]
                    probs = torch.softmax(logits, dim=1)  # [batch, 3]
                    class_pred = torch.argmax(probs, dim=1).item()  # 0, 1, or 2
                    confidence = probs[0, class_pred].item()  # Probability of predicted class
                
                # Map class to action
                actions = ["SELL", "HOLD", "BUY"]
                act = actions[class_pred]
                c = float(confidence)
                
                self.logger.i(f"{sym} → {act} (class={class_pred}, conf={c:.3f})")
                return {"symbol":sym,"action":act,"confidence":c,"confidence_std":0.1,"version":self.version}
                
            except Exception as e:
                self.logger.e(f"PyTorch prediction failed: {e}")
                # Fall through to dummy
        
        # Fallback: dummy prediction
        self.logger.w(f"{sym} → HOLD (dummy fallback, no model)")
        return {"symbol":sym,"action":"HOLD","confidence":0.5,"confidence_std":0.1,"version":self.version}
        act={0:"SELL",1:"HOLD",2:"BUY"}[i]
        c,s=float(np.max(p)),float(np.std(p)) if len(p.shape)>1 else (0.7, 0.1)
        self.logger.i(f"{sym} → {act} (conf={c:.3f},std={s:.3f})")
        return {"symbol":sym,"action":act,"confidence":c,"confidence_std":s,"version":self.version}

# Backward compatibility
XGBAgent = XGBoostAgent
