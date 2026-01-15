#!/usr/bin/env python3
"""
MetaPredictorAgent v5 – Meta-Learning Layer
-------------------------------------------
Lærer direkte av ensemble-outputene fra XGBoost, LightGBM, PatchTST og N-HiTS.
Input: 8 features (4 confidences + 4 actions)
Output: Final prediction (BUY/HOLD/SELL)
"""
import os, json, numpy as np, torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pathlib import Path

# ---------- LOGGER ----------
def safe_log(msg):
    """Simple logger for meta agent"""
    print(msg, flush=True)
    try:
        logfile = Path("/var/log/quantum/meta-agent.log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} | {msg}\n")
    except Exception:
        pass

# ---------- NEURAL NETWORK ----------
class MetaNet(nn.Module):
    """Simple feedforward network for meta-learning"""
    def __init__(self, input_dim=8, hidden=32, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ---------- META AGENT ----------
class MetaPredictorAgent:
    """
    Meta-learning agent that combines predictions from all 4 base models.
    
    Input features (8 total):
        - xgb_conf, lgbm_conf, patch_conf, nhits_conf (4 floats: 0-1)
        - xgb_action, lgbm_action, patch_action, nhits_action (4 ints: 0=SELL, 1=HOLD, 2=BUY)
    
    Output:
        - {"action": "BUY"|"HOLD"|"SELL", "confidence": float, "source": "meta_v5"}
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        self.name = "Meta-Agent"
        self.model = None
        self.scaler = None
        self.device = torch.device("cpu")
        self.mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
        self.reverse_mapping = {"SELL": 0, "HOLD": 1, "BUY": 2}
        self.ready = False
        self.version = "v5"
        
        # Model directory
        base = os.path.dirname(os.path.dirname(__file__))
        self.model_dir = os.path.join(base, "models")
        
        # Try to load model
        if model_path or self._find_latest():
            self._load(model_path, scaler_path)
    
    def _find_latest(self):
        """Find latest meta_v model"""
        try:
            files = [os.path.join(self.model_dir, x) 
                    for x in os.listdir(self.model_dir)
                    if x.startswith("meta_v") and x.endswith(".pth")]
            return max(files, key=os.path.getmtime) if files else None
        except Exception:
            return None
    
    def _load(self, model_path=None, scaler_path=None):
        """Load model and scaler"""
        try:
            model_path = model_path or self._find_latest()
            if not model_path:
                safe_log(f"[{self.name}] No model found - agent disabled")
                return
            
            scaler_path = scaler_path or model_path.replace(".pth", "_scaler.pkl")
            
            # Load model
            self.model = MetaNet(input_dim=8, hidden=32, output_dim=3)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            
            # Load scaler
            import joblib
            self.scaler = joblib.load(scaler_path)
            
            self.ready = True
            safe_log(f"[{self.name}] ✅ Loaded {os.path.basename(model_path)}")
            
        except Exception as e:
            safe_log(f"[{self.name}] ❌ Load error: {e}")
            self.ready = False
    
    def is_ready(self):
        """Check if agent is ready"""
        return self.ready and self.model is not None and self.scaler is not None
    
    def predict(self, ensemble_vector: dict, symbol: str = "UNKNOWN"):
        """
        Make prediction based on ensemble outputs
        
        Args:
            ensemble_vector: {"xgb": {"action": "BUY", "confidence": 0.73}, "lgbm": {...}, ...}
            symbol: Trading pair symbol
        
        Returns:
            {"action": str, "confidence": float, "source": str, "version": str}
        """
        if not self.is_ready():
            return {"action": "HOLD", "confidence": 0.0, "source": "meta_v5", "version": self.version}
        
        try:
            # Extract features in consistent order
            acts, confs = [], []
            for key in ["xgb", "lgbm", "patch", "nhits"]:
                v = ensemble_vector.get(key, {})
                action = v.get("action", "HOLD")
                confidence = float(v.get("confidence", 0.5))
                
                # Encode action as integer
                acts.append(self.reverse_mapping.get(action, 1))
                confs.append(confidence)
            
            # Create feature vector: [conf1, conf2, conf3, conf4, act1, act2, act3, act4]
            x = np.array(confs + acts).reshape(1, -1)
            
            # Scale
            x_scaled = self.scaler.transform(x)
            
            # Predict
            with torch.no_grad():
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
            
            # Get prediction
            cls = int(np.argmax(probs))
            action = self.mapping[cls]
            confidence = float(np.max(probs))
            
            # Log
            safe_log(f"[{self.name}] {symbol} → {action} (conf={confidence:.3f})")
            
            return {
                "action": action,
                "confidence": confidence,
                "confidence_std": float(np.std(probs)),
                "source": "meta_v5",
                "version": self.version
            }
            
        except Exception as e:
            safe_log(f"[{self.name}] Prediction error: {e}")
            return {"action": "HOLD", "confidence": 0.0, "source": "meta_v5", "version": self.version}
