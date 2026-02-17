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
# These must match EXACTLY the architectures in backend/domains/learning/model_training.py
if TORCH_AVAILABLE:
    class NHiTSModel(nn.Module):
        """N-HiTS architecture matching model_training.py (uses blocks, output_layer)"""
        def __init__(self, input_dim, hidden_dim, output_dim, n_blocks=3, mlp_units=None, dropout=0.1):
            super().__init__()
            if mlp_units is None:
                mlp_units = [512, 512]
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.n_blocks = n_blocks
            
            # Multi-rate blocks (uses 'blocks' to match saved weights)
            self.blocks = nn.ModuleList([
                self._create_block(input_dim, hidden_dim, output_dim, mlp_units, dropout)
                for _ in range(n_blocks)
            ])
            
            # Output layer (uses 'output_layer' to match saved weights)
            self.output_layer = nn.Linear(output_dim * n_blocks, output_dim)
        
        def _create_block(self, input_dim, hidden_dim, output_dim, mlp_units, dropout):
            layers = []
            layers.append(nn.Linear(input_dim, mlp_units[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for i in range(len(mlp_units) - 1):
                layers.append(nn.Linear(mlp_units[i], mlp_units[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(mlp_units[-1], output_dim))
            return nn.Sequential(*layers)
        
        def forward(self, x):
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten
            block_outputs = []
            for block in self.blocks:
                block_out = block(x)
                block_outputs.append(block_out)
            concatenated = torch.cat(block_outputs, dim=-1)
            return self.output_layer(concatenated)
    
    class PatchTSTModel(nn.Module):
        """PatchTST architecture matching model_training.py (uses patch_embedding, pos_encoding, output_proj)"""
        def __init__(self, input_dim, hidden_dim, output_dim, patch_len=16, stride=8, n_heads=4, n_layers=3, dropout=0.1):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.patch_len = patch_len
            self.stride = stride
            
            # Patch embedding (uses 'patch_embedding' to match saved weights)
            self.patch_embedding = nn.Linear(patch_len * input_dim, hidden_dim)
            
            # Positional encoding (uses 'pos_encoding' to match saved weights)
            self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Output projection (uses 'output_proj' to match saved weights)
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            if x.dim() == 2:
                # x: [batch, features] -> treat as single timestep, duplicate for patches
                batch_size, features = x.size()
                x = x.unsqueeze(1).expand(-1, self.patch_len, -1)  # [batch, patch_len, features]
            
            batch_size, seq_len, features = x.size()
            
            # Create patches
            patches = []
            for i in range(0, max(1, seq_len - self.patch_len + 1), self.stride):
                patch = x[:, i:i + self.patch_len, :]
                patch = patch.reshape(batch_size, -1)
                patches.append(patch)
            
            if not patches:
                patches = [x.reshape(batch_size, -1)]
            
            patches = torch.stack(patches, dim=1)
            embedded = self.patch_embedding(patches)
            n_patches = embedded.size(1)
            embedded = embedded + self.pos_encoding[:, :n_patches, :]
            transformed = self.transformer(embedded)
            pooled = transformed.mean(dim=1)
            return self.output_proj(pooled)

    class GatedResidualNetwork(nn.Module):
        """Gated Residual Network for TFT"""
        def __init__(self, input_size, hidden_size, output_size=None, dropout=0.1):
            super().__init__()
            if output_size is None:
                output_size = input_size
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.gate = nn.Linear(input_size + hidden_size, output_size)
            self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(output_size)

        def forward(self, x):
            residual = self.skip(x)
            h = torch.relu(self.fc1(x))
            h = self.dropout(h)
            h = self.fc2(h)
            gate_input = torch.cat([x, torch.relu(self.fc1(x))], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            return self.layer_norm(gate * h + residual)

    class VariableSelectionNetwork(nn.Module):
        """Variable Selection Network for TFT"""
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.feature_transform = nn.Linear(input_size, hidden_size)
            self.gating = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
                nn.Softmax(dim=-1)
            )
            self.grn = GatedResidualNetwork(hidden_size, hidden_size)

        def forward(self, x):
            weights = self.gating(x)
            weighted = x * weights
            transformed = self.feature_transform(weighted)
            return self.grn(transformed)

    class TemporalFusionBlock(nn.Module):
        """Temporal Fusion Block for TFT"""
        def __init__(self, hidden_size, num_layers=2, dropout=0.1):
            super().__init__()
            self.static_encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, 
                                        batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.enrichment_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )

        def forward(self, x, hidden=None):
            if x.dim() == 3:
                static = self.static_encoder(x[:, -1, :])
                out, hidden = self.decoder_lstm(x, hidden)
                gate_input = torch.cat([out[:, -1, :], static], dim=-1)
                gate = self.enrichment_gate(gate_input)
                return out[:, -1, :] * gate
            else:
                return self.static_encoder(x)

    class TFTModel(nn.Module):
        """Temporal Fusion Transformer model with 49-feature support (Feb 2026)"""
        def __init__(self, input_size=49, hidden_size=128, num_heads=8, num_layers=3, 
                     num_classes=3, dropout=0.1):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            self.vsn = VariableSelectionNetwork(input_size, hidden_size)
            self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                                        batch_first=True, bidirectional=True, dropout=dropout)
            self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.grn1 = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)
            self.grn2 = GatedResidualNetwork(hidden_size, hidden_size * 2, hidden_size, dropout)
            self.fusion = TemporalFusionBlock(hidden_size, num_layers=2, dropout=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [batch, 1, features]
            
            batch_size, seq_len, _ = x.shape
            vsn_out = []
            for t in range(seq_len):
                vsn_out.append(self.vsn(x[:, t, :]))
            x = torch.stack(vsn_out, dim=1)
            
            enc_out, _ = self.encoder_lstm(x)
            enc_out = self.encoder_projection(enc_out)
            attn_out, _ = self.attention(enc_out, enc_out, enc_out)
            attn_out = self.grn1(attn_out[:, -1, :])
            processed = self.grn2(attn_out)
            fused = self.fusion(enc_out)
            combined = self.layer_norm(processed + fused)
            logits = self.classifier(combined)
            
            return logits

else:
    # Fallback if torch not available
    NHiTSModel = None
    PatchTSTModel = None
    TFTModel = None

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
                # Go up 2 levels: unified_agents.py -> agents -> ai_engine
                agents_dir = os.path.dirname(os.path.abspath(__file__))
                ai_engine_dir = os.path.dirname(agents_dir)
                self.model_dir = os.path.join(ai_engine_dir, "models")
            except:
                # Fallback to hardcoded path for systemd services
                self.model_dir = "/home/qt/quantum_trader/ai_engine/models"
        
        if not os.path.exists(self.model_dir):
            self.logger.w(f"Model directory not found: {self.model_dir}, trying fallback")
            self.model_dir = "/home/qt/quantum_trader/ai_engine/models"
            
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
        
        # FIX: Handle checkpoint format where state_dict is nested under 'model_state_dict' key
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            self.logger.i("Detected checkpoint format, extracting model_state_dict")
            checkpoint = state_dict
            actual_state_dict = checkpoint['model_state_dict']
            # Extract architecture params from checkpoint - map to new param names
            hidden_dim = checkpoint.get('hidden_dim', checkpoint.get('d_model', checkpoint.get('hidden_size', 128)))
            n_heads = checkpoint.get('n_heads', checkpoint.get('nhead', 4))
            n_layers = checkpoint.get('n_layers', checkpoint.get('num_layers', 3))
            dropout = checkpoint.get('dropout', 0.1)
            input_dim = checkpoint.get('input_dim', checkpoint.get('num_features', checkpoint.get('input_size', 18)))
            output_dim = checkpoint.get('output_dim', 3)  # 3 classes
            patch_len = checkpoint.get('patch_len', 16)
            stride = checkpoint.get('stride', 8)
            self.logger.i(f"Checkpoint params: hidden_dim={hidden_dim}, n_heads={n_heads}, n_layers={n_layers}")
        else:
            actual_state_dict = state_dict
            hidden_dim, n_heads, n_layers, dropout, input_dim, output_dim = 128, 4, 3, 0.1, 18, 3
            patch_len, stride = 16, 8
            # Load architecture params from metadata file
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                arch = meta.get('architecture', {})
                hidden_dim = arch.get('hidden_dim', arch.get('d_model', hidden_dim))
                n_heads = arch.get('n_heads', arch.get('nhead', n_heads))
                n_layers = arch.get('n_layers', arch.get('num_layers', n_layers))
                dropout = arch.get('dropout', dropout)
                input_dim = meta.get('input_dim', meta.get('num_features', input_dim))
                output_dim = arch.get('output_dim', output_dim)
                patch_len = arch.get('patch_len', patch_len)
                stride = arch.get('stride', stride)
                self.logger.i(f"Metadata params: hidden_dim={hidden_dim}, n_heads={n_heads}")
            except Exception as e:
                self.logger.w(f"Could not read metadata, using defaults: {e}")
        
        # Infer params from state_dict if possible
        try:
            # Get hidden_dim from patch_embedding
            if 'patch_embedding.weight' in actual_state_dict:
                hidden_dim = actual_state_dict['patch_embedding.weight'].shape[0]
                self.logger.i(f"Inferred hidden_dim={hidden_dim} from state_dict")
            
            # Get output_dim from output_proj
            if 'output_proj.weight' in actual_state_dict:
                output_dim = actual_state_dict['output_proj.weight'].shape[0]
                self.logger.i(f"Inferred output_dim={output_dim} from state_dict")
            
            # Count transformer layers
            layer_keys = [k for k in actual_state_dict.keys() if 'transformer.layers.' in k]
            if layer_keys:
                layer_indices = set(int(k.split('.')[2]) for k in layer_keys)
                n_layers = len(layer_indices)
                self.logger.i(f"Inferred n_layers={n_layers} from state_dict")
        except Exception as e:
            self.logger.w(f"Could not infer params from state_dict: {e}")
        
        self.logger.i(f"Reconstructing PatchTST: hidden_dim={hidden_dim}, n_heads={n_heads}, n_layers={n_layers}")
        
        # Instantiate model with correct params for model_training.py architecture
        model = PatchTSTModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Load state dict
        try:
            model.load_state_dict(actual_state_dict)
            model.eval()  # Set to evaluation mode
            
            # Validate model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.logger.e("Model has zero parameters")
                return None
            self.logger.i(f"✅ State dict loaded ({param_count:,} parameters)")
            
            # Validate model produces non-constant output
            with torch.no_grad():
                x1 = torch.randn(1, input_dim)
                x2 = torch.randn(1, input_dim)
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
        
        # FIX: Handle checkpoint format where state_dict is nested under 'model_state_dict' key
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            self.logger.i("Detected checkpoint format, extracting model_state_dict")
            checkpoint = state_dict
            actual_state_dict = checkpoint['model_state_dict']
            # Extract architecture params from checkpoint - map to new param names
            hidden_dim = checkpoint.get('hidden_dim', checkpoint.get('hidden_size', 128))
            n_blocks = checkpoint.get('n_blocks', checkpoint.get('num_stacks', 3))
            input_dim = checkpoint.get('input_dim', checkpoint.get('num_features', checkpoint.get('input_size', 18)))
            output_dim = checkpoint.get('output_dim', 3)  # 3 classes: SELL, HOLD, BUY
            dropout = checkpoint.get('dropout', 0.1)
            self.logger.i(f"Checkpoint params: input_dim={input_dim}, hidden_dim={hidden_dim}, n_blocks={n_blocks}")
        else:
            actual_state_dict = state_dict
            hidden_dim, n_blocks, input_dim, output_dim, dropout = 128, 3, 18, 3, 0.1
            # Load architecture params from metadata file
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                arch = meta.get('architecture', {})
                hidden_dim = arch.get('hidden_dim', arch.get('hidden_size', hidden_dim))
                n_blocks = arch.get('n_blocks', arch.get('num_stacks', n_blocks))
                input_dim = meta.get('input_dim', meta.get('num_features', input_dim))
                output_dim = arch.get('output_dim', output_dim)
                dropout = arch.get('dropout', dropout)
                self.logger.i(f"Metadata params: hidden_dim={hidden_dim}, n_blocks={n_blocks}")
            except Exception as e:
                self.logger.w(f"Could not read metadata, using defaults: {e}")
        
        # Infer params from state_dict if needed
        try:
            # Count blocks from state_dict keys (blocks.X.*)
            block_keys = [k for k in actual_state_dict.keys() if k.startswith('blocks.')]
            if block_keys:
                block_indices = set(int(k.split('.')[1]) for k in block_keys)
                n_blocks = len(block_indices)
                self.logger.i(f"Inferred n_blocks={n_blocks} from state_dict")
            
            # Get input_dim from first block's first layer
            if 'blocks.0.0.weight' in actual_state_dict:
                input_dim = actual_state_dict['blocks.0.0.weight'].shape[1]
                self.logger.i(f"Inferred input_dim={input_dim} from state_dict")
            
            # Get output_dim from output_layer
            if 'output_layer.weight' in actual_state_dict:
                output_dim = actual_state_dict['output_layer.weight'].shape[0]
                self.logger.i(f"Inferred output_dim={output_dim} from state_dict")
        except Exception as e:
            self.logger.w(f"Could not infer params from state_dict: {e}")
        
        self.logger.i(f"Reconstructing N-HiTS: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, n_blocks={n_blocks}")
        
        # Instantiate model with correct params for model_training.py architecture
        model = NHiTSModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_blocks=n_blocks,
            mlp_units=[512, 512],  # Default from model_training.py
            dropout=dropout
        )
        
        # Load state dict
        try:
            model.load_state_dict(actual_state_dict)
            model.eval()  # Set to evaluation mode
            
            # Validate model has parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.logger.e("Model has zero parameters")
                return None
            self.logger.i(f"✅ State dict loaded ({param_count:,} parameters)")
            
            # Validate model produces non-constant output
            with torch.no_grad():
                x1 = torch.randn(1, input_dim)
                x2 = torch.randn(1, input_dim)
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


class TFTAgent(BaseAgent):
    """Temporal Fusion Transformer agent with 49-feature support (Feb 2026)"""
    def __init__(self):
        super().__init__("TFT-Agent", "tft_v")
        self.pytorch_model = None
        self._load()
    
    def _load_pytorch_model(self, state_dict, meta_path):
        """Reconstruct TFT model from checkpoint"""
        if not TORCH_AVAILABLE or TFTModel is None:
            self.logger.e("PyTorch not available, cannot reconstruct TFT model")
            return None
        
        try:
            # Handle checkpoint format
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.logger.i("Detected checkpoint format, extracting model_state_dict")
                checkpoint = state_dict
                actual_state_dict = checkpoint['model_state_dict']
                # Extract architecture params
                input_size = checkpoint.get('input_size', checkpoint.get('num_features', 49))
                hidden_size = checkpoint.get('hidden_size', 128)
                num_heads = checkpoint.get('num_heads', 8)
                num_layers = checkpoint.get('num_layers', 3)
                num_classes = checkpoint.get('num_classes', 3)
                dropout = checkpoint.get('dropout', 0.1)
                self.logger.i(f"Checkpoint params: input_size={input_size}, hidden_size={hidden_size}")
            else:
                actual_state_dict = state_dict
                # Load from metadata
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    arch = meta.get('architecture', {})
                    input_size = meta.get('input_size', meta.get('num_features', 49))
                    hidden_size = arch.get('hidden_size', 128)
                    num_heads = arch.get('num_heads', 8)
                    num_layers = arch.get('num_layers', 3)
                    num_classes = arch.get('num_classes', 3)
                    dropout = arch.get('dropout', 0.1)
                except:
                    # Defaults
                    input_size, hidden_size, num_heads, num_layers, num_classes, dropout = 49, 128, 8, 3, 3, 0.1
            
            # Reconstruct model
            model = TFTModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
            
            # Load state dict
            model.load_state_dict(actual_state_dict, strict=False)
            model.eval()
            
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            if param_count == 0:
                self.logger.e("Model has zero parameters")
                return None
            self.logger.i(f"✅ TFT state dict loaded ({param_count:,} parameters)")
            
            # Validate output
            with torch.no_grad():
                x1 = torch.randn(1, input_size)
                x2 = torch.randn(1, input_size)
                logits1 = model(x1)
                logits2 = model(x2)
                if torch.allclose(logits1, logits2, atol=1e-6):
                    self.logger.e("Model output is constant - reconstruction failed")
                    return None
            
            self.logger.i("✅ TFT model validation passed")
            return model
        except Exception as e:
            self.logger.e(f"Failed to load/validate TFT model: {e}")
            return None
    
    def predict(self, sym, feat):
        """Predict using 49-feature schema"""
        # Import 49-feature schema
        try:
            from ai_engine.common_features import FEATURES_V6, get_feature_default
            features = FEATURES_V6
        except ImportError:
            # Fallback to hardcoded 49-feature list
            features = [
                'returns', 'log_returns', 'price_range', 'body_size', 'upper_wick', 'lower_wick',
                'is_doji', 'is_hammer', 'is_engulfing', 'gap_up', 'gap_down',
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'roc',
                'ema_9', 'ema_9_dist', 'ema_21', 'ema_21_dist', 'ema_50', 'ema_50_dist', 
                'ema_200', 'ema_200_dist',
                'sma_20', 'sma_50',
                'adx', 'plus_di', 'minus_di',
                'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                'atr', 'atr_pct', 'volatility',
                'volume_sma', 'volume_ratio', 'obv', 'obv_ema', 'vpt',
                'momentum_5', 'momentum_10', 'momentum_20', 'acceleration', 'relative_spread'
            ]
            def get_feature_default(name):
                if name.startswith('is_'):
                    return 0
                elif name == 'rsi':
                    return 50
                elif name in ['volatility', 'atr_pct']:
                    return 0.01
                return 0.0
        
        # Build feature vector
        feature_values = []
        for f in features:
            val = feat.get(f, get_feature_default(f))
            feature_values.append(float(val))
        
        # Check for PyTorch model
        if self.pytorch_model is not None and TORCH_AVAILABLE:
            try:
                # Convert to tensor
                X_tensor = torch.FloatTensor([feature_values])  # [1, 49]
                
                # Scale if scaler available
                expected_dim = self.scaler.n_features_in_ if self.scaler else len(features)
                if self.scaler and expected_dim != len(features):
                    self.logger.w(f"Scaler expects {expected_dim} features but got {len(features)}. Bypassing scaler.")
                    self.scaler = None
                
                if self.scaler:
                    X_scaled = self.scaler.transform([feature_values])
                    X_tensor = torch.FloatTensor(X_scaled)
                
                # Forward pass
                with torch.no_grad():
                    logits = self.pytorch_model(X_tensor)  # [1, 3]
                    probs = torch.softmax(logits, dim=1)
                    class_pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, class_pred].item()
                
                # Map class to action
                actions = ["SELL", "HOLD", "BUY"]
                act = actions[class_pred]
                
                self.logger.i(f"{sym} → {act} (TFT, conf={confidence:.3f})")
                return {
                    "symbol": sym,
                    "action": act,
                    "confidence": float(confidence),
                    "confidence_std": 0.1,
                    "version": self.version
                }
                
            except Exception as e:
                self.logger.e(f"TFT prediction failed: {e}")
        
        # Fallback
        self.logger.w(f"{sym} → HOLD (dummy fallback, no TFT model)")
        return {
            "symbol": sym,
            "action": "HOLD",
            "confidence": 0.5,
            "confidence_std": 0.1,
            "version": self.version
        }


# Backward compatibility
XGBAgent = XGBoostAgent
