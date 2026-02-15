#!/usr/bin/env python3
import sys

# Read the file
with open('/opt/quantum/ai_engine/agents/unified_agents.py', 'r') as f:
    content = f.read()

# Check if already exists
if 'PyTorchClassifierWrapper' in content:
    print('PyTorchClassifierWrapper already exists in unified_agents.py')
    sys.exit(0)

# Add wrapper class after the imports
wrapper_code = '''
# ---------- PYTORCH WRAPPER (for NHiTS/PatchTST compatibility) ----------
import torch
import torch.nn as nn

class PyTorchClassifierWrapper:
    """Wrapper that gives PyTorch model a sklearn-like .predict() interface"""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, X):
        """Returns class predictions (0, 1, 2)"""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            outputs = self.model(X)
            preds = outputs.argmax(dim=1).cpu().numpy()
        return preds
    
    def predict_proba(self, X):
        """Returns probability distribution over classes"""
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            X = X.to(self.device)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs


# NHiTS model architecture
class NHiTS(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, num_stacks=4, num_classes=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.stacks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for i in range(num_stacks)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        for i, stack in enumerate(self.stacks):
            if i == 0:
                x = stack(x)
            else:
                x = stack(x) + x
        return self.fc(x)


# PatchTST model architecture
class PatchTST(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)

'''

# Insert after 'from pathlib import Path'
insert_marker = 'from pathlib import Path'
if insert_marker in content:
    content = content.replace(insert_marker, insert_marker + wrapper_code)
    with open('/opt/quantum/ai_engine/agents/unified_agents.py', 'w') as f:
        f.write(content)
    print('Added PyTorchClassifierWrapper and model classes to unified_agents.py')
else:
    print('Could not find insertion point!')
    sys.exit(1)
