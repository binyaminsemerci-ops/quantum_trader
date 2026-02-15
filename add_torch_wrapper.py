#!/usr/bin/env python3
"""Add TorchRegressorWrapper and SimpleMLP to unified_agents.py"""
import sys

with open('/opt/quantum/ai_engine/agents/unified_agents.py', 'r') as f:
    content = f.read()

# Check if already exists
if 'TorchRegressorWrapper' in content:
    print('TorchRegressorWrapper already exists')
    sys.exit(0)

# New code to add
new_code = '''

# ---------- SIMPLE TORCH REGRESSOR (for NHiTS/PatchTST v8) ----------
class SimpleMLP(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchRegressorWrapper:
    """Wrapper with .predict() interface for PyTorch regression models"""
    def __init__(self, state_dict, config):
        self.state_dict = state_dict
        self.config = config
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            self._model = SimpleMLP(
                input_size=self.config.get('input_size', 18),
                hidden_size=self.config.get('hidden_size', 128),
                output_size=1
            )
            self._model.load_state_dict(self.state_dict)
            self._model.eval()
        return self._model
    
    def predict(self, X):
        """Returns PnL% predictions"""
        model = self._get_model()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            outputs = model(X).numpy()
        return outputs

'''

# Find insertion point after existing PyTorchClassifierWrapper or after imports
if 'PyTorchClassifierWrapper' in content:
    # Insert after PyTorchClassifierWrapper class
    marker = "class PyTorchClassifierWrapper:"
    # Find end of that class
    idx = content.find(marker)
    if idx != -1:
        # Find next class definition
        next_class = content.find("\nclass ", idx + len(marker))
        if next_class != -1:
            content = content[:next_class] + new_code + content[next_class:]
        else:
            content += new_code
else:
    # Insert after torch imports
    marker = "import torch.nn as nn"
    if marker in content:
        content = content.replace(marker, marker + new_code)
    else:
        # Insert at end
        content += new_code

with open('/opt/quantum/ai_engine/agents/unified_agents.py', 'w') as f:
    f.write(content)
print('Added TorchRegressorWrapper and SimpleMLP to unified_agents.py')
