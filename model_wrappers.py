"""
Model wrappers for PyTorch models to provide sklearn-like interface.
This module must exist on the AI Engine side for joblib to unpickle properly.
"""
import numpy as np
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple MLP for regression (predicts PnL%)"""
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
    """
    Wrapper that gives PyTorch model a sklearn-like .predict() interface.
    Stores state_dict and config so model can be reconstructed on load.
    """
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
        """Returns PnL% predictions - same interface as XGBoost regressor"""
        model = self._get_model()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            # Handle batch of 1
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            outputs = model(X).numpy()
        return outputs
