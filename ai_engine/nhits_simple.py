"""
Simplified N-HiTS model for debugging NaN issues.
Removes complex multi-rate stacking, focuses on basic MLP.
"""
import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleNHiTS(nn.Module):
    """
    Simplified N-HiTS without multi-rate stacking.
    Just a basic MLP for classification from sequences.
    """
    
    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 256,
        num_features: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_features = num_features
        
        logger.info("🏗️ Building SIMPLE N-HiTS:")
        logger.info(f"   Input: {input_size} × {num_features}")
        logger.info(f"   Hidden: {hidden_size}")
        
        # Flatten sequence
        self.flatten = nn.Flatten()
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size * num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 3)  # BUY/HOLD/SELL
        )
        
        # Conservative init
        self.apply(self._init_weights)
        
        logger.info("[OK] Simple N-HiTS complete!")
    
    def _init_weights(self, module):
        """Conservative initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, num_features]
        
        Returns:
            logits: [batch, 3]
            dummy_forecast: [batch, 1] (for compatibility)
        """
        # Flatten
        x_flat = self.flatten(x)  # [batch, seq_len * num_features]
        
        # MLP
        logits = self.mlp(x_flat)
        
        # Dummy forecast for compatibility
        dummy_forecast = logits[:, :1]  # Just take first logit
        
        return logits, dummy_forecast


# Alias for compatibility
NHiTS = SimpleNHiTS
