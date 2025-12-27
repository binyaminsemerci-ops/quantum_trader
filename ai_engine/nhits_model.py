"""
N-HiTS (Neural Hierarchical Interpolation for Time Series)
State-of-the-art model from Nixtla (2022)

Key features:
- Multi-rate sampling (sees both short and long-term patterns)
- Hierarchical interpolation
- Much faster than TFT
- Better accuracy on volatile time series (perfect for crypto!)

Paper: https://arxiv.org/abs/2201.12886
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class NHiTSBlock(nn.Module):
    """
    Single N-HiTS block with multi-layer perceptron and basis expansion.
    
    Each block operates at a different temporal resolution:
    - Stack 1: Short-term (high frequency)
    - Stack 2: Medium-term  
    - Stack 3: Long-term (low frequency)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.1,
        pool_kernel_size: int = 1
    ):
        super().__init__()
        
        self.input_size = input_size
        self.horizon = horizon
        self.pool_kernel_size = pool_kernel_size
        
        # Pooling for multi-rate sampling
        if pool_kernel_size > 1:
            self.pool = nn.MaxPool1d(
                kernel_size=pool_kernel_size,
                stride=pool_kernel_size,
                ceil_mode=True
            )
            pooled_size = (input_size + pool_kernel_size - 1) // pool_kernel_size
        else:
            self.pool = None
            pooled_size = input_size
        
        # MLP for feature extraction
        layers = []
        prev_size = pooled_size
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if i < num_layers - 1:  # No dropout on last layer
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.mlp = nn.Sequential(*layers)
        
        # Backcast and forecast heads
        self.backcast_head = nn.Linear(hidden_size, input_size)
        self.forecast_head = nn.Linear(hidden_size, horizon)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq_len]
        
        Returns:
            backcast: [batch, seq_len] - Reconstruction of input
            forecast: [batch, horizon] - Prediction for future
        """
        # Multi-rate pooling
        if self.pool is not None:
            # x: [batch, seq_len] -> [batch, 1, seq_len]
            x_pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        else:
            x_pooled = x
        
        # MLP
        h = self.mlp(x_pooled)
        
        # Backcast and forecast
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        
        return backcast, forecast


class NHiTSStack(nn.Module):
    """
    Stack of N-HiTS blocks at same temporal resolution.
    
    Uses doubly residual stacking:
    - Residual connections on backcast (input reconstruction)
    - Accumulation on forecast (predictions)
    """
    
    def __init__(
        self,
        num_blocks: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float = 0.1,
        pool_kernel_size: int = 1
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                horizon=horizon,
                dropout=dropout,
                pool_kernel_size=pool_kernel_size
            )
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq_len]
        
        Returns:
            residual: [batch, seq_len] - What remains after this stack
            forecast: [batch, horizon] - Cumulative forecast from this stack
        """
        residual = x
        forecast = torch.zeros(x.size(0), self.blocks[0].horizon, device=x.device)
        
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast  # Doubly residual
            forecast = forecast + block_forecast  # Accumulate forecasts
        
        return residual, forecast


class NHiTS(nn.Module):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series
    
    Multi-stack architecture:
    - Stack 1: High frequency (short-term patterns, pool=1)
    - Stack 2: Medium frequency (medium-term patterns, pool=2)  
    - Stack 3: Low frequency (long-term trends, pool=4)
    
    Each stack sees the data at different resolution!
    """
    
    def __init__(
        self,
        input_size: int = 120,
        hidden_size: int = 512,
        num_blocks_per_stack: int = 3,
        num_layers_per_block: int = 2,
        horizon: int = 1,
        dropout: float = 0.1,
        num_features: int = 14
    ):
        super().__init__()
        
        self.input_size = input_size
        self.horizon = horizon
        self.num_features = num_features
        
        logger.info("🏗️ Building N-HiTS Architecture:")
        logger.info(f"   Input: {input_size} timesteps × {num_features} features")
        logger.info(f"   Hidden: {hidden_size} units")
        logger.info(f"   Stacks: 3 (multi-rate)")
        logger.info(f"   Blocks per stack: {num_blocks_per_stack}")
        
        # Feature embedding
        self.feature_embed = nn.Linear(num_features, 1)
        
        # Stack 1: High frequency (no pooling)
        self.stack1 = NHiTSStack(
            num_blocks=num_blocks_per_stack,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers_per_block,
            horizon=horizon,
            dropout=dropout,
            pool_kernel_size=1  # No pooling - sees all detail
        )
        
        # Stack 2: Medium frequency (pool by 2)
        self.stack2 = NHiTSStack(
            num_blocks=num_blocks_per_stack,
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers_per_block,
            horizon=horizon,
            dropout=dropout,
            pool_kernel_size=2  # 2x pooling - sees medium patterns
        )
        
        # Stack 3: Low frequency (pool by 4)
        self.stack3 = NHiTSStack(
            num_blocks=num_blocks_per_stack,
            input_size=input_size,
            hidden_size=hidden_size // 4,
            num_layers=num_layers_per_block,
            horizon=horizon,
            dropout=dropout,
            pool_kernel_size=4  # 4x pooling - sees long-term trends
        )
        
        # Classification head (BUY/SELL/HOLD)
        self.classifier = nn.Sequential(
            nn.Linear(horizon, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # 3 classes
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info("[OK] N-HiTS Architecture complete!")
    
    def _init_weights(self, module):
        """Xavier initialization with conservative scaling."""
        if isinstance(module, nn.Linear):
            # Use smaller init to prevent gradient explosion
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq_len, num_features]
        
        Returns:
            logits: [batch, 3] - Class probabilities
            forecast: [batch, horizon] - Raw forecast values
        """
        # Clamp input to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Embed features: [batch, seq_len, features] -> [batch, seq_len]
        x_embed = self.feature_embed(x).squeeze(-1)
        
        # Clamp embedding output
        x_embed = torch.clamp(x_embed, min=-10.0, max=10.0)
        
        # Stack 1: High frequency
        residual1, forecast1 = self.stack1(x_embed)
        
        # Stack 2: Medium frequency
        residual2, forecast2 = self.stack2(residual1)
        
        # Stack 3: Low frequency
        _, forecast3 = self.stack3(residual2)
        
        # Combine forecasts from all stacks
        forecast = forecast1 + forecast2 + forecast3
        
        # Classification
        logits = self.classifier(forecast)
        
        return logits, forecast


def save_model(
    model: nn.Module,
    path: str,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None
):
    """Save N-HiTS model with normalization stats."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': getattr(model, 'input_size', 120),
        'horizon': getattr(model, 'horizon', 1),  # Default for SimpleNHiTS
        'num_features': model.num_features,
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }
    torch.save(checkpoint, path)
    logger.info(f"[OK] Model saved to {path}")


def load_model(path: str, device: str = 'cpu') -> Tuple[NHiTS, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load N-HiTS model with normalization stats."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = NHiTS(
        input_size=checkpoint['input_size'],
        horizon=checkpoint['horizon'],
        num_features=checkpoint['num_features']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')
    
    logger.info(f"[OK] Model loaded from {path}")
    
    return model, feature_mean, feature_std


class NHiTSTrainer:
    """Trainer for N-HiTS model."""
    
    def __init__(self, model: NHiTS, device: str = 'cpu', learning_rate: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("[TARGET] N-HiTS Trainer initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Learning rate: {learning_rate}")
    
    def train_epoch(self, train_loader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits, _ = self.model(sequences)
            loss = self.criterion(logits, targets.long())
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"   Batch {batch_idx}/{num_batches}: Loss={loss.item():.4f}")
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                logits, _ = self.model(sequences)
                loss = self.criterion(logits, targets.long())
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
