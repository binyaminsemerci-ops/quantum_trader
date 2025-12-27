import os
import json
import time
import logging
import torch
import numpy as np
import redis
from datetime import datetime, timedelta
from torch import nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Configuration
MEMORY_PATH = "/app/policy_memory"
os.makedirs(MEMORY_PATH, exist_ok=True)

FORECAST_INTERVAL = int(os.getenv("FORECAST_INTERVAL", 3600 * 6))  # 6 hours
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", 50))  # Minimum samples needed
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 32))  # Time steps
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", 5))

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL REGIME PREDICTOR (TRANSFORMER-BASED)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalRegimePredictor(nn.Module):
    """
    Transformer-based model for predicting market regimes from historical
    strategy performance patterns.
    
    Input: Sequence of strategy metrics (score, sharpe, drawdown, parameters)
    Output: Probability distribution over 4 regimes (bull, bear, volatile, neutral)
    """
    
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Encoder: Project input features to model dimension
        self.encoder = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(d_model // 2, 4)  # 4 regimes
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, input_dim)
        
        Returns:
            regime_probs: (batch_size, 4) - probabilities for each regime
        """
        # Encode input
        x = self.encoder(x)  # (batch, seq, d_model)
        
        # Apply transformer
        x = self.transformer(x)  # (batch, seq, d_model)
        
        # Take the last time step
        x = x[:, -1, :]  # (batch, d_model)
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Softmax for probabilities
        return torch.softmax(x, dim=-1)


class PolicyMemoryDataset(Dataset):
    """
    Dataset for training regime predictor from historical strategy data
    """
    
    def __init__(self, memory_data, sequence_length=32):
        """
        Args:
            memory_data: (num_samples, feature_dim) numpy array
            sequence_length: Number of time steps in each sequence
        """
        self.data = memory_data
        self.sequence_length = sequence_length
        
        # Create labels based on performance trends
        self.labels = self._create_labels(memory_data)
        
    def _create_labels(self, data):
        """
        Create regime labels based on performance patterns
        
        0: Bull (high returns, positive momentum)
        1: Bear (negative returns, high drawdown)
        2: Volatile (high variance, unstable)
        3: Neutral (stable, moderate returns)
        """
        labels = []
        
        for i in range(len(data)):
            # Extract key metrics
            score = data[i, 0]  # Composite score
            sharpe = data[i, 1]  # Sharpe ratio
            drawdown = data[i, 2]  # Drawdown
            
            # Regime classification logic
            if sharpe > 0.5 and drawdown < 15 and score > 2.0:
                label = 0  # Bull - good performance
            elif sharpe < -0.2 or drawdown > 30:
                label = 1  # Bear - poor performance
            elif drawdown > 20 or abs(score) > 5:
                label = 2  # Volatile - unstable
            else:
                label = 3  # Neutral - stable
            
            labels.append(label)
        
        return np.array(labels)
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        x = self.data[idx:idx + self.sequence_length]
        
        # Get label for next time step
        y = self.labels[idx + self.sequence_length]
        
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )


def load_memory_vector():
    """
    Load strategy memory from Phase 10's memory bank and convert to feature vectors
    
    Returns:
        np.array of shape (num_strategies, num_features)
    """
    try:
        # Get all strategy files sorted by timestamp
        files = sorted([f for f in os.listdir(MEMORY_PATH) if f.endswith(".json")])
        
        if not files:
            logging.warning("[QPM] No strategy files found in memory bank")
            return np.array([])
        
        # Load last 200 strategies (or all if fewer)
        files = files[-200:]
        
        vectors = []
        
        for filename in files:
            try:
                filepath = os.path.join(MEMORY_PATH, filename)
                with open(filepath, 'r') as f:
                    strategy = json.load(f)
                
                # Extract features
                # Feature vector: [score, sharpe, sortino, drawdown, risk_factor, 
                #                  momentum_sensitivity, mean_reversion, position_scaler]
                vector = [
                    strategy.get("score", strategy.get("composite_score", 0)),
                    strategy.get("sharpe", 0),
                    strategy.get("sortino", 0),
                    strategy.get("drawdown", 0),
                    strategy.get("risk_factor", 1.0),
                    strategy.get("momentum_sensitivity", 1.0),
                    strategy.get("mean_reversion", 0.5),
                    strategy.get("position_scaler", 1.0)
                ]
                
                vectors.append(vector)
                
            except Exception as e:
                logging.warning(f"[QPM] Could not load {filename}: {e}")
                continue
        
        if not vectors:
            logging.warning("[QPM] No valid strategy vectors loaded")
            return np.array([])
        
        memory_array = np.array(vectors, dtype=np.float32)
        logging.info(f"[QPM] Loaded {len(memory_array)} strategy vectors from memory bank")
        
        return memory_array
    
    except Exception as e:
        logging.error(f"[QPM] Error loading memory: {e}")
        return np.array([])


def regime_name(index):
    """Convert regime index to human-readable name"""
    regime_names = ["bull", "bear", "volatile", "neutral"]
    return regime_names[index]


def train_model(data, epochs=5):
    """
    Train the temporal regime predictor
    
    Args:
        data: (num_samples, feature_dim) numpy array
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    logging.info(f"[QPM] Training regime predictor with {len(data)} samples...")
    
    # Create model
    model = TemporalRegimePredictor(
        input_dim=data.shape[1],
        d_model=64,
        nhead=4,
        num_layers=3
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and dataloader
    dataset = PolicyMemoryDataset(data, sequence_length=SEQUENCE_LENGTH)
    
    if len(dataset) == 0:
        logging.warning("[QPM] Dataset is empty, cannot train")
        return model
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training loop
    model.train()
    total_loss = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        for x_batch, y_batch in loader:
            # Forward pass
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / max(batch_count, 1)
        total_loss += avg_loss
        
        logging.info(f"[QPM] Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
    
    avg_total_loss = total_loss / epochs
    logging.info(f"[QPM] Training complete - Average loss: {avg_total_loss:.4f}")
    
    return model


def forecast_regime(model, data):
    """
    Use trained model to forecast the next regime
    
    Args:
        model: Trained TemporalRegimePredictor
        data: Historical data array
    
    Returns:
        regime_name: str (bull, bear, volatile, neutral)
        probabilities: list of 4 floats
    """
    model.eval()
    
    with torch.no_grad():
        # Take last sequence_length samples
        if len(data) < SEQUENCE_LENGTH:
            # Pad if not enough data
            padding = np.zeros((SEQUENCE_LENGTH - len(data), data.shape[1]))
            x = np.vstack([padding, data])
        else:
            x = data[-SEQUENCE_LENGTH:]
        
        # Convert to tensor and add batch dimension
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)
        
        # Get predictions
        probs = model(x_tensor)
        probs_array = probs.squeeze().numpy()
        
        # Get predicted regime
        regime_idx = probs_array.argmax()
        regime = regime_name(regime_idx)
        
        return regime, probs_array.tolist()


def calculate_regime_confidence(probabilities):
    """
    Calculate confidence score for the predicted regime
    
    Returns confidence between 0 and 1
    """
    max_prob = max(probabilities)
    # Entropy-based confidence
    entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
    max_entropy = np.log(4)  # Maximum entropy for 4 classes
    
    # Normalize: higher confidence = lower entropy
    confidence = 1.0 - (entropy / max_entropy)
    
    return round(confidence * max_prob, 3)


def run_loop():
    """
    Main loop: periodically train model and forecast regime
    """
    logging.info("")
    logging.info("    ╔═══════════════════════════════════════════════════════════╗")
    logging.info("    ║  PHASE 11: QUANTUM POLICY MEMORY & REGIME FORECASTING     ║")
    logging.info("    ║  Status: ACTIVE                                           ║")
    logging.info("    ╚═══════════════════════════════════════════════════════════╝")
    logging.info("")
    logging.info(f"[QPM] Configuration:")
    logging.info(f"[QPM]   - Forecast Interval: {FORECAST_INTERVAL} seconds ({FORECAST_INTERVAL/3600:.1f} hours)")
    logging.info(f"[QPM]   - Minimum Samples: {MIN_SAMPLES}")
    logging.info(f"[QPM]   - Sequence Length: {SEQUENCE_LENGTH}")
    logging.info(f"[QPM]   - Training Epochs: {TRAINING_EPOCHS}")
    logging.info(f"[QPM]   - Memory Path: {MEMORY_PATH}")
    logging.info("")
    logging.info("[QPM] 🧠 Starting quantum policy memory and regime forecasting...")
    logging.info("")
    
    # Perform initial forecast
    logging.info("[QPM] Performing initial regime forecast...")
    
    while True:
        try:
            logging.info("[QPM] ═══════════════════════════════════════")
            logging.info("[QPM] Starting forecast cycle...")
            
            # Load memory from Phase 10's evolution memory bank
            mem_data = load_memory_vector()
            
            if len(mem_data) == 0:
                logging.warning("[QPM] Memory bank is empty, waiting for strategies...")
                time.sleep(1800)  # 30 minutes
                continue
            
            if len(mem_data) < MIN_SAMPLES:
                logging.info(f"[QPM] Insufficient samples ({len(mem_data)}/{MIN_SAMPLES}), waiting for more data...")
                time.sleep(1800)  # 30 minutes
                continue
            
            # Train model on historical data
            model = train_model(mem_data, epochs=TRAINING_EPOCHS)
            
            # Forecast next regime
            regime, probs = forecast_regime(model, mem_data)
            
            # Calculate confidence
            confidence = calculate_regime_confidence(probs)
            
            # Store forecast in Redis
            forecast_data = {
                "regime": regime,
                "bull": round(probs[0], 3),
                "bear": round(probs[1], 3),
                "volatile": round(probs[2], 3),
                "neutral": round(probs[3], 3),
                "confidence": confidence,
                "samples_used": len(mem_data),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            r.hset("quantum_regime_forecast", mapping=forecast_data)
            
            # Also store as JSON for history
            r.set("latest_regime_forecast", json.dumps(forecast_data))
            
            # Log results
            logging.info(f"[QPM] ╔════════════════════════════════════════╗")
            logging.info(f"[QPM] ║   REGIME FORECAST COMPLETE             ║")
            logging.info(f"[QPM] ╚════════════════════════════════════════╝")
            logging.info(f"[QPM] 🎯 Predicted Regime: {regime.upper()}")
            logging.info(f"[QPM] 📊 Probabilities:")
            logging.info(f"[QPM]    Bull:     {probs[0]:.3f} ({probs[0]*100:.1f}%)")
            logging.info(f"[QPM]    Bear:     {probs[1]:.3f} ({probs[1]*100:.1f}%)")
            logging.info(f"[QPM]    Volatile: {probs[2]:.3f} ({probs[2]*100:.1f}%)")
            logging.info(f"[QPM]    Neutral:  {probs[3]:.3f} ({probs[3]*100:.1f}%)")
            logging.info(f"[QPM] 🎲 Confidence: {confidence} ({confidence*100:.1f}%)")
            logging.info(f"[QPM] 📈 Samples: {len(mem_data)}")
            logging.info(f"[QPM] ═══════════════════════════════════════")
            logging.info("")
            
            # Store forecast history
            history_key = f"regime_forecast_history:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            r.setex(history_key, 7 * 24 * 3600, json.dumps(forecast_data))  # Keep for 7 days
            
            # Update statistics
            stats = {
                "total_forecasts": int(r.get("qpm_total_forecasts") or 0) + 1,
                "last_forecast_time": datetime.utcnow().isoformat(),
                "last_regime": regime,
                "last_confidence": confidence
            }
            r.set("qpm_stats", json.dumps(stats))
            r.incr("qpm_total_forecasts")
            
            logging.info(f"[QPM] Sleeping for {FORECAST_INTERVAL/3600:.1f} hours until next forecast...")
            logging.info("")
            time.sleep(FORECAST_INTERVAL)
            
        except Exception as e:
            logging.error(f"[QPM] Error in forecast loop: {e}")
            import traceback
            traceback.print_exc()
            logging.info("[QPM] Retrying in 20 minutes...")
            time.sleep(1200)


if __name__ == "__main__":
    run_loop()
