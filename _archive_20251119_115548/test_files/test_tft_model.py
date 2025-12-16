"""
Quick test to verify TFT model loads correctly
"""
import torch
from ai_engine.tft_model import TemporalFusionTransformer

print("Loading TFT model...")
model = TemporalFusionTransformer(
    input_size=14,
    hidden_size=128,
    num_layers=3,
    num_heads=8,
    num_classes=3,
    dropout=0.1
)

checkpoint = torch.load('ai_engine/models/tft_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("[OK] Model loaded successfully!")
print(f"[CHART] Epoch: {checkpoint['epoch']}")
print(f"[TARGET] Best accuracy: {checkpoint['best_accuracy']*100:.2f}%")
print(f"📏 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with dummy input
import numpy as np
dummy_input = torch.randn(1, 60, 14)  # batch, seq_len, features
with torch.no_grad():
    logits, quantiles, attention = model(dummy_input)
    predictions = torch.argmax(logits, dim=1)
    
print(f"[OK] Inference test passed!")
print(f"   Output shape: {logits.shape}")
print(f"   Prediction: {predictions.item()} (0=SELL, 1=HOLD, 2=BUY)")
