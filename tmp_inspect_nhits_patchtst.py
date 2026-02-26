import torch, numpy as np, sys
sys.path.insert(0, "/home/qt/quantum_trader")

print("=== NHiTS state_dict keys ===")
ck = torch.load("/app/models/nhits_v20260224_103403_v2.pth", map_location="cpu", weights_only=False)
for k,v in ck["model_state_dict"].items():
    print(f"  {k}: {tuple(v.shape)}")
print("val_accuracy:", ck.get("val_accuracy"))
fmean = np.array(ck["feature_mean"]) if ck.get("feature_mean") is not None else None
fstd  = np.array(ck["feature_std"])  if ck.get("feature_std")  is not None else None

print()
print("=== PatchTST state_dict (first 5 keys) ===")
ck2 = torch.load("/app/models/patchtst_v3_20260217_003223.pth", map_location="cpu", weights_only=False)
cfg = ck2.get("model_config", {})
print("model_config:", cfg)
sm = ck2.get("scaler_mean"); sv = ck2.get("scaler_var")
if sm is not None: print("scaler built-in: YES, n_features =", len(np.array(sm)))
keys_sd = list(ck2.get("model_state_dict", {}).keys())
print("First 8 state_dict keys:", keys_sd[:8])
print("test_accuracy:", ck2.get("test_accuracy"))

print("=== NHiTS detailed ===")
ck = torch.load("/app/models/nhits_v20260224_103403_v2.pth", map_location="cpu", weights_only=False)
print("val_accuracy:", ck.get("val_accuracy", "MISSING"))
print("trained_at:", ck.get("trained_at", "MISSING"))
fmean = np.array(ck["feature_mean"]) if ck.get("feature_mean") is not None else None
fstd  = np.array(ck["feature_std"])  if ck.get("feature_std")  is not None else None
if fmean is not None:
    print("feature_mean[:3]:", fmean[:3].round(4), "  feature_std[:3]:", fstd[:3].round(4))

import torch.nn as nn
class SimpleNHiTS(nn.Module):
    def __init__(self, input_size, hidden_size, num_features, num_stacks=3, num_blocks=2):
        super().__init__(); self.num_features = num_features; self.input_size = input_size
        self.blocks = nn.ModuleList(); in_dim = input_size * num_features
        for _ in range(num_stacks * num_blocks):
            self.blocks.append(nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, hidden_size), nn.ReLU()))
            in_dim = hidden_size
        self.classifier = nn.Linear(hidden_size, 3)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for b in self.blocks: x = b(x)
        return self.classifier(x), None

model = SimpleNHiTS(120, 256, 49); model.load_state_dict(ck["model_state_dict"]); model.eval()

np.random.seed(42); cnts = [0,0,0]
for _ in range(1000):
    X = torch.FloatTensor(np.random.randn(120, 49).astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(X)[0], dim=1)[0]
        cnts[p.argmax().item()] += 1
print(f"NHiTS random N(0,1) x1000: SELL={cnts[0]} HOLD={cnts[1]} BUY={cnts[2]}")

z = torch.zeros(1,120,49)
with torch.no_grad():
    p = torch.softmax(model(z)[0], dim=1)[0]
    print(f"NHiTS zeros: SELL={p[0]:.3f} HOLD={p[1]:.3f} BUY={p[2]:.3f}")

if fmean is not None:
    cnts2=[0,0,0]
    for _ in range(500):
        raw = (fmean + np.random.randn(120,49).astype(np.float32)*fstd)
        norm = (raw - fmean)/(fstd+1e-8)
        X = torch.FloatTensor(norm).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(X)[0], dim=1)[0]
            cnts2[p.argmax().item()] += 1
    print(f"NHiTS normalized around training mean x500: SELL={cnts2[0]} HOLD={cnts2[1]} BUY={cnts2[2]}")

print()
print("=== PatchTST detailed ===")
ck2 = torch.load("/app/models/patchtst_v3_20260217_003223.pth", map_location="cpu", weights_only=False)
cfg = ck2.get("model_config", {})
print("model_config:", cfg)
sm = ck2.get("scaler_mean"); sv = ck2.get("scaler_var")
if sm is not None: print("scaler_mean[:3]:", np.array(sm)[:3].round(4), "  scaler_var[:3]:", np.array(sv)[:3].round(4))
print("test_accuracy:", ck2.get("test_accuracy"))
