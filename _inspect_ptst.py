import torch, sys
m = torch.load('/home/qt/quantum_trader/ai_engine/models/patchtst_v3_20260217_003223.pth',
               map_location='cpu', weights_only=False)
sd = m['model_state_dict']
print("=== PatchTST v3 ALL KEYS ===")
for k, v in sd.items():
    print(f"  {k}: {v.shape}")
print(f"\n  Total keys: {len(sd)}")
print(f"\n  model_config: {m.get('model_config')}")
