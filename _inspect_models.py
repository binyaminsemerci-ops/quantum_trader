import sys, torch, json
sys.path.insert(0, '/home/qt/quantum_trader')

def inspect(name, path):
    print(f"\n=== {name} ===")
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"  LOAD ERROR: {e}")
        return

    print(f"  type: {type(obj)}")
    if isinstance(obj, dict):
        print(f"  top-level keys: {list(obj.keys())}")
        if 'model_state_dict' in obj:
            sd = obj['model_state_dict']
            for k, v in {k: v for k, v in obj.items() if k != 'model_state_dict'}.items():
                print(f"  meta [{k}]: {v}")
        else:
            sd = obj
        keys = list(sd.keys())
        print(f"  state_dict keys ({len(keys)} total, first 20):")
        for k in keys[:20]:
            print(f"    {k}: {sd[k].shape}")
    else:
        print(f"  (not a dict, type={type(obj)})")

inspect("NHiTS", "/home/qt/quantum_trader/ai_engine/models/nhits_v20260224_103403_v2.pth")
inspect("PatchTST", "/home/qt/quantum_trader/ai_engine/models/patchtst_v3_20260217_003223.pth")

# Also check /opt/quantum newer versions
import os
for label, path in [
    ("NHiTS-v7 (opt)", "/opt/quantum/ai_engine/models/nhits_v7_20260305_035242.pth"),
    ("PatchTST-v7 (opt)", "/opt/quantum/ai_engine/models/patchtst_v7_20260305_025903.pth"),
]:
    if os.path.exists(path):
        inspect(label, path)
    else:
        print(f"\n  {label}: NOT FOUND at {path}")
