import torch, os, numpy as np

for path in ['/app/models/patchtst_v3_20260217_003223.pth', '/app/models/patchtst_model.pth']:
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        continue
    ck = torch.load(path, map_location='cpu', weights_only=False)
    print(f'=== {os.path.basename(path)} ===')
    keys = list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__
    print('  keys:', keys)
    if isinstance(ck, dict):
        print('  num_classes:', ck.get('num_classes', ck.get('output_dim', '?')))
        print('  num_features:', ck.get('num_features', ck.get('input_dim', '?')))
        print('  accuracy:', ck.get('accuracy', ck.get('val_acc', 'N/A')))
        sd = ck.get('model_state_dict', ck)
        for k, v in list(sd.items())[:5]:
            print(f'  {k}: {tuple(v.shape)}')
    print()
