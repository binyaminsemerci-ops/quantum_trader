import torch, os

for path in ['/app/models/nhits_model.pth', '/app/models/nhits_v6_20260223_012955.pth']:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    print('=== ' + os.path.basename(path) + ' ===')
    print('  checkpoint keys:', list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__)
    for k, v in list(sd.items())[:10]:
        print('  ' + k + ': ' + str(tuple(v.shape)))
    print()
