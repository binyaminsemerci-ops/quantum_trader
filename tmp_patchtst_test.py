import torch, numpy as np, sys, json
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.agents.patchtst_agent_v3 import SimplePatchTST

ck = torch.load('/app/models/patchtst_v3_20260217_003223.pth', map_location='cpu', weights_only=False)
meta_path = '/app/models/patchtst_v3_20260217_003223_metadata.json'

import os
if os.path.exists(meta_path):
    meta = json.load(open(meta_path))
    print('meta keys:', list(meta.keys()))
    num_features = len(meta.get('features', [])) or 49
    print('num_features from meta:', num_features)
    acc = meta.get('test_accuracy', meta.get('accuracy', 'N/A'))
    print('accuracy:', acc)
else:
    print('no meta found, using 49')
    num_features = 49

model = SimplePatchTST(num_features=num_features)
model.load_state_dict(ck['model_state_dict'])
model.eval()
print('Model loaded OK')

# Test with zeros
x = torch.zeros(1, num_features)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
print('probs(zeros):', probs.numpy().round(4))
print('action:', ['SELL','HOLD','BUY'][torch.argmax(probs).item()])

# Distribution over 100 random inputs
actions = []
for _ in range(100):
    x = torch.randn(1, num_features)
    with torch.no_grad():
        logits = model(x)
        p = torch.softmax(logits, dim=1)[0]
    actions.append(torch.argmax(p).item())

from collections import Counter
cnt = Counter(actions)
print('distribution (100 random):', {str(k)+' '+['SELL','HOLD','BUY'][k]: v for k,v in sorted(cnt.items())})
mean_probs = np.array(all_probs).mean(axis=0)
print('mean probs 100 random [SELL, HOLD, BUY]:', mean_probs.round(4))
from collections import Counter
cnt = Counter(actions)
print('class distribution:', {['SELL','HOLD','BUY'][k]: v for k,v in sorted(cnt.items())})
