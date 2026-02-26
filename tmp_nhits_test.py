import torch, numpy as np, sys
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.nhits_simple import SimpleNHiTS

# Load old nhits_model.pth
ck = torch.load('/app/models/nhits_model.pth', map_location='cpu', weights_only=False)
model = SimpleNHiTS(input_size=120, hidden_size=256, num_features=12)
model.load_state_dict(ck['model_state_dict'])
model.eval()

print('=== nhits_model.pth (12 features) ===')
# Test with zeros
x = torch.zeros(1, 120, 12)
with torch.no_grad():
    logits, _ = model(x)
    probs = torch.softmax(logits, dim=1)[0]
print('probs(zeros):', probs.numpy().round(4))
print('class:', torch.argmax(probs).item(), '-> action:', {0:'SELL',1:'HOLD',2:'BUY'}[torch.argmax(probs).item()])

# 100 random inputs
actions = []
for _ in range(100):
    x = torch.randn(1, 120, 12)
    with torch.no_grad():
        logits, _ = model(x)
        p = torch.softmax(logits, dim=1)[0]
    actions.append(torch.argmax(p).item())

from collections import Counter
cnt = Counter(actions)
print('distribution (100 random):', {str(k)+' '+{0:"SELL",1:"HOLD",2:"BUY"}[k]: v for k,v in sorted(cnt.items())})
