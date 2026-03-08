#!/usr/bin/env python3
"""Build train_nhits_v7.py from v6, replacing model architecture to match unified_agents.py"""
import re

with open('/opt/quantum/ops/retrain/train_nhits_v6.py') as f:
    src = f.read()

new_model_block = '''\
class NHiTSModel(nn.Module):
    """Exact architecture from unified_agents.py: blocks + output_layer."""
    def __init__(self, input_dim=49, hidden_dim=256, output_dim=3, n_blocks=3,
                 mlp_units=None, dropout=0.1):
        super().__init__()
        if mlp_units is None:
            mlp_units = [256, 256]
        self.mlp_units = mlp_units
        self.blocks = nn.ModuleList([
            self._mk_block(input_dim, output_dim, mlp_units, dropout)
            for _ in range(n_blocks)])
        self.output_layer = nn.Linear(output_dim * n_blocks, output_dim)

    def _mk_block(self, input_dim, output_dim, mlp_units, dropout):
        layers = [nn.Linear(input_dim, mlp_units[0]), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(len(mlp_units) - 1):
            layers += [nn.Linear(mlp_units[i], mlp_units[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(mlp_units[-1], output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.output_layer(torch.cat([b(x) for b in self.blocks], dim=-1))

'''

# Replace NHiTSModel class block up to the instantiation line
src = re.sub(
    r'class NHiTSModel\(nn\.Module\):.*?(?=\nmodel\s*=)',
    new_model_block,
    src,
    flags=re.DOTALL
)

# Fix model instantiation
src = re.sub(
    r'model\s*=\s*NHiTSModel\([^)]+\)\.to\(DEVICE\)',
    'model = NHiTSModel(input_dim=input_dim, hidden_dim=256, output_dim=3, n_blocks=3, mlp_units=[256, 256]).to(DEVICE)',
    src
)

# Fix checkpoint: update hidden_dim and add mlp_units key
src = src.replace(
    "    'hidden_dim': 128,",
    "    'hidden_dim': 256,\n    'mlp_units': [256, 256],"
)

# Update version strings
src = src.replace('nhits_v6_', 'nhits_v7_')
src = src.replace('NHiTS v6', 'NHiTS v7')

with open('/opt/quantum/ops/retrain/train_nhits_v7.py', 'w') as f:
    f.write(src)

print('train_nhits_v7.py written OK')
print('output_layer present:', 'self.output_layer' in src)
print('mlp_units in checkpoint:', "'mlp_units': [256, 256]" in src)
print('old head gone:', 'self.head' not in src[src.find('class NHiTSModel'):src.find('model = NHiTSModel')])
print('instantiation uses mlp_units:', 'mlp_units=[256, 256]' in src)
