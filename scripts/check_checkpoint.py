import torch
import numpy as np

c = torch.load('ai_engine/models/tft_model.pth', weights_only=False)
print('Keys:', list(c.keys()))
print('Has feature_mean:', 'feature_mean' in c)
print('Has feature_std:', 'feature_std' in c)

if 'feature_mean' in c:
    print('\nfeature_mean shape:', c['feature_mean'].shape)
    print('feature_mean[0:5]:', c['feature_mean'][0:5])
    print('feature_std[0:5]:', c['feature_std'][0:5])
    print('\nfeature_mean min/max:', c['feature_mean'].min(), '/', c['feature_mean'].max())
    print('feature_std min/max:', c['feature_std'].min(), '/', c['feature_std'].max())
