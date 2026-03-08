#!/usr/bin/env python3
"""Update ai-engine.env on VPS to include dlinear in ensemble model lists."""
src = '/tmp/ai_env.env'
dst = '/tmp/ai_env_new.env'

with open(src) as f:
    data = f.read()

data = data.replace(
    'ENABLED_MODELS=xgb,lgbm,nhits,patchtst,tft',
    'ENABLED_MODELS=xgb,lgbm,nhits,patchtst,tft,dlinear'
)

old = 'ENSEMBLE_MODELS=["xgb","lgbm","nhits","patchtst","tft"]'
new = 'ENSEMBLE_MODELS=["xgb","lgbm","nhits","patchtst","tft","dlinear"]'
data = data.replace(old, new)

with open(dst, 'w') as f:
    f.write(data)

# Verify
for line in data.splitlines():
    if 'ENABLED_MODELS' in line or 'ENSEMBLE_MODELS' in line:
        print(line)
print("OK")
