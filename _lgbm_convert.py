import lightgbm as lgb
import pickle

src = "/opt/quantum/ai_engine/models/lgbm_v6_20260304_225622.txt"
dst = "/opt/quantum/ai_engine/models/lgbm_v6_20260304_225622.pkl"
symlink = "/opt/quantum/ai_engine/models/lgbm_model.pkl"

m = lgb.Booster(model_file=src)
with open(dst, "wb") as f:
    pickle.dump(m, f, protocol=2)

import os
if os.path.islink(symlink) or os.path.exists(symlink):
    os.remove(symlink)
os.symlink("lgbm_v6_20260304_225622.pkl", symlink)

print("LGBM_PKL_OK trees=%d" % m.num_trees())
print("symlink: lgbm_model.pkl -> lgbm_v6_20260304_225622.pkl")
