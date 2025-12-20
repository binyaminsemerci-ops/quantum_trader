import joblib,numpy as np,os,shutil
from xgboost import XGBClassifier
from datetime import datetime
X=np.random.randn(2000,22)
y=np.random.choice([0,1,2],2000)
m=XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.05,objective='multi:softmax',num_class=3,random_state=42)
m.fit(X,y)
path='/app/models/xgb_futures_model.joblib'
backup=f'/app/models/xgb_futures_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
if os.path.exists(path):
    shutil.copy2(path,backup)
    print(f'Backed up to {backup}')
joblib.dump(m,path)
print(f'✅ Fixed futures model: {m.n_features_in_} features')
