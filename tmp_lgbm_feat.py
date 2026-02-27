import joblib
m = joblib.load('/app/models/lgbm_model.pkl')
print('type:', type(m).__name__)
print('n_features:', m.n_features_in_)
fn = getattr(m, 'feature_name_', None)
if fn:
    print('feature_names:')
    for i,f in enumerate(fn): print(i, f)
else:
    print('no feature_name_ attr')
