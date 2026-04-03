import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pickle, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.optimize import nnls
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

TARGETS      = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S']
LOG_TARGETS  = ['B', 'Cu', 'Zn', 'Mn', 'S']
CV_SPLITS    = 5

def get_base_learners(target):
    xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, 
                       subsample=0.8, colsample_bytree=0.8, 
                       verbosity=0, tree_method='hist')
    cat = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, 
                            verbose=0)
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=2, n_jobs=-1)
    return [('xgb',xgb), ('cat',cat), ('rf',rf)]

def nnls_meta_weights(oof_preds_dict, y_true):
    keys = list(oof_preds_dict.keys())
    A = np.column_stack([oof_preds_dict[k] for k in keys])
    w, _ = nnls(A, y_true)
    w /= (w.sum() + 1e-12)
    return dict(zip(keys, w))

def train_target(target, X, y, clusters):
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    folds = list(skf.split(X, clusters))
    learners = get_base_learners(target)

    oof_preds = {}
    for name, est in learners:
        oof_preds[name] = cross_val_predict(est, X, y, cv=folds, n_jobs=1)
    weights = nnls_meta_weights(oof_preds, y)
    print(f"  [{target}] NNLS weights: { {k:round(v,3) for k,v in weights.items()} }", flush=True)

    xgb_best = [e for n,e in learners if n=='xgb'][0]
    selector  = SelectFromModel(xgb_best, max_features=400, threshold='median', prefit=False)
    selector.fit(X, y)
    X_sel = selector.transform(X)

    fitted = {}
    for name, est in learners:
        est.fit(X_sel, y)
        fitted[name] = est

    return fitted, weights, selector

if __name__ == '__main__':
    X        = np.load('cache/X_full.npy')
    clusters = np.load('cache/clusters.npy')
    with open('cache/Y_log.pkl','rb') as f: Y_log = pickle.load(f)

    ensemble = {}
    for t in TARGETS:
        print(f"\nTraining [{t}]...", flush=True)
        fitted, weights, selector = train_target(t, X, Y_log[t], clusters)
        ensemble[t] = {'models': fitted, 'weights': weights, 'selector': selector}

    os.makedirs('models/', exist_ok=True)
    with open('models/ensemble.pkl','wb') as f: pickle.dump(ensemble, f)
    print("\nEnsemble saved -> models/ensemble.pkl")
