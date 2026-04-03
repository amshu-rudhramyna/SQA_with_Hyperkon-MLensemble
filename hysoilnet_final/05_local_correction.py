import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pickle, lightgbm as lgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

TARGETS     = ['B', 'Fe', 'Zn', 'Cu', 'Mn', 'S']
LOG_TARGETS = ['B', 'Cu', 'Zn', 'Mn', 'S']
K_NEIGHBORS = 30

def global_predict(ensemble, X, target):
    rec = ensemble[target]
    X_sel = rec['selector'].transform(X)
    preds = np.zeros(len(X))
    for name, model in rec['models'].items():
        preds += rec['weights'][name] * model.predict(X_sel)
    return preds

def local_lgbm_residual(X_train, residuals, X_test, k=K_NEIGHBORS):
    sim       = cosine_similarity(X_test, X_train)
    corrected = np.zeros(len(X_test))
    for i in range(len(X_test)):
        nn_idx = np.argsort(sim[i])[-k:]
        lgb_m  = lgb.LGBMRegressor(n_estimators=100, num_leaves=15,
                                     learning_rate=0.05, verbose=-1, n_jobs=-1)
        lgb_m.fit(X_train[nn_idx], residuals[nn_idx])
        corrected[i] = lgb_m.predict(X_test[[i]])[0]
    return corrected

if __name__ == '__main__':
    X        = np.load('cache/X_full.npy')
    clusters = np.load('cache/clusters.npy')
    with open('cache/Y_log.pkl','rb') as f: Y_log = pickle.load(f)
    with open('models/ensemble.pkl','rb') as f: ensemble = pickle.load(f)

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    correctors = {}

    for t in TARGETS:
        print(f"Fitting local corrector [{t}]...", flush=True)
        y = Y_log[t]
        oof_global    = np.zeros_like(y)
        oof_corrected = np.zeros_like(y)

        for tr_idx, val_idx in skf.split(X, clusters):
            g_pred_val   = global_predict(ensemble, X[val_idx], t)
            g_pred_tr    = global_predict(ensemble, X[tr_idx], t)
            residuals_tr = y[tr_idx] - g_pred_tr
            correction   = local_lgbm_residual(X[tr_idx], residuals_tr, X[val_idx])
            oof_global[val_idx]    = g_pred_val
            oof_corrected[val_idx] = g_pred_val + correction

        g_pred_full    = global_predict(ensemble, X, t)
        residuals_full = y - g_pred_full
        correctors[t]  = {'X_train': X, 'residuals': residuals_full}
        print(f"  [{t}] global OOF -> corrected OOF saved", flush=True)

    os.makedirs('models/', exist_ok=True)
    with open('models/correctors.pkl','wb') as f: pickle.dump(correctors, f)
    print("Local correctors saved -> models/correctors.pkl")
